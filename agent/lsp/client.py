"""Async LSP client over stdin/stdout.

One :class:`LSPClient` corresponds to one ``(language_server, workspace_root)``
pair — exactly what OpenCode keys clients on, and the same shape Claude
Code uses.  The client owns a child process, drives the JSON-RPC
exchange, and exposes:

- :meth:`open_file` / :meth:`change_file` — text document sync
- :meth:`wait_for_diagnostics` — block until the server emits fresh
  diagnostics for a specific file (or a timeout fires)
- :meth:`diagnostics_for` — read the current per-file diagnostic store
- :meth:`shutdown` — graceful close + SIGTERM/SIGKILL fallback

The class is designed for async use from a single asyncio event loop.
The :class:`agent.lsp.manager.LSPService` runs an event loop in a
background thread so the synchronous file_operations layer can call
into it via :func:`agent.lsp.manager.LSPService.touch_file`.

Implementation notes:

- All per-document state lives in one :class:`_DocState` keyed by
  absolute path.  Freshness is tracked with **document versions**,
  not timestamps: every didChange bumps ``version``, and each stored
  push/pull result is tagged with the version it describes.  A
  result is fresh iff its tag >= the version being waited on, so a
  didChange implicitly invalidates everything older — no clearing,
  no clock comparisons, no race windows.  This is what prevents
  "ghost diagnostics": a slow server's leftovers from the previous
  edit can never masquerade as a verdict on the current content.

- Whole-document sync.  Even when the server advertises incremental
  sync, we send a single ``contentChanges`` entry replacing the
  entire document.  Pretending to be incremental while sending a
  full replacement is well-tolerated by every major server and saves
  range bookkeeping.  See OpenCode's ``client.ts:584-659`` for the
  same trick.

- The "touch-file dance": every ``open_file`` call also fires a
  ``workspace/didChangeWatchedFiles`` notification (CREATED on the
  first open, CHANGED thereafter).  Some servers (clangd, eslint)
  only re-scan when this notification fires, even though the LSP spec
  doesn't strictly require it.

- ``ContentModified`` (-32801) errors get retried with exponential
  backoff up to 3 times.  This matches Claude Code's
  ``LSPServerInstance.sendRequest``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from urllib.parse import quote, unquote, urlsplit

from agent.lsp.protocol import (
    ERROR_CONTENT_MODIFIED,
    ERROR_METHOD_NOT_FOUND,
    LSPProtocolError,
    LSPRequestError,
    classify_message,
    encode_message,
    make_error_response,
    make_notification,
    make_request,
    make_response,
    read_message,
)
from hermes_cli._subprocess_compat import windows_batch_command

logger = logging.getLogger("agent.lsp.client")

# Timeouts (seconds) — mirror OpenCode's constants, scaled to seconds.
INITIALIZE_TIMEOUT = 45.0
DIAGNOSTICS_DOCUMENT_WAIT = 5.0
DIAGNOSTICS_FULL_WAIT = 10.0
DIAGNOSTICS_REQUEST_TIMEOUT = 3.0
PUSH_DEBOUNCE = 0.15
SHUTDOWN_GRACE = 1.0  # seconds between SIGTERM and SIGKILL

# Retry policy for transient ContentModified errors.
MAX_CONTENT_MODIFIED_RETRIES = 3
RETRY_BASE_DELAY = 0.5  # 0.5, 1.0, 2.0 — exponential


def _absolute_path(path: str) -> str:
    """Return an absolute normalized path without changing its case."""
    return os.path.abspath(os.path.normpath(path))


def _path_key(path: str) -> str:
    """Return a stable key for per-file LSP state."""
    parsed = urlsplit(path)
    is_windows_drive = (
        len(parsed.scheme) == 1 and len(path) >= 2 and path[1] == ":"
    )
    if parsed.scheme and parsed.scheme.lower() != "file" and not is_windows_drive:
        # LSP also permits non-file document URIs such as ``untitled:``.
        # They are opaque identifiers, not local paths: making them absolute
        # or case-folding them changes their identity.
        return path
    if parsed.scheme.lower() == "file":
        path = uri_to_path(path)
    normalized = _absolute_path(path)
    return os.path.normcase(normalized) if os.name == "nt" else normalized


def file_uri(path: str) -> str:
    """Return ``file://`` URI for an absolute filesystem path.

    Mirrors Node's ``pathToFileURL`` — handles spaces, unicode, and
    Windows drive letters (``C:\\foo`` → ``file:///C:/foo``).
    """
    abs_path = _absolute_path(path)
    if os.name == "nt":
        # Windows: backslash → forward slash. UNC paths already start with
        # ``//`` and therefore carry their file-URI authority directly.
        abs_path = abs_path.replace("\\", "/")
        if abs_path.startswith("//"):
            return "file:" + quote(abs_path, safe="/:")
        if not abs_path.startswith("/"):
            abs_path = "/" + abs_path
    return "file://" + quote(abs_path, safe="/:")


def uri_to_path(uri: str) -> str:
    """Inverse of :func:`file_uri`, normalized for stable dictionary keys."""
    if not uri.startswith("file://"):
        return uri
    if os.name != "nt":
        return os.path.normpath(unquote(uri[len("file://"):]))

    parsed = urlsplit(uri)
    raw = unquote(parsed.path)
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        raw = f"//{parsed.netloc}{raw}"
    elif raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]  # strip leading slash before drive letter
    return _path_key(raw)


def _end_position(text: str) -> Dict[str, int]:
    """Return the LSP Position at the end of ``text``.

    Used to construct a single-range "replace whole document" change
    for ``textDocument/didChange`` regardless of the server's declared
    sync mode.
    """
    if not text:
        return {"line": 0, "character": 0}
    lines = text.splitlines(keepends=False)
    last_line = len(lines) - 1
    last_col = len(lines[-1]) if lines else 0
    # If the text ends with a trailing newline, ``splitlines`` won't
    # represent it.  The end position is then the start of the next
    # (empty) line — line index is len(lines), column 0.
    if text.endswith(("\n", "\r")):
        return {"line": last_line + 1, "character": 0}
    return {"line": last_line, "character": last_col}


@dataclass
class _DocState:
    """Everything the client tracks for one open document.

    ``version`` is the LSP document version we last sent (didOpen=0,
    each didChange +1).  It doubles as the freshness token: stored
    push/pull results are tagged with the version they describe
    (``push_version`` / ``pull_version``), and a result is *fresh*
    iff its tag has caught up to ``version``.  Bumping the version on
    didChange therefore invalidates all older results implicitly —
    no store-clearing, no timestamps.

    ``push_version``/``pull_version`` start at -1 = "no data yet".
    Servers that echo a document version in publishDiagnostics get
    exact tagging; those that don't are credited with the current
    version at receipt time (a push observed after we sent the
    change describes the changed content or newer).
    """

    version: int = 0
    text: str = ""
    push: List[Dict[str, Any]] = field(default_factory=list)
    pull: List[Dict[str, Any]] = field(default_factory=list)
    push_version: int = -1
    pull_version: int = -1
    push_counter: int = 0
    push_time: float = -1.0
    seed_seen: bool = False

    def fresh_push(self, version: Optional[int] = None) -> bool:
        return self.push_version >= (self.version if version is None else version)

    def fresh_pull(self, version: Optional[int] = None) -> bool:
        return self.pull_version >= (self.version if version is None else version)


class LSPClient:
    """Async LSP client tied to one server process and one workspace root.

    Lifecycle:

        c = LSPClient(server_id, workspace_root, command, args, init_options)
        await c.start()       # spawn + initialize
        ver = await c.open_file("/path/to/foo.py")
        await c.wait_for_diagnostics("/path/to/foo.py", ver)
        diags = c.diagnostics_for("/path/to/foo.py")
        await c.shutdown()
    """

    # ------------------------------------------------------------------
    # construction + lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        server_id: str,
        workspace_root: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        initialization_options: Optional[Dict[str, Any]] = None,
        seed_diagnostics_on_first_push: bool = False,
    ) -> None:
        self.server_id = server_id
        self.workspace_root = workspace_root
        self._command = list(command)
        self._env = env
        self._cwd = cwd or workspace_root
        self._init_options = initialization_options or {}
        self._seed_first_push = seed_diagnostics_on_first_push

        # Process + streams
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._reader_task: Optional[asyncio.Task] = None

        # Request/response correlation
        self._next_id: int = 0
        self._pending: Dict[int, asyncio.Future] = {}

        # Server-side request handlers (server → client requests).
        # Kept small and explicit; everything else returns method-not-found.
        self._request_handlers: Dict[str, Callable[[Any], Awaitable[Any]]] = {
            "window/workDoneProgress/create": self._handle_work_done_create,
            "workspace/configuration": self._handle_workspace_configuration,
            "client/registerCapability": self._handle_register_capability,
            "client/unregisterCapability": self._handle_unregister_capability,
            "workspace/workspaceFolders": self._handle_workspace_folders,
            "workspace/diagnostic/refresh": self._handle_diagnostic_refresh,
        }
        # Notifications (server → client) we care about.
        self._notification_handlers: Dict[str, Callable[[Any], None]] = {
            "textDocument/publishDiagnostics": self._handle_publish_diagnostics,
            # Everything else (window/showMessage, $/progress, etc.)
            # is silently dropped by default.
        }

        # Per-document state (version, text, diagnostic stores, and
        # their freshness tags), keyed by canonical absolute file path.
        # See _DocState for the version-based freshness model.
        self._docs: Dict[str, _DocState] = {}
        # Keep document versions and notification ordering atomic when two
        # file-tool calls touch the same client concurrently.
        self._document_lock = asyncio.Lock()
        # Push-sequence baselines are tied to the version returned by
        # ``open_file`` so overlapping edits retain distinct freshness bounds.
        self._diagnostic_baselines: Dict[tuple[str, int], int] = {}
        # Capability registrations — only diagnostic ones are tracked.
        self._diagnostic_registrations: Dict[str, Dict[str, Any]] = {}

        # State machine
        self._state: str = "stopped"
        self._initialize_result: Optional[Dict[str, Any]] = None
        self._sync_kind: int = 1  # 1=Full, 2=Incremental
        self._stopping: bool = False

        # Push event for waiters.
        self._push_event = asyncio.Event()
        # Monotonic counter incremented on every publishDiagnostics push.
        # Waiters snapshot it on entry and treat any increase as
        # "something happened, recheck the predicate".  Avoids the
        # asyncio.Event sticky-state trap.
        self._push_counter = 0
        # Registration change event so wait_for_diagnostics can re-loop
        # when the server announces a new dynamic provider.
        self._registration_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._state == "running" and self._proc is not None and self._proc.returncode is None

    @property
    def state(self) -> str:
        return self._state

    async def start(self) -> None:
        """Spawn the server and complete the initialize handshake.

        Raises any exception encountered during spawn/init.  On failure
        the process is killed and the client is left in state
        ``"error"`` — re-call ``start()`` to retry.
        """
        if self._state in {"running", "starting"}:
            return
        self._state = "starting"
        try:
            await self._spawn()
            await self._initialize()
            self._state = "running"
        except Exception:
            self._state = "error"
            await self._cleanup_process()
            raise

    @staticmethod
    def _win_shell_command(cmd: List[str], env: Dict[str, str]) -> str:
        """Build a safely quoted command for a Windows batch launcher.

        ``cmd.exe`` treats characters such as ``&`` as operators even when
        Python passes arguments as a list. Put each argument in the child
        environment and expand it inside quotes so valid Windows paths and
        arguments containing shell metacharacters retain their identity.
        """
        return windows_batch_command(cmd, env, prefix="HERMES_LSP_COMMAND")

    async def _spawn(self) -> None:
        env = dict(os.environ)
        if self._env:
            env.update(self._env)

        cmd = self._command
        use_windows_shell = sys.platform == "win32" and cmd[0].lower().endswith(
            (".cmd", ".bat")
        )

        try:
            # start_new_session=True detaches the LSP server into its own
            # process group / session. Without this, the LSP server inherits
            # the gateway's pgid (= TUI parent PID). When mcp_tool's
            # _kill_orphaned_mcp_children races with LSP spawn and sweeps the
            # gateway's child set, it captures the LSP PID, records the
            # inherited pgid, and killpg() then kills the TUI parent itself.
            # See tui_gateway_crash.log "killpg → SIGTERM received" stacks.
            spawn_kwargs = {
                "stdin": asyncio.subprocess.PIPE,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "env": env,
                "cwd": self._cwd,
                "start_new_session": True,
            }
            if use_windows_shell:
                command_line = self._win_shell_command(cmd, env)
                self._proc = await asyncio.create_subprocess_shell(
                    command_line, **spawn_kwargs
                )
            else:
                self._proc = await asyncio.create_subprocess_exec(
                    cmd[0], *cmd[1:], **spawn_kwargs
                )
        except FileNotFoundError as e:
            raise LSPProtocolError(
                f"LSP server binary not found: {cmd[0]} ({e})"
            ) from e

        # Drain stderr at debug level — if we don't, the pipe buffer
        # fills and the server hangs.
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        # Start the reader loop.
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("[%s] stderr: %s", self.server_id, text[:1000])
        except (asyncio.CancelledError, OSError):
            pass

    async def _reader_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            while True:
                msg = await read_message(self._proc.stdout)
                if msg is None:
                    logger.debug("[%s] server closed stdout cleanly", self.server_id)
                    break
                kind, key = classify_message(msg)
                if kind == "response":
                    self._dispatch_response(key, msg)
                elif kind == "request":
                    asyncio.create_task(self._dispatch_request(key, msg))
                elif kind == "notification":
                    self._dispatch_notification(key, msg)
                else:
                    logger.warning("[%s] dropping invalid message: %r", self.server_id, msg)
        except LSPProtocolError as e:
            logger.warning("[%s] protocol error in reader loop: %s", self.server_id, e)
        except (asyncio.CancelledError, OSError):
            pass
        finally:
            # Wake up any pending requests so they can fail fast.
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(LSPProtocolError("server connection closed"))
            self._pending.clear()

    async def _initialize(self) -> None:
        params = {
            "rootUri": file_uri(self.workspace_root),
            "rootPath": self.workspace_root,
            "processId": os.getpid(),
            "workspaceFolders": [
                {"name": "workspace", "uri": file_uri(self.workspace_root)}
            ],
            "initializationOptions": self._init_options,
            "capabilities": {
                "window": {"workDoneProgress": True},
                "workspace": {
                    "configuration": True,
                    "workspaceFolders": True,
                    "didChangeWatchedFiles": {"dynamicRegistration": True},
                    "diagnostics": {"refreshSupport": False},
                },
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": False,
                        "didOpen": True,
                        "didChange": True,
                        "didSave": True,
                        "willSave": False,
                        "willSaveWaitUntil": False,
                    },
                    "diagnostic": {
                        "dynamicRegistration": True,
                        "relatedDocumentSupport": True,
                    },
                    "publishDiagnostics": {
                        "relatedInformation": True,
                        "tagSupport": {"valueSet": [1, 2]},
                        "versionSupport": True,
                        "codeDescriptionSupport": True,
                        "dataSupport": False,
                    },
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {"linkSupport": True},
                    "references": {},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                },
                "general": {"positionEncodings": ["utf-16"]},
            },
        }

        result = await asyncio.wait_for(
            self._send_request("initialize", params),
            timeout=INITIALIZE_TIMEOUT,
        )
        self._initialize_result = result
        self._sync_kind = self._extract_sync_kind(result.get("capabilities") or {})

        await self._send_notification("initialized", {})
        if self._init_options:
            # Some servers (vtsls, eslint) want config pushed via
            # didChangeConfiguration even if it was sent in
            # initializationOptions.
            await self._send_notification(
                "workspace/didChangeConfiguration",
                {"settings": self._init_options},
            )

    @staticmethod
    def _extract_sync_kind(capabilities: dict) -> int:
        sync = capabilities.get("textDocumentSync")
        if isinstance(sync, int):
            return sync
        if isinstance(sync, dict):
            change = sync.get("change")
            if isinstance(change, int):
                return change
        return 1  # default to Full

    async def shutdown(self) -> None:
        """Best-effort graceful shutdown.

        Sends ``shutdown`` + ``exit``, then SIGTERMs/SIGKILLs the
        process if it doesn't exit cleanly.  Idempotent.
        """
        if self._stopping:
            return
        self._stopping = True
        try:
            if self.is_running:
                try:
                    await asyncio.wait_for(self._send_request("shutdown", None), timeout=2.0)
                except (asyncio.TimeoutError, LSPRequestError, LSPProtocolError):
                    pass
                try:
                    await self._send_notification("exit", None)
                except Exception:
                    pass
        finally:
            self._state = "stopped"
            await self._cleanup_process()

    async def _cleanup_process(self) -> None:
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.returncode is None:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=SHUTDOWN_GRACE)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass
            except ProcessLookupError:
                pass

    # ------------------------------------------------------------------
    # request / notification plumbing
    # ------------------------------------------------------------------

    async def _send_request(self, method: str, params: Any) -> Any:
        if self._proc is None or self._proc.stdin is None or self._proc.stdin.is_closing():
            raise LSPProtocolError(f"cannot send {method!r}: stdin closed")
        loop = asyncio.get_running_loop()
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        try:
            self._proc.stdin.write(encode_message(make_request(req_id, method, params)))
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self._pending.pop(req_id, None)
            raise LSPProtocolError(f"send failed for {method!r}: {e}") from e
        try:
            return await fut
        finally:
            self._pending.pop(req_id, None)

    async def _send_request_with_retry(self, method: str, params: Any, *, timeout: float) -> Any:
        """Send a request, retrying on ``ContentModified`` (-32801).

        Other errors propagate.  The retry policy matches Claude Code's
        ``LSPServerInstance.sendRequest`` — 3 attempts with delays
        0.5s, 1.0s, 2.0s.
        """
        for attempt in range(MAX_CONTENT_MODIFIED_RETRIES + 1):
            try:
                return await asyncio.wait_for(self._send_request(method, params), timeout=timeout)
            except LSPRequestError as e:
                if e.code == ERROR_CONTENT_MODIFIED and attempt < MAX_CONTENT_MODIFIED_RETRIES:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                raise

    async def _send_notification(self, method: str, params: Any) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdin.is_closing():
            return
        try:
            self._proc.stdin.write(encode_message(make_notification(method, params)))
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.debug("[%s] notify %s failed: %s", self.server_id, method, e)

    async def _send_response(self, req_id: Any, result: Any) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdin.is_closing():
            return
        try:
            self._proc.stdin.write(encode_message(make_response(req_id, result)))
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    async def _send_error_response(self, req_id: Any, code: int, message: str) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdin.is_closing():
            return
        try:
            self._proc.stdin.write(encode_message(make_error_response(req_id, code, message)))
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _dispatch_response(self, req_id: int, msg: dict) -> None:
        fut = self._pending.get(req_id)
        if fut is None or fut.done():
            return
        if "error" in msg:
            err = msg["error"] or {}
            fut.set_exception(
                LSPRequestError(
                    code=int(err.get("code", -32000)),
                    message=str(err.get("message", "unknown")),
                    data=err.get("data"),
                )
            )
        else:
            fut.set_result(msg.get("result"))

    async def _dispatch_request(self, req_id: Any, msg: dict) -> None:
        method = msg.get("method", "")
        params = msg.get("params")
        handler = self._request_handlers.get(method)
        if handler is None:
            await self._send_error_response(req_id, ERROR_METHOD_NOT_FOUND, f"method not found: {method}")
            return
        try:
            result = await handler(params)
        except Exception as e:  # noqa: BLE001 — protocol must not blow up
            logger.warning("[%s] request handler %s failed: %s", self.server_id, method, e)
            await self._send_error_response(req_id, -32000, f"handler failed: {e}")
            return
        await self._send_response(req_id, result)

    def _dispatch_notification(self, method: str, msg: dict) -> None:
        handler = self._notification_handlers.get(method)
        if handler is None:
            return
        try:
            handler(msg.get("params"))
        except Exception as e:  # noqa: BLE001
            logger.debug("[%s] notification handler %s failed: %s", self.server_id, method, e)

    # ------------------------------------------------------------------
    # built-in server-→-client request handlers
    # ------------------------------------------------------------------

    async def _handle_work_done_create(self, params: Any) -> Any:
        # Acknowledge progress tokens — required by some servers.
        return None

    async def _handle_workspace_configuration(self, params: Any) -> Any:
        # Walk dotted sections through initializationOptions.  Mirrors
        # OpenCode's `client.ts:198-220` — return null when missing.
        if not isinstance(params, dict):
            return [None]
        items = params.get("items") or []
        out: List[Any] = []
        for item in items:
            if not isinstance(item, dict):
                out.append(None)
                continue
            section = item.get("section")
            if not section or not self._init_options:
                out.append(self._init_options or None)
                continue
            cur: Any = self._init_options
            for part in str(section).split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    cur = None
                    break
            out.append(cur)
        return out

    async def _handle_register_capability(self, params: Any) -> Any:
        if not isinstance(params, dict):
            return None
        for reg in params.get("registrations") or []:
            if not isinstance(reg, dict):
                continue
            method = reg.get("method")
            reg_id = reg.get("id")
            if method == "textDocument/diagnostic" and reg_id:
                self._diagnostic_registrations[str(reg_id)] = reg
                self._registration_event.set()
        return None

    async def _handle_unregister_capability(self, params: Any) -> Any:
        if not isinstance(params, dict):
            return None
        for unreg in params.get("unregisterations") or []:
            if not isinstance(unreg, dict):
                continue
            reg_id = unreg.get("id")
            if reg_id:
                self._diagnostic_registrations.pop(str(reg_id), None)
        return None

    async def _handle_workspace_folders(self, params: Any) -> Any:
        return [{"name": "workspace", "uri": file_uri(self.workspace_root)}]

    async def _handle_diagnostic_refresh(self, params: Any) -> Any:
        # We don't honour refresh — we re-pull on every touchFile.
        return None

    # ------------------------------------------------------------------
    # publishDiagnostics handler
    # ------------------------------------------------------------------

    def _handle_publish_diagnostics(self, params: Any) -> None:
        if not isinstance(params, dict):
            return
        uri = params.get("uri")
        if not isinstance(uri, str):
            return
        path_key = _path_key(uri_to_path(uri))
        diagnostics = params.get("diagnostics") or []
        if not isinstance(diagnostics, list):
            diagnostics = []
        version = params.get("version")

        doc = self._docs.setdefault(path_key, _DocState(version=-1))
        if self._seed_first_push and not doc.seed_seen:
            # First push: seed the store WITHOUT a freshness tag.  It
            # arrives before the user-triggered didChange could've
            # produced fresh diagnostics, so it must never satisfy a
            # waiter — it's baseline data only.
            doc.seed_seen = True
            doc.push = diagnostics
            return

        doc.seed_seen = True
        doc.push = diagnostics
        # Tag with the echoed document version when the server provides
        # one; otherwise credit the current version — a push observed
        # after we sent the change describes the changed content (or
        # newer).  Note doc.version is -1 for never-opened paths
        # (e.g. relatedDocuments spillover), keeping them unfresh.
        doc.push_version = version if isinstance(version, int) else doc.version
        # Bump the monotonic push counter and wake every waiter.  We
        # keep the Event sticky-set so any wait already in progress
        # resolves; waiters re-check their predicate after waking and
        # decide whether to keep waiting.  ``_push_counter`` is what
        # they actually compare against to detect a fresh event.
        self._push_counter += 1
        doc.push_counter = self._push_counter
        doc.push_time = asyncio.get_event_loop().time()
        self._push_event.set()

    # ------------------------------------------------------------------
    # public file-sync API
    # ------------------------------------------------------------------

    async def open_file(self, path: str, *, language_id: str = "plaintext") -> int:
        """Send didOpen (first time) or didChange (subsequent) for ``path``.

        Returns the new document version number that the agent's
        ``wait_for_diagnostics`` should match against.
        """
        async with self._document_lock:
            return await self._open_file_locked(path, language_id=language_id)

    async def _open_file_locked(
        self, path: str, *, language_id: str = "plaintext"
    ) -> int:
        """Implement :meth:`open_file` while ``_document_lock`` is held."""
        if not self.is_running:
            raise LSPProtocolError("client not running")

        abs_path = _absolute_path(path)
        path_key = _path_key(abs_path)
        try:
            text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise LSPProtocolError(f"cannot read {abs_path}: {e}") from e

        uri = file_uri(abs_path)
        doc = self._docs.get(path_key)

        if doc is not None and doc.version >= 0:
            # Re-open: bump version, fire didChangeWatchedFiles + didChange.
            await self._send_notification(
                "workspace/didChangeWatchedFiles",
                {"changes": [{"uri": uri, "type": 2}]},  # 2 = CHANGED
            )
            diagnostic_counter = self._push_counter
            new_version = doc.version + 1
            old_text = doc.text
            content_changes: List[Dict[str, Any]]
            if self._sync_kind == 2:
                content_changes = [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": _end_position(old_text),
                        },
                        "text": text,
                    }
                ]
            else:
                content_changes = [{"text": text}]
            # Advance local state before the write can yield to the reader
            # task. A fast unversioned push received during ``drain()`` must
            # be tagged with the version carried by this didChange.
            doc.version = new_version
            doc.text = text
            self._diagnostic_baselines[(path_key, new_version)] = diagnostic_counter
            await self._send_notification(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": new_version},
                    "contentChanges": content_changes,
                },
            )
            return new_version

        # First open: didChangeWatchedFiles CREATED + didOpen.
        await self._send_notification(
            "workspace/didChangeWatchedFiles",
            {"changes": [{"uri": uri, "type": 1}]},  # 1 = CREATED
        )
        diagnostic_counter = self._push_counter
        # Fresh doc state — anything stashed under this path by a
        # pre-open push (relatedDocuments spillover etc.) is discarded.
        self._docs[path_key] = _DocState(version=0, text=text)
        self._diagnostic_baselines[(path_key, 0)] = diagnostic_counter
        await self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 0,
                    "text": text,
                }
            },
        )
        return 0

    async def save_file(self, path: str) -> None:
        """Send didSave for ``path``.  Some linters re-scan only on save."""
        if not self.is_running:
            return
        abs_path = _absolute_path(path)
        await self._send_notification(
            "textDocument/didSave",
            {"textDocument": {"uri": file_uri(abs_path)}},
        )

    # ------------------------------------------------------------------
    # diagnostics: pull + wait
    # ------------------------------------------------------------------

    async def _pull_document_diagnostics(self, path: str) -> None:
        """Send ``textDocument/diagnostic`` for one file.

        Stores results into the doc's pull store, tagged with the
        document version captured at request send time.  If a didChange
        races past the in-flight request, the version bump makes the
        stored result stale automatically — no explicit invalidation.
        Silently no-ops on errors (server may not support the pull
        endpoint).
        """
        abs_path = _absolute_path(path)
        path_key = _path_key(abs_path)
        doc = self._docs.get(path_key)
        sent_version = doc.version if doc else -1
        try:
            params: Dict[str, Any] = {
                "textDocument": {"uri": file_uri(abs_path)}
            }
            result = await self._send_request_with_retry(
                "textDocument/diagnostic",
                params,
                timeout=DIAGNOSTICS_REQUEST_TIMEOUT,
            )
        except (LSPRequestError, LSPProtocolError, asyncio.TimeoutError) as e:
            logger.debug("[%s] document diagnostic pull failed: %s", self.server_id, e)
            return
        if not isinstance(result, dict):
            return
        items = result.get("items")
        if isinstance(items, list):
            doc = self._docs.setdefault(path_key, _DocState(version=-1))
            doc.pull = items
            doc.pull_version = sent_version
        related = result.get("relatedDocuments")
        if isinstance(related, dict):
            for uri, sub in related.items():
                if not isinstance(sub, dict):
                    continue
                sub_items = sub.get("items")
                if isinstance(sub_items, list):
                    rel = self._docs.setdefault(
                        _path_key(uri_to_path(uri)), _DocState(version=-1)
                    )
                    rel.pull = sub_items
                    # Same send-anchored tagging: fresh only if that
                    # doc hasn't changed since the request went out.
                    rel.pull_version = rel.version

    async def wait_for_diagnostics(
        self,
        path: str,
        version: int,
        *,
        mode: str = "document",
        timeout: Optional[float] = None,
    ) -> bool:
        """Wait for the server to publish diagnostics for ``path`` at ``version``.

        ``mode`` is ``"document"`` (5s budget, document pulls) or
        ``"full"`` (10s budget, also workspace pulls).  ``timeout``
        overrides the mode's default budget when provided — this is
        how the user's ``lsp.wait_timeout`` config reaches the wait
        loop (slow servers like tsserver on big projects need more
        than the 5s default).

        Returns ``True`` when *fresh* diagnostics arrived (a push at
        or after our didChange, or a pull answered after it) and
        ``False`` on timeout.  Callers must treat ``False`` as "no
        data", NOT as "no errors" — the diagnostic stores may still
        hold stale entries from the previous edit at that point.
        Best-effort — never throws if the server doesn't support pull
        diagnostics; we still get the push side.
        """
        if timeout is not None and timeout > 0:
            budget = timeout
        else:
            budget = DIAGNOSTICS_FULL_WAIT if mode == "full" else DIAGNOSTICS_DOCUMENT_WAIT
        deadline = asyncio.get_event_loop().time() + budget
        abs_path = _absolute_path(path)
        path_key = _path_key(abs_path)
        baseline = self._diagnostic_baselines.pop(
            (path_key, version), self._push_counter
        )

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return False

            # Concurrent: document pull + push wait.
            pull_task = asyncio.create_task(self._pull_document_diagnostics(abs_path))
            push_task = asyncio.create_task(
                self._wait_for_fresh_push(path_key, version, remaining, baseline)
            )
            try:
                done, _ = await asyncio.wait(
                    {pull_task, push_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Push-only servers such as typescript-language-server can
                # reject textDocument/diagnostic immediately. Keep the push
                # waiter alive instead of issuing unsupported pulls in a loop.
                doc = self._docs.get(path_key)
                if (
                    pull_task in done
                    and not push_task.done()
                    and not (doc and doc.fresh_pull(version))
                ):
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining > 0:
                        try:
                            await asyncio.wait_for(push_task, timeout=remaining)
                        except asyncio.TimeoutError:
                            pass
            finally:
                for task in (pull_task, push_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(pull_task, push_task, return_exceptions=True)

            # If we got a fresh post-edit push for our version, we're done.
            doc = self._docs.get(path_key)
            if doc and doc.fresh_push(version) and doc.push_counter > baseline:
                return True

            # Pull may have answered for the current version — that's
            # also success.
            if doc and doc.fresh_pull(version):
                return True

            # Loop until budget runs out.

    async def _wait_for_fresh_push(
        self, path: str, version: int, timeout: float, baseline: int
    ) -> None:
        """Wait until a fresh publishDiagnostics arrives for ``path`` at ``version``+."""
        deadline = asyncio.get_event_loop().time() + timeout
        observed_counter = self._push_counter
        while True:
            doc = self._docs.get(path)
            if doc and doc.fresh_push(version) and doc.push_counter > baseline:
                # TypeScript can publish stale and corrected unversioned
                # snapshots in quick succession. Debounce quietness for this
                # path only so unrelated files neither satisfy nor prolong it.
                while True:
                    push_counter = doc.push_counter
                    now = asyncio.get_event_loop().time()
                    quiet_remaining = min(
                        PUSH_DEBOUNCE - (now - doc.push_time),
                        deadline - now,
                    )
                    if quiet_remaining <= 0:
                        break
                    self._push_event.clear()
                    current_doc = self._docs.get(path)
                    if current_doc is None:
                        break
                    if current_doc.push_counter != push_counter:
                        doc = current_doc
                        continue
                    try:
                        await asyncio.wait_for(
                            self._push_event.wait(), timeout=quiet_remaining
                        )
                    except asyncio.TimeoutError:
                        break
                    doc = self._docs.get(path)
                    if doc is None:
                        break
                return
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return
            if self._push_counter != observed_counter:
                observed_counter = self._push_counter
                continue
            self._push_event.clear()
            if self._push_counter != observed_counter:
                observed_counter = self._push_counter
                continue
            try:
                await asyncio.wait_for(self._push_event.wait(), timeout=min(remaining, 0.5))
            except asyncio.TimeoutError:
                continue
            observed_counter = self._push_counter

    def diagnostics_for(self, path: str, *, fresh_only: bool = False) -> List[Dict[str, Any]]:
        """Return current merged + deduped diagnostics for one file.

        Diagnostics from push and pull stores are concatenated and
        deduplicated by ``(severity, code, message, range)`` content
        key.  Empty list if the server hasn't published anything.

        With ``fresh_only=True``, a store only contributes when its
        version tag has caught up to the document's current version —
        stale leftovers from the previous edit cycle are excluded.
        This is what report paths should use: after an edit, "stale
        errors" and "no errors" must not be conflated.
        """
        doc = self._docs.get(_path_key(path))
        if doc is None:
            return []
        if fresh_only:
            return _dedupe(
                doc.push if doc.fresh_push() else [],
                doc.pull if doc.fresh_pull() else [],
            )
        return _dedupe(doc.push, doc.pull)


def _dedupe(*lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for lst in lists:
        for d in lst:
            if not isinstance(d, dict):
                continue
            key = _diagnostic_key(d)
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
    return out


def _diagnostic_key(d: Dict[str, Any]) -> str:
    """Content-equality key for a diagnostic.

    Matches the structural-equality used in claude-code's
    ``areDiagnosticsEqual`` — message + severity + source + code +
    range coords.  The range is reduced to a tuple to keep the key
    stable across dict orderings.
    """
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    code = d.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)
    return "\x00".join(
        [
            str(d.get("severity") or 1),
            str(code or ""),
            str(d.get("source") or ""),
            str(d.get("message") or "").strip(),
            f"{start.get('line', 0)}:{start.get('character', 0)}-{end.get('line', 0)}:{end.get('character', 0)}",
        ]
    )


__all__ = [
    "LSPClient",
    "file_uri",
    "uri_to_path",
    "INITIALIZE_TIMEOUT",
    "DIAGNOSTICS_DOCUMENT_WAIT",
    "DIAGNOSTICS_FULL_WAIT",
]
