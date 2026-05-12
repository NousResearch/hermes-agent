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

- Push diagnostics are stored per-URI in :attr:`_push_diagnostics` from
  ``textDocument/publishDiagnostics`` notifications.  Pull diagnostics
  go in :attr:`_pull_diagnostics`.  The merged view dedupes by content.

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
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from urllib.parse import quote, unquote

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


def file_uri(path: str) -> str:
    """Return ``file://`` URI for an absolute filesystem path.

    Mirrors Node's ``pathToFileURL`` — handles spaces, unicode, and
    Windows drive letters (``C:\\foo`` → ``file:///C:/foo``).
    """
    abs_path = os.path.abspath(path)
    if os.name == "nt":
        # Windows: backslash → forward slash, prepend extra slash so
        # the drive letter shows up as part of the path component.
        abs_path = abs_path.replace("\\", "/")
        if not abs_path.startswith("/"):
            abs_path = "/" + abs_path
    return "file://" + quote(abs_path, safe="/:")


def uri_to_path(uri: str) -> str:
    """Inverse of :func:`file_uri`."""
    if not uri.startswith("file://"):
        return uri
    raw = uri[len("file://"):]
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]  # strip leading slash before drive letter
    return os.path.normpath(unquote(raw))


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

        # Tracked file state — required for didChange version bumps.
        self._files: Dict[str, Dict[str, Any]] = {}
        # Diagnostic stores, keyed by file path (NOT URI).
        self._push_diagnostics: Dict[str, List[Dict[str, Any]]] = {}
        self._pull_diagnostics: Dict[str, List[Dict[str, Any]]] = {}
        # Per-path "last published" time so wait-for-fresh logic works.
        self._published: Dict[str, float] = {}
        # Per-path version of the latest push (matches our didChange
        # version when the server respects it).
        self._published_version: Dict[str, int] = {}
        # First-push seen flag, for typescript-style seed-on-first-push.
        self._first_push_seen: Set[str] = set()
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
        if self._state in ("running", "starting"):
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

    async def _spawn(self) -> None:
        env = dict(os.environ)
        if self._env:
            env.update(self._env)

        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._command[0],
                *self._command[1:],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self._cwd,
            )
        except FileNotFoundError as e:
            raise LSPProtocolError(
                f"LSP server binary not found: {self._command[0]} ({e})"
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
        path = uri_to_path(uri)
        diagnostics = params.get("diagnostics") or []
        if not isinstance(diagnostics, list):
            diagnostics = []
        version = params.get("version")
        loop_time = asyncio.get_event_loop().time()

        if self._seed_first_push and path not in self._first_push_seen:
            # First push: seed without firing the event so a waiter
            # doesn't resolve on the very first push (which arrives
            # before the user-triggered didChange could've produced
            # fresh diagnostics).
            self._first_push_seen.add(path)
            self._push_diagnostics[path] = diagnostics
            self._published[path] = loop_time
            if isinstance(version, int):
                self._published_version[path] = version
            return

        self._push_diagnostics[path] = diagnostics
        self._published[path] = loop_time
        if isinstance(version, int):
            self._published_version[path] = version
        self._first_push_seen.add(path)
        # Bump the monotonic push counter and wake every waiter.  We
        # keep the Event sticky-set so any wait already in progress
        # resolves; waiters re-check their predicate after waking and
        # decide whether to keep waiting.  ``_push_counter`` is what
        # they actually compare against to detect a fresh event.
        self._push_counter += 1
        self._push_event.set()

    # ------------------------------------------------------------------
    # public file-sync API
    # ------------------------------------------------------------------

    async def open_file(self, path: str, *, language_id: str = "plaintext") -> int:
        """Send didOpen (first time) or didChange (subsequent) for ``path``.

        Returns the new document version number that the agent's
        ``wait_for_diagnostics`` should match against.
        """
        if not self.is_running:
            raise LSPProtocolError("client not running")

        abs_path = os.path.abspath(path)
        try:
            text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise LSPProtocolError(f"cannot read {abs_path}: {e}") from e

        uri = file_uri(abs_path)
        existing = self._files.get(abs_path)

        if existing is not None:
            # Re-open: bump version, fire didChangeWatchedFiles + didChange.
            await self._send_notification(
                "workspace/didChangeWatchedFiles",
                {"changes": [{"uri": uri, "type": 2}]},  # 2 = CHANGED
            )
            new_version = existing["version"] + 1
            old_text = existing["text"]
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
            await self._send_notification(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": new_version},
                    "contentChanges": content_changes,
                },
            )
            self._files[abs_path] = {"version": new_version, "text": text}
            return new_version

        # First open: didChangeWatchedFiles CREATED + didOpen.
        await self._send_notification(
            "workspace/didChangeWatchedFiles",
            {"changes": [{"uri": uri, "type": 1}]},  # 1 = CREATED
        )
        # Clear any stale push/pull entries — fresh open should start
        # from scratch.
        self._push_diagnostics.pop(abs_path, None)
        self._pull_diagnostics.pop(abs_path, None)
        self._published.pop(abs_path, None)
        self._published_version.pop(abs_path, None)
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
        self._files[abs_path] = {"version": 0, "text": text}
        return 0

    async def save_file(self, path: str) -> None:
        """Send didSave for ``path``.  Some linters re-scan only on save."""
        if not self.is_running:
            return
        abs_path = os.path.abspath(path)
        await self._send_notification(
            "textDocument/didSave",
            {"textDocument": {"uri": file_uri(abs_path)}},
        )

    # ------------------------------------------------------------------
    # diagnostics: pull + wait
    # ------------------------------------------------------------------

    async def _pull_document_diagnostics(self, path: str) -> None:
        """Send ``textDocument/diagnostic`` for one file.

        Stores results into :attr:`_pull_diagnostics`.  Silently
        no-ops on errors (server may not support the pull endpoint).
        """
        try:
            params: Dict[str, Any] = {
                "textDocument": {"uri": file_uri(os.path.abspath(path))}
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
            self._pull_diagnostics[os.path.abspath(path)] = items
        related = result.get("relatedDocuments")
        if isinstance(related, dict):
            for uri, sub in related.items():
                if not isinstance(sub, dict):
                    continue
                sub_items = sub.get("items")
                if isinstance(sub_items, list):
                    self._pull_diagnostics[uri_to_path(uri)] = sub_items

    async def wait_for_diagnostics(
        self,
        path: str,
        version: int,
        *,
        mode: str = "document",
    ) -> None:
        """Wait for the server to publish diagnostics for ``path`` at ``version``.

        ``mode`` is ``"document"`` (5s budget, document pulls) or
        ``"full"`` (10s budget, also workspace pulls).  Best-effort —
        returns silently on timeout.  Does NOT throw if the server
        doesn't support pull diagnostics; we still get the push side.
        """
        budget = DIAGNOSTICS_FULL_WAIT if mode == "full" else DIAGNOSTICS_DOCUMENT_WAIT
        deadline = asyncio.get_event_loop().time() + budget
        abs_path = os.path.abspath(path)

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return

            # Concurrent: document pull + push wait.
            pull_task = asyncio.create_task(self._pull_document_diagnostics(abs_path))
            push_task = asyncio.create_task(self._wait_for_fresh_push(abs_path, version, remaining))
            done, pending = await asyncio.wait(
                {pull_task, push_task},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in pending:
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

            # If we got a fresh push for our version, we're done.
            current_v = self._published_version.get(abs_path)
            if abs_path in self._published and (
                current_v is None or current_v >= version
            ):
                return

            # Pull may have populated _pull_diagnostics — that's also
            # success.
            if abs_path in self._pull_diagnostics:
                return

            # Loop until budget runs out.

    async def _wait_for_fresh_push(self, path: str, version: int, timeout: float) -> None:
        """Wait until a publishDiagnostics arrives for ``path`` at ``version``+."""
        deadline = asyncio.get_event_loop().time() + timeout
        baseline = self._push_counter
        while True:
            current_v = self._published_version.get(path)
            if path in self._published and (current_v is None or current_v >= version):
                # Debounce — wait a tick in case more diagnostics arrive
                # immediately after.  TS often emits in pairs.  We
                # snapshot the counter so we wake on a *new* push, not
                # on the one that satisfied us a moment ago.
                debounce_baseline = self._push_counter
                debounce_deadline = asyncio.get_event_loop().time() + PUSH_DEBOUNCE
                while self._push_counter == debounce_baseline:
                    remaining = debounce_deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    self._push_event.clear()
                    try:
                        await asyncio.wait_for(self._push_event.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                return
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return
            if self._push_counter > baseline:
                # New event arrived but predicate still false — re-check
                # immediately without waiting again.
                baseline = self._push_counter
                continue
            self._push_event.clear()
            try:
                await asyncio.wait_for(self._push_event.wait(), timeout=min(remaining, 0.5))
            except asyncio.TimeoutError:
                continue

    def diagnostics_for(self, path: str) -> List[Dict[str, Any]]:
        """Return current merged + deduped diagnostics for one file.

        Diagnostics from push and pull stores are concatenated and
        deduplicated by ``(severity, code, message, range)`` content
        key.  Empty list if the server hasn't published anything.
        """
        abs_path = os.path.abspath(path)
        push = self._push_diagnostics.get(abs_path) or []
        pull = self._pull_diagnostics.get(abs_path) or []
        return _dedupe(push, pull)


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
