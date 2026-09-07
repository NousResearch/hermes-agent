"""Async LSP client over stdin/stdout — one per ``(server, workspace_root)``.

Freshness is tracked with **document versions**, not timestamps: every didChange
bumps ``version`` and each stored push/pull result is tagged with the version it
describes, so a slow server's leftovers never masquerade as a verdict on the
current content ("ghost diagnostics").  Whole-document sync is always sent, every
``open_file`` also fires ``didChangeWatchedFiles`` (clangd/eslint only re-scan on
it), and ``ContentModified`` (-32801) errors are retried with exponential backoff.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from urllib.parse import quote, unquote

from hermes_cli._subprocess_compat import windows_hide_flags

from agent.lsp.protocol import (
    ERROR_CONTENT_MODIFIED, ERROR_METHOD_NOT_FOUND, LSPProtocolError, LSPRequestError, classify_message,
    encode_message, make_error_response, make_notification, make_request, make_response, read_message,
)

logger = logging.getLogger("agent.lsp.client")

# Timeouts (seconds).
INITIALIZE_TIMEOUT = 45.0
DIAGNOSTICS_DOCUMENT_WAIT = 5.0
DIAGNOSTICS_FULL_WAIT = 10.0
DIAGNOSTICS_REQUEST_TIMEOUT = 3.0
PUSH_DEBOUNCE = 0.15
SHUTDOWN_GRACE = 1.0  # seconds between SIGTERM and SIGKILL
# Retry policy for transient ContentModified errors: 0.5, 1.0, 2.0s.
MAX_CONTENT_MODIFIED_RETRIES = 3
RETRY_BASE_DELAY = 0.5

_WRITE_ERRORS = (BrokenPipeError, ConnectionResetError, OSError)
_LIVE_STATES = {"starting", "running"}

_CLIENT_CAPABILITIES: Dict[str, Any] = {
    "window": {"workDoneProgress": True},
    "workspace": {"configuration": True, "workspaceFolders": True,
                  "didChangeWatchedFiles": {"dynamicRegistration": True}, "diagnostics": {"refreshSupport": False}},
    "textDocument": {
        "synchronization": {"dynamicRegistration": False, "didOpen": True, "didChange": True,
                            "didSave": True, "willSave": False, "willSaveWaitUntil": False},
        "diagnostic": {"dynamicRegistration": True, "relatedDocumentSupport": True},
        "publishDiagnostics": {"relatedInformation": True, "tagSupport": {"valueSet": [1, 2]},
                               "versionSupport": True, "codeDescriptionSupport": True, "dataSupport": False},
        "hover": {"contentFormat": ["markdown", "plaintext"]},
        "definition": {"linkSupport": True},
        "references": {},
        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
    },
    "general": {"positionEncodings": ["utf-16"]},
}


def file_uri(path: str) -> str:
    """Return a ``file://`` URI for a path (handles spaces, unicode, Windows drive letters)."""
    abs_path = os.path.abspath(path)
    if os.name == "nt":
        # ``C:\foo`` → ``file:///C:/foo``: the drive letter must be a path component.
        abs_path = abs_path.replace("\\", "/")
        abs_path = abs_path if abs_path.startswith("/") else "/" + abs_path
    return "file://" + quote(abs_path, safe="/:")


def _folder(root: str) -> Dict[str, str]:
    """Build an LSP ``WorkspaceFolder`` for ``root``."""
    return {"name": os.path.basename(root.rstrip(os.sep)) or root, "uri": file_uri(root)}


def uri_to_path(uri: str) -> str:
    """Inverse of :func:`file_uri`."""
    if not uri.startswith("file://"):
        return uri
    raw = uri[len("file://"):]
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]  # strip leading slash before drive letter
    return os.path.normpath(unquote(raw))


def _end_position(text: str) -> Dict[str, int]:
    """LSP Position at the end of ``text`` (for a whole-document replace range)."""
    if not text:
        return {"line": 0, "character": 0}
    lines = text.splitlines(keepends=False)
    # splitlines drops a trailing newline: the end is then the start of the next (empty) line.
    if text.endswith(("\n", "\r")):
        return {"line": len(lines), "character": 0}
    return {"line": len(lines) - 1, "character": len(lines[-1])}


@dataclass
class _DocState:
    """Per-document state.  ``version`` is the LSP document version last sent (didOpen=0, +1 per
    didChange) and doubles as the freshness token: ``push_version`` / ``pull_version`` tag stored
    results, fresh iff tag >= version; -1 means "no data yet".  Servers that echo a version in
    publishDiagnostics get exact tagging; others are credited with the current version at receipt."""
    version: int = 0
    text: str = ""
    push: List[Dict[str, Any]] = field(default_factory=list)
    pull: List[Dict[str, Any]] = field(default_factory=list)
    push_version: int = -1
    pull_version: int = -1
    seed_seen: bool = False

    def fresh_push(self, version: Optional[int] = None) -> bool:
        return self.push_version >= (self.version if version is None else version)

    def fresh_pull(self, version: Optional[int] = None) -> bool:
        return self.pull_version >= (self.version if version is None else version)

    def fresh(self, version: Optional[int] = None) -> bool:
        return self.fresh_push(version) or self.fresh_pull(version)


class LSPClient:
    """One server process + one workspace root.  ``start()`` → ``open_file()`` → ``wait_for_diagnostics()`` →
    ``diagnostics_for()`` → ``shutdown()``."""

    def __init__(self, *, server_id: str, workspace_root: str, command: List[str],
                 env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None,
                 initialization_options: Optional[Dict[str, Any]] = None,
                 seed_diagnostics_on_first_push: bool = False) -> None:
        self.server_id = server_id
        self.workspace_root = workspace_root
        # Roots this server serves.  Single-root servers only ever hold ``workspace_root``;
        # multi-root servers (pyright) grow this via ``add_workspace_folder`` instead of a second process.
        self.workspace_folders: List[str] = [workspace_root]
        self._command = list(command)
        self._env = env
        self._cwd = cwd or workspace_root
        self._init_options = initialization_options or {}
        self._seed_first_push = seed_diagnostics_on_first_push

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._cleanup_lock = asyncio.Lock()
        self._next_id: int = 0
        self._pending: Dict[int, asyncio.Future] = {}

        # Server → client requests; anything else gets method-not-found.  Capability (un)registration
        # and diagnostic refresh are acknowledged but not acted on: we re-pull on every touch anyway.
        self._request_handlers: Dict[str, Callable[[Any], Awaitable[Any]]] = {
            "window/workDoneProgress/create": self._handle_null,
            "workspace/configuration": self._handle_workspace_configuration,
            "client/registerCapability": self._handle_null,
            "client/unregisterCapability": self._handle_null,
            "workspace/workspaceFolders": self._handle_workspace_folders,
            "workspace/diagnostic/refresh": self._handle_null,
        }
        # Server → client notifications; others (showMessage, $/progress) are dropped.
        self._notification_handlers: Dict[str, Callable[[Any], None]] = {
            "textDocument/publishDiagnostics": self._handle_publish_diagnostics,
        }

        self._docs: Dict[str, _DocState] = {}  # keyed by absolute path (NOT URI)
        self._state: str = "stopped"
        self._sync_kind: int = 1  # 1=Full, 2=Incremental
        self._stopping: bool = False
        # Waiters snapshot ``_push_counter`` and treat any increase as "recheck the
        # predicate" — avoids the asyncio.Event sticky-state trap.
        self._push_event = asyncio.Event()
        self._push_counter = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == "running" and self._connection_is_open()

    def _connection_is_open(self) -> bool:
        proc, reader = self._proc, self._reader_task
        return (
            self._state in _LIVE_STATES
            and proc is not None and proc.returncode is None
            and proc.stdin is not None and not proc.stdin.is_closing()
            and reader is not None and not reader.done()
        )

    # ---- lifecycle ----

    async def start(self) -> None:
        """Spawn + initialize handshake.  On failure the process is killed and state is ``"error"``; re-call to retry."""
        if self._state in _LIVE_STATES:
            return
        self._state = "starting"
        try:
            await self._spawn()
            await self._initialize()
            if not self._connection_is_open():
                raise LSPProtocolError("server connection closed during initialization")
            self._state = "running"
        except Exception:
            self._state = "error"
            await self._cleanup_process()
            raise

    async def _spawn(self) -> None:
        cmd = self._command
        if sys.platform == "win32" and cmd[0].lower().endswith((".cmd", ".bat")):
            cmd = ["cmd.exe", "/c", *cmd]  # CreateProcess can't run .cmd/.bat shims directly
        try:
            # start_new_session=True gives the server its own process group; otherwise it inherits
            # the gateway's pgid and mcp_tool's orphan sweeper can killpg() the TUI parent with it.
            # windows_hide_flags() suppresses the console window a .cmd shim would flash from a
            # console-less host (CREATE_NO_WINDOW; 0 on POSIX).
            self._proc = await asyncio.create_subprocess_exec(
                cmd[0], *cmd[1:],
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **(self._env or {})}, cwd=self._cwd,
                start_new_session=True, creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as e:
            raise LSPProtocolError(f"LSP server binary not found: {cmd[0]} ({e})") from e
        # stderr must be drained or the pipe buffer fills and the server hangs.
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            while line := await self._proc.stderr.readline():
                if text := line.decode("utf-8", errors="replace").rstrip():
                    logger.debug("[%s] stderr: %s", self.server_id, text[:1000])
        except (asyncio.CancelledError, OSError):
            pass

    def _dispatch(self, msg: dict) -> None:
        kind, key = classify_message(msg)
        if kind == "response":
            self._dispatch_response(key, msg)
        elif kind == "request":
            asyncio.create_task(self._dispatch_request(key, msg))
        elif kind == "notification":
            self._dispatch_notification(key, msg)
        else:
            logger.warning("[%s] dropping invalid message: %r", self.server_id, msg)

    async def _reader_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            while (msg := await read_message(self._proc.stdout)) is not None:
                self._dispatch(msg)
            logger.debug("[%s] server closed stdout cleanly", self.server_id)
        except LSPProtocolError as e:
            logger.warning("[%s] protocol error in reader loop: %s", self.server_id, e)
        except (asyncio.CancelledError, OSError):
            pass
        finally:
            unexpected_close = not self._stopping and self._state in _LIVE_STATES
            if unexpected_close:
                self._state = "error"
            for fut in list(self._pending.values()):  # fail pending requests fast
                if not fut.done():
                    fut.set_exception(LSPProtocolError("server connection closed"))
            self._pending.clear()
            if unexpected_close:
                await self._cleanup_process()

    def _workspace_folders(self) -> List[Dict[str, str]]:
        return [_folder(r) for r in self.workspace_folders]

    async def add_workspace_folder(self, root: str) -> None:
        """Attach another root to a running multi-root server.  Idempotent; the folder is recorded
        before the notification is sent so concurrent callers for the same root only announce once."""
        if root in self.workspace_folders:
            return
        self.workspace_folders.append(root)
        await self._send_notification(
            "workspace/didChangeWorkspaceFolders", {"event": {"added": [_folder(root)], "removed": []}},
        )

    async def _initialize(self) -> None:
        params = {
            "rootUri": file_uri(self.workspace_root), "rootPath": self.workspace_root, "processId": os.getpid(),
            "workspaceFolders": self._workspace_folders(),
            "initializationOptions": self._init_options, "capabilities": _CLIENT_CAPABILITIES,
        }
        result = await asyncio.wait_for(self._send_request("initialize", params), timeout=INITIALIZE_TIMEOUT)
        sync = (result.get("capabilities") or {}).get("textDocumentSync")
        if isinstance(sync, dict):
            sync = sync.get("change")
        self._sync_kind = sync if isinstance(sync, int) else 1  # default to Full
        await self._send_notification("initialized", {})
        if self._init_options:  # vtsls/eslint only pick config up via didChangeConfiguration
            await self._send_notification("workspace/didChangeConfiguration", {"settings": self._init_options})

    async def shutdown(self) -> None:
        """Best-effort graceful shutdown: ``shutdown`` + ``exit``, then SIGTERM/SIGKILL.  Idempotent."""
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
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._state = "stopped"
            await self._cleanup_process()

    async def _cleanup_process(self) -> None:
        async with self._cleanup_lock:
            tasks = [self._reader_task, self._stderr_task]
            self._reader_task = self._stderr_task = None
            proc, self._proc = self._proc, None
            live = [t for t in tasks if t is not None and not t.done() and t is not asyncio.current_task()]
            for t in live:
                t.cancel()
            await asyncio.gather(*live, return_exceptions=True)
            if proc is None or proc.returncode is not None:
                return
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=SHUTDOWN_GRACE)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            except ProcessLookupError:
                pass

    # ---- request / notification plumbing ----

    async def _write(self, msg: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(encode_message(msg))
        await self._proc.stdin.drain()

    def _require_open(self, method: str) -> None:
        if not self._connection_is_open():
            raise LSPProtocolError(f"cannot send {method!r}: server connection closed")

    async def _send_request(self, method: str, params: Any) -> Any:
        self._require_open(method)
        req_id, self._next_id = self._next_id, self._next_id + 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._write(make_request(req_id, method, params))
        except _WRITE_ERRORS as e:
            self._pending.pop(req_id, None)
            raise LSPProtocolError(f"send failed for {method!r}: {e}") from e
        try:
            return await fut
        finally:
            self._pending.pop(req_id, None)

    async def _send_request_with_retry(self, method: str, params: Any, *, timeout: float) -> Any:
        """Send a request, retrying ``ContentModified`` (-32801) with backoff; other errors propagate."""
        for attempt in range(MAX_CONTENT_MODIFIED_RETRIES + 1):
            try:
                return await asyncio.wait_for(self._send_request(method, params), timeout=timeout)
            except LSPRequestError as e:
                if e.code != ERROR_CONTENT_MODIFIED or attempt >= MAX_CONTENT_MODIFIED_RETRIES:
                    raise
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))

    async def _send_notification(self, method: str, params: Any) -> None:
        self._require_open(method)
        try:
            await self._write(make_notification(method, params))
        except _WRITE_ERRORS as e:
            logger.debug("[%s] notify %s failed: %s", self.server_id, method, e)

    async def _send_reply(self, msg: dict) -> None:
        """Send a response to a server→client request; silently no-ops when the pipe is gone."""
        if self._proc is not None and self._proc.stdin is not None and not self._proc.stdin.is_closing():
            try:
                await self._write(msg)
            except _WRITE_ERRORS:
                pass

    def _dispatch_response(self, req_id: int, msg: dict) -> None:
        fut = self._pending.get(req_id)
        if fut is None or fut.done():
            return
        if "error" not in msg:
            fut.set_result(msg.get("result"))
            return
        err = msg["error"] or {}
        fut.set_exception(LSPRequestError(int(err.get("code", -32000)), str(err.get("message", "unknown")), err.get("data")))

    async def _dispatch_request(self, req_id: Any, msg: dict) -> None:
        method = msg.get("method", "")
        handler = self._request_handlers.get(method)
        if handler is None:
            reply = make_error_response(req_id, ERROR_METHOD_NOT_FOUND, f"method not found: {method}")
        else:
            try:
                reply = make_response(req_id, await handler(msg.get("params")))
            except Exception as e:  # noqa: BLE001 — protocol must not blow up
                logger.warning("[%s] request handler %s failed: %s", self.server_id, method, e)
                reply = make_error_response(req_id, -32000, f"handler failed: {e}")
        await self._send_reply(reply)

    def _dispatch_notification(self, method: str, msg: dict) -> None:
        handler = self._notification_handlers.get(method)
        if handler is None:
            return
        try:
            handler(msg.get("params"))
        except Exception as e:  # noqa: BLE001
            logger.debug("[%s] notification handler %s failed: %s", self.server_id, method, e)

    # ---- built-in server-→-client request handlers ----

    async def _handle_null(self, params: Any) -> Any:
        return None

    async def _handle_workspace_folders(self, params: Any) -> Any:
        return self._workspace_folders()

    async def _handle_workspace_configuration(self, params: Any) -> Any:
        """Walk dotted ``section`` paths through initializationOptions; null when missing."""
        if not isinstance(params, dict):
            return [None]
        return [self._config_section(item) for item in params.get("items") or []]

    def _config_section(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return None
        section = item.get("section")
        if not section or not self._init_options:
            return self._init_options or None
        cur: Any = self._init_options
        for part in str(section).split("."):
            if not (isinstance(cur, dict) and part in cur):
                return None
            cur = cur[part]
        return cur

    def _handle_publish_diagnostics(self, params: Any) -> None:
        if not isinstance(params, dict) or not isinstance(params.get("uri"), str):
            return
        diagnostics = params.get("diagnostics") or []
        version = params.get("version")
        doc = self._docs.setdefault(uri_to_path(params["uri"]), _DocState(version=-1))
        is_seed = self._seed_first_push and not doc.seed_seen
        doc.seed_seen = True
        doc.push = diagnostics if isinstance(diagnostics, list) else []
        if is_seed:
            # First push is baseline data only: it predates any didChange we sent,
            # so it's stored WITHOUT a freshness tag and never satisfies a waiter.
            return
        # Tag with the echoed version when provided; otherwise credit the current
        # version (a push observed after our change describes it or newer).  doc.version
        # is -1 for never-opened paths (relatedDocuments spillover), keeping them unfresh.
        doc.push_version = version if isinstance(version, int) else doc.version
        # Keep the Event sticky-set so in-progress waits resolve; waiters
        # compare ``_push_counter`` to detect a genuinely new push.
        self._push_counter += 1
        self._push_event.set()

    # ---- public file-sync API ----

    async def open_file(self, path: str, *, language_id: str = "plaintext") -> int:
        """Send didOpen (first time) or didChange (subsequent); return the new document version."""
        if not self.is_running:
            raise LSPProtocolError("client not running")
        abs_path = os.path.abspath(path)
        try:
            text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise LSPProtocolError(f"cannot read {abs_path}: {e}") from e
        uri = file_uri(abs_path)
        doc = self._docs.get(abs_path)
        if doc is not None and doc.version < 0:
            doc = None  # never opened (relatedDocuments spillover): treat as new
        # FileChangeType: 1 = CREATED, 2 = CHANGED.
        await self._send_notification(
            "workspace/didChangeWatchedFiles", {"changes": [{"uri": uri, "type": 1 if doc is None else 2}]}
        )
        if doc is None:
            # Fresh state: anything a pre-open push stashed under this path (relatedDocuments spillover) is discarded.
            self._docs[abs_path] = _DocState(version=0, text=text)
            await self._send_notification(
                "textDocument/didOpen",
                {"textDocument": {"uri": uri, "languageId": language_id, "version": 0, "text": text}},
            )
            return 0
        change: Dict[str, Any] = {"text": text}
        if self._sync_kind == 2:
            change["range"] = {"start": {"line": 0, "character": 0}, "end": _end_position(doc.text)}
        new_version = doc.version + 1
        await self._send_notification(
            "textDocument/didChange",
            {"textDocument": {"uri": uri, "version": new_version}, "contentChanges": [change]},
        )
        # Bumping the version is the whole invalidation story (see _DocState).
        doc.version, doc.text = new_version, text
        return new_version

    async def save_file(self, path: str) -> None:
        """Send didSave for ``path``.  Some linters re-scan only on save."""
        if self.is_running:
            await self._send_notification(
                "textDocument/didSave", {"textDocument": {"uri": file_uri(os.path.abspath(path))}}
            )

    # ---- diagnostics: pull + wait ----

    async def _pull_document_diagnostics(self, path: str) -> None:
        """Send ``textDocument/diagnostic`` for one file into the pull store.  Results are tagged with the
        version captured at send time, so a didChange racing past the request makes them stale
        automatically.  Silently no-ops on errors (server may not support pull)."""
        abs_path = os.path.abspath(path)
        doc = self._docs.get(abs_path)
        sent_version = doc.version if doc else -1
        try:
            result = await self._send_request_with_retry(
                "textDocument/diagnostic", {"textDocument": {"uri": file_uri(abs_path)}},
                timeout=DIAGNOSTICS_REQUEST_TIMEOUT,
            )
        except (LSPRequestError, LSPProtocolError, asyncio.TimeoutError) as e:
            logger.debug("[%s] document diagnostic pull failed: %s", self.server_id, e)
            return
        if not isinstance(result, dict):
            return
        related = result.get("relatedDocuments")
        reports = [(abs_path, result, sent_version)]
        if isinstance(related, dict):
            # Related docs get the same send-anchored tagging: fresh only if unchanged since.
            reports += [(uri_to_path(uri), sub, None) for uri, sub in related.items()]
        for doc_path, report, tag in reports:
            items = report.get("items") if isinstance(report, dict) else None
            if isinstance(items, list):
                d = self._docs.setdefault(doc_path, _DocState(version=-1))
                d.pull = items
                d.pull_version = d.version if tag is None else tag

    async def wait_for_diagnostics(self, path: str, version: int, *, mode: str = "document",
                                   timeout: Optional[float] = None) -> bool:
        """Wait for fresh diagnostics for ``path`` at ``version``; True iff fresh data arrived in budget.

        ``mode`` is ``"document"`` (5s) or ``"full"`` (10s); ``timeout`` overrides the budget (how
        ``lsp.wait_timeout`` reaches the loop).  Callers must treat False as "no data", NOT "no errors" —
        the stores may still hold stale entries.  Never throws for servers lacking pull support.
        """
        if not (timeout is not None and timeout > 0):
            timeout = DIAGNOSTICS_FULL_WAIT if mode == "full" else DIAGNOSTICS_DOCUMENT_WAIT
        now = asyncio.get_event_loop().time
        deadline = now() + timeout
        abs_path = os.path.abspath(path)
        while True:
            if not self._connection_is_open():
                raise LSPProtocolError("server connection closed while waiting for diagnostics")
            remaining = deadline - now()
            if remaining <= 0:
                return False
            # Concurrent: document pull + push wait.
            tasks = {
                asyncio.create_task(self._pull_document_diagnostics(abs_path)),
                asyncio.create_task(self._wait_for_fresh_push(abs_path, version, remaining)),
            }
            _done, pending = await asyncio.wait(tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            doc = self._docs.get(abs_path)
            if doc and doc.fresh(version):
                return True

    async def _await_push(self, timeout: float) -> bool:
        """Block until the next publishDiagnostics or ``timeout``; True iff a push woke us."""
        self._push_event.clear()
        try:
            await asyncio.wait_for(self._push_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True

    async def _wait_for_fresh_push(self, path: str, version: int, timeout: float) -> None:
        """Wait until a fresh publishDiagnostics arrives for ``path`` at ``version``+."""
        now = asyncio.get_event_loop().time
        deadline = now() + timeout
        baseline = self._push_counter
        while True:
            doc = self._docs.get(path)
            if doc and doc.fresh_push(version):
                # Debounce: TS often emits in pairs.  Snapshot the counter so
                # we wake on a *new* push, not the one that just satisfied us.
                debounce_baseline = self._push_counter
                debounce_deadline = now() + PUSH_DEBOUNCE
                while self._push_counter == debounce_baseline:
                    remaining = debounce_deadline - now()
                    if remaining <= 0 or not await self._await_push(remaining):
                        break
                return
            remaining = deadline - now()
            if remaining <= 0:
                return
            if self._push_counter > baseline:
                # New push but predicate still false — re-check without waiting.
                baseline = self._push_counter
                continue
            await self._await_push(min(remaining, 0.5))

    def diagnostics_for(self, path: str, *, fresh_only: bool = False) -> List[Dict[str, Any]]:
        """Merged + deduped push/pull diagnostics for one file.  With ``fresh_only=True`` a store only
        contributes once its version tag has caught up to the document's — report paths must use this
        so "stale" and "clean" aren't conflated."""
        doc = self._docs.get(os.path.abspath(path))
        if doc is None:
            return []
        push = doc.push if not fresh_only or doc.fresh_push() else []
        pull = doc.pull if not fresh_only or doc.fresh_pull() else []
        return _dedupe(push, pull)


def _dedupe(*lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for d in (d for lst in lists for d in lst if isinstance(d, dict)):
        if (key := _diagnostic_key(d)) not in seen:
            seen.add(key)
            out.append(d)
    return out


def _diagnostic_key(d: Dict[str, Any]) -> str:
    """Content-equality key: severity + code + source + message + range.  Shared with the manager's
    cross-edit delta filter (``_diag_key``) so both layers agree on identity.  Range is included so an
    identical error at a second site still surfaces as new (the manager line-shifts its baseline first)."""
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    code = d.get("code")
    return "\x00".join([
        str(d.get("severity") or 1), "" if code is None else str(code), str(d.get("source") or ""),
        str(d.get("message") or "").strip(),
        f"{start.get('line', 0)}:{start.get('character', 0)}-{end.get('line', 0)}:{end.get('character', 0)}",
    ])


__all__ = ["LSPClient", "file_uri", "uri_to_path", "INITIALIZE_TIMEOUT", "DIAGNOSTICS_DOCUMENT_WAIT", "DIAGNOSTICS_FULL_WAIT"]
