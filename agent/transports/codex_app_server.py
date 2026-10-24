"""Codex app-server JSON-RPC client (newline-delimited JSON-RPC 2.0 over stdio, codex 0.125+).

``initialize`` handshake, then ``thread/start`` + ``turn/start`` with streaming
``item/*`` notifications until ``turn/completed``. Wire-level speaker only —
projection, approvals and transcript handling live in sibling modules.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Optional

from agent.deadline import kill_process_tree
from agent.transports.hermes_tools_mcp_server import HERMES_TOOLS_MCP_SERVER_NAME
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

MIN_CODEX_VERSION = (0, 125, 0)

_VALID_SANDBOX_MODES = {"read-only", "workspace-write", "danger-full-access"}
_VALID_APPROVAL_POLICIES = {"untrusted", "on-request", "never"}
_VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}


def _optional_choice(config: dict[str, Any], key: str, choices: set[str]) -> str:
    if key not in config:
        return ""
    value = config[key]
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if normalized not in choices:
        raise ValueError(
            f"model.codex_app_server.{key} must be one of: {', '.join(sorted(choices))}"
        )
    return normalized


def resolve_codex_app_server_args(
    *,
    config: Optional[dict[str, Any]] = None,
    kanban_root: Optional[str] = None,
    kanban_bounded: bool = False,
) -> list[str]:
    """Validate and resolve session-local Codex ``-c`` overrides.

    ``kanban_bounded`` clamps the sandbox without granting any root (a Kanban
    descendant); ``kanban_root`` is the one writable root of an owned worker.
    """
    kanban_bounded = kanban_bounded or bool(kanban_root)
    cfg = config if isinstance(config, dict) else {}
    overrides: list[tuple[str, Any]] = []
    effort = _optional_choice(cfg, "model_reasoning_effort", _VALID_REASONING_EFFORTS)
    if effort:
        overrides.append(("model_reasoning_effort", effort))

    configured_sandbox_mode = _optional_choice(
        cfg, "sandbox_mode", _VALID_SANDBOX_MODES
    )
    approval = _optional_choice(cfg, "ask_for_approval", _VALID_APPROVAL_POLICIES)

    # Validate everything first so a malformed value fails the same way in a
    # Kanban worker as in a normal session, and only then clamp.
    if "network_access" in cfg and not isinstance(cfg["network_access"], bool):
        raise ValueError("model.codex_app_server.network_access must be a boolean")
    configured_roots = cfg.get("writable_roots", [])
    if not isinstance(configured_roots, list) or any(
        not isinstance(root, str) or not os.path.isabs(root)
        for root in configured_roots
    ):
        raise ValueError(
            "model.codex_app_server.writable_roots must contain only absolute paths"
        )

    sandbox_mode = "workspace-write" if kanban_bounded else configured_sandbox_mode
    if kanban_bounded:
        ignored = [
            key
            for key in ("sandbox_mode", "network_access", "writable_roots")
            if key in cfg
        ]
        if ignored:
            logger.warning(
                "model.codex_app_server.%s ignored: Kanban-lineage sessions are "
                "bounded to workspace-write and no network; only an owned worker "
                "gets its Kanban root",
                "/".join(ignored),
            )
    elif sandbox_mode != "workspace-write":
        ignored = [key for key in ("network_access", "writable_roots") if key in cfg]
        if ignored:
            logger.warning(
                "model.codex_app_server.%s ignored: they only apply with "
                "sandbox_mode: workspace-write",
                "/".join(ignored),
            )
    if sandbox_mode:
        overrides.append(("sandbox_mode", sandbox_mode))
    if approval:
        overrides.append(("approval_policy", approval))

    if sandbox_mode == "workspace-write":
        if "network_access" in cfg or kanban_bounded:
            overrides.append(
                (
                    "sandbox_workspace_write.network_access",
                    False if kanban_bounded else cfg["network_access"],
                )
            )
        roots: list[str] = []
        seen_roots: set[str] = set()
        for candidate in [*([] if kanban_bounded else configured_roots), kanban_root]:
            if not candidate:
                continue
            dedupe_key = os.path.normcase(os.path.normpath(candidate))
            if dedupe_key not in seen_roots:
                seen_roots.add(dedupe_key)
                roots.append(candidate)
        if roots:
            overrides.append(("sandbox_workspace_write.writable_roots", roots))

    args: list[str] = []
    for key, value in overrides:
        args.extend(
            (
                "-c",
                f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}",
            )
        )
    return args


@dataclass
class CodexAppServerError(RuntimeError):
    """Raised on JSON-RPC errors from the app-server."""

    code: int
    message: str
    data: Optional[Any] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"codex app-server error {self.code}: {self.message}"


class CodexAppServerTransportError(CodexAppServerError):
    """The JSON-RPC transport is gone: a write failed or close() drained the request.

    Distinct from server-reported errors so session boundaries can retire the
    session without swallowing unrelated ``RuntimeError`` programming defects.
    """

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


_TRANSPORT_LOST_CODE = -32000


def _snapshot_descendants(pid: int) -> list[Any]:
    """psutil handles for ``pid``'s current descendants ([] when psutil is unavailable)."""
    try:
        import psutil
        return psutil.Process(int(pid)).children(recursive=True)
    except Exception:
        return []


def _reap_snapshotted(descendants: list[Any]) -> None:
    """SIGTERM the snapshotted descendants (deepest first), then SIGKILL survivors after a
    bounded wait. psutil identity checks make a recycled PID a no-op."""
    if not descendants:
        return
    import psutil
    live = []
    for child in reversed(descendants):
        with contextlib.suppress(Exception):
            if child.is_running():
                child.terminate()
                live.append(child)
    try:
        _, alive = psutil.wait_procs(live, timeout=1.0)
    except Exception:
        alive = live
    for child in alive:
        with contextlib.suppress(Exception):
            child.kill()


class CodexAppServerClient:
    """Minimal synchronous JSON-RPC 2.0 client for ``codex app-server`` over stdio.

    One reader thread routes replies to pending queues and notifications / server
    requests to queues; another captures stderr. Deliberately NOT async:
    AIAgent.run_conversation() is synchronous and cancels via ``turn/interrupt``.
    """

    def __init__(
        self, codex_bin: str = "codex", codex_home: Optional[str] = None,
        extra_args: Optional[list[str]] = None, env: Optional[dict[str, str]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._codex_bin = codex_bin
        # codex needs LLM provider creds but must not receive Tier-1 Hermes secrets (gateway/GitHub/infra tokens).
        # codex app-server is a model-driving CLI executor: it runs a model-chosen agentic loop that
        # executes shell commands, so it legitimately needs LLM provider credentials
        # (inherit_credentials=True) to authenticate against the model endpoint. But the previous
        # `os.environ.copy()` also handed it every Tier-1 Hermes secret — gateway bot tokens, GitHub auth,
        # Modal/Daytona infra tokens, the dashboard session token, AUXILIARY_* side-LLM keys,
        # GATEWAY_RELAY_* auth — none of which a coding subprocess has any use for. Route through the
        # centralized helper so Tier-1 + dynamic-internal secrets are always stripped while provider creds
        # still flow, matching copilot_acp_client (#29157 sibling spawn-site gap).
        spawn_env = hermes_subprocess_env(inherit_credentials=True)
        if env:
            spawn_env.update(env)
        if codex_home:
            spawn_env["CODEX_HOME"] = codex_home

        from agent.delegation_context import (
            DELEGATED_CHILD_ENV_MARKER, KANBAN_ENV_KEYS,
            delegated_child_subprocess_env, is_delegated_child_process_context,
            owned_kanban_task,
        )
        # Native shell children remain unowned. Only Hermes' managed MCP tool
        # endpoint acts for this worker; grant it scope via its existing per-server
        # environment (the entry the runtime migration registers), never by granting
        # the whole executor process ownership.
        owned_task = bool(owned_kanban_task())
        # The sandbox bound follows Kanban lineage, not ownership: descendants of a
        # worker (delegate child, cron run, spawned process) keep HERMES_KANBAN_DB
        # plus the fence marker and must not widen the sandbox via profile config.
        # The marker alone (a delegate without Kanban) is no lineage, and its value
        # is the fenced root: never a root to write to.
        kanban_bounded = owned_task or bool(
            (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
            or (is_delegated_child_process_context() and os.environ.get("HERMES_KANBAN_DB"))
        )
        managed_args: list[str] = []
        if owned_task:
            for key in (*KANBAN_ENV_KEYS, "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
                if key in os.environ:
                    managed_args += ["-c", f"mcp_servers.{HERMES_TOOLS_MCP_SERVER_NAME}.env.{key}={json.dumps(os.environ[key])}"]
            managed_args += ["-c", f'mcp_servers.{HERMES_TOOLS_MCP_SERVER_NAME}.env.{DELEGATED_CHILD_ENV_MARKER}=""']
        spawn_env = delegated_child_subprocess_env(spawn_env)
        kanban_root = None
        if owned_task:
            kanban_db = spawn_env.get("HERMES_KANBAN_DB")
            default_root = os.path.join(spawn_env.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "kanban")
            if kanban_db:
                kanban_root = os.path.dirname(kanban_db)
            else:
                kanban_root = spawn_env.get("HERMES_KANBAN_ROOT", default_root)
        cmd = [
            codex_bin, "app-server",
            *(extra_args or []),
            *managed_args,
            *resolve_codex_app_server_args(
                config=config, kanban_root=kanban_root, kanban_bounded=kanban_bounded
            ),
        ]
        # Codex emits tracing to stderr; default WARN keeps it quiet for users.
        spawn_env.setdefault("RUST_LOG", "warn")

        # Hide the console the codex child would otherwise flash on Windows (#56747).
        # Hide-only — stdio pipes stay intact for the app-server wire.
        # See #56747.
        from hermes_cli._subprocess_compat import windows_hide_flags

        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, env=spawn_env, creationflags=windows_hide_flags(),
        )
        self._next_id = 1
        self._pending: dict[int, queue.Queue] = {}  # request id -> single-slot reply queue
        self._pending_lock = threading.Lock()
        self._notifications: queue.Queue = queue.Queue()
        self._server_requests: queue.Queue = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._closed = False
        self._initialized = False

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()

    def initialize(
        self, client_name: str = "hermes", client_title: str = "Hermes Agent",
        client_version: str = "0.1", capabilities: Optional[dict] = None, timeout: float = 10.0,
    ) -> dict:
        """Send ``initialize`` + ``initialized``; return the server's InitializeResponse."""
        if self._initialized:
            raise RuntimeError("already initialized")
        params = {
            "clientInfo": {"name": client_name, "title": client_title, "version": client_version},
            "capabilities": capabilities or {},
        }
        result = self.request("initialize", params, timeout=timeout)
        self.notify("initialized")
        self._initialized = True
        return result

    def close(self, timeout: float = 3.0) -> None:
        """Close stdin and wait for the subprocess TREE to exit, escalating to kill.

        Codex app-server owns stdio MCP descendants that may sit in their own process
        groups (``setsid``). Once the root exits they reparent and a parent walk can no
        longer find them, so descendants are snapshotted BEFORE the root is retired and
        the proven identities (PID + create time) are swept afterwards."""
        if self._closed:
            return
        self._closed = True
        self._fail_pending_requests("codex app-server client is closing")
        descendants = _snapshot_descendants(self._proc.pid)
        with contextlib.suppress(Exception):
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                kill_process_tree(self._proc.pid)
                self._proc.wait(timeout=1.0)
        finally:
            _reap_snapshotted(descendants)

    def _fail_pending_requests(self, reason: str) -> None:
        """Unblock every thread currently sitting in request() instead of
        leaving them to ride out their own per-call timeout (up to 30s by
        default) after the transport they're waiting on has already died.
        Mirrors _read_stdout's own pop-then-deliver dispatch under the same
        lock, so a reply that lands at the exact same moment still wins the
        race cleanly instead of being dropped or double-delivered."""
        with self._pending_lock:
            pending_items = list(self._pending.items())
            self._pending.clear()
        if not pending_items:
            return
        synthetic = {"error": {"code": _TRANSPORT_LOST_CODE, "message": reason}, "transportLost": True}
        for _rid, pending in pending_items:
            pending.put_nowait(synthetic)

    def __enter__(self) -> CodexAppServerClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict:
        """Send a request and block for ``result``; raise CodexAppServerError on ``error``."""
        rid, self._next_id = self._next_id, self._next_id + 1
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = q
        try:
            self._send({"id": rid, "method": method, "params": params or {}})
        except CodexAppServerTransportError:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"codex app-server method {method!r} timed out after {timeout}s")
        if "error" in msg:
            err = msg["error"]
            cls = CodexAppServerTransportError if msg.get("transportLost") else CodexAppServerError
            raise cls(code=err.get("code", -1), message=err.get("message", ""), data=err.get("data"))
        return msg.get("result", {})

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        self._send({"method": method, "params": params or {}})

    def respond(self, request_id: Any, result: dict) -> None:
        """Reply to a server-initiated request (e.g. approval prompts)."""
        self._send({"id": request_id, "result": result})

    def respond_error(self, request_id: Any, code: int, message: str, data: Optional[Any] = None) -> None:
        """Reply to a server-initiated request with an error."""
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"id": request_id, "error": err})

    @staticmethod
    def _take(q: queue.Queue, timeout: float) -> Optional[dict]:
        try:
            return q.get_nowait() if timeout <= 0 else q.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_notification(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next streaming notification, or None on timeout (0 = non-blocking)."""
        return self._take(self._notifications, timeout)

    def take_server_request(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next server-initiated request (e.g. exec/applyPatch approval)."""
        return self._take(self._server_requests, timeout)

    def stderr_tail(self, n: int = 20) -> list[str]:
        """Return last n lines of codex's stderr (for error reports)."""
        if self._proc.poll() is not None:
            # The reader can still hold the crash text when the exit is first observed (#121467).
            self._stderr_reader.join(timeout=1.0)
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def _send(self, obj: dict) -> None:
        if self._closed:
            raise CodexAppServerTransportError(code=_TRANSPORT_LOST_CODE, message="codex app-server client is closed")
        if self._proc.stdin is None:
            raise CodexAppServerTransportError(code=_TRANSPORT_LOST_CODE, message="codex app-server stdin not available")
        try:
            self._proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:  # BrokenPipe, EINVAL on a torn-down pipe, write on closed file
            raise CodexAppServerTransportError(
                code=_TRANSPORT_LOST_CODE, message=f"codex app-server stdin closed unexpectedly: {exc}",
            ) from exc

    def _append_stderr(self, line: str) -> None:
        with self._stderr_lock:
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 500:  # bound memory
                self._stderr_lines = self._stderr_lines[-500:]

    def _read_stdout(self) -> None:
        if self._proc.stdout is None:
            return
        try:
            for line in iter(self._proc.stdout.readline, b""):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON stdout is unexpected; surface it via the stderr buffer.
                    self._append_stderr(f"<non-json on stdout> {line[:200]!r}")
                    continue
                self._dispatch(msg)
        except Exception as exc:
            self._append_stderr(f"<stdout reader error> {exc}")
        finally:
            # EOF (codex died) or a reader failure: nobody will ever answer the
            # requests still waiting, so fail them now instead of letting each
            # ride out its per-call timeout.
            self._fail_pending_requests("codex app-server stdout closed")

    def _dispatch(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):  # reply
            with self._pending_lock:
                pending = self._pending.pop(msg["id"], None)
            if pending is not None:
                with contextlib.suppress(queue.Full):  # pragma: no cover - defensive
                    pending.put_nowait(msg)
        elif "method" in msg:  # server-initiated request (has id) or notification
            (self._server_requests if "id" in msg else self._notifications).put(msg)

    def _read_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        with contextlib.suppress(Exception):  # pragma: no cover
            for line in iter(self._proc.stderr.readline, b""):
                self._append_stderr(line.decode("utf-8", "replace").rstrip())


def parse_codex_version(output: str) -> Optional[tuple[int, int, int]]:
    """Parse ``codex --version`` output ("codex-cli 0.130.0 ...") into (major, minor, patch)."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output or "")
    return tuple(int(g) for g in match.groups()) if match else None


def check_codex_binary(
    codex_bin: str = "codex", min_version: tuple[int, int, int] = MIN_CODEX_VERSION
) -> tuple[bool, str]:
    """Verify codex CLI is installed and meets minimum version. Returns (ok, message)."""
    try:
        proc = subprocess.run(
            [codex_bin, "--version"], capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=10, stdin=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return False, f"codex CLI not found at {codex_bin!r}. Install with: npm i -g @openai/codex"
    except subprocess.TimeoutExpired:
        return False, "codex --version timed out"
    if proc.returncode != 0:
        return False, f"codex --version exited {proc.returncode}: {proc.stderr.strip()}"
    version = parse_codex_version(proc.stdout)
    if version is None:
        return False, f"could not parse codex version from: {proc.stdout!r}"
    have = ".".join(map(str, version))
    if version < min_version:
        return False, f"codex {have} is older than required {'.'.join(map(str, min_version))}. Run: npm i -g @openai/codex"
    return True, have
