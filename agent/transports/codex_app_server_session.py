"""Session adapter for codex app-server runtime.

Owns one Codex thread per Hermes session: drives ``turn/start``, consumes
streaming notifications via CodexEventProjector, bridges server-initiated
approval requests, translates cancellation, and returns a TurnResult that
AIAgent.run_conversation() splices into ``messages``. Synchronous: the client's
reader threads feed queues that this adapter polls, like the chat_completions loop.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.codex_responses_adapter import _format_responses_error
from agent.redact import redact_sensitive_text
from agent.transports.codex_app_server import (
    CodexAppServerClient, CodexAppServerError, CodexAppServerTransportError,
)
from agent.transports.codex_event_projector import CodexEventProjector, ProjectionResult
from agent.transports.hermes_tools_mcp_server import HERMES_TOOLS_MCP_SERVER_NAME

logger = logging.getLogger(__name__)


_STDERR_TAIL_LINES = 12  # stderr tail on generic errors: legible, yet enough for a config/auth diagnostic

# Hermes' tools.terminal.security_mode -> Codex permissions profile id.
# Missing config -> workspace-write (Codex's own default).
_HERMES_TO_CODEX_PERMISSION_PROFILE = {
    "auto": "workspace-write", "approval-required": "read-only-with-approval",
    "unrestricted": "full-access", "yolo": "full-access",  # yolo: backstop alias used by some skills/tests
}


@dataclass
class TurnResult:
    """Result of one user→assistant→tool turn through the codex app-server."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None  # non-recoverable turn error
    turn_id: Optional[str] = None
    thread_id: Optional[str] = None
    # Exact turn/start text distinguishes the input echo from a new user event.
    submitted_user_text: Optional[str] = None
    token_usage_last: Optional[dict[str, Any]] = None
    model_context_window: Optional[int] = None
    compacted: bool = False
    # Codex likely wedged (turn timeout, dead subprocess, token refresh failure): caller respawns next turn.
    should_retire: bool = False


# Some codex versions stream ``<turn_aborted>`` as raw agentMessage text when an
# interrupt/upstream error tears the turn down without emitting turn/completed.
_TURN_ABORTED_MARKERS = ("<turn_aborted>", "<turn_aborted/>")


def _first_scope_id(*lookups: tuple[Any, str, str]) -> Any:
    """``src.get(a) or src.get(b)`` over successive dict sources until one is not None."""
    for src, primary, fallback in lookups:
        if isinstance(src, dict):
            observed = src.get(primary) or src.get(fallback)
            if observed is not None:
                return observed
    return None


def _notification_scope_ids(note: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract the thread/turn identity carried by a notification (top-level, then turn/item)."""
    params = (note.get("params") or {}) if isinstance(note, dict) else None
    if not isinstance(params, dict):
        return None, None
    turn, item = params.get("turn") or {}, params.get("item") or {}
    return (
        _first_scope_id((params, "threadId", "thread_id"), (turn, "threadId", "thread_id"), (item, "threadId", "thread_id")),
        _first_scope_id((params, "turnId", "turn_id"), (turn, "id", "turnId"), (item, "turnId", "turn_id")),
    )


def _notification_belongs_to_turn(note: dict, *, thread_id: Optional[str], turn_id: Optional[str]) -> bool:
    """Whether a multiplexed notification belongs to this turn.

    One connection can carry parent and hosted subagent threads; an explicitly
    foreign thread/turn event must not mutate this transcript. Unscoped
    notifications remain accepted for protocol compatibility.
    """
    if not isinstance(note, dict):
        return False
    observed = _notification_scope_ids(note)
    return not any(
        expected is not None and seen is not None and str(seen) != str(expected)
        for expected, seen in zip((thread_id, turn_id), observed)
    )


_TEXT_PART_TYPES = frozenset({"text", "input_text"})
_IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})
_IMAGE_URL_SCHEMES = ("data:", "http://", "https://")


def _image_part_to_turn_input(item: dict) -> Optional[dict]:
    """Map one Hermes image part onto the app-server ``UserInput`` shape.

    ``turn/start`` accepts ``{type: image, url}`` (data:/http URLs) and ``{type: localImage, path}``
    natively (protocol schema ``v2/UserInput``), so nothing here is flattened into a text marker.
    """
    ref = item.get("image_url") or item.get("url") or item.get("path") or item.get("image")
    if isinstance(ref, dict):
        ref = ref.get("url") or ref.get("path")
    ref = (ref or "").strip() if isinstance(ref, str) else ""
    if not ref:
        return None
    if ref.startswith(_IMAGE_URL_SCHEMES):
        return {"type": "image", "url": ref}
    if ref.startswith("file://"):
        ref = ref[len("file://"):]
    return {"type": "localImage", "path": ref}


def _build_turn_input(user_input: Any) -> tuple[list[dict], str]:
    """Build the ``turn/start`` ``input`` list plus the text the wire will echo back.

    Text parts stay text; image parts ride natively (#51053 — a text marker in their place left the
    model blind to the attachment). Returns ``(input_items, submitted_text)``.
    """
    if isinstance(user_input, str):
        return [{"type": "text", "text": user_input}], user_input
    if not isinstance(user_input, list):
        text = "" if user_input is None else str(user_input)
        return [{"type": "text", "text": text}], text
    texts: list[str] = []
    images: list[dict] = []
    for item in user_input:
        if not isinstance(item, dict):
            if item.strip() if isinstance(item, str) else item is not None:
                texts.append(str(item))
        elif item.get("type") in _TEXT_PART_TYPES:
            texts.append(str(item.get("text") or item.get("content") or ""))
        elif item.get("type") in _IMAGE_PART_TYPES:
            mapped = _image_part_to_turn_input(item)
            if mapped is not None:
                images.append(mapped)
    text = "\n\n".join(t for t in texts if t).strip()
    if not text and images:
        text = "What do you see in this image?"
    items: list[dict] = [{"type": "text", "text": text}] if text or not images else []
    return items + images, text


# Strong credential-failure signals: trusted whether they appear in the primary
# JSON-RPC error or in ambient app-server stderr.
_OAUTH_REFRESH_FAILURE_HINTS = (
    "invalid_grant", "invalid grant", "refresh token", "refresh_token", "token refresh", "token_refresh",
    "token has expired", "expired_token", "token_expired", "expired token", "not authenticated", "unauthenticated",
    "re-authenticate", "reauthenticate", "please log in", "please login", "no auth profile",
)
# Generic auth words are authoritative only in the primary error. codex writes
# independent ChatGPT plugin prewarm failures ("HTTP 401 Unauthorized") to stderr,
# so there they must not mask an unrelated RPC error or timeout (#75167).
_PRIMARY_ONLY_OAUTH_HINTS = ("401 unauthorized", "unauthorized", "oauth", "auth profile")

_OAUTH_REAUTH_HINT = (
    "Codex authentication failed — your ChatGPT/Codex login looks expired or invalid. Run `codex login` to refresh, "
    "then retry. (Fall back to default runtime with `/codex-runtime auto` if the issue persists.)"
)


def _classify_oauth_failure(primary: str = "", *, stderr: str = "") -> Optional[str]:
    """Re-auth hint when ``primary`` (the operation's own error) or ``stderr`` proves the codex login is broken."""
    primary_l = (primary or "").lower()
    stderr_l = (stderr or "").lower()
    if any(n in primary_l for n in _OAUTH_REFRESH_FAILURE_HINTS + _PRIMARY_ONLY_OAUTH_HINTS):
        return _OAUTH_REAUTH_HINT
    return _OAUTH_REAUTH_HINT if any(n in stderr_l for n in _OAUTH_REFRESH_FAILURE_HINTS) else None


@dataclass
class _ServerRequestRouting:
    """Default approval policies when no interactive approval_callback is wired in (tests, cron)."""

    auto_approve_exec: bool = False
    auto_approve_apply_patch: bool = False
    prompt_mcp_elicitation_servers: frozenset[str] = frozenset()


def _is_empty_confirmation_elicitation(params: dict) -> bool:
    """Return whether Codex is asking for a binary form confirmation."""
    if params.get("mode") != "form":
        return False
    schema = params.get("requestedSchema")
    if not isinstance(schema, dict):
        return False
    return (
        schema.get("type") == "object"
        and schema.get("properties") == {}
        and schema.get("required") in (None, [])
    )


class CodexThreadResumeError(CodexAppServerError):
    """``thread/resume`` did not hand back the stored thread (unknown/garbage id, rollout still locked by
    a killed app-server, or codex answered with a different thread). The caller decides the policy."""

    def __init__(self, thread_id: str, detail: str) -> None:
        super().__init__(code=-32600, message=f"codex thread {thread_id[:8]} could not be resumed: {detail}")
        self.thread_id = thread_id


def _extract_thread_id(result: dict) -> Optional[str]:
    """Different codex versions serialize the id under thread.id / sessionId / threadId."""
    thread_obj = result.get("thread") or {}
    return thread_obj.get("id") or thread_obj.get("sessionId") or result.get("sessionId") or result.get("threadId")


class CodexAppServerSession:
    """One Codex thread per Hermes session, lifetime owned by AIAgent. Not thread-safe: one caller at a time."""

    def __init__(
        self, *, cwd: Optional[str] = None, codex_bin: str = "codex",
        codex_home: Optional[str] = None, permission_profile: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        mcp_elicitation_callback: Optional[Callable[..., str]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        request_routing: Optional[_ServerRequestRouting] = None,
        client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
        model: Optional[str] = None, model_provider: Optional[str] = None,
        developer_instructions: Optional[str] = None, resume_thread_id: Optional[str] = None,
        history_seed: Optional[str] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._codex_bin = codex_bin
        self._codex_home = codex_home
        # A codex thread id persisted by an earlier process for this Hermes session: the first
        # ``ensure_started`` issues ``thread/resume`` for it instead of ``thread/start``.
        self._resume_thread_id = resume_thread_id
        # ``thread/start.model`` / ``.modelProvider``: select a provider from codex's own
        # ``[model_providers.<id>]`` table. Only the id travels; codex reads base_url/env_key itself.
        self._model = (model or "").strip() or None
        self._model_provider = (model_provider or "").strip() or None
        # Hermes' composed system prompt (SOUL.md, memory, channel overrides). Sent ONCE per thread as
        # ``thread/start.developerInstructions``: codex keeps its own base instructions (tool guidance) and
        # inserts this as the first developer message of every model request. ``baseInstructions`` would
        # REPLACE codex's base and ``instructions`` is accepted but ignored (verified against codex 0.147).
        self._developer_instructions = developer_instructions
        # Hermes' prior transcript, appended to developerInstructions ONLY when a thread is started from
        # scratch: a resumed thread already holds the conversation (agent/codex_runtime_history_seed.py).
        self._history_seed = history_seed
        self._permission_profile = permission_profile or _HERMES_TO_CODEX_PERMISSION_PROFILE.get(
            os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"), "workspace-write"
        )
        self._approval_callback = approval_callback
        self._mcp_elicitation_callback = mcp_elicitation_callback
        self._on_event = on_event  # Display hook (kawaii spinner ticks etc.)
        self._routing = request_routing or _ServerRequestRouting()
        self._client_factory = client_factory or CodexAppServerClient

        self._client: Optional[CodexAppServerClient] = None
        self._thread_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._active_turn_id: Optional[str] = None
        self._active_turn_lock = threading.Lock()
        # In-progress fileChange items by id (item/started -> item/completed):
        # approval params don't carry the changeset, so this feeds the prompt summary.
        self._pending_file_changes: dict[str, str] = {}
        self._closed = False

    def ensure_started(self) -> str:
        """Spawn, handshake, and ``thread/start`` (or ``thread/resume`` for a stored id); idempotent, returns
        the codex thread id. A failed resume raises :class:`CodexThreadResumeError` once; the next call
        starts a fresh thread on the same handshaken client."""
        if self._thread_id is not None:
            return self._thread_id
        if self._client is None:
            self._client = self._client_factory(codex_bin=self._codex_bin, codex_home=self._codex_home)
            self._client.initialize(client_name="hermes", client_title="Hermes Agent", client_version=_get_hermes_version())
        # Permissions are NOT sent on thread/start: codex gates ``thread/start.permissions``
        # behind experimentalApi + a matching ``[permissions]`` table in ~/.codex/config.toml.
        # Hermes supplies the agent identity through its own system prompt; ``personality: "none"`` strips
        # codex's built-in "# Personality" section from the base instructions so it cannot compete (#72104).
        params: dict[str, Any] = {"cwd": self._cwd, "personality": "none"}
        if self._developer_instructions and self._developer_instructions.strip():
            params["developerInstructions"] = self._developer_instructions
        if self._model_provider:
            params["modelProvider"] = self._model_provider
        if self._model:
            params["model"] = self._model
        if self._resume_thread_id:
            wanted, self._resume_thread_id = self._resume_thread_id, None  # one attempt per stored id
            thread_id = self._resume_thread(wanted, params)
            logger.info("codex app-server thread resumed: id=%s cwd=%s", thread_id[:8], self._cwd)
        else:
            if self._history_seed:
                params["developerInstructions"] = "\n\n".join(
                    part for part in (params.get("developerInstructions"), self._history_seed) if part)
            result = self._client.request("thread/start", params, timeout=15)
            thread_id = _extract_thread_id(result)
            if not thread_id:
                raise CodexAppServerError(
                    code=-32603, message=f"codex thread/start returned no thread id (payload keys: {sorted(result.keys())})",
                )
            logger.info("codex app-server thread started: id=%s profile=%s cwd=%s", thread_id[:8], self._permission_profile, self._cwd)
        self._thread_id = thread_id
        return thread_id

    def _resume_thread(self, wanted: str, params: dict[str, Any]) -> str:
        """``thread/resume`` for the stored id; the same thread/start params ride along so the resumed thread
        carries the CURRENT prompt composition and provider (accepted by the resume schema, codex 0.147)."""
        assert self._client is not None
        try:
            result = self._client.request("thread/resume", {"threadId": wanted, **params}, timeout=15)
        except CodexAppServerError as exc:
            raise CodexThreadResumeError(wanted, exc.message) from exc
        thread_id = _extract_thread_id(result)
        if thread_id != wanted:
            raise CodexThreadResumeError(wanted, f"app-server answered with thread {str(thread_id)[:8]!r}")
        return wanted

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._active_turn_lock:
            self._active_turn_id = None
        if self._client is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - best-effort cleanup
                self._client.close()
        self._client = None
        self._thread_id = None

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to issue turn/interrupt and unwind."""
        self._interrupt_event.set()

    def request_steer(self, text: str) -> bool:
        """Append user guidance to the active Codex turn via ``turn/steer``."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        with self._active_turn_lock:
            turn_id, thread_id, client = self._active_turn_id, self._thread_id, self._client
        if not turn_id or not thread_id or client is None:
            return False
        try:
            response = client.request(
                "turn/steer",
                {"threadId": thread_id, "input": [{"type": "text", "text": cleaned}], "expectedTurnId": turn_id}, timeout=10,
            )
        except (CodexAppServerError, TimeoutError):
            logger.debug("turn/steer rejected for active Codex turn", exc_info=True)
            return False
        accepted_turn_id = response.get("turnId") if isinstance(response, dict) else None
        return accepted_turn_id in {None, turn_id}

    def _format_error_with_stderr(self, prefix: str, exc: Any = "", *, tail_lines: int = _STDERR_TAIL_LINES) -> str:
        """User-facing error string plus the force-redacted stderr tail (keeps secrets out of chat output)."""
        exc_str = "" if exc is None else str(exc)
        base = f"{prefix}: {exc_str}" if exc_str else prefix
        try:
            tail = self._client.stderr_tail(tail_lines) if self._client is not None else []
        except Exception:  # pragma: no cover - diagnostic best-effort
            return base
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return base
        return f"{base}\ncodex stderr (last {len(tail)} lines):\n{redact_sensitive_text(joined, force=True)}"

    def _stderr_blob(self, n: int) -> str:
        client = self._client
        return "" if client is None else "\n".join(client.stderr_tail(n))

    @staticmethod
    def _retire(result: TurnResult, error: str) -> None:
        """Record a terminal error and flag the session for respawn on the next turn."""
        result.error = error
        result.should_retire = True

    def _set_classified_error(self, result: TurnResult, prefix: str, classify_text: str, detail: Any) -> None:
        """OAuth failures -> re-auth hint AND retire (token store broken though JSON-RPC is fine); else stderr tail."""
        hint = _classify_oauth_failure(classify_text, stderr=self._stderr_blob(40))
        if hint is not None:
            self._retire(result, hint)
        else:
            result.error = self._format_error_with_stderr(prefix, detail)

    def _start_for(self, result: TurnResult) -> bool:
        """ensure_started(); startup failures become a retiring TurnResult.error instead of raw exceptions."""
        try:
            self.ensure_started()
        except (CodexAppServerError, TimeoutError) as exc:
            self._retire(result, self._format_error_with_stderr("codex app-server startup failed", exc))
            return False
        assert self._client is not None and self._thread_id is not None
        result.thread_id = self._thread_id
        return True

    def _request_for(self, result: TurnResult, method: str, params: dict, label: str) -> Optional[dict]:
        """Issue ``method``; on failure fill ``result.error`` and return None. A timeout always retires."""
        try:
            return self._client.request(method, params, timeout=10)
        except CodexAppServerTransportError as exc:
            self._retire(result, self._format_error_with_stderr(f"{label} failed", exc))
        except CodexAppServerError as exc:
            self._set_classified_error(result, f"{label} failed", exc.message, exc)
        except TimeoutError as exc:
            hint = _classify_oauth_failure(stderr=self._stderr_blob(40))
            self._retire(result, hint or self._format_error_with_stderr(f"{label} timed out", exc))
        return None

    def _subprocess_died(self, result: TurnResult, client: Optional[CodexAppServerClient]) -> bool:
        """Bail out early (rather than waiting on the deadline) when codex exited or close() ran.

        ``client`` is the loop's snapshot: close() on another thread nulls ``self._client``
        mid-turn (session expiry), which must end the turn, not raise AttributeError. A
        ``None`` snapshot means close() already landed before the loop started.
        """
        if client is None or self._closed or self._client is not client:
            result.interrupted = True
            self._retire(result, "codex app-server session closed while the turn was in flight")
            return True
        if client.is_alive():
            return False
        hint = _classify_oauth_failure(stderr=self._stderr_blob(60))
        self._retire(result, hint or self._format_error_with_stderr("codex app-server subprocess exited unexpectedly", tail_lines=20))
        return True

    def _absorb_notification(
        self, result: TurnResult, projector: CodexEventProjector, note: dict
    ) -> tuple[ProjectionResult, bool]:
        """Fan one in-scope notification out to display, accounting, file-change tracking and the projector.

        Returns (projection, aborted); aborted = agent text carried a terminal ``<turn_aborted>`` marker.
        """
        if self._on_event is not None:
            try:
                self._on_event(note)
            except Exception:  # pragma: no cover - display callback
                logger.debug("on_event callback raised", exc_info=True)
        _apply_accounting_notification(result, note)
        self._track_pending_file_change(note)
        projection = projector.project(note)
        if projection.messages:
            result.projected_messages.extend(projection.messages)
        if projection.is_tool_iteration:
            result.tool_iterations += 1
        aborted = False
        if projection.final_text is not None:
            # Multiple agentMessage items per turn: the last one is canonical.
            result.final_text = projection.final_text
            aborted = _has_turn_aborted_marker(projection.final_text)
            if aborted:
                result.interrupted = True
                result.error = result.error or "codex reported turn_aborted"
        return projection, aborted

    def run_turn(
        self, user_input: Any, *, turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25, post_tool_quiet_timeout: float = 90.0,
    ) -> TurnResult:
        """Send a user message and block until turn/completed, bridging approvals and projecting items.

        post_tool_quiet_timeout: if codex emits a tool completion and then goes quiet for this many seconds
        without emitting another item or `turn/completed`, log a warning (once per tool result) and keep
        waiting. Wire silence is not evidence of a wedged process: after a large tool output codex can
        reason for minutes without emitting a single event while the app-server still answers RPCs
        (#112928). Only subprocess death or ``turn_timeout`` retires the session.
        """
        result = TurnResult()
        if self._start_for(result):
            # Do not clear first: a hard stop arriving during ensure_started() must
            # be honored before launching a Codex turn.
            if self._interrupt_event.is_set():
                result.interrupted = True
            else:
                input_items, result.submitted_user_text = _build_turn_input(user_input)
                ts = self._request_for(
                    result, "turn/start",
                    {"threadId": self._thread_id, "input": input_items},
                    "turn/start",
                )
                if ts is not None:
                    self._run_started_turn(result, ts, turn_timeout, notification_poll_timeout, post_tool_quiet_timeout)
        self._interrupt_event.clear()
        return result

    def _run_started_turn(
        self, result: TurnResult, ts: dict, turn_timeout: float, notification_poll_timeout: float,
        post_tool_quiet_timeout: float,
    ) -> None:
        """Drive an accepted ``turn/start`` to completion: quiet warning, approvals, projection."""
        projector = CodexEventProjector()
        client = self._client
        result.turn_id = (ts.get("turn") or {}).get("id")
        with self._active_turn_lock:
            self._active_turn_id = result.turn_id
        # Post-tool quiet timer: armed on each tool completion, cleared by any other activity.
        # Observability only — it never interrupts or retires (see run_turn docstring).
        last_tool_completion_at: Optional[float] = None

        def warn_if_quiet() -> bool:
            nonlocal last_tool_completion_at
            if last_tool_completion_at is None or (time.monotonic() - last_tool_completion_at) <= post_tool_quiet_timeout:
                return False
            last_tool_completion_at = None
            logger.warning(
                "codex has emitted no events for %.0fs after a tool result; still waiting (turn deadline %.0fs)",
                post_tool_quiet_timeout, turn_timeout,
            )
            return False

        def on_server_request(sreq: dict) -> bool:
            nonlocal last_tool_completion_at
            # Drain pending notifications first (bounded) so _pending_file_changes is
            # current for the approval decision and display events still reach on_event.
            turn_complete = False
            for _ in range(8):
                pending = client.take_notification(timeout=0)
                if pending is None:
                    break
                if not _notification_belongs_to_turn(pending, thread_id=self._thread_id, turn_id=result.turn_id):
                    logger.debug("ignoring foreign codex notification while draining server request: method=%s", pending.get("method"))
                    continue
                proj, aborted = self._absorb_notification(result, projector, pending)
                if proj.is_tool_iteration:
                    last_tool_completion_at = time.monotonic()
                turn_complete = turn_complete or aborted
            self._handle_server_request(sreq)
            # An approval round-trip is live signal — don't let it trip the quiet warning.
            last_tool_completion_at = None
            return turn_complete

        def on_note(note: dict, method: str) -> bool:
            nonlocal last_tool_completion_at
            projection, aborted = self._absorb_notification(result, projector, note)
            if projection.is_tool_iteration:
                last_tool_completion_at = time.monotonic()
            elif projection.messages or projection.final_text is not None:
                last_tool_completion_at = None
            if method != "turn/completed":
                return aborted
            turn_obj = (note.get("params") or {}).get("turn") or {}
            turn_status = turn_obj.get("status")
            if turn_status and turn_status not in {"completed", "interrupted"} and turn_obj.get("error"):
                err_msg = _format_responses_error(turn_obj["error"], str(turn_status))
                self._set_classified_error(result, f"turn ended status={turn_status}", err_msg, err_msg)
            return True

        self._drive_turn(
            result, turn_timeout=turn_timeout, notification_poll_timeout=notification_poll_timeout,
            timeout_label="turn", before_poll=warn_if_quiet, on_server_request=on_server_request,
            on_note=on_note, accept_final_text_at_deadline=True,
        )
        with self._active_turn_lock:
            self._active_turn_id = None

    def _drive_turn(
        self, result: TurnResult, *, turn_timeout: float, notification_poll_timeout: float,
        timeout_label: str, on_server_request: Callable[[dict], bool],
        on_note: Callable[[dict, str], bool], before_poll: Optional[Callable[[], bool]] = None,
        pre_scope_filter: Optional[Callable[[dict, str], bool]] = None,
        accept_final_text_at_deadline: bool = False,
    ) -> None:
        """Shared poll loop for run_turn / compact_thread until turn/completed or deadline.

        Per iteration: interrupt -> subprocess death -> ``before_poll`` (quiet warning) ->
        server requests (answered first so codex isn't blocked) -> one notification,
        filtered by ``pre_scope_filter`` then turn scope, handed to ``on_note``. Hooks
        return True to complete the turn. Deadline without completion interrupts and
        retires the session.
        """
        deadline = time.monotonic() + turn_timeout
        turn_complete = False
        client = self._client
        while time.monotonic() < deadline and not turn_complete:
            if self._interrupt_event.is_set():
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                break
            if self._subprocess_died(result, client) or client is None:  # `is None` narrows only; already retired
                break
            if before_poll is not None and before_poll():
                break
            sreq = client.take_server_request(timeout=0)
            if sreq is not None:
                try:
                    turn_complete = on_server_request(sreq)
                except CodexAppServerTransportError as exc:
                    self._retire(result, self._format_error_with_stderr("codex app-server request response failed", exc))
                    break
                continue
            note = client.take_notification(timeout=notification_poll_timeout)
            if note is None:
                continue
            method = note.get("method", "")
            if pre_scope_filter is not None and not pre_scope_filter(note, method):
                continue
            if not _notification_belongs_to_turn(note, thread_id=self._thread_id, turn_id=result.turn_id):
                logger.debug("ignoring foreign codex notification: method=%s", method)
                continue
            turn_complete = on_note(note, method)

        if accept_final_text_at_deadline and not turn_complete and not result.interrupted and result.final_text and result.error is None:
            logger.warning(
                "codex app-server turn reached deadline after a completed assistant message but before "
                "turn/completed; accepting the assistant text as the terminal response"
            )
            turn_complete = True

        if not turn_complete and not result.interrupted:
            self._issue_interrupt(result.turn_id)
            result.interrupted = True
            if not result.error:
                result.error = self._format_error_with_stderr(f"{timeout_label} timed out after {turn_timeout}s")
            result.should_retire = True

    def compact_thread(
        self, *, turn_timeout: float = 600.0, notification_poll_timeout: float = 0.25
    ) -> TurnResult:
        """Trigger Codex-native history compaction for the current thread.

        ``thread/compact/start`` returns immediately with no turn id; progress streams
        as normal turn/item notifications, so wait for the matching ``turn/completed``.
        """
        result = TurnResult()
        if not self._start_for(result):
            return result
        self._interrupt_event.clear()
        projector = CodexEventProjector()

        if self._request_for(result, "thread/compact/start", {"threadId": self._thread_id}, "thread/compact/start") is None:
            return result

        def pre_scope_filter(note: dict, method: str) -> bool:
            if result.turn_id is not None:
                return True
            observed_thread_id, observed_turn_id = _notification_scope_ids(note)
            if method == "turn/started":
                if observed_thread_id is not None and str(observed_thread_id) != str(self._thread_id):
                    logger.debug("ignoring foreign compact turn/started: thread=%s", observed_thread_id)
                    return False
                if observed_turn_id is None:
                    logger.debug("ignoring compact turn/started without a turn id")
                    return False
                result.turn_id = str(observed_turn_id)
            elif observed_turn_id is not None or method in {"item/completed", "turn/completed"}:
                # Before the new turn/started, terminal/projectable events are stale or unattributable.
                logger.debug("ignoring codex notification before compact turn start: method=%s", method)
                return False
            return True

        def on_note(note: dict, method: str) -> bool:
            _, aborted = self._absorb_notification(result, projector, note)
            if method not in {"turn/started", "turn/completed"}:
                return aborted
            turn_obj = (note.get("params") or {}).get("turn") or {}
            result.turn_id = turn_obj.get("id") or result.turn_id
            if method == "turn/started":
                return aborted
            turn_status = turn_obj.get("status")
            if turn_status == "interrupted":
                result.interrupted = True
                result.error = result.error or "compact turn interrupted"
            elif turn_status and turn_status != "completed":
                err_msg = _format_responses_error(turn_obj.get("error"), str(turn_status))
                self._set_classified_error(result, f"compact turn ended status={turn_status}", err_msg, err_msg)
            return True

        def on_server_request(sreq: dict) -> bool:
            self._handle_server_request(sreq)
            return False

        self._drive_turn(
            result, turn_timeout=turn_timeout, notification_poll_timeout=notification_poll_timeout,
            timeout_label="compact turn", on_server_request=on_server_request, on_note=on_note,
            pre_scope_filter=pre_scope_filter,
        )
        return result

    def _issue_interrupt(self, turn_id: Optional[str]) -> None:
        client = self._client
        if client is None or self._thread_id is None or turn_id is None:
            return
        try:
            client.request("turn/interrupt", {"threadId": self._thread_id, "turnId": turn_id}, timeout=5)
        except CodexAppServerError as exc:
            # "no active turn to interrupt" is fine — already done.
            logger.debug("turn/interrupt non-fatal: %s", exc)
        except TimeoutError:
            logger.warning("turn/interrupt timed out")

    def _handle_server_request(self, req: dict) -> None:
        """Answer a codex server request (approval / elicitation) via Hermes' approval flow.

        Permission escalations are always declined (the user chose their profile in
        ~/.codex/config.toml); unknown methods get a JSON-RPC error so codex doesn't hang.
        """
        client = self._client
        if client is None:
            return
        method = req.get("method", "")
        rid = req.get("id")
        params = req.get("params") or {}
        handler = self._SERVER_REQUEST_HANDLERS.get(method)
        if handler is None:
            logger.warning("Unknown codex server request: %s", method)
            client.respond_error(rid, code=-32601, message=f"Unsupported method: {method}")
            return
        client.respond(rid, handler(self, params))

    def _respond_elicitation(self, params: dict) -> dict:
        """Route a Codex MCP confirmation through the configured Hermes policy."""
        return self._decide_mcp_elicitation(params)

    _SERVER_REQUEST_HANDLERS: dict[str, Callable[..., dict]] = {
        "item/commandExecution/requestApproval": lambda self, p: {"decision": self._decide_exec_approval(p)},
        "item/fileChange/requestApproval": lambda self, p: {"decision": self._decide_apply_patch_approval(p)},
        "item/permissions/requestApproval": lambda self, p: {"decision": "decline"},
        "mcpServer/elicitation/request": _respond_elicitation,
    }

    def _run_approval_callback(self, auto_approve: bool, prompt: Callable[[], tuple[str, str]], log_label: str) -> str:
        """Protocol routing only: auto-approve, fail-closed without a callback, else ask via ``prompt()``.

        Approval mode/timeout resolution lives upstream (codex_runtime.py derives the
        auto flags; the callback runs the shared gate). Do not re-read config here.
        """
        if auto_approve:
            return "accept"
        if self._approval_callback is None:
            return "decline"
        command, description = prompt()
        try:
            choice = self._approval_callback(command, description, allow_permanent=False)
            return _approval_choice_to_codex_decision(choice)
        except Exception:
            logger.exception("approval_callback raised on %s", log_label)
            return "decline"

    def _decide_exec_approval(self, params: dict) -> str:
        def prompt() -> tuple[str, str]:
            # ``cwd`` is Optional on codex's side; fall back so the prompt is never empty.
            description = f"Codex requests exec in {params.get('cwd') or self._cwd or '<unknown>'}"
            if params.get("reason"):
                description += f" — {params['reason']}"
            return params.get("command") or "", description

        return self._run_approval_callback(self._routing.auto_approve_exec, prompt, "exec request")

    def _decide_apply_patch_approval(self, params: dict) -> str:
        def prompt() -> tuple[str, str]:
            # Params carry reason + grantRoot only; the changeset comes from _track_pending_file_change.
            reason, grant_root = params.get("reason"), params.get("grantRoot")
            change_summary = self._pending_file_changes.get(params.get("itemId") or "") or None
            parts = [p for p in (reason, change_summary, grant_root and f"grants write to {grant_root}") if p]
            detail = change_summary or reason
            return (
                f"apply_patch: {detail}" if detail else "apply_patch",
                "; ".join(parts) if parts else "Codex requests to apply a patch",
            )

        return self._run_approval_callback(self._routing.auto_approve_apply_patch, prompt, "apply_patch")

    def _decide_mcp_elicitation(self, params: dict) -> dict:
        """Route an MCP confirmation through Hermes' consent UI.

        Codex uses an empty form schema for binary MCP tool-call approval.
        Only that shape is compatible with Hermes' existing approval UI.
        URL elicitations and forms requesting user-provided fields remain
        fail-closed: accepting either without collecting their input would
        produce a misleading or schema-invalid response.
        """
        response = {"action": "decline", "content": None, "_meta": None}
        raw_server_name = params.get("serverName")
        if not isinstance(raw_server_name, str) or not raw_server_name.strip():
            return response
        server_name = raw_server_name.strip()

        # The internal callback remains separately trusted. It only exposes
        # Hermes tools that the user already enabled for this runtime.
        if server_name == HERMES_TOOLS_MCP_SERVER_NAME:
            return {"action": "accept", "content": None, "_meta": None}

        if server_name not in self._routing.prompt_mcp_elicitation_servers:
            return response
        if not _is_empty_confirmation_elicitation(params):
            return response
        if self._mcp_elicitation_callback is None:
            return response

        raw_message = params.get("message")
        message = (
            raw_message
            if isinstance(raw_message, str) and raw_message.strip()
            else f"Allow MCP server '{server_name}' to perform this action?"
        )
        description = (
            f"Codex requests confirmation for MCP server '{server_name}'."
        )
        try:
            action = self._mcp_elicitation_callback(
                message,
                description,
                surface=f"mcp-elicitation/{server_name}",
            )
        except Exception:
            logger.exception(
                "MCP elicitation callback raised for server %s", server_name
            )
            return response

        if action in {"accept", "decline", "cancel"}:
            response["action"] = action
        return response

    def _track_pending_file_change(self, note: dict) -> None:
        """Track fileChange items (item/started -> item/completed) so the apply_patch prompt can show the changeset."""
        method = note.get("method", "")
        item = (note.get("params") or {}).get("item") or {}
        item_id = item.get("id") or ""
        if item.get("type") != "fileChange" or not item_id:
            return
        if method == "item/completed":
            self._pending_file_changes.pop(item_id, None)
        elif method == "item/started":
            self._pending_file_changes[item_id] = _summarize_file_changes(item.get("changes") or [])


def _summarize_file_changes(raw_changes: list) -> str:
    """One-line ``"<n> add, <m> update: a.py, b.py, +k more"`` summary of a fileChange item's changes."""
    if not raw_changes:
        return "1 change pending"
    changes = [ch for ch in raw_changes if isinstance(ch, dict)]
    kinds: dict[str, int] = {}
    for ch in changes:
        kind = (ch.get("kind") or {}).get("type") or "update"
        kinds[kind] = kinds.get(kind, 0) + 1
    paths: list[str] = [ch["path"] for ch in changes if ch.get("path")]
    counts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
    preview = ", ".join(paths[:3])
    if len(paths) > 3:
        preview += f", +{len(paths) - 3} more"
    return f"{counts}: {preview}" if preview else counts


def _apply_accounting_notification(result: TurnResult, note: dict) -> None:
    """Capture token usage (thread/tokenUsage/updated, not turn/completed) and compaction
    boundaries (a contextCompaction item on recent builds, deprecated thread/compacted on older)."""
    if not isinstance(note, dict):
        return
    method = note.get("method") or ""
    params = note.get("params") or {}
    if not isinstance(params, dict):
        return
    if method == "thread/tokenUsage/updated":
        token_usage = params.get("tokenUsage") or {}
        if isinstance(token_usage, dict):
            last, window = token_usage.get("last"), token_usage.get("modelContextWindow")
            if isinstance(last, dict):
                result.token_usage_last = dict(last)
            if isinstance(window, int) and window > 0:
                result.model_context_window = window
        return
    item = params.get("item") if method in {"item/started", "item/completed"} else None
    if method == "thread/compacted" or (isinstance(item, dict) and item.get("type") == "contextCompaction"):
        result.compacted = True
        result.thread_id = params.get("threadId") or result.thread_id
        result.turn_id = params.get("turnId") or result.turn_id


# Hermes approval choice -> codex decision (app-server-protocol v2). "deny" and
# "timeout" both decline — codex has no "prompt expired" wire value.
_APPROVAL_CHOICE_TO_DECISION = {"once": "accept", "session": "acceptForSession", "always": "acceptForSession"}


def _approval_choice_to_codex_decision(choice: str) -> str:
    """Map a Hermes approval choice onto codex's approval decision wire value."""
    return _APPROVAL_CHOICE_TO_DECISION.get(choice, "decline")


def _has_turn_aborted_marker(text: str) -> bool:
    """True if ``text`` carries a raw ``<turn_aborted>`` marker (terminal without turn/completed)."""
    return bool(text) and any(marker in text for marker in _TURN_ABORTED_MARKERS)


def _get_hermes_version() -> str:
    """Best-effort Hermes version string for codex's userAgent line."""
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:  # pragma: no cover
        return "0.0.0"
