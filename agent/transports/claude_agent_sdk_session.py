"""Session adapter for the claude-agent-sdk runtime.

Owns one Claude Agent SDK client per Hermes session — the structural twin of
``codex_app_server_session.py``, with the Codex JSON-RPC subprocess replaced
by Anthropic's official ``claude-agent-sdk`` (which manages the Claude Code
CLI subprocess, its agent loop, and — critically — **subscription OAuth**:
``CLAUDE_CODE_OAUTH_TOKEN`` / the ``~/.claude`` credential store by default;
known metered sources and Extra Usage fail closed unless the operator opts in).
See GitHub issue #25267.

Lifecycle:
    session = ClaudeAgentSdkSession(cwd="/home/x/proj", model="claude-opus-4-8")
    session.ensure_started()                       # loop thread + SDK connect
    result = session.run_turn(user_input="hello")  # blocks until ResultMessage
    session.close()                                # disconnect + stop loop

Threading model: the SDK is async-first, but AIAgent.run_conversation() is
synchronous (the same constraint that made CodexAppServerClient thread-based).
The adapter owns a dedicated background thread running one asyncio event loop
for the whole session lifetime; every SDK coroutine is marshaled onto it with
``asyncio.run_coroutine_threadsafe`` and awaited with a timeout, so the SDK
client keeps stable loop affinity and ``run_turn`` stays blocking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

# TurnResult is the shared contract with the runtime glue — reused verbatim
# from the codex session module (same fields, same semantics) so
# ``run_claude_agent_sdk_turn`` mirrors ``run_codex_app_server_turn`` 1:1.
from agent.transports.codex_app_server_session import TurnResult
from agent.transports.claude_sdk_event_projector import ClaudeSdkEventProjector
from agent.transports.claude_agent_sdk_session_availability import (
    _safe_sdk_error_text,
    _safe_sdk_traceback,
    check_claude_sdk_available,
    classify_auth_failure,
)
from agent.transports.claude_agent_sdk_session_sanitize import (
    _SDK_CALLBACK_CHOICE_MAX_UTF8_BYTES,
    _SDK_CALLBACK_REASON_MAX_UTF8_BYTES,
    _SDK_FIXED_LOG_TOOL_IDENTITIES,
    _canonical_sdk_tool_request,
    _is_bounded_sdk_callback_string,
    _safe_sdk_deny_log_reason,
    _safe_sdk_tool_use_id,
    safe_sdk_tool_presentation_from_canonical,
    validate_canonical_sdk_request_serialization,
)
from agent.transports.claude_agent_sdk_session_child import (
    ClaudeSdkChildProcessMixin,
    _SDK_DISCONNECT_TIMEOUT_S,
    _force_kill_sdk_child,
    _own_sdk_child_process,
    _sdk_child_pid,
)
from agent.transports.claude_agent_sdk_session_input import (
    _coerce_turn_input,
    _sdk_image_content_block,
    _sdk_user_message_stream,
)
from agent.transports.claude_agent_sdk_session_config import (
    _HERMES_TO_SDK_PERMISSION_MODE,
    _build_hermes_tools_mcp_config,
    _configured_hybrid_exclude,
    _configured_max_buffer_size,
    _configured_permission_mode,
    _configured_post_tool_quiet_timeout,
    _configured_setting_sources,
    _configured_turn_timeout,
    _http_mcp_entries_from_config,
    _is_subscription_oauth_token,
    _mcp_registry_tool_specs,
    _provider_config,
    _provider_flag,
    _sdk_env_overrides,
)
from agent.transports.claude_agent_sdk_session_watchdog import (
    _DEFAULT_POST_TOOL_QUIET_STREAMING,
    _DEFAULT_TURN_TIMEOUT,
    _POLL_STALL_FACTOR,
    _StreamEnd,
    _TURN_ABORT_GRACE,
    _TurnWatch,
    _swallow_interrupt_result,
    _swallow_steer_result,
)
from agent.transports.claude_agent_sdk_session_billing import (
    ClaudeSdkBillingMixin,
)
from agent.transports.claude_agent_sdk_session_compaction import (
    ClaudeSdkCompactionMixin,
)
from agent.transports.claude_agent_sdk_session_notify import (
    ClaudeSdkNotifyMixin,
)

logger = logging.getLogger(__name__)

# The names other packages import through this facade (the class, the runtime's and the
# auxiliary client's config/input entry points, the approval gateway's validators).
__all__ = [
    "ClaudeAgentSdkSession",
    "_coerce_turn_input",
    "_provider_config",
    "_provider_flag",
    "_sdk_env_overrides",
    "_sdk_image_content_block",
    "_sdk_user_message_stream",
    "safe_sdk_tool_presentation_from_canonical",
    "validate_canonical_sdk_request_serialization",
]


# Exact names only: these are the SDK profile's two bounded readers. Never
# widen this to an MCP/server wildcard or an unexposed mutation identity.
_SDK_AUTO_ALLOWED_MCP_TOOLS = frozenset({
    "mcp__hermes-tools__read_file",
    "mcp__hermes-tools__search_files",
})


class ClaudeAgentSdkSession(ClaudeSdkNotifyMixin, ClaudeSdkCompactionMixin, ClaudeSdkBillingMixin, ClaudeSdkChildProcessMixin):
    """One SDK client per Hermes session, lifetime owned by AIAgent.

    Not thread-safe from the caller's side — one caller drives it at a time,
    matching AIAgent.run_conversation(). Internally owns a loop thread."""

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        permission_mode: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        approval_bypass_provider: Optional[Callable[[], bool]] = None,
        on_tool_started: Optional[Callable[[str, str, dict], None]] = None,
        max_budget_usd: Optional[float] = None,
        client_factory: Optional[Callable[..., Any]] = None,
        include_hermes_tools: bool = True,
        hermes_session_id: Optional[str] = None,
        resume_session_id: Optional[str] = None,
        on_stream_delta: Optional[Callable[[str], None]] = None,
        on_interim_assistant: Optional[Callable[[str], None]] = None,
        on_tool_iteration: Optional[Callable[[], None]] = None,
        on_unsolicited_result: Optional[Callable[[list[str]], None]] = None,
        on_compaction: Optional[Callable[[str], None]] = None,
        on_compact_boundary: Optional[Callable[[str], None]] = None,
        # Hybrid MCP bridge (ported from PR #56413): the explicit config
        # opt-in plus both `agent` and `tools` activate in-process servers
        # exposing the full Hermes registry (including proxified third-party
        # MCPs). Missing either input or the opt-in preserves fcava's original
        # stdio-only behavior.
        agent: Optional[Any] = None,
        tools: Optional[list[dict]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._model = model
        requested_permission_mode = (
            permission_mode
            or _configured_permission_mode()
            or _HERMES_TO_SDK_PERMISSION_MODE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "default",
            )
        )
        self._sdk_approval_bypass_requested = (
            requested_permission_mode == "bypassPermissions"
        )
        # The SDK auto-approves before can_use_tool in bypassPermissions mode.
        # Keep Hermes in callback-capable mode and carry the bypass intent into
        # this mandatory session wrapper, where canonical validation and
        # immutable hardline/sudo/user-deny floors run before every selected
        # gateway, CLI, ACP, plugin, or custom callback.
        self._permission_mode = (
            "default"
            if self._sdk_approval_bypass_requested
            else requested_permission_mode
        )
        self._system_prompt_append = system_prompt_append
        self._approval_callback = approval_callback
        self._approval_bypass_provider = approval_bypass_provider
        self._on_tool_started = on_tool_started
        self._on_compaction = on_compaction
        self._on_compact_boundary = on_compact_boundary
        self._max_budget_usd = max_budget_usd
        self._client_factory = client_factory  # test seam
        self._include_hermes_tools = include_hermes_tools
        # Hermes-side session id, exported to the hermes-tools MCP subprocess
        # so the stateless session_search shim can exclude its own lineage.
        self._hermes_session_id = hermes_session_id
        # SDK-side session id to resume (#25267 continuity). Verified live:
        # resume restores the model context and keeps the SAME session id; a
        # stale id fails the session start (the caller retires + retries
        # fresh).
        self._resume_session_id = resume_session_id
        # Display-only partial-text consumer (W4 streaming). Deltas never
        # enter the projected transcript; the gateway's stream consumer
        # handles rate limiting and the already_sent final-send dedup.
        self._turn_callback_lock = threading.RLock()
        self._on_stream_delta = on_stream_delta
        # Hybrid MCP bridge inputs (see param docstring above).
        self._agent = agent
        self._tools = tools
        # Completed assistant prose that accompanies a tool call is a true
        # interim status, not final-answer text. It follows the gateway's
        # existing commentary callback so platforms can render it separately.
        # These callbacks are refreshed before EVERY run_turn: a session may
        # safely span several Hermes turns, while visibility must stay scoped
        # to the current one.
        self._on_interim_assistant = on_interim_assistant
        self._on_tool_iteration = on_tool_iteration

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._session_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._closed = False
        # Activity evidence for the in-flight turn (None between turns).
        self._turn_watch: Optional[_TurnWatch] = None
        # Snapshot the streaming posture once: the quiet-watchdog default is
        # derived from it, and a mid-session config edit must not flip the
        # watchdog's semantics while the live client still has the old
        # include_partial_messages option.
        self._streaming = _provider_flag("streaming")
        # Stream ownership (see _reader_loop). The reader task is the ONLY
        # consumer of the SDK stream; `_turn_inbox` is non-None exactly while a
        # turn is in flight, which is what makes "unsolicited" decidable.
        self._reader_task: Any = None
        self._turn_inbox: Any = None
        self._turn_claims: Any = None
        self._turn_claim_requested = False
        self._unsolicited_results = 0
        self._stream_ended: Optional[_StreamEnd] = None
        # Delivery half of the stream-ownership fix (dasbrow-hermes-coder#2):
        # a finished background Agent task's answer is captured here and
        # handed to the callback — it must never enter a turn's result, but
        # dropping it entirely left completed work silently undelivered
        # (observed live 2026-07-29: answers sat in the CLI session until the
        # operator poked). No callback wired = the historical drop semantics.
        self._on_unsolicited_result = on_unsolicited_result
        self._unsolicited_text: list[str] = []
        self._unsolicited_delivered: set[str] = set()
        # One immutable billing posture per child. The startup environment,
        # post-start evidence guard, and accounting label must never disagree
        # because config.yaml changed while a long-lived SDK session was live.
        self._allow_metered = _provider_flag("allow_metered_key")
        self._billing_evidence: dict[str, Any] = {}
        self._billing_guard_error: Optional[str] = None

    # ---------- lifecycle ----------

    def ensure_started(self) -> str:
        """Start the loop thread, build the SDK client, connect. Idempotent —
        returns the session marker (SDK session ids arrive on first result)."""
        if self._client is not None:
            return self._session_id or "pending"
        # Hard default, enforced fail-closed: this provider targets the Claude
        # SUBSCRIPTION. If a metered ANTHROPIC_API_KEY is present the
        # underlying CLI would silently prefer it — refuse to start instead.
        # ANTHROPIC_TOKEN is in the set because it alone authenticates
        # Hermes' NATIVE Anthropic lane (x-api-key, metered) — but Hermes
        # also persists subscription setup tokens there, so only an
        # API-key-shaped value counts as a metered vector.
        for metered_var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
            value = os.environ.get(metered_var)
            if not value or self._allow_metered:
                continue
            if metered_var == "ANTHROPIC_TOKEN" and _is_subscription_oauth_token(value):
                continue
            raise RuntimeError(
                f"claude-agent-sdk runtime refuses to start: {metered_var} "
                "is set, which would silently switch billing from the "
                "Claude subscription to metered API usage. Unset it, or "
                "set agent.claude_agent_sdk.allow_metered_key: true in "
                "config.yaml to explicitly allow it."
            )
        if self._client_factory is None:
            ok, msg = check_claude_sdk_available()
            if not ok:
                raise RuntimeError(msg)

        self._start_loop_thread()
        client = self._build_client()
        # Assign BEFORE connect: a connect timeout/cancel leaves a
        # half-connected client whose CLI subprocess close() must still reap
        # — a None _client would skip disconnect and orphan it.
        self._client = client
        self._run_coro(client.connect(), timeout=60.0)
        # From here on exactly ONE consumer owns the SDK stream (_reader_loop).
        # Started after connect so the client is live, before any turn so no
        # message can arrive unowned.
        self._start_reader()
        logger.info(
            "claude-agent-sdk session started: model=%s mode=%s cwd=%s",
            self._model or "cli-default",
            self._permission_mode,
            self._cwd,
        )
        return self._session_id or "pending"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Cancel the reader BEFORE disconnect so it unwinds on a live stream
        # instead of raising against a torn-down one.
        self._stop_reader()
        if self._client is not None and self._loop is not None:
            # Budget must exceed the SDK transport's own escalation ladder
            # (stdin lock + graceful wait + SIGTERM + SIGKILL); on timeout the
            # loop thread dies below and its shielded reap never finishes, so
            # fall back to killing the child directly.
            pid = _sdk_child_pid(self._client)
            # Capture the psutil identity before waiting on disconnect. If the
            # original child exits and its PID is reused during the timeout,
            # this object will fail psutil's identity check instead of
            # resolving the reused PID as a fresh kill target.
            child_process = _own_sdk_child_process(pid) if pid else None
            try:
                self._run_coro(
                    self._client.disconnect(), timeout=_SDK_DISCONNECT_TIMEOUT_S
                )
            except Exception:  # pragma: no cover - best-effort cleanup
                if child_process is not None:
                    _force_kill_sdk_child(pid, process=child_process)
            self._client = None
        self._stop_loop_thread()

    def __enter__(self) -> "ClaudeAgentSdkSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def consume_interrupt(self) -> None:
        """Clear a pending interrupt signal — the caller honored it through
        another path (e.g. the runtime's cold-agent short-circuit)."""
        self._interrupt_event.clear()

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to interrupt and unwind."""
        self._interrupt_event.set()
        if self._client is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._client.interrupt(), self._loop
                )
                future.add_done_callback(_swallow_interrupt_result)
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "SDK interrupt scheduling failed: %s",
                    _safe_sdk_error_text(exc),
                )

    def steer(self, text: str) -> bool:
        """Inject a mid-turn user message using the SDK's own streaming input.

        Why this exists: on this lane the SDK owns tool execution, so Hermes'
        tool-batch drain points (agent/tool_executor.py) are never reached and
        a /steer would strand in ``_pending_steer`` until the turn finalizer
        handed it back — where the gateway only redelivers it if no other
        message is queued. Steers were therefore silently lost.

        ``ClaudeSDKClient.query()`` is documented as "send a new request in
        streaming mode", and we already connect in streaming mode (``connect()``
        with no prompt), so a second query() on the LIVE session is the SDK's
        own steering contract — not an interrupt and not a session restart,
        which is what the claude-cli-live transport has to do.

        Verified 2026-08-15 against SDK 0.2.120: a query() issued while a tool
        call was in flight did not raise and was honored at the next turn
        boundary ~3s later. Note it also produces a SECOND ResultMessage once
        the superseded work unwinds; that arrives unclaimed and is handled by
        the existing unsolicited-result path (``_on_unsolicited_result``), so
        ``deliver_background_results`` is load-bearing here.

        Returns True only when the steer was actually scheduled onto a live
        turn. False means "not applicable" — the caller must fall back to the
        ordinary pending-steer stash. Crucially this is never both: returning
        True suppresses the stash, so a steer cannot be delivered twice.
        """
        if not text or not text.strip():
            return False
        # No claimed turn means there is nothing to steer INTO. Sending anyway
        # would open a fresh unclaimed turn on this session whose output the
        # reader routes to the unsolicited path — a reply appearing from
        # nowhere. Decline instead and let the caller queue it normally.
        if self._turn_inbox is None:
            return False
        client = self._client
        loop = self._loop
        if client is None or loop is None:
            return False
        cleaned = text.strip()
        try:
            future = asyncio.run_coroutine_threadsafe(client.query(cleaned), loop)
            future.add_done_callback(_swallow_steer_result)
        except Exception:
            logger.debug("SDK steer scheduling failed", exc_info=True)
            return False
        logger.info(
            "claude-agent-sdk: steered live turn via streaming input (%d chars)",
            len(cleaned),
        )
        return True

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: Optional[float] = None,
        post_tool_quiet_timeout: Optional[float] = None,
        watch_poll_interval: float = 1.0,
        abort_grace: float = _TURN_ABORT_GRACE,
    ) -> TurnResult:
        """Send a user message and block until the SDK's ResultMessage,
        projecting the typed stream into Hermes' messages shape.

        Turn lifetime is activity-aware, not a bare wall clock (production
        forensics: four of six 600s kills were actively-working turns).
        `turn_timeout` (explicit arg > agent.claude_agent_sdk.turn_timeout >
        600s) is a SOFT budget: it only fires when nothing is outstanding —
        no tool running, no approval awaiting a human — and the stream has
        ALSO been quiet ≥ min(30s, budget). A post-tool quiet watchdog
        (`post_tool_quiet_timeout`; default 90s with streaming on, disabled
        with streaming off) catches wedges early: armed when a tool result
        arrives, cleared by any later activity. On a trip the CLI is
        interrupted and given `abort_grace` to unwind — a clean unwind keeps
        the partial transcript and the resumable session id (no retire);
        only a grace expiry hard-cancels and retires."""
        result = TurnResult()
        prompt = _coerce_turn_input(user_input)
        if isinstance(prompt, str) and not prompt.strip():
            self._interrupt_event.clear()
            result.final_text = (
                "This Claude Agent SDK route can't process an empty message. "
                "Please send text or a supported image."
            )
            result.error = result.final_text
            result.api_call_made = False
            return result
        try:
            self.ensure_started()
        except Exception as exc:
            safe_exc = _safe_sdk_error_text(exc)
            hint = classify_auth_failure(safe_exc)
            result.error = hint or f"claude-agent-sdk startup failed: {safe_exc}"
            result.should_retire = True
            # Keep the traceback: `str(exc)` alone collapses a KeyError to a
            # bare quoted key (e.g. "'anyio'"), which is undiagnosable from the
            # user-facing error string. WARNING so it reaches errors.log.
            logger.warning(
                "claude-agent-sdk startup failed (%s: %s)\n"
                "Traceback (most recent call last):\n%s",
                type(exc).__name__,
                safe_exc,
                _safe_sdk_traceback(exc),
            )
            # A refusal to start is fatal to the run, not turn-scoped: the
            # metered-key guard and an uninstallable SDK are config errors
            # ("startup" — NOT "billing": kanban maps failure_reason
            # "billing" to the transient EX_TEMPFAIL requeue sentinel, and
            # retrying cannot fix a present metered key); an auth-classified
            # failure is "auth".
            result.fatal_reason = "auth" if hint else "startup"
            return result

        # Text buffered before this turn whose terminal ResultMessage never
        # arrived is partial mid-burst content — a later unrelated result
        # must never pick it up (misattribution is the proven worse failure;
        # texts parked DURING the turn re-buffer via the residue drain and
        # are unaffected). Never a silent drop: WARN what is discarded.
        stale = list(self._unsolicited_text)
        if stale:
            self._unsolicited_text.clear()
            logger.warning(
                "claude-agent-sdk: discarding %d stale unsolicited text(s) "
                "(%d chars) buffered before this turn — their terminal "
                "ResultMessage never arrived; attaching them to a later "
                "unrelated result is the proven worse failure",
                len(stale), sum(len(t) for t in stale),
            )

        # An interrupt that arrived between turns or during connect (up to
        # 60s) targets THIS turn — honor it instead of erasing it. (The old
        # unconditional clear() silently swallowed that window.)
        if self._interrupt_event.is_set():
            self._interrupt_event.clear()
            result.interrupted = True
            return result

        import concurrent.futures

        budget = (
            float(turn_timeout)
            if turn_timeout is not None
            else (_configured_turn_timeout() or _DEFAULT_TURN_TIMEOUT)
        )
        if post_tool_quiet_timeout is not None:
            quiet = float(post_tool_quiet_timeout)
        else:
            configured_quiet = _configured_post_tool_quiet_timeout()
            if configured_quiet is not None:
                quiet = configured_quiet  # 0 = explicitly disabled
            else:
                quiet = (
                    _DEFAULT_POST_TOOL_QUIET_STREAMING if self._streaming else 0.0
                )
        poll = max(0.01, float(watch_poll_interval))

        assert self._loop is not None, "loop thread not started"
        watch = _TurnWatch()
        self._turn_watch = watch
        trip: Optional[str] = None
        trip_elapsed = trip_idle = 0.0
        hard_trip = False
        turn_data: Optional[dict[str, Any]] = None
        future = asyncio.run_coroutine_threadsafe(
            self._consume_turn(prompt), self._loop
        )
        try:
            pending_verdict: Optional[str] = None
            prev_poll = time.monotonic()
            while turn_data is None and trip is None:
                try:
                    turn_data = future.result(timeout=poll)
                except (TimeoutError, concurrent.futures.TimeoutError):
                    # py3.11 unifies TimeoutError/asyncio.TimeoutError/
                    # concurrent.futures.TimeoutError — a DONE future here
                    # means the COROUTINE settled in the race window: with an
                    # exception (typically its own TimeoutError — a socket/
                    # pipe timeout under the CLI), re-raise into the
                    # classification path instead of misreading it as a poll
                    # expiry (which would busy-spin until the budget); with a
                    # RESULT, harvest it — the turn completed.
                    if future.done():
                        if future.exception() is not None:
                            raise
                        turn_data = future.result()
                        continue
                    now = time.monotonic()
                    if now - prev_poll > _POLL_STALL_FACTOR * poll:
                        # The PROCESS was descheduled (swap/OOM), not the
                        # turn — that gap proves nothing about the stream.
                        watch.rebaseline()
                        prev_poll = now
                        pending_verdict = None
                        continue
                    prev_poll = now
                    verdict = watch.check(budget=budget, quiet=quiet)
                    if verdict is None:
                        pending_verdict = None
                        continue
                    if pending_verdict != verdict:
                        # Debounce: two consecutive polls must agree before
                        # a trip (closes the tick-vs-check race window).
                        pending_verdict = verdict
                        continue
                    now = time.monotonic()
                    trip = verdict
                    trip_elapsed = now - watch.started
                    trip_idle = now - watch.last_activity
            if trip is not None and turn_data is None:
                if future.done():
                    # Completed between check() and the trip decision —
                    # completion wins, the trip is void.
                    turn_data = future.result()
                    trip = None
                else:
                    # Interrupt through request_interrupt(): setting
                    # _interrupt_event is REQUIRED — _consume_turn's W22 EDE
                    # mask keys on the turn-local flag, so the CLI's
                    # interrupt-ack is masked instead of delivered as a
                    # fresh error.
                    self.request_interrupt()
                    try:
                        turn_data = future.result(timeout=abort_grace)
                    except (TimeoutError, concurrent.futures.TimeoutError):
                        if future.done():
                            if future.exception() is not None:
                                raise
                            turn_data = future.result()
                        elif not future.cancel():
                            # Settled in the cancel race window — a real
                            # result beats a fabricated hard trip.
                            if future.exception() is not None:
                                raise
                            turn_data = future.result()
                        else:
                            hard_trip = True
            if trip is not None and turn_data is not None and not hard_trip:
                if (
                    not turn_data["error"]
                    and turn_data.get("result_uuid")
                    and turn_data["final_text"]
                ):
                    # The turn FINISHED inside the grace window with a real
                    # answer — deliver it in full; the trip never happened.
                    # final_text is REQUIRED: the W22-masked interrupt ack
                    # also has error=None + a result uuid, but its final_text
                    # is empty (projection stopped at the interrupt) — that
                    # shape must stay a trip, or it degrades into a silent
                    # dropped turn. Consume our own interrupt signal so the
                    # mapping below cannot misread it. (A real user /stop
                    # racing this exact window loses its signal too — the
                    # completed answer it targeted is delivered, same as any
                    # near-boundary stop today.)
                    self._interrupt_event.clear()
                    logger.info(
                        "claude-agent-sdk: turn completed during watchdog "
                        "grace (%.0fs elapsed) — delivered in full",
                        time.monotonic() - watch.started,
                    )
                    trip = None
        except Exception as exc:
            self._interrupt_event.clear()
            safe_exc = _safe_sdk_error_text(exc)
            hint = classify_auth_failure(safe_exc)
            result.error = hint or f"claude-agent-sdk turn failed: {safe_exc}"
            result.should_retire = True
            if hint is not None:
                # Auth failures are fatal; other mid-turn exceptions stay
                # transient (retire + retry semantics unchanged).
                result.fatal_reason = "auth"
            return result
        finally:
            self._turn_watch = None

        if turn_data is None:
            # Hard trip: the CLI ignored the interrupt for the whole grace —
            # nothing was harvested. Retire (today's shape, now the rare
            # fallback for a genuinely unresponsive CLI).
            self._interrupt_event.clear()
            result.interrupted = True
            result.error = self._format_trip_error(
                trip, budget, quiet, trip_elapsed, trip_idle
            )
            result.should_retire = True
            return result

        result.final_text = turn_data["final_text"]
        result.projected_messages = turn_data["messages"]
        result.tool_iterations = turn_data["tool_iterations"]
        result.token_usage_last = turn_data["usage"]
        result.token_usage_total = turn_data["usage"]
        result.model_last = turn_data.get("model")
        result.billing_mode = turn_data.get("billing_mode", "unknown")
        result.billing_evidence = turn_data.get("billing_evidence", {})
        result.total_cost_usd = turn_data.get("total_cost_usd")
        result.api_call_made = turn_data.get("api_call_made", True)
        result.thread_id = self._session_id
        result.turn_id = turn_data.get("result_uuid")
        result.interrupted = self._interrupt_event.is_set()
        if result.interrupted:
            # Consume the honored interrupt so it cannot bleed into the
            # next turn on this session object.
            self._interrupt_event.clear()
        if turn_data["error"]:
            # A prior MCP tool use is not evidence that this terminal SDK
            # error belongs to MCP; preserve fail-closed Claude auth handling
            # unless the error text itself identifies an MCP origin.
            hint = classify_auth_failure(turn_data["error"])
            result.error = hint or turn_data["error"]
            if hint is not None:
                result.should_retire = True
                result.fatal_reason = "auth"
        if turn_data.get("billing_guard_violation"):
            # This is durable account/config state, not a transient provider
            # billing error: the generic "billing" fatal_reason is a retry
            # sentinel, which would repeatedly spend calls against the same
            # unsafe lane. "startup" stops the run until the operator fixes
            # Extra Usage/credentials or explicitly opts into metering.
            result.should_retire = True
            result.fatal_reason = "startup"
        if turn_data.get("stream_ended"):
            # A dead stream is permanent on this session object: the reader
            # exited, _stream_ended stays set, and ensure_started() keeps
            # returning early while _client is non-None — so WITHOUT retire,
            # every later turn short-circuits to this same error with zero
            # model calls, forever (the CLI died once and poisoned the
            # session; observed as the unrecoverable half of the 2026-08-09
            # desync incident). Retire lets the runtime rebuild a fresh CLI
            # on the next attempt/turn. Model-level result subtypes
            # (error_max_turns, error_max_budget_usd) stay non-retiring.
            result.should_retire = True
        if trip is not None:
            # Clean-ack trip: the CLI honored our interrupt inside the grace,
            # the partial transcript above is preserved, and the session id
            # is resumable — mirror the user-interrupt lane (close-but-keep-
            # resume-id) instead of retiring. The trip text WINS over
            # whatever error the drain surfaced (typically the W22-masked
            # interrupt ack) — EXCEPT auth and stream-death, which keep
            # their retire verdicts from the mapping above.
            if result.fatal_reason != "auth":
                result.error = self._format_trip_error(
                    trip, budget, quiet, trip_elapsed, trip_idle
                )
            result.interrupted = True
        return result

    def _format_trip_error(
        self,
        kind: Optional[str],
        budget: float,
        quiet: float,
        elapsed: float,
        idle: float,
    ) -> str:
        """≤200 chars (the gateway truncates at str(err)[:200]); keeps the
        literal "turn timed out" needle; names the cause."""
        if kind == "post_tool_quiet":
            return (
                f"turn timed out: no SDK activity for {idle:.0f}s after a "
                f"tool result (turn ran {elapsed:.0f}s, quiet limit "
                f"{quiet:.0f}s)"
            )
        return (
            f"turn timed out after {elapsed:.0f}s "
            f"(budget {budget:.0f}s, idle {idle:.0f}s)"
        )

    # ---------- internals ----------

    async def _consume_turn(self, prompt: Any) -> dict[str, Any]:
        """The async side of one turn: query, then read THIS turn's messages
        off the reader loop's inbox until the ResultMessage.

        Deliberately NOT `receive_response()` — that helper serves the oldest
        buffered ResultMessage, which is not necessarily ours. See
        `_reader_loop` for why that is a permanent-corruption bug."""
        projector = ClaudeSdkEventProjector()
        out: dict[str, Any] = {
            "final_text": "",
            "messages": [],
            "tool_iterations": 0,
            "usage": None,
            "error": None,
            "result_uuid": None,
            "model": None,
            "stream_ended": False,
            "mcp_tool_seen": False,
            "billing_guard_violation": False,
            "billing_mode": self._reported_billing_mode(),
            "billing_evidence": dict(self._billing_evidence),
            "total_cost_usd": None,
            "api_call_made": True,
        }
        ended = self._stream_ended
        if ended is not None:
            out["error"] = "SDK message stream ended before this turn" + (
                f": {_safe_sdk_error_text(ended.error)}" if ended.error else ""
            )
            out["stream_ended"] = True
            return out
        if self._billing_guard_error is not None:
            out["error"] = self._billing_guard_error
            out["billing_guard_violation"] = True
            out["billing_mode"] = self._reported_billing_mode()
            out["billing_evidence"] = dict(self._billing_evidence)
            out["api_call_made"] = False
            return out
        inbox: Any = asyncio.Queue()
        # Ask the sole reader to drain everything already ahead of this turn
        # in the resumed client's FIFO, then atomically install our inbox.  A
        # direct assignment here races an already-buffered background result:
        # the reader can observe the new inbox before it observes that older
        # result and mis-serve it as this query's answer.
        claims = self._turn_claims
        if claims is None:
            out["error"] = "SDK message reader is not ready to claim this turn"
            out["stream_ended"] = True
            return out
        claim_ack = asyncio.get_running_loop().create_future()
        self._turn_claim_requested = True
        claims.put_nowait(("claim", inbox, claim_ack))
        interrupted = False
        billing_guarded = False
        try:
            try:
                await claim_ack
            finally:
                self._turn_claim_requested = False
            ended = self._stream_ended
            if ended is not None:
                out["error"] = "SDK message stream ended before this turn" + (
                    f": {_safe_sdk_error_text(ended.error)}" if ended.error else ""
                )
                out["stream_ended"] = True
                return out
            query_input = (
                _sdk_user_message_stream(prompt)
                if isinstance(prompt, list)
                else prompt
            )
            await self._client.query(query_input)
            while True:
                message = await inbox.get()
                watch = self._turn_watch
                if watch is not None:
                    # Any stream message is liveness — stamp before anything
                    # else so the caller-thread watchdog sees it.
                    watch.tick()
                if isinstance(message, _StreamEnd):
                    out["error"] = (
                        "SDK message stream ended before this turn's result"
                        + (f": {_safe_sdk_error_text(message.error)}" if message.error else "")
                    )
                    out["stream_ended"] = True
                    break
                # Capture the SDK session id from ANY message that carries it —
                # the init SystemMessage announces it first, so even a turn
                # interrupted before its ResultMessage keeps a resumable id.
                early_sid = getattr(message, "session_id", None)
                if early_sid:
                    self._session_id = early_sid
                self._handle_compact_boundary(message)
                if self._billing_guard_error is not None and not billing_guarded:
                    billing_guarded = True
                    out["error"] = self._billing_guard_error
                    out["billing_guard_violation"] = True
                    try:
                        await self._client.interrupt()
                    except Exception:
                        logger.debug(
                            "claude-agent-sdk billing-guard interrupt failed",
                            exc_info=True,
                        )
                if self._interrupt_event.is_set():
                    # Previously `break`. Bailing out before this turn's
                    # ResultMessage orphaned it in the shared stream, and the
                    # NEXT turn was then served that stale result — the same
                    # off-by-one corruption `_reader_loop` exists to prevent,
                    # reachable through the interrupt path with no Agent tool
                    # involved. Keep draining to the result; stop projecting.
                    interrupted = True
                if type(message).__name__ == "StreamEvent":
                    if watch is not None:
                        # A partial delta proves the post-tool model call is
                        # alive — the quiet watchdog stands down.
                        watch.disarm_post_tool()
                    if not interrupted and not billing_guarded:
                        self._forward_stream_delta(message)
                    continue
                for block in getattr(message, "content", None) or []:
                    if (
                        type(block).__name__ == "ToolUseBlock"
                        and str(getattr(block, "name", "")).startswith("mcp__")
                    ):
                        out["mcp_tool_seen"] = True
                if not interrupted and not billing_guarded:
                    self._notify_tool_started(message)
                    self._notify_interim_assistant(message)
                projection = projector.project(message)
                if watch is not None:
                    # Outstanding-tool evidence: ToolUseBlocks issue, tool
                    # results resolve (server tools never enter — the
                    # projector resolves them inside their own assistant
                    # message). A turn with a tool in flight is suspended
                    # from BOTH watchdog rules.
                    issued = sum(
                        len(m.get("tool_calls") or [])
                        for m in projection.messages
                        if m.get("role") == "assistant"
                    )
                    if issued:
                        watch.note_tools_issued(issued)
                    if projection.is_tool_iteration:
                        watch.note_tools_resolved(
                            sum(
                                1
                                for m in projection.messages
                                if m.get("role") == "tool"
                            )
                        )
                        # Codex-parity arm point: a tool result just landed;
                        # silence from here on is the wedge signature.
                        watch.arm_post_tool()
                    elif projection.messages or projection.final_text is not None:
                        watch.disarm_post_tool()
                if projection.model:
                    # Last reported id wins; captured even on interrupted
                    # turns — the tokens were still spent on that model.
                    out["model"] = projection.model
                # A genuine terminal SDK error carries a result string, but it
                # is transport diagnostics, not an assistant answer. The outer
                # runtime needs `out["error"]` to activate provider fallback;
                # emitting this text first persists it as an assistant message
                # and poisons the fallback's history with a false “I am
                # blocked” claim. Contradictory success envelopes remain a
                # success exactly as handled below.
                _result_subtype = getattr(message, "subtype", "") or ""
                _result_is_error = bool(
                    projection.is_result
                    and (
                        getattr(message, "is_error", False)
                        or _result_subtype not in ("", "success")
                    )
                )
                _result_is_contradictory_success = bool(
                    _result_is_error
                    and _result_subtype == "success"
                    and not (getattr(message, "errors", None) or [])
                    and not getattr(message, "api_error_status", None)
                )
                if not interrupted and not billing_guarded:
                    if projection.messages:
                        out["messages"].extend(projection.messages)
                    if projection.is_tool_iteration:
                        out["tool_iterations"] += 1
                        self._notify_tool_iteration()
                    if projection.final_text is not None and (
                        not _result_is_error or _result_is_contradictory_success
                    ):
                        out["final_text"] = projection.final_text
                if projection.is_result:
                    usage = getattr(message, "usage", None)
                    if isinstance(usage, dict):
                        out["usage"] = dict(usage)
                    sid = getattr(message, "session_id", None)
                    if sid:
                        self._session_id = sid
                    out["result_uuid"] = getattr(message, "uuid", None)
                    out["total_cost_usd"] = getattr(
                        message, "total_cost_usd", None
                    )
                    subtype = getattr(message, "subtype", "") or ""
                    if getattr(message, "is_error", False):
                        errors = getattr(message, "errors", None) or []
                        api_error_status = getattr(message, "api_error_status", None)
                        if subtype == "success" and not errors and not api_error_status:
                            # Contradictory envelope: is_error=True yet
                            # subtype="success" with nothing in errors. The
                            # CLI emits this shape rarely (2026-08-11: it
                            # killed a cron run as "RuntimeError: SDK result
                            # error (subtype=success): success" — a dead job
                            # over a turn that had actually produced its
                            # answer). There is no error to report, so the
                            # error flag loses to the subtype; kept loud for
                            # diagnosis. A genuine failure carries a non-empty
                            # errors list or a non-success subtype and still
                            # takes the honest path below.
                            logger.warning(
                                "claude-agent-sdk: contradictory result "
                                "envelope (is_error=True, subtype=success, "
                                "no errors) — treated as success (diagnostic omitted)",
                            )
                            break
                        detail = _safe_sdk_error_text(
                            "; ".join(str(e) for e in errors)
                            or getattr(message, "result", None)
                            or subtype
                        )
                        err_text = f"SDK result error (subtype={subtype}): {detail}"
                        if api_error_status:
                            err_text += f" (HTTP {api_error_status})"
                        # A turn WE interrupted before any assistant content
                        # ends as is_error/error_during_execution in the CLI
                        # ("[ede_diagnostic] result_type=user…") — that is the
                        # interrupt being honored, not a failure; surfacing it
                        # paged the operator with a false "Processing stopped"
                        # (2026-08-09 barge-in incident). Mask ONLY that exact
                        # shape, only when THIS turn's interrupt flag is set
                        # (never a fresh event read — a late /stop must not
                        # reclassify a genuine error), and only when no auth
                        # needle is present (an auth failure must keep its
                        # error + retire path regardless of interrupts).
                        # A real transient failure colliding with an interrupt
                        # re-fires next turn with no interrupt pending and
                        # takes the honest path; the masked text is INFO-kept.
                        if (
                            interrupted
                            and subtype == "error_during_execution"
                            and classify_auth_failure(err_text) is None
                        ):
                            logger.info(
                                "claude-agent-sdk: masked %s on requested "
                                "interrupt (interrupt honored, not a "
                                "failure): %s",
                                subtype,
                                _safe_sdk_error_text(err_text),
                            )
                        else:
                            if not billing_guarded:
                                out["error"] = err_text
                    elif subtype not in ("", "success") and not billing_guarded:
                        # e.g. error_max_turns / error_max_budget_usd — surface
                        # honestly; the partial transcript is still projected.
                        out["error"] = f"SDK turn ended: {subtype}"
                    break
        finally:
            if self._turn_inbox is inbox:
                # Relinquish ownership through the same reader arbiter.  It
                # drains immediately buffered post-result residue into this
                # inbox before acknowledging release, preserving the existing
                # overlap/own-answer dedup handling below.
                if self._stream_ended is None and self._turn_claims is not None:
                    release_ack = asyncio.get_running_loop().create_future()
                    self._turn_claims.put_nowait(("release", inbox, release_ack))
                    await release_ack
                else:
                    self._turn_inbox = None
                # Stream death can win the release handshake after the
                # terminal ResultMessage was already delivered.  The reader's
                # death path acknowledges queued operations so waiters wake,
                # but it deliberately does not install/apply their ownership.
                # Re-check after the acknowledgement: preserve the valid turn
                # result while retiring the now-dead session, and never leave
                # a between-turn inbox falsely claimed.
                if self._stream_ended is not None:
                    if self._turn_inbox is inbox:
                        self._turn_inbox = None
                    out["stream_ended"] = True
            # Anything the reader parked after our ResultMessage belongs to a
            # CLI-initiated turn that overlapped ours. Route it now — left in
            # a discarded queue it would be lost, and left in the stream it
            # would become the next turn's answer. EXCEPT a residue result
            # that repeats THIS turn's own answer: delivering it through the
            # background lane re-presents the agent's own text as a fake
            # completion (the 2026-08-06 echo class) — suppress it, dedup-mark
            # it, and consume its burst buffer. Genuinely different residue is
            # a real background completion and still delivers.
            final_norm = " ".join(str(out["final_text"]).split())
            while not inbox.empty():
                residue = inbox.get_nowait()
                if final_norm and type(residue).__name__ == "ResultMessage":
                    res_text = getattr(residue, "result", None)
                    if (
                        isinstance(res_text, str)
                        and " ".join(res_text.split()) == final_norm
                    ):
                        uuid = getattr(residue, "uuid", None)
                        if uuid:
                            self._unsolicited_delivered.add(uuid)
                        self._unsolicited_results += 1
                        buffered = len(self._unsolicited_text)
                        self._unsolicited_text.clear()
                        logger.warning(
                            "claude-agent-sdk: residue ResultMessage %s "
                            "matches this turn's own answer — suppressed, "
                            "never delivered as a background result "
                            "(%d buffered text(s) consumed)",
                            uuid, buffered,
                        )
                        continue
                self._handle_unsolicited(residue)
        out["billing_mode"] = self._reported_billing_mode()
        out["billing_evidence"] = dict(self._billing_evidence)
        return out

    # ---------- stream ownership ----------

    async def _reader_loop(self) -> None:
        """Sole owner of the SDK message stream for the client's lifetime.

        Hermes used to call `receive_response()` once per turn. That helper
        returns the OLDEST buffered ResultMessage — not necessarily this
        turn's — and the SDK offers no per-turn correlator (`query()` takes
        only a `session_id`, which is constant across turns; `ResultMessage`
        has a `uuid` with no request-side counterpart). Ordering is therefore
        the ONLY thing keeping answers matched to questions.

        The Claude Code CLI breaks that ordering by itself: when a background
        Agent task completes it injects a `<task-notification>` and runs a
        FULL assistant turn nobody asked for, leaving an unconsumed
        ResultMessage in the shared FIFO. Every later turn then pops a stale
        result. Because a desynced turn still looks like success — no error,
        no timeout, no interrupt — no retire path fires, so the skew is
        permanent and silent. Observed live 2026-07-25: four such turns left a
        constant off-by-4 for hours (dasbrow-hermes-coder#2).

        One reader for the whole client lifetime makes the rule enforceable:
        a message arriving while no turn is in flight is unsolicited BY
        DEFINITION, and gets routed away instead of poisoning the next turn."""
        end: _StreamEnd
        iterator = self._client.receive_messages().__aiter__()
        message_task = asyncio.ensure_future(iterator.__anext__())
        claim_task = asyncio.ensure_future(self._turn_claims.get())
        try:
            while True:
                done, _pending = await asyncio.wait(
                    (message_task, claim_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if message_task in done:
                    try:
                        message = message_task.result()
                    except StopAsyncIteration:
                        end = _StreamEnd(error=None)
                        break
                    self._observe_billing_evidence(message)
                    inbox = self._turn_inbox
                    if inbox is not None:
                        inbox.put_nowait(message)
                    else:
                        self._handle_unsolicited(message)
                    # Message wins ties with a claim. Re-arm first, then loop:
                    # every immediately available pre-claim FIFO entry is
                    # classified unsolicited before the claim is acknowledged.
                    message_task = asyncio.ensure_future(iterator.__anext__())
                    continue

                operation, inbox, claim_ack = claim_task.result()
                claim_task = asyncio.ensure_future(self._turn_claims.get())
                if claim_ack.cancelled():
                    continue
                if operation == "claim":
                    if self._turn_inbox is not None:
                        claim_ack.set_exception(
                            RuntimeError("SDK message stream already has a turn owner")
                        )
                        continue
                    self._turn_inbox = inbox
                elif operation == "release":
                    if self._turn_inbox is not inbox:
                        claim_ack.set_exception(
                            RuntimeError("SDK message stream release owner mismatch")
                        )
                        continue
                    self._turn_inbox = None
                else:  # pragma: no cover - internal invariant
                    claim_ack.set_exception(
                        RuntimeError(f"unknown SDK stream ownership operation: {operation}")
                    )
                    continue
                claim_ack.set_result(None)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            message_task.cancel()
            claim_task.cancel()
            raise
        except Exception as exc:  # pragma: no cover - stream torn down
            logger.debug(
                "claude-agent-sdk reader loop ended: %s",
                _safe_sdk_error_text(exc),
            )
            end = _StreamEnd(error=_safe_sdk_error_text(exc))
        finally:
            if not message_task.done():
                message_task.cancel()
            if not claim_task.done():
                claim_task.cancel()
        # The stream is gone (CLI exited or transport died). Mark it before
        # acknowledging every queued ownership operation.  A foreground turn
        # can pass its initial stream check, enqueue a claim behind buffered
        # messages, and then lose the reader to EOF; resolving the claim makes
        # it re-check this terminal state instead of waiting out turn_timeout.
        self._stream_ended = end
        pending_claims = []
        if claim_task.done() and not claim_task.cancelled():
            try:
                pending_claims.append(claim_task.result())
            except Exception:  # pragma: no cover - queue task failed
                pass
        claims = self._turn_claims
        if claims is not None:
            while True:
                try:
                    pending_claims.append(claims.get_nowait())
                except asyncio.QueueEmpty:
                    break
        for _operation, _owner, claim_ack in pending_claims:
            if not claim_ack.done():
                claim_ack.set_result(None)
        inbox = self._turn_inbox
        if inbox is not None:
            inbox.put_nowait(end)

    def _handle_unsolicited(self, message: Any) -> None:
        """Route a message that arrived with no turn in flight.

        These are real CLI output (typically a finished background Agent task
        reporting in). They answer nothing Hermes asked, so they must never
        enter a turn's result — but their CONTENT is completed work the user
        is waiting on: with a delivery callback wired, capture each top-level
        assistant message's text and hand the FULL burst over as an ordered
        list on the terminal ResultMessage (uuid-deduped). Without a
        callback, the historical WARN-drop stands."""
        if isinstance(message, _StreamEnd):
            return
        sid = getattr(message, "session_id", None)
        if sid:
            self._session_id = sid
        name = type(message).__name__
        if name == "ResultMessage":
            self._unsolicited_results += 1
            if self._on_unsolicited_result is None:
                self._unsolicited_text.clear()
                logger.warning(
                    "claude-agent-sdk: dropped unsolicited ResultMessage (no "
                    "turn in flight, total=%d) — CLI-initiated turn; see "
                    "dasbrow-hermes-coder#2",
                    self._unsolicited_results,
                )
                return
            uuid = getattr(message, "uuid", None)
            if uuid and uuid in self._unsolicited_delivered:
                self._unsolicited_text.clear()
                logger.debug(
                    "claude-agent-sdk: duplicate unsolicited ResultMessage "
                    "%s ignored", uuid,
                )
                return
            result_text = getattr(message, "result", None)
            texts = list(self._unsolicited_text)
            self._unsolicited_text.clear()
            if isinstance(result_text, str) and result_text.strip():
                # The CLI's result text repeats the turn's final assistant
                # message — never hand the same text over twice.
                if not texts or texts[-1] != result_text:
                    texts.append(result_text)
            if uuid:
                self._unsolicited_delivered.add(uuid)
            if not texts:
                logger.warning(
                    "claude-agent-sdk: unsolicited ResultMessage carried no "
                    "text (total=%d) — nothing to deliver",
                    self._unsolicited_results,
                )
                return
            logger.info(
                "claude-agent-sdk: delivering unsolicited result burst "
                "(background task finished, total=%d, %d message(s), "
                "%d chars)",
                self._unsolicited_results, len(texts),
                sum(len(t) for t in texts),
            )
            try:
                self._on_unsolicited_result(texts)
            except Exception:
                logger.warning(
                    "claude-agent-sdk: unsolicited-result delivery callback "
                    "raised — answer may be lost", exc_info=True,
                )
        elif name == "AssistantMessage":
            # Buffer top-level text, one entry per message so the burst
            # delivers in message granularity; subagent streams
            # (parent_tool_use_id set) are noise — the same gate
            # _forward_stream_delta uses.
            if (
                self._on_unsolicited_result is not None
                and not getattr(message, "parent_tool_use_id", None)
            ):
                parts = [
                    getattr(block, "text", "") or ""
                    for block in getattr(message, "content", None) or []
                    if type(block).__name__ == "TextBlock"
                ]
                message_text = "\n".join(p for p in parts if p)
                if message_text:
                    self._unsolicited_text.append(message_text)
            logger.info(
                "claude-agent-sdk: unsolicited %s outside a turn", name,
            )
        else:
            logger.debug(
                "claude-agent-sdk: unsolicited %s outside a turn", name,
            )

    def _start_reader(self) -> None:
        """Spawn the reader on the session loop. Idempotent."""
        if self._reader_task is not None:
            return

        async def _spawn() -> Any:
            self._turn_claims = asyncio.Queue()
            return asyncio.ensure_future(self._reader_loop())

        self._reader_task = self._run_coro(_spawn(), timeout=10.0)

    def _stop_reader(self) -> None:
        task = self._reader_task
        self._reader_task = None
        if task is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(task.cancel)
        except Exception:  # pragma: no cover - loop already gone
            pass

    def build_option_fields(self) -> dict[str, Any]:
        """The ClaudeAgentOptions field dict — plain data so tests can assert
        on it without importing the SDK."""
        mcp_servers: dict[str, Any] = {}
        # Hybrid in-process MCP bridge (ported from PR #56413) — exposes the
        # full Hermes tool registry, including proxified third-party MCP
        # servers, which the stdio `hermes-tools` wrapper cannot reach
        # because credentials live in the gateway process env and don't
        # propagate to the subprocess. Activation requires ALL of:
        #   1. the operator opted in via
        #      ``agent.claude_agent_sdk.hybrid_mcp_bridge: true``
        #      (checked here as well as at the runtime call site);
        #   2. the caller supplied both ``agent`` and ``tools``.
        # The bridge splits into two in-process servers:
        #   - ``hermes-tools`` — stdio-legacy names (see
        #     ``HERMES_TOOLS_LEGACY_NAMES``). Preserves operator grants
        #     stored in ``~/.claude/settings.json`` that key on
        #     ``mcp__hermes-tools__<tool>``: a box on
        #     ``permission_mode: default`` would otherwise face an
        #     approval storm for tools it already granted.
        #   - ``hermes-hybrid`` — everything else (proxified MCPs +
        #     agent-level tools).
        # The exclude list from config is applied to BOTH buckets so an
        # operator can keep the wide bridge for proxified MCPs without
        # inheriting a specific tool (delegate_task, cron_*, terminal, ...).
        hybrid_active = False
        hybrid_opted_in = _provider_flag("hybrid_mcp_bridge", default=False)
        if hybrid_opted_in and self._agent is not None and self._tools:
            try:
                from agent.transports.hermes_hybrid_mcp import (
                    HERMES_TOOLS_SERVER,
                    HYBRID_SERVER,
                    build_hybrid_mcp_server,
                )
                from agent.transports.hermes_tool_exposure import (
                    HERMES_TOOLS_LEGACY_NAMES,
                )

                exclude_names = _configured_hybrid_exclude()
                mcp_servers[HERMES_TOOLS_SERVER] = build_hybrid_mcp_server(
                    self._agent,
                    self._tools,
                    server_name=HERMES_TOOLS_SERVER,
                    only_names=HERMES_TOOLS_LEGACY_NAMES,
                    exclude_names=exclude_names,
                )
                # Registry diff rides the "extras" bucket only: the legacy
                # bucket stays byte-identical so ``mcp__hermes-tools__*``
                # grants keep matching. When the diff is empty the snapshot
                # object is forwarded untouched, so a deployment with nothing
                # late-registered builds exactly the bridge it built before.
                registry_extra = _mcp_registry_tool_specs(
                    self._agent, self._tools
                )
                hybrid_tools = (
                    list(self._tools) + registry_extra
                    if registry_extra
                    else self._tools
                )
                mcp_servers[HYBRID_SERVER] = build_hybrid_mcp_server(
                    self._agent,
                    hybrid_tools,
                    server_name=HYBRID_SERVER,
                    exclude_names=(
                        list(exclude_names) + list(HERMES_TOOLS_LEGACY_NAMES)
                    ),
                )
                hybrid_active = True
            except Exception as exc:  # noqa: BLE001
                # Never break session start on hybrid bridge failure — the
                # stdio wrapper still provides the curated tool set.
                logger.warning(
                    "hybrid MCP bridge failed to build (%s) — falling back "
                    "to stdio hermes-tools only",
                    exc,
                )
        # The stdio ``hermes-tools`` wrapper exposes ~25 curated tools that
        # the hybrid bridge already re-exposes under the same server name
        # (``hermes-tools``) when active — registering both concurrently
        # would send Claude two copies of every curated tool under the same
        # server. Skip stdio when hybrid is active. (Hybrid failure above
        # falls back to stdio via ``hybrid_active`` staying False.)
        if self._include_hermes_tools and not hybrid_active:
            mcp_servers["hermes-tools"] = _build_hermes_tools_mcp_config(
                hermes_session_id=self._hermes_session_id
            )

        # Headerless third-party HTTP MCPs configured in Hermes (config.yaml
        # mcp_servers.<name>.url) can be exposed directly to the SDK because
        # the registry snapshot may not contain their late/proxified tools.
        # This is part of the SAME wide-surface security choice as the hybrid
        # bridge, never an independent back door: discovery runs only after
        # the explicit opt-in passed and both in-process bridge buckets built
        # successfully. Header-bearing entries are refused by the loader
        # because the SDK puts its MCP config in the Claude CLI argv.
        if hybrid_active:
            for entry_name, entry_cfg in _http_mcp_entries_from_config().items():
                if entry_name in mcp_servers:
                    continue
                mcp_servers[entry_name] = entry_cfg

        system_prompt: Any = {"type": "preset", "preset": "claude_code"}
        if self._system_prompt_append:
            system_prompt = {
                "type": "preset",
                "preset": "claude_code",
                "append": self._system_prompt_append,
            }

        # This wrapper is Hermes' mandatory permission-policy owner, including
        # when runtime callback selection legitimately produced no downstream
        # callback.  Callbackless non-bypass sessions intentionally fail closed
        # here instead of falling through to SDK-native prompting/permission
        # behavior; bounded readers and trusted bypass still resolve only after
        # canonical validation, correlation validation, and immutable floors.
        can_use_tool = self._make_can_use_tool()

        # Metered-vector scrub for the spawned CLI (see _METERED_ENV_DENYLIST).
        # agent.claude_agent_sdk.allow_metered_key: true is the operator's
        # explicit "bill me metered" opt-in (the same flag the startup guard
        # honors), so it disables the scrub too — otherwise the documented
        # escape hatch would hand the CLI an environment with the key blanked.
        env_overrides = _sdk_env_overrides(
            metered_allowed=self._allow_metered
        )

        fields = {
            "model": self._model,
            "cwd": self._cwd,
            "permission_mode": self._permission_mode,
            "system_prompt": system_prompt,
            "mcp_servers": mcp_servers,
            "max_budget_usd": self._max_budget_usd,
            "can_use_tool": can_use_tool,
            "env": env_overrides,
            # Explicit SDK isolation. None (the SDK default, verified against
            # claude-agent-sdk 0.2.120) means the CLI loads ALL filesystem
            # settings — ~/.claude/settings.json, .claude/settings.json,
            # .claude/settings.local.json — so ambient operator/project
            # settings could re-permission tools or install hooks UNDERNEATH
            # the permission_mode/can_use_tool posture configured above. The
            # empty list is the SDK's documented isolation mode: settings
            # come from Hermes config only. Accepted side effect: "project"
            # is also what loads CLAUDE.md, which this runtime doesn't want —
            # it composes its own system-prompt append. Operators whose
            # deployment stores tool grants in ~/.claude/settings.json
            # (unattended cron turns with nobody to answer a prompt) opt
            # back in via agent.claude_agent_sdk.setting_sources — see
            # _configured_setting_sources.
            "setting_sources": _configured_setting_sources(),
            # Explicit, because the SDK's 1 MiB default is below what this
            # lane's own tool results routinely produce and overflowing it
            # kills the turn outright — see _configured_max_buffer_size.
            "max_buffer_size": _configured_max_buffer_size(),
            # AskUserQuestion has no native answer bridge. Native Read stays
            # behind the protected-path-aware, bounded Hermes MCP surface in
            # every supported SDK permission mode.
            "disallowed_tools": ["AskUserQuestion", "Read"],
        }
        # The CLI owns compaction on this lane, so its PreCompact hook is the
        # only honest signal that a turn stalled to compact. Registered only
        # when a consumer asked for it, so the default option set is unchanged.
        _compaction_hooks = self._build_compaction_hooks()
        if _compaction_hooks is not None:
            fields["hooks"] = _compaction_hooks
        if self._resume_session_id:
            fields["resume"] = self._resume_session_id
        # Default OFF (upstream-conservative): partial messages only when the
        # operator opts in via agent.claude_agent_sdk.streaming in config.yaml.
        # Reads the __init__ snapshot so option and quiet-watchdog semantics
        # can never diverge across a mid-session config edit.
        if self._streaming:
            fields["include_partial_messages"] = True
        return fields

    def _build_client(self) -> Any:
        fields = self.build_option_fields()
        if self._client_factory is not None:
            return self._client_factory(options=fields)
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        return ClaudeSDKClient(options=ClaudeAgentOptions(**fields))

    def _sdk_approval_bypass_active(self) -> bool:
        """Resolve only trusted session/process/config bypass intent."""
        if self._sdk_approval_bypass_requested:
            return True
        provider = self._approval_bypass_provider
        if provider is not None:
            try:
                return provider() is True
            except Exception:
                return False
        try:
            from tools.approval import is_approval_bypass_active_for_session

            return is_approval_bypass_active_for_session(
                self._hermes_session_id or "",
            )
        except Exception:
            return False

    def _make_can_use_tool(self) -> Any:
        """Bridge SDK permission requests onto Hermes' approval callback.
        Fail-closed: any callback failure denies.

        Silent-deny observability (P2.d): every deny that was NOT an
        operator's choice is logged at INFO here — this is the choke point
        every SDK-lane deny transits with tool name and honest reason in
        hand (the 2026-08-06 incident's silent denies had no log line at
        all). "denied by user" (with or without ": <text>") appears IFF a
        human chose deny — W8/W11 reserved that wording — so the prefix is
        the discriminator; operator denies are not silent and not logged
        here. Honesty boundary: settings deny-rule hits on the SDK side
        (the CLI consulting ~/.claude/settings.json deny rules, e.g. an
        installer-only skills-dir rule) never invoke can_use_tool and never
        transit hermes code — they are UNLOGGABLE here by construction;
        operator-facing relief for that class is a separate deny-notice
        feature decision."""
        approval_callback = self._approval_callback
        hermes_session_id = self._hermes_session_id

        async def _can_use_tool(tool_name: str, tool_input: dict, context: Any):
            from claude_agent_sdk import (
                PermissionResultAllow,
                PermissionResultDeny,
            )

            # Capture the watch OBJECT at entry: an approval that outlives
            # its turn (orphaned wait, self-expiring at approvals.timeout)
            # must decrement the watch it suspended, never a later turn's.
            # None = no Hermes turn in flight (unsolicited CLI-side turn) —
            # nothing to suspend.
            watch = self._turn_watch
            if watch is not None:
                # A human is being asked — their think time is not the
                # turn's silence. Both watchdog rules stand down until the
                # tap (or the approval machinery's own timeout) resolves.
                watch.approval_begin()
            try:
                return await self._resolve_can_use_tool(
                    tool_name, tool_input, context,
                    approval_callback, hermes_session_id,
                    PermissionResultAllow, PermissionResultDeny,
                )
            finally:
                if watch is not None:
                    watch.approval_end()

        return _can_use_tool

    async def _resolve_can_use_tool(
        self,
        tool_name: str,
        tool_input: dict,
        context: Any,
        approval_callback: Any,
        hermes_session_id: Any,
        PermissionResultAllow: Any,
        PermissionResultDeny: Any,
    ) -> Any:
        try:
            canonical_tool_input = _canonical_sdk_tool_request(tool_name, tool_input)
            checked_request = validate_canonical_sdk_request_serialization(
                canonical_tool_input,
            )
            presentation = safe_sdk_tool_presentation_from_canonical(
                canonical_tool_input,
            )
        except Exception:
            checked_request = None
            presentation = None
        if checked_request is None or presentation is None:
            return PermissionResultDeny(message="canonical request is unassessable")
        frozen_tool_input = checked_request[1]["tool_input"]
        # Validate correlation metadata before every policy decision. Legacy
        # callbacks retain their exact ABI, but no floor or bypass decision can
        # be made from callback markers or callback-controlled text.
        safe_tool_use_id = _safe_sdk_tool_use_id(context)
        callback_command, callback_description = presentation
        log_tool_identity = (
            tool_name if tool_name in _SDK_FIXED_LOG_TOOL_IDENTITIES else "sdk-tool"
        )
        # Native Read remains unavailable even if this callback is invoked
        # directly despite the SDK disallowed_tools option.
        if tool_name == "Read":
            return PermissionResultDeny(message="native SDK Read is disallowed")
        if tool_name == "Bash":
            try:
                from tools.approval_sdk_gateway import sdk_bash_immutable_floor_reason

                floor_reason = sdk_bash_immutable_floor_reason(
                    frozen_tool_input.get("command"),
                )
            except Exception:
                floor_reason = "canonical request is unassessable"
            if floor_reason is not None:
                return PermissionResultDeny(message="approval denied by callback")
        if tool_name in _SDK_AUTO_ALLOWED_MCP_TOOLS:
            return PermissionResultAllow(updated_input=frozen_tool_input)
        if self._sdk_approval_bypass_active():
            return PermissionResultAllow(updated_input=frozen_tool_input)
        if approval_callback is None:
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, "approval callback unavailable",
            )
            return PermissionResultDeny(message="approval callback unavailable")
        try:
            kwargs: dict = {"allow_permanent": False}
            try:
                from tools.approval_sdk_gateway import (
                    is_trusted_sdk_gateway_approval_callback,
                )

                trusted_gateway_callback = (
                    is_trusted_sdk_gateway_approval_callback(approval_callback)
                )
            except Exception:
                trusted_gateway_callback = False
            # Correlation and canonical JSON are additive only for callbacks
            # that explicitly advertise the corresponding SDK ABI extension.
            if getattr(approval_callback, "_accepts_tool_use_id", False):
                kwargs["tool_use_id"] = safe_tool_use_id
            if getattr(approval_callback, "_accepts_canonical_tool_input", False):
                kwargs["canonical_tool_input"] = canonical_tool_input
            result = await asyncio.to_thread(
                approval_callback,
                callback_command,
                callback_description,
                **kwargs,
            )
        except Exception:
            logger.warning("SDK approval callback failed at protected boundary")
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, "approval callback failed",
            )
            return PermissionResultDeny(message="approval callback failed")
        # Widened callback contract: a plain choice string, or a dict
        # {"choice": str, "reason": str} carrying an honest deny reason
        # (no-approver / timeout / notify-failure / teardown-expiry).
        # "denied by user" is reserved for a real human deny — a
        # reason-bearing deny must never be attributed to the user.
        try:
            reason = None
            operator_denied = False
            reason_shape = False
            operator_denial_shape = False
            choice = result
            if type(result) is dict:
                # The callback result is untrusted. Protocol dicts have exactly
                # one of three tiny widths; reject all others before collecting,
                # copying, hashing, or otherwise traversing callback keys:
                # {choice}, {choice, reason}, or the trusted gateway-only
                # {choice, operator_denial, reason} structural denial.
                width = len(result)
                if width == 1 and "choice" in result:
                    choice = result["choice"]
                elif width == 2 and "choice" in result and "reason" in result:
                    choice = result["choice"]
                    reason = result["reason"]
                    reason_shape = True
                elif (
                    width == 3
                    and "choice" in result
                    and "operator_denial" in result
                    and "reason" in result
                ):
                    choice = result["choice"]
                    reason = result["reason"]
                    reason_shape = True
                    operator_denial_shape = True
                else:
                    raise ValueError("malformed callback result shape")
            elif type(result) is not str:
                raise TypeError("unsupported callback result")
            if not _is_bounded_sdk_callback_string(
                choice,
                _SDK_CALLBACK_CHOICE_MAX_UTF8_BYTES,
                allow_space=False,
            ) or (
                reason is not None
                and not _is_bounded_sdk_callback_string(
                    reason,
                    _SDK_CALLBACK_REASON_MAX_UTF8_BYTES,
                    allow_space=True,
                )
            ):
                raise TypeError("malformed callback result")
            if reason_shape and choice != "deny":
                raise ValueError("reason is valid only for deny")
            if operator_denial_shape:
                operator_denied = (
                    choice == "deny"
                    and result["operator_denial"] is True
                    and trusted_gateway_callback
                )
                if not operator_denied:
                    raise ValueError("untrusted operator-denial result")
            if choice not in {"once", "session", "always", "deny", "timeout"}:
                raise ValueError("unknown callback choice")
        except Exception:
            logger.warning("SDK approval callback failed at protected boundary")
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, "approval callback failed",
            )
            return PermissionResultDeny(message="approval callback failed")
        if choice in ("once", "session", "always"):
            return PermissionResultAllow(updated_input=frozen_tool_input)
        if operator_denied:
            message = "denied by user"
            if reason:
                message = f"{message}: {reason}"
        elif choice == "timeout":
            message = "approval timed out — no operator response"
        else:
            safe_reason = _safe_sdk_deny_log_reason(reason)
            message = (
                safe_reason
                if safe_reason != "non-operator denial"
                else "approval denied by callback"
            )
        if not operator_denied:
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, _safe_sdk_deny_log_reason(message),
            )
        return PermissionResultDeny(message=message)
