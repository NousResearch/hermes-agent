"""Session creation and per-turn wiring for the claude-agent-sdk runtime.

What used to be closures nested in ``run_claude_agent_sdk_turn`` are small
owner objects and module-level functions that take the agent explicitly; the
session receives the per-agent ones bound with ``functools.partial``. Lifetime
split kept visible: ``_make_visibility_callbacks`` refreshes every turn, while
``_create_session`` (approval callback, prompt append, budget, bridge inputs) is
session-creation work. Extracted from ``claude_sdk_runtime.py``.
"""

from __future__ import annotations

import functools
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent.redact import redact_sensitive_text
from agent.claude_sdk_runtime_compaction import _on_compact_boundary, _on_compaction
from agent.claude_sdk_runtime_continuity import (
    _canonical_sdk_cwd,
    _persisted_sdk_session_id,
    _render_continuity_digest,
    _store_sdk_session_id,
)
from agent.claude_sdk_runtime_prompt import build_system_prompt_append
from agent.claude_sdk_runtime_state import _SdkTurnState
from agent.claude_sdk_runtime_tools import (
    _hybrid_bridge_enabled,
    _snapshot_agent_tools_with_mcp_refresh,
)

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


class _TurnVisibility:
    """Visibility callbacks fenced to one exact Hermes turn.

    Created once per turn. The epoch / turn-id pair it records on the agent lets
    a late callback from a superseded turn (or a replaced session) recognise that
    it is stale and drop itself; the iteration counter and the stream-sink flag
    start fresh with it.
    """

    def __init__(self, agent) -> None:
        self._agent = agent
        self._turn_id = str(getattr(agent, "_current_turn_id", "") or "")
        lock = getattr(agent, "_sdk_visibility_lock", None)
        if lock is None:
            lock = threading.RLock()
            agent._sdk_visibility_lock = lock
        self._lock = lock
        with lock:
            agent._sdk_visibility_epoch = getattr(agent, "_sdk_visibility_epoch", 0) + 1
            self._epoch = agent._sdk_visibility_epoch
            agent._sdk_visibility_turn_id = self._turn_id
            agent._sdk_visibility_iteration_count = 0
            agent._sdk_stream_sink_accepted = False

    def is_current(self) -> bool:
        agent = self._agent
        with self._lock:
            return (
                getattr(agent, "_sdk_visibility_epoch", None) == self._epoch
                and getattr(agent, "_sdk_visibility_turn_id", None) == self._turn_id
                and getattr(agent, "_current_turn_id", None) == self._turn_id
                and not getattr(agent, "_interrupt_requested", False)
            )

    def on_tool_iteration(self) -> None:
        agent = self._agent
        with self._lock:
            if not self.is_current():
                return
            agent._sdk_visibility_iteration_count += 1
        try:
            agent._touch_activity("completed SDK tool iteration")
        except Exception:
            logger.debug("claude-sdk iteration activity update failed", exc_info=True)

    def relay_interim_assistant(self, text: str) -> None:
        agent = self._agent
        if not self.is_current() or not isinstance(text, str):
            return
        visible = agent._strip_think_blocks(text).strip()
        if visible:
            from agent.redact import redact_sensitive_text
            visible = redact_sensitive_text(visible)
        if not visible or visible == "(empty)" or agent._interim_text_was_delivered(visible):
            return
        callback = getattr(agent, "interim_assistant_callback", None)
        if callback is None:
            return
        # Mirror the native lane (run_agent.py::_emit_interim_assistant_
        # message): compute the flag instead of hardcoding it. With
        # `agent.claude_agent_sdk.streaming` on, this prose has already
        # been painted by the delta sink, and a False here makes the
        # surface re-render it as fresh commentary on top of the
        # streaming buffer instead of sealing the segment — the text
        # visibly appears, is dropped, then reappears.
        # Two signals, not one: the accumulator says the text was
        # released, the flag says a surface accepted it. Sealing a segment
        # nobody painted would drop the prose from the UI entirely.
        already_streamed = bool(
            getattr(agent, "_sdk_stream_sink_accepted", False)
        ) and agent._interim_content_was_streamed(visible)
        try:
            callback(visible, already_streamed=already_streamed)
            agent._record_delivered_interim_text(visible)
        except Exception:
            logger.debug("interim assistant relay raised", exc_info=True)


def _make_visibility_callbacks(agent):
    """Create visibility callbacks fenced to this exact Hermes turn."""
    visibility = _TurnVisibility(agent)
    return visibility.relay_interim_assistant, visibility.on_tool_iteration


def _approval_bypass_active(agent) -> bool:
    """Resolve live trusted bypass posture for the foreign SDK thread."""
    try:
        from tools.approval import is_approval_bypass_active_for_session

        ctx = getattr(agent, "_sdk_approval_turn_ctx", None)
        session_key = (
            ctx.get("session_key", "") if type(ctx) is dict else ""
        )
        return is_approval_bypass_active_for_session(session_key)
    except Exception:
        return False


def _on_tool_started(agent, tool_name: str, preview: str, args: dict) -> None:
    # Claude SDK tool calls bypass the native tool executor, so mirror
    # its shared activity updates here. The gateway heartbeat reads
    # get_activity_summary(), which derives its useful current action
    # from these fields; without this, an active SDK turn remains
    # stuck at its initial "initializing" state.
    agent._sdk_issued_tool_effect = True
    agent._current_tool = tool_name
    try:
        agent._touch_activity(f"executing tool: {tool_name}")
    except Exception:
        logger.debug("claude-sdk activity update failed", exc_info=True)
    progress_callback = getattr(agent, "tool_progress_callback", None)
    if progress_callback is None:
        return
    try:
        progress_callback("tool.started", tool_name, preview, args)
    except Exception:
        logger.debug(
            "claude-sdk tool-progress callback raised", exc_info=True
        )


def _relay_stream_delta(agent, text: str) -> None:
    # Late-bound: the gateway assigns stream_delta_callback per turn
    # AFTER the session exists (and clears it between turns).
    # Fan out to BOTH display sinks, mirroring the native runtimes
    # (run_agent.py: [self.stream_delta_callback, self._stream_callback]).
    # `stream_delta_callback` is the CLI/TUI sink. `_stream_callback` is
    # the one the JSON-RPC gateway installs via run_conversation's
    # `stream_callback=` kwarg, and that is the sink the DESKTOP listens
    # on (it feeds the `message.delta` notification). Relaying only to
    # the first meant the desktop never streamed on this runtime, no
    # matter how the operator set display.streaming.
    callbacks = [
        cb
        for cb in (
            getattr(agent, "stream_delta_callback", None),
            getattr(agent, "_stream_callback", None),
        )
        if cb is not None
    ]
    if not callbacks:
        return
    # Record BEFORE the sinks run, deliberately: a sink that raises
    # *after* handing text to the user must still count as streamed, or
    # the turn fails over and replays output the user already saw
    # (pinned by test_stream_relay_records_delivery_before_display_
    # callback). The accumulator answers "was this released?", not
    # "did a surface paint it?".
    agent._record_streamed_assistant_text(text)
    for cb in callbacks:
        try:
            cb(text)
        except Exception:
            logger.debug("stream delta relay raised", exc_info=True)
        else:
            # Separate signal for the interim relay: sealing a segment
            # is only safe once a sink actually accepted a delta.
            agent._sdk_stream_sink_accepted = True


@dataclass
class _BackgroundResultDelivery:
    """Enqueue a finished background answer burst for DIRECT platform delivery.

    The completion is the AGENT'S OWN finished answer — it must go straight to
    the platform outbound lane, never back into the model as a synthetic
    delegation (2026-08-06 self-echo: the model recognized its own text, refused
    to "relay" it, and the report never left the box). The watcher delivers each
    payload as its own outbound message, in order.

    The creation-time snapshots survive as FALLBACKS only — the SDK session
    outlives hermes session rotations, so anything read at creation can be stale
    by the time a background completion fires. Parent/route are resolved AT
    DELIVERY TIME: a completion firing after a hermes session rotation must carry
    the LIVE session id, not the creation-time snapshot — the gateway classifies a
    rotated-away parent as permanently gone and drops the delivery.
    """

    agent: Any
    session_key: str
    parent_session_id: Any
    model: Any

    def __call__(self, texts: list[str]) -> None:
        agent = self.agent
        try:
            from tools.approval_context import (
                get_current_session_key as _live_key_fn,
            )

            _live_key = _live_key_fn() or ""
        except Exception:
            _live_key = ""
        # This callback fires on the SDK loop thread, where the
        # get_current_session_key contextvar may be unset — an empty
        # live read falls back to the creation-time snapshot rather
        # than losing the route.
        session_key = _live_key or self.session_key
        parent_session_id = (
            getattr(agent, "session_id", None) or self.parent_session_id
        )
        model = getattr(agent, "model", None) or self.model
        try:
            import time as _time

            from tools.process_registry import process_registry

            now = _time.time()
            process_registry.completion_queue.put({
                "type": "sdk_background_result",
                "payloads": list(texts),
                "session_key": session_key,
                "parent_session_id": parent_session_id,
                "model": model,
                "dispatched_at": now,
                "completed_at": now,
            })
        except Exception:
            logger.warning(
                "claude-sdk background-result enqueue failed — "
                "answer may be lost", exc_info=True,
            )


def _build_approval_callback(agent):
    """The per-session approval callback: the CLI's thread-local one, else the gateway bridge."""
    try:
        from tools.terminal_tool import _get_approval_callback
        approval_callback = _get_approval_callback()
    except Exception:
        approval_callback = None
    if approval_callback is None:
        # Gateway turns have no thread-local CLI callback — without this
        # bridge the SDK denies every un-allowlisted tool silently, no
        # prompt reaching the user, even though the gateway registers a
        # notify channel around every turn (production finding on a 24/7
        # telegram deployment). The builder returns None for surfaces
        # that are not gateway-shaped, so CLI posture is unchanged; the
        # context_provider hands it the per-turn snapshot refreshed by
        # run_claude_agent_sdk_turn, so cron-ness and the session key are
        # resolved per CALL (a cron-born session must not be frozen into
        # forever-deny).
        try:
            from tools.approval_sdk_gateway import build_sdk_gateway_approval_callback
            approval_callback = build_sdk_gateway_approval_callback(
                context_provider=lambda: (
                    getattr(agent, "_sdk_approval_turn_ctx", None) or {}
                ),
            )
        except Exception:
            approval_callback = None
    return approval_callback


def _background_result_sink(agent) -> Optional[_BackgroundResultDelivery]:
    """Delivery half of the stream-ownership fix, config-gated (default OFF).

    When the CLI finishes a background Agent task between turns, the session
    captures the answer burst and this callback enqueues it as an
    "sdk_background_result" completion event; the gateway watcher sends it
    DIRECTLY on the platform outbound lane (completion_queue →
    _async_delegation_watcher → adapter send). In-memory at-least-once, same as
    the watcher's requeue semantics. Default OFF per the block's
    upstream-conservative contract (every default falsy — pinned by
    test_canonical_defaults); gateway-bot deployments opt in.
    """
    from agent.transports.claude_agent_sdk_session import _provider_flag

    if not _provider_flag("deliver_background_results", default=False):
        return None
    # Creation-time snapshots survive as FALLBACKS only — the SDK
    # session outlives hermes session rotations, so anything read
    # here can be stale by the time a background completion fires.
    try:
        from tools.approval_context import get_current_session_key

        _bg_session_key = get_current_session_key() or ""
    except Exception:
        _bg_session_key = ""
    return _BackgroundResultDelivery(
        agent=agent,
        session_key=_bg_session_key,
        parent_session_id=getattr(agent, "session_id", None),
        model=getattr(agent, "model", None),
    )


def _configured_max_budget_usd() -> Optional[float]:
    """agent.claude_agent_sdk.max_budget_usd from config.yaml.

    Forwarded to the SDK's ``max_budget_usd`` option: the query stops with an
    ``error_max_budget_usd`` result once exceeded (which run_turn already
    surfaces as "SDK turn ended: error_max_budget_usd"). None/absent — the
    canonical default — means no budget, i.e. current behavior. Non-numeric
    or non-positive values are ignored with a warning rather than passed
    through: a 0 cap would fail every turn instantly, and a typo must never
    become a silent behavior change."""
    from agent.transports.claude_agent_sdk_session import _provider_config

    raw = _provider_config().get("max_budget_usd")
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        # YAML `true` would float() to 1.0 — a nonsense budget, reject it.
        logger.warning(
            "agent.claude_agent_sdk.max_budget_usd=%r is not a number — "
            "ignoring (no budget cap).", raw,
        )
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "agent.claude_agent_sdk.max_budget_usd=%r is not a number — "
            "ignoring (no budget cap).", raw,
        )
        return None
    if value <= 0:
        logger.warning(
            "agent.claude_agent_sdk.max_budget_usd=%r must be positive — "
            "ignoring (no budget cap).", raw,
        )
        return None
    return value


def _create_session(
    agent,
    *,
    resume_id: Optional[str],
    on_interim_assistant,
    on_tool_iteration,
) -> None:
    """Build the SDK session for this agent (session-creation work, not per turn)."""
    from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    cwd = str(resolve_agent_cwd())
    context_cwd = resolve_context_cwd()
    approval_callback = _build_approval_callback(agent)

    append = build_system_prompt_append(
        platform=getattr(agent, "platform", None),
        session_id=getattr(agent, "session_id", None),
        model=getattr(agent, "model", None),
        cwd=str(context_cwd) if context_cwd is not None else None,
        include_project_context=not bool(
            getattr(agent, "skip_context_files", False)
        ),
    )

    on_unsolicited_result = _background_result_sink(agent)

    agent._claude_sdk_session = ClaudeAgentSdkSession(
        cwd=cwd,
        model=getattr(agent, "model", None) or None,
        approval_callback=approval_callback,
        approval_bypass_provider=functools.partial(_approval_bypass_active, agent),
        on_tool_started=functools.partial(_on_tool_started, agent),
        system_prompt_append=append,
        hermes_session_id=getattr(agent, "session_id", None),
        resume_session_id=resume_id,
        on_stream_delta=functools.partial(_relay_stream_delta, agent),
        on_interim_assistant=on_interim_assistant,
        on_tool_iteration=on_tool_iteration,
        on_unsolicited_result=on_unsolicited_result,
        on_compaction=functools.partial(_on_compaction, agent),
        on_compact_boundary=functools.partial(_on_compact_boundary, agent),
        # Operator budget cap (agent.claude_agent_sdk.max_budget_usd);
        # None = no budget. Read per session creation so a config edit
        # applies on the next session, same as the append snapshot.
        max_budget_usd=_configured_max_budget_usd(),
        # Hybrid MCP bridge inputs (ported from PR #56413). Passing the
        # live agent + its OpenAI-format tool list activates an in-process
        # MCP server that exposes the full Hermes tool registry — so
        # proxified third-party MCP servers become reachable from inside
        # the SDK loop, not just the ~25 curated stdio tools.
        #
        # Off by default (agent.claude_agent_sdk.hybrid_mcp_bridge:
        # false) so a green-field upgrade is byte-identical to fcava's
        # stdio-only behaviour — the wide bridge exposes agent-level
        # tools whose enablement is a security choice. Operators opt in
        # explicitly.
        #
        # agent.tools is a snapshot taken at agent build time and never
        # re-reads the registry (see tools/mcp_tool.py::refresh_agent_mcp_tools
        # docstring). If an HTTP MCP finished connecting AFTER that snapshot
        # (e.g. slow initial handshake, or /reload-mcp), its tools would be
        # invisible to the hybrid bridge. Force a refresh here so the bridge
        # sees the current registry — the same call turn_context.py does
        # between turns, but pulled forward so it also applies to the
        # session-creation build.
        agent=(agent if _hybrid_bridge_enabled() else None),
        tools=(
            _snapshot_agent_tools_with_mcp_refresh(agent)
            if _hybrid_bridge_enabled()
            else None
        ),
    )
    # The prologue persisted Hermes' native composed prompt — a prompt
    # this runtime never sends. Overwrite the snapshot with the
    # EFFECTIVE prompt so the audit trail tells the truth.
    try:
        if getattr(agent, "_session_db", None) and agent.session_id:
            agent._session_db.update_system_prompt(
                agent.session_id, "[claude_code preset]\n\n" + (append or "")
            )
    except Exception:
        logger.debug("effective-prompt snapshot failed", exc_info=True)


def _refresh_turn_visibility(agent, state: _SdkTurnState) -> None:
    """Per-turn: fresh visibility callbacks, pushed onto an already-live session.

    Also the workspace fence. A cached agent can be reused after the workspace
    moved; its live SDK session is still bound to the OLD cwd, and the child
    CLI's cwd is fixed at spawn. Retire it here — before the callbacks are
    pushed onto a session we are about to discard — so the attempt loop creates
    a fresh session in the current workspace.
    """
    state.on_interim_assistant, state.on_tool_iteration = _make_visibility_callbacks(agent)
    live_session = getattr(agent, "_claude_sdk_session", None)
    live_cwd = getattr(live_session, "_cwd", None) if live_session is not None else None
    if isinstance(live_cwd, str):
        if _canonical_sdk_cwd(live_cwd) != _canonical_sdk_cwd():
            logger.info(
                "claude-agent-sdk: retiring live session after workspace change"
            )
            try:
                live_session.close()
            except Exception:
                logger.debug("workspace-change session close failed", exc_info=True)
            agent._claude_sdk_session = None
            live_session = None
    if live_session is not None:
        try:
            live_session.set_turn_visibility_callbacks(
                on_interim_assistant=state.on_interim_assistant,
                on_tool_iteration=state.on_tool_iteration,
            )
        except Exception:
            logger.debug("claude-sdk visibility callback refresh failed", exc_info=True)


def _run_sdk_attempts(agent, state: _SdkTurnState) -> Optional[Dict[str, Any]]:
    """Drive the turn through the session: resume when an id is persisted,
    retire and retry ONCE with the continuity digest on a failed or stale
    resume.

    Sets ``state.turn`` / ``state.resumed``. Returns the failed result dict
    when the session RAISED (a dead turn, not a recoverable partial), else
    None.
    """
    user_input = state.user_input
    messages = state.messages
    turn = None
    resumed = False
    send_input = user_input
    for attempt in (0, 1):
        if not hasattr(agent, "_claude_sdk_session") or agent._claude_sdk_session is None:
            resume_id = _persisted_sdk_session_id(agent) if attempt == 0 else None
            resumed = bool(resume_id)
            send_input = user_input
            if not resume_id and len(messages) > 1:
                digest = _render_continuity_digest(messages[:-1])
                if digest:
                    if isinstance(user_input, list):
                        send_input = [
                            {"type": "text", "text": digest},
                            *user_input,
                        ]
                    else:
                        send_input = digest + user_input
            _create_session(
                agent,
                resume_id=resume_id,
                on_interim_assistant=state.on_interim_assistant,
                on_tool_iteration=state.on_tool_iteration,
            )

        turn_session_cwd = getattr(agent._claude_sdk_session, "_cwd", None)
        if not isinstance(turn_session_cwd, str):
            # The live session is the authority, but it can be absent here (a
            # retired client, a stand-in). Resolve the fallback NOW rather than
            # leaving None for the persist phase to fill in: persist runs after
            # run_turn, so a workspace that moved mid-turn would be sampled
            # post-move and the binding would always match itself. Sampling at
            # turn start is what makes the check able to fail.
            from agent.runtime_cwd import resolve_agent_cwd

            try:
                turn_session_cwd = str(resolve_agent_cwd())
            except Exception:
                logger.debug(
                    "claude-agent-sdk: turn workspace unresolvable", exc_info=True
                )
                turn_session_cwd = None
        state.turn_session_cwd = turn_session_cwd
        try:
            turn = agent._claude_sdk_session.run_turn(user_input=send_input)
        except Exception as exc:
            safe_exc = redact_sensitive_text(str(exc), force=True)
            interrupted = bool(getattr(agent, "_interrupt_requested", False))
            # A PreCompact hook may have opened a transient user-visible status.
            # This exception bypasses the normal terminal edge below; clear it
            # here so a later unrelated turn cannot announce stale completion.
            if getattr(agent, "_sdk_compaction_pending", False):
                agent._sdk_compaction_pending = False
                try:
                    emit = getattr(agent, "_emit_status", None)
                    if callable(emit):
                        emit("⚠️ Context compaction interrupted")
                except Exception:
                    logger.debug("failed to close interrupted compaction status", exc_info=True)
            # Do not use logger.exception here: it appends the raw exception
            # string after the redacted message to the log record.
            logger.error("claude-agent-sdk turn failed: %s", safe_exc)
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None
            # The sample above was taken BEFORE close(). Tearing the transport
            # down is not instantaneous, so a stop admitted during cleanup is
            # still a stop against this turn — re-read the flag now that the
            # session is gone. Deciding on the stale pre-close sample would
            # replay the prompt the user just asked to abandon and leave the
            # agent-level flag set to poison the next turn. Sticky OR: a stop
            # observed before close stays observed regardless of cleanup.
            interrupted = interrupted or bool(
                getattr(agent, "_interrupt_requested", False)
            )
            if interrupted:
                # The session close above consumes transport-local interrupt
                # state, and the session object is discarded either way, so
                # no live transport survives to carry it. Consume the agent
                # layer too: this dead turn honored the user's stop and must
                # not reject the next message.
                agent._interrupt_requested = False
            if resumed and attempt == 0 and not interrupted:
                # A raising RESUMED session is a suspect resume — clear the
                # id and give the turn one fresh chance (digest included).
                # Never replay a turn that concurrently received /stop.
                _store_sdk_session_id(agent, None)
                resumed = False
                continue
            return {
                "final_response": f"claude-agent-sdk turn failed: {safe_exc}",
                "messages": messages,
                "api_calls": 0,
                "completed": False,
                "partial": True,
                "interrupted": interrupted,
                # run_turn consumes its own exceptions into TurnResult, so
                # anything RAISING here is a dead turn, not a recoverable
                # partial — mark it failed so one-shot runs exit nonzero
                # (mirrors conversation_loop's generic non-retryable return).
                "failed": not interrupted,
                "error": safe_exc,
            }

        if getattr(turn, "should_retire", False):
            logger.warning(
                "claude-agent-sdk session retired (turn error: %s)",
                redact_sensitive_text(str(turn.error or ""), force=True),
            )
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None
            # Error/timeout retire always clears the persisted resume id —
            # never resume a conversation that just failed.
            _store_sdk_session_id(agent, None)
            if (
                resumed
                and attempt == 0
                and not getattr(turn, "interrupted", False)
                and not getattr(agent, "_interrupt_requested", False)
            ):
                # Stale/failed resume: one fresh retry with digest. Never for
                # an INTERRUPTED retire (user /stop that killed the CLI, or a
                # hard watchdog trip) — re-running the stopped turn in full
                # would evaporate the stop and deliver the answer anyway.
                resumed = False
                continue
        break

    state.turn = turn
    state.resumed = resumed
    return None
