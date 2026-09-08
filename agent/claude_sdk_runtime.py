"""claude-agent-sdk runtime — the subscription-Claude agent-loop path.

The structural twin of ``agent/codex_runtime.py``'s app-server path: hands the
entire turn to Anthropic's official ``claude-agent-sdk`` (which drives the
Claude Code CLI's own agent loop under subscription OAuth by default, with
known metered lanes refused unless explicitly enabled) and projects its typed
message stream back into Hermes'
messages list so transcript persistence and recall keep working. GitHub
issue #25267.

* ``run_claude_agent_sdk_turn`` — drives one turn through a lazily-created
  ``ClaudeAgentSdkSession`` (used when ``agent.api_mode == "claude_agent_sdk"``).
"""

from __future__ import annotations

import copy
import functools
import logging
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text
from agent.claude_sdk_runtime_continuity import (
    _persisted_sdk_session_id,
    _render_continuity_digest,
    _store_sdk_session_id,
)
from agent.claude_sdk_runtime_fallback import (
    ClaudeSdkTurnEffects,
    _sdk_provider_failover_reason,
)
from agent.claude_sdk_runtime_prompt import (
    build_system_prompt_append,
)
from agent.claude_sdk_runtime_compaction import (
    _emit_dangling_compaction_done,
    _on_compact_boundary,
    _on_compaction,
)
from agent.claude_sdk_runtime_session import (
    _approval_bypass_active,
    _make_visibility_callbacks,
)
from agent.claude_sdk_runtime_tools import (
    _hybrid_bridge_enabled,
    _snapshot_agent_tools_with_mcp_refresh,
)
from agent.claude_sdk_runtime_usage import (
    _record_claude_sdk_usage,
)

logger = logging.getLogger(__name__)


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


def run_claude_agent_sdk_turn(
    agent,
    *,
    user_message: Any,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """claude-agent-sdk runtime path. Hands the entire turn to the SDK's
    agent loop and projects its messages back into Hermes' list.

    Called from run_conversation() when agent.api_mode == "claude_agent_sdk".
    Returns the same dict shape as the chat_completions path.

    Continuity retire matrix (#25267):
      /new, session expiry      → NEW Hermes session row → no persisted id → fresh
      gateway restart/eviction  → same row, id persisted  → RESUME
      error/timeout retire      → id CLEARED → next turn fresh + digest
      stale/failed resume       → retire → clear → ONE fresh retry with digest
    """
    from agent.transports.claude_agent_sdk_session import (
        ClaudeAgentSdkSession,
        _coerce_turn_input,
    )

    user_input = _coerce_turn_input(user_message)
    if isinstance(user_input, str) and not user_input.strip():
        agent._interrupt_requested = False
        live_session = getattr(agent, "_claude_sdk_session", None)
        if live_session is not None:
            try:
                live_session.consume_interrupt()
            except Exception:
                logger.debug("consume_interrupt failed", exc_info=True)
        rejection = (
            "This Claude Agent SDK route can't process an empty message. "
            "Please send text or a supported image."
        )
        return {
            "final_response": rejection,
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": rejection,
            "interrupted": False,
            "agent_persisted": True,
        }

    # P1.b: refresh the approval-context snapshot EVERY turn (including
    # session-reuse turns). This runs on the agent turn thread, where the
    # session contextvars are visible; the SDK invokes the approval callback
    # from its own loop thread, where they never are — the callback reads
    # this holder instead. A cron turn writes gateway=False (honest deny, no
    # block); a later interactive turn on the SAME SDK session rewrites it
    # and un-freezes the cron-born session.
    try:
        from tools.approval_sdk_gateway import current_approval_turn_context

        agent._sdk_approval_turn_ctx = current_approval_turn_context()
    except Exception:
        logger.debug("approval turn-context refresh failed", exc_info=True)

    def _create_session(resume_id: Optional[str]) -> None:
        from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd

        cwd = str(resolve_agent_cwd())
        context_cwd = resolve_context_cwd()
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
            # context_provider hands it the per-turn snapshot refreshed
            # above, so cron-ness and the session key are resolved per CALL
            # (a cron-born session must not be frozen into forever-deny).
            try:
                from tools.approval_sdk_gateway import build_sdk_gateway_approval_callback
                approval_callback = build_sdk_gateway_approval_callback(
                    context_provider=lambda: (
                        getattr(agent, "_sdk_approval_turn_ctx", None) or {}
                    ),
                )
            except Exception:
                approval_callback = None

        def _on_tool_started(tool_name: str, preview: str, args: dict) -> None:
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

        def _relay_stream_delta(text: str) -> None:
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

        append = build_system_prompt_append(
            platform=getattr(agent, "platform", None),
            session_id=getattr(agent, "session_id", None),
            model=getattr(agent, "model", None),
            cwd=str(context_cwd) if context_cwd is not None else None,
            include_project_context=not bool(
                getattr(agent, "skip_context_files", False)
            ),
        )

        # Delivery half of the stream-ownership fix: when the CLI finishes a
        # background Agent task between turns, the session captures the
        # answer burst and this callback enqueues it as an
        # "sdk_background_result" completion event; the gateway watcher sends
        # it DIRECTLY on the platform outbound lane (completion_queue →
        # _async_delegation_watcher → adapter send). In-memory at-least-once,
        # same as the watcher's requeue semantics. Config-gated, default OFF
        # per the block's upstream-conservative contract (every default falsy
        # — pinned by test_canonical_defaults); gateway-bot deployments opt
        # in.
        on_unsolicited_result = None
        from agent.transports.claude_agent_sdk_session import _provider_flag

        if _provider_flag("deliver_background_results", default=False):
            # Creation-time snapshots survive as FALLBACKS only — the SDK
            # session outlives hermes session rotations, so anything read
            # here can be stale by the time a background completion fires.
            try:
                from tools.approval_context import get_current_session_key

                _bg_session_key = get_current_session_key() or ""
            except Exception:
                _bg_session_key = ""
            _bg_parent_session_id = getattr(agent, "session_id", None)
            _bg_model = getattr(agent, "model", None)

            def _deliver_background_result(texts: list[str]) -> None:
                # The completion is the AGENT'S OWN finished answer — it must
                # go straight to the platform outbound lane, never back into
                # the model as a synthetic delegation (2026-08-06 self-echo:
                # the model recognized its own text, refused to "relay" it,
                # and the report never left the box). The watcher delivers
                # each payload as its own outbound message, in order.
                #
                # Parent/route are resolved AT DELIVERY TIME: a completion
                # firing after a hermes session rotation must carry the LIVE
                # session id, not the creation-time snapshot — the gateway
                # classifies a rotated-away parent as permanently gone and
                # drops the delivery.
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
                session_key = _live_key or _bg_session_key
                parent_session_id = (
                    getattr(agent, "session_id", None) or _bg_parent_session_id
                )
                model = getattr(agent, "model", None) or _bg_model
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

            on_unsolicited_result = _deliver_background_result

        agent._claude_sdk_session = ClaudeAgentSdkSession(
            cwd=cwd,
            model=getattr(agent, "model", None) or None,
            approval_callback=approval_callback,
            approval_bypass_provider=functools.partial(_approval_bypass_active, agent),
            on_tool_started=_on_tool_started,
            system_prompt_append=append,
            hermes_session_id=getattr(agent, "session_id", None),
            resume_session_id=resume_id,
            on_stream_delta=_relay_stream_delta,
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

    # NOTE: the user message is ALREADY appended to messages by the standard
    # run_conversation() flow before the early return reaches us. Do NOT
    # append again — that would duplicate. (Same contract as codex_runtime.)

    # An interrupt that landed before the SDK session exists (first turn, or
    # right after a retire) only set agent._interrupt_requested — honor it
    # here, mirroring the native loop's top-of-loop check, and consume the
    # flag so the NEXT turn runs normally.
    if getattr(agent, "_interrupt_requested", False):
        agent._interrupt_requested = False
        live_session = getattr(agent, "_claude_sdk_session", None)
        if live_session is not None:
            # interrupt() also set the live session's event; consume it here
            # or the NEXT legitimate message dies on the stale event with no
            # model call.
            try:
                live_session.consume_interrupt()
            except Exception:
                logger.debug("consume_interrupt failed", exc_info=True)
        return {
            "final_response": "",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            # Without this key the gateway's empty-response normalizer has no
            # branch to take (interrupted absent, api_calls 0, partial True →
            # every arm defeated) and the user's message dies in SILENCE.
            # With it, api_calls==0 + interrupted surfaces the honest
            # "interrupted before processing — send it again" path.
            "interrupted": True,
            "error": None,
            "agent_persisted": True,
        }

    # Stream/replay state belongs to this SDK attempt, never to the cached
    # gateway agent.  Reset before session startup too: an authoritative auth
    # failure may happen before a model call and must not inherit prior output.
    agent._current_streamed_assistant_text = ""
    agent._sdk_issued_tool_effect = False
    _messages_before_sdk_attempt = copy.deepcopy(messages)

    on_interim_assistant, on_tool_iteration = _make_visibility_callbacks(agent)
    live_session = getattr(agent, "_claude_sdk_session", None)
    if live_session is not None:
        try:
            live_session.set_turn_visibility_callbacks(
                on_interim_assistant=on_interim_assistant,
                on_tool_iteration=on_tool_iteration,
            )
        except Exception:
            logger.debug("claude-sdk visibility callback refresh failed", exc_info=True)

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
            _create_session(resume_id)

        try:
            turn = agent._claude_sdk_session.run_turn(user_input=send_input)
        except Exception as exc:
            safe_exc = redact_sensitive_text(str(exc), force=True)
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
            if resumed and attempt == 0:
                # A raising RESUMED session is a suspect resume — clear the
                # id and give the turn one fresh chance (digest included).
                _store_sdk_session_id(agent, None)
                resumed = False
                continue
            return {
                "final_response": f"claude-agent-sdk turn failed: {safe_exc}",
                "messages": messages,
                "api_calls": 0,
                "completed": False,
                "partial": True,
                # run_turn consumes its own exceptions into TurnResult, so
                # anything RAISING here is a dead turn, not a recoverable
                # partial — mark it failed so one-shot runs exit nonzero
                # (mirrors conversation_loop's generic non-retryable return).
                "failed": True,
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
            ):
                # Stale/failed resume: one fresh retry with digest. Never for
                # an INTERRUPTED retire (user /stop that killed the CLI, or a
                # hard watchdog trip) — re-running the stopped turn in full
                # would evaporate the stop and deliver the answer anyway.
                resumed = False
                continue
        break

    _sdk_effects = ClaudeSdkTurnEffects(
        tool=(
            bool(getattr(agent, "_sdk_issued_tool_effect", False))
            or int(getattr(turn, "tool_iterations", 0) or 0) > 0
        ),
        streamed=bool(getattr(agent, "_current_streamed_assistant_text", "")),
        projected=bool(getattr(turn, "projected_messages", None)),
        interrupted=bool(
            getattr(turn, "interrupted", False)
            or getattr(agent, "_interrupt_requested", False)
        ),
        mutated=messages != _messages_before_sdk_attempt,
    )
    _sdk_failover_reason = None
    if getattr(turn, "error", None) and _sdk_effects.replay_safe:
        _sdk_failover_reason = _sdk_provider_failover_reason(
            agent,
            str(turn.error),
            getattr(turn, "fatal_reason", None),
        )
        if _sdk_failover_reason is not None and agent._claude_sdk_session is not None:
            # A provider switch must never retain transport/session state from
            # the failed SDK backend.
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None
            _store_sdk_session_id(agent, None)

    # FALLBACK ONLY. _on_compact_boundary above is the real terminal edge and
    # normally clears the flag mid-turn; reaching here means the CLI started a
    # compaction and never streamed its compact_boundary (older/newer CLI, or a
    # turn that died mid-compaction). A completed turn still proves the
    # compaction ended, so this closes the dangling status rather than leaving
    # "🗜️ Compacting..." as the user's last word from the turn.
    #
    # Do NOT promote this back to the primary path: end-of-turn is exactly where
    # end-of-turn progress cleanup deletes the message, so the notice is emitted
    # and destroyed in the same instant (see _handle_compact_boundary).
    _emit_dangling_compaction_done(agent)

    # Interrupt handoff (codex_runtime parity, its ~739-746): capture BEFORE
    # the consume below zeroes the agent flag — the result dict needs it, and
    # without an "interrupted" key the gateway's queued-drain classifies an
    # interrupted turn as a plain partial error and DELIVERS the abandoned
    # turn's error text ("⚠️ Processing stopped… Try again", 2026-08-09
    # barge-in incident) instead of discarding it via its interrupted branch.
    _user_interrupted = bool(
        getattr(turn, "interrupted", False)
        and getattr(agent, "_interrupt_requested", False)
    )

    if getattr(turn, "interrupted", False):
        # The interrupt was honored by THIS turn — consume the agent-level
        # flag so the next turn is not short-circuited by it.
        agent._interrupt_requested = False
        if agent._claude_sdk_session is not None:
            # The abandoned stream may still hold the interrupted turn's
            # ResultMessage; a REUSED client would serve it as the NEXT
            # turn's answer. Retire the client — the persisted id below lets
            # the next turn RESUME the same SDK conversation cleanly.
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None

    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        # Early-return path bypasses conversation_loop's per-step persistence;
        # flush the new projected rows ourselves (idempotent via the intrinsic
        # _DB_PERSISTED_MARKER — the user turn was flushed at turn start).
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "claude-sdk projected-message flush failed", exc_info=True
                )

    if not getattr(turn, "should_retire", False) and _sdk_failover_reason is None:
        # Persist the SDK session id for restart/eviction/interrupt resume.
        # AFTER the flush on purpose: the flush's _ensure_db_session retry is
        # what (re)creates the session row when turn-start persistence hit a
        # transient lock — storing first would silently discard the id.
        thread_id = getattr(turn, "thread_id", None)
        if thread_id:
            _store_sdk_session_id(agent, thread_id)

    # Counter ticks — _turns_since_memory/_user_turn_count are incremented by
    # run_conversation()'s pre-loop block; only _iters_since_skill is ours.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    usage_result = (
        _record_claude_sdk_usage(agent, turn)
        if getattr(turn, "api_call_made", True)
        else {}
    )

    should_review_skills = False
    # Skill-review cadence belongs to review policy, not foreground tool
    # availability. If routed, a distinct normal runtime owns optional writes.
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    if (
        turn.final_text
        and not turn.interrupted
        and not agent.skip_background_review
        and (should_review_memory or should_review_skills)
    ):
        # #25267 suppressed this spawn unconditionally: the fork inherits
        # api_mode="claude_agent_sdk" and early-returns into a fresh SDK
        # session whose tool surface has no `memory` / `skill_manage`, so it
        # burned a subscription turn and wrote nothing.
        #
        # That reasoning only holds when the fork stays on THIS runtime. With
        # auxiliary.background_review naming a concrete different
        # provider/model, _resolve_review_runtime reports routed=True and the
        # review runs over there — on a normal tool surface that can write.
        # Suppressing it in that case is what left this lane unable to record
        # anything durable across sessions.
        #
        # Unrouted, #25267 still applies: skip, and let the counters above
        # keep ticking for a bounded replacement pass.
        routed = False
        try:
            from agent.background_review import _resolve_review_runtime

            routed = bool(_resolve_review_runtime(agent).get("routed"))
        except Exception:
            logger.debug("review-runtime resolve failed", exc_info=True)

        if routed:
            try:
                agent._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=should_review_memory,
                    review_skills=should_review_skills,
                )
            except Exception:
                logger.debug("background review spawn raised", exc_info=True)
        else:
            logger.debug(
                "claude-sdk runtime: background review skipped "
                "(memory=%s, skills=%s) — the review fork cannot write on "
                "this runtime",
                should_review_memory,
                should_review_skills,
            )

    result = {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": int(getattr(turn, "api_call_made", True)),
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "failed": bool(turn.error) and not _sdk_effects.interrupted,
        "error": redact_sensitive_text(str(turn.error or ""), force=True) if turn.error else None,
        "interrupted": _user_interrupted,
        "sdk_effects": _sdk_effects.as_result_dict(),
        **(
            {"failover_reason": _sdk_failover_reason.value}
            if _sdk_failover_reason is not None
            else {}
        ),
        # Same persistence contract as the codex app-server path: we flushed
        # the projected rows ourselves, so the gateway must not re-write the
        # user turn (append_message has no dedup).
        "agent_persisted": True,
        "claude_sdk_session_id": turn.thread_id,
        **usage_result,
    }
    # Fatal startup/auth/billing refusals surface the same machine-readable
    # fields the chat_completions path sets (conversation_loop), so the -Q
    # exit contract and gateway consumers see a real failure instead of a
    # recoverable partial. getattr: test doubles build TurnResult-shaped
    # namespaces that may predate the field.
    _fatal = getattr(turn, "fatal_reason", None)
    if _fatal:
        result["failed"] = True
        result["failure_reason"] = _fatal
    return result


__all__ = ["run_claude_agent_sdk_turn"]
