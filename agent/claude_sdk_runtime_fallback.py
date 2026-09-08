"""Replay-safe provider hand-off for the claude-agent-sdk runtime.

``ClaudeSdkTurnEffects`` (the observable-effects ledger that decides whether a
failed turn may be replayed on another provider) and the failover-reason
classifier. Extracted from ``claude_sdk_runtime.py``; the consumer half of the
contract stays in ``agent.turn_runtime_handoff``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from agent.claude_sdk_runtime_compaction import _emit_dangling_compaction_done
from agent.claude_sdk_runtime_continuity import _store_sdk_session_id
from agent.claude_sdk_runtime_state import _SdkTurnState

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


@dataclass(frozen=True)
class ClaudeSdkTurnEffects:
    """Replay-safety ledger for one SDK provider attempt."""

    tool: bool = False
    streamed: bool = False
    projected: bool = False
    interrupted: bool = False
    mutated: bool = False

    @property
    def replay_safe(self) -> bool:
        return not any(asdict(self).values())

    def as_result_dict(self) -> Dict[str, bool]:
        return asdict(self)


_SDK_PROVIDER_FAILOVER_REASONS = frozenset(
    {
        "auth",
        "auth_permanent",
        "rate_limit",
        "upstream_rate_limit",
        "overloaded",
        "server_error",
        "timeout",
    }
)


def _sdk_provider_failover_reason(agent, error: str, fatal_reason: Optional[str]):
    """Return one canonical provider handoff reason, or ``None``.

    SDK failures are serialized into result text, so recover a status code when
    present and delegate classification to the shared API error taxonomy.  The
    explicit allowlist is intentionally narrower than ``should_fallback``:
    local/configuration errors, request-shape failures, policy refusals, and
    billing-safety guards must remain terminal on this runtime.
    """
    from agent.error_classifier import FailoverReason, classify_api_error

    if fatal_reason:
        try:
            reason = FailoverReason(str(fatal_reason))
        except ValueError:
            return None
        return reason if reason.value in _SDK_PROVIDER_FAILOVER_REASONS else None

    text = str(error or "").strip()
    if not text:
        return None

    class _SerializedSdkProviderError(RuntimeError):
        status_code: Optional[int] = None

    exc = _SerializedSdkProviderError(text)
    match = re.search(
        r"(?:api\s+error|status(?:_code)?|http)\s*[:=]?\s*(\d{3})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        exc.status_code = int(match.group(1))
    classified = classify_api_error(
        exc,
        provider=str(getattr(agent, "provider", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
    )
    return (
        classified.reason
        if classified.reason.value in _SDK_PROVIDER_FAILOVER_REASONS
        else None
    )


def _reconcile_turn_outcome(agent, state: _SdkTurnState) -> None:
    """Settle the turn's terminal/stop state, then decide provider fallback.

    Order is load-bearing: the effects ledger is read AFTER the attempt loop
    retired or resumed the session and BEFORE the interrupt flag is consumed,
    so a stopped turn is never replay-safe; the dangling-compaction fallback
    and the interrupt hand-off follow.
    """
    turn = state.turn
    state.effects = ClaudeSdkTurnEffects(
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
        mutated=state.messages != state.messages_before_attempt,
    )
    state.failover_reason = None
    if getattr(turn, "error", None) and state.effects.replay_safe:
        state.failover_reason = _sdk_provider_failover_reason(
            agent,
            str(turn.error),
            getattr(turn, "fatal_reason", None),
        )
        if state.failover_reason is not None and agent._claude_sdk_session is not None:
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
    state.user_interrupted = bool(
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
