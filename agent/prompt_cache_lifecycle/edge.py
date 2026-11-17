"""The cache edge: once a chat's keep-alives are over, compact it (step 2) or offload its old tool output
(step 3). ``PromptCacheWarmer.claim_edge`` asks these gates under its lock; owners never call them."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent.prompt_cache_lifecycle.warmer import PromptCacheWarmer

#: Shortest cache window a chat is compacted before: below it, ordinary pauses outlast the window.
MIN_COMPACT_WINDOW_SECONDS = 30 * 60
#: A chat is compacted only once its context is at least this share of the compaction threshold.
COMPACT_CONTEXT_FRACTION = 0.5
#: How often the surface owners poll ``PromptCacheWarmer.claim_edge``.
CHECK_INTERVAL_SECONDS = 30.0
#: Telemetry trigger name for cache-edge compactions.
TRIGGER = "cache_expiry"


@dataclass(frozen=True)
class EdgeDecision:
    """What the cache edge calls for now: ``action`` is ``"compact"`` (step 2), ``"offload"`` (step 3)
    or ``None``; ``compact`` / ``offload`` say why each step is or is not due (``"due"`` when it is)."""

    action: Optional[str]
    compact: str
    offload: str


@dataclass(frozen=True)
class _Spent:
    """A warm budget that ran out with the entry still warm: the moment step 2 decides."""

    due_at: float  # when the next keep-alive would have gone out
    tier_s: float
    witness: tuple


def human_paced(agent: Any) -> bool:
    """A chat a person types into (CLI, TUI, Desktop, messaging). Subagents, side agents (/bg),
    forks (/btw) and machine-paced sources (cron, one-shots, the API server, ...) are not."""
    from agent.prompt_caching import MACHINE_PACED_SOURCES
    from agent.session_source import session_source_for

    if int(getattr(agent, "_delegate_depth", 0) or 0) > 0 or getattr(agent, "side_agent", False) is True:
        return False
    if getattr(agent, "_skip_prompt_cache_warming", False):
        return False
    return session_source_for(getattr(agent, "platform", None)) not in MACHINE_PACED_SOURCES


def _context_tokens(agent: Any) -> int:
    tokens = getattr(getattr(agent, "context_compressor", None), "last_prompt_tokens", 0)
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else 0


def _compressor_blocked(agent: Any) -> Optional[str]:
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return "no_compressor"
    if getattr(compressor, "awaiting_real_usage_after_compression", False) is True:
        return "awaiting_usage"
    cooldown = getattr(compressor, "get_active_compression_failure_cooldown", None)
    if callable(cooldown) and cooldown():
        return "cooldown"
    return None


def _rough_tokens(agent: Any, messages: Optional[list] = None) -> int:
    from agent.conversation_compression_manual import estimate_request_tokens

    return estimate_request_tokens(agent, list(getattr(agent, "_session_messages", None) or ())
                                   if messages is None else messages)


def _compaction_baseline(agent: Any) -> int:
    """Rough size the last compaction left: the compressor's own figure or, on an agent rebuilt since
    (the gateway evicts after compacting), the transcript through its newest summary handoff plus
    the tail budget kept verbatim after it. 0 when the session never compacted."""
    compressor = agent.context_compressor
    last = getattr(compressor, "last_compression_rough_tokens", 0)
    if isinstance(last, int) and not isinstance(last, bool) and last > 0:
        return last
    messages = list(getattr(agent, "_session_messages", None) or ())
    is_summary = getattr(compressor, "_is_context_summary_message", None)
    handoff = next((i for i in range(len(messages) - 1, -1, -1) if callable(is_summary) and is_summary(messages[i])), None)
    if handoff is None:
        return 0
    kept_tail = getattr(compressor, "tail_token_budget", 0)
    return _rough_tokens(agent, messages[:handoff + 1]) + (kept_tail if isinstance(kept_tail, int) else 0)


def _grew_enough(agent: Any) -> bool:
    """Growth guard: a chat that stays large after compacting must gain a floor's worth of new
    content (the compressor's post-compaction target) before it is summarised again."""
    compressor = agent.context_compressor
    try:
        floor = int(compressor.threshold_tokens * compressor.summary_target_ratio)
    except (AttributeError, TypeError, ValueError):
        floor = 0
    baseline = _compaction_baseline(agent)
    return baseline <= 0 or _rough_tokens(agent) > baseline + floor


def _session_gate(agent: Any) -> Optional[str]:
    """Why *agent*'s chat is never compacted at the cache edge, or None (cheap checks only)."""
    if agent is None or not getattr(agent, "compression_enabled", False):
        return "disabled"
    if getattr(agent, "api_mode", None) == "codex_app_server":
        return "codex_app_server"
    if not human_paced(agent):
        return "machine_paced"
    threshold = getattr(getattr(agent, "context_compressor", None), "threshold_tokens", 0) or 0
    if _context_tokens(agent) < COMPACT_CONTEXT_FRACTION * threshold or threshold <= 0:
        return "small_context"
    return _compressor_blocked(agent)


def _window_gate(agent: Any, spent: _Spent, now: float) -> Optional[str]:
    """Why the spent budget's window does not compact now, or None."""
    from agent.prompt_cache_clock import DECLARED, PROVEN, cache_window, context_witness

    if spent.tier_s < MIN_COMPACT_WINDOW_SECONDS:
        return f"short_window:{int(spent.tier_s)}s"
    if now < spent.due_at:
        return "not_yet"
    window = cache_window(agent)
    if window is None or window.confidence not in (PROVEN, DECLARED):
        return "window_not_proven"
    # No warmth check: the summary is a fresh auxiliary request, not a replay of the cached prefix, so
    # a cold prefix costs it nothing extra (and the gateway's 300 s housekeeping lands past the 5/6 mark).
    if context_witness(agent) != spent.witness:
        return "context_changed"
    return None


def _compaction_reason(agent: Any, warmer: PromptCacheWarmer, releasing: bool, now: float) -> str:
    """Why step 2 does not run now (``"due"`` when it does). Caller holds ``warmer._lock``.

    Due only when this chat's warm budget is over (spent or missed; at a release, a live budget ends
    now) on a proven/declared window of at least 30 minutes, its context is at least half the
    compaction threshold and grew since the last compaction, and no compaction ran this stretch."""
    spent, run = warmer._spent, warmer._run
    if spent is None and releasing and run is not None:
        spent = _Spent(now, run.ttl_s, run.witness)
    if spent is None:
        return "compacted_this_stretch" if warmer._compacted else "budget_not_spent"
    if releasing:
        spent = replace(spent, due_at=min(spent.due_at, now))
    reason = _session_gate(agent) or _window_gate(agent, spent, now)
    if reason is None and not _grew_enough(agent):  # last: the estimate walks the transcript
        reason = "no_growth"
    return reason or "due"


def _offload_reason(agent: Any, warmer: PromptCacheWarmer, now: float) -> str:
    """Why step 3's offload does not run now (``"due"`` when it does): the chat is idle and its cache
    is over (past the clock's proven/declared end, or its route changed). Once per stretch; the cold
    prune applies its own gates (protected tail, >= 16k reclaim). Caller holds ``warmer._lock``."""
    from agent.prompt_cache_clock import prompt_cache_is_cold

    if warmer._offloaded:
        return "offloaded_this_stretch"
    if warmer._run is not None:
        return "still_warming"
    if not getattr(agent, "compression_enabled", False):
        return "disabled"
    return "due" if prompt_cache_is_cold(agent, now=now) else "not_cold"
