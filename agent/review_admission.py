"""Admission + replay bounds for the post-turn background review (``agent/background_review.py``).

Two independent cost leaks are closed here, both observed on the same turn: a 65-call turn
finished, the automatic review forked with the WHOLE snapshot (~205K tokens), and a follow-up
that had been queued during that turn started ~100ms later — two >200K contexts decoding at once
for one session.

* **Foreground exclusion.** ``run_conversation`` registers its turn on the session here BEFORE it
  cancels an in-flight review, so a review that has not yet issued its first request refuses
  admission instead of racing the cancel with a full transcript already on the wire. A follow-up
  the gateway has queued but not yet started counts too (``agent.followup_pending_callback``): it
  IS the next live turn. Self-improvement never competes with a user-facing turn — the review is
  skipped, and the next eligible review receives the freshest bounded snapshot. The registry is
  keyed by **profile AND session**: a multiplexed gateway derives session IDs per platform
  conversation, so two independent profiles can hold the same ID and must not suppress each other.
* **Replay bound.** The un-routed fork replays the snapshot verbatim for warm-cache parity, which
  is right for an ordinary session and ruinous for a long one. Past
  ``auxiliary.background_review.max_replay_tokens`` the fork keeps only the widest recent,
  user-led suffix that fits; if no complete suffix fits, the automatic review is skipped.

Reasons are short, stable slugs — never message bodies — so logs stay greppable and silo-safe.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Deterministic, body-free skip/bound reasons (log + test surface).
REASON_LIVE_TURN = "live_turn_active"
REASON_QUEUED_FOLLOWUP = "queued_followup_pending"
REASON_OVERSIZED = "oversized_snapshot"

# Verbatim replay ceiling for one review fork. Well above an ordinary session (so normal learning
# keeps the warm-cache replay) and well below the ~205K incident.
MAX_REPLAY_TOKENS_DEFAULT = 120_000

_lock = threading.RLock()
_live_turns: Dict[Tuple[str, str], Set[int]] = {}
# token -> the key it was registered under, so a release is exact whatever profile the releasing
# thread is acting for by then.
_turn_keys: Dict[int, Tuple[str, str]] = {}
_tokens = itertools.count(1)


def admission_lock() -> Any:
    """Process-wide foreground admission lock shared with automatic review request startup."""
    return _lock


def current_profile_key() -> str:
    """Canonical identity of the profile/home the calling thread is acting for.

    ``hermes_home_key`` is the same resolved+normcased key the tool registry uses, so a profile
    reached through a symlink or a trailing slash lands in one bucket. An unresolvable home reads
    as the empty key rather than raising: admission must never break a turn.
    """
    try:
        from hermes_constants import hermes_home_key

        return hermes_home_key()
    except Exception:  # noqa: BLE001 — an unresolvable home must not break the turn/review path
        logger.debug(
            "Could not resolve the profile key for review admission", exc_info=True
        )
        return ""


def _admission_key(session_id: Any, profile_key: Optional[str]) -> Tuple[str, str]:
    """``(profile, session)`` bucket; ``profile_key`` pins a caller-supplied profile (a deferred
    dispatch runs on the queue thread, which is not inside the turn's profile scope)."""
    profile = current_profile_key() if profile_key is None else str(profile_key)
    return profile, str(session_id or "")


def note_turn_started(session_id: str, profile_key: Optional[str] = None) -> int:
    """Register a live turn on ``session_id``; the token identifies it to the review gate."""
    key = _admission_key(session_id, profile_key)
    with _lock:
        token = next(_tokens)
        _live_turns.setdefault(key, set()).add(token)
        _turn_keys[token] = key
    return token


def note_turn_finished(
    session_id: str, token: int, profile_key: Optional[str] = None
) -> None:
    """Balance :func:`note_turn_started`. Idempotent — a double release is not an error.

    The key recorded at registration wins over the arguments, so a turn that changed profile
    context mid-flight still releases the bucket it actually occupies.
    """
    fallback = _admission_key(session_id, profile_key)
    with _lock:
        key = _turn_keys.pop(token, None) or fallback
        live = _live_turns.get(key)
        if live is None:
            return
        live.discard(token)
        if not live:
            _live_turns.pop(key, None)


def other_live_turn(
    session_id: str, token: Optional[int], profile_key: Optional[str] = None
) -> bool:
    """Is a turn OTHER than ``token`` live on this profile's session? ``token`` is the turn that
    spawned the review (still live inside its own finalizer), so it can never block its own
    review."""
    key = _admission_key(
        session_id, profile_key
    )  # resolved off the lock (may touch the FS)
    with _lock:
        live = _live_turns.get(key)
        if not live:
            return False
        return bool(live - {token}) if token is not None else True


def followup_pending(agent: Any) -> bool:
    """Has the host queued a follow-up that will become the next live turn for this session?

    ``followup_pending_callback`` is installed per turn by the gateway (the only host with a busy
    queue). A probe that raises reads False: an unknown foreground state must not starve learning
    forever, and the turn-token gate still covers the follow-up once it actually starts.
    """
    probe = getattr(agent, "followup_pending_callback", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001 — a broken host probe must not break the review path
        logger.debug(
            "followup_pending_callback raised; treating the session as free",
            exc_info=True,
        )
        return False


def foreground_block_reason(
    agent: Any, turn_token: Optional[int] = None, profile_key: Optional[str] = None
) -> Optional[str]:
    """Reason the foreground owns this session right now, or None when a review may run.

    ``profile_key`` is the profile the review belongs to (its spawning turn's). Pass it whenever
    the check may run off the turn's thread — the idle-queue dispatcher and the review fork are
    outside the gateway's per-turn profile scope, so "current" is not their answer.
    """
    if other_live_turn(
        getattr(agent, "session_id", None) or "", turn_token, profile_key
    ):
        return REASON_LIVE_TURN
    if followup_pending(agent):
        return REASON_QUEUED_FOLLOWUP
    return None


def replay_token_budget(task_cfg: Optional[Dict[str, Any]]) -> Optional[int]:
    """``auxiliary.background_review.max_replay_tokens`` (``None`` means unlimited)."""
    config = task_cfg or {}
    if "max_replay_tokens" not in config:
        return MAX_REPLAY_TOKENS_DEFAULT

    raw = config.get("max_replay_tokens")
    if raw is None or isinstance(raw, bool):
        logger.warning(
            "Invalid auxiliary.background_review.max_replay_tokens=%r; using default %d",
            raw,
            MAX_REPLAY_TOKENS_DEFAULT,
        )
        return MAX_REPLAY_TOKENS_DEFAULT
    try:
        budget = int(raw)
    except (OverflowError, TypeError, ValueError):
        logger.warning(
            "Invalid auxiliary.background_review.max_replay_tokens=%r; using default %d",
            raw,
            MAX_REPLAY_TOKENS_DEFAULT,
        )
        return MAX_REPLAY_TOKENS_DEFAULT
    return budget if budget > 0 else None


def _is_complete_user_anchor(message: Any) -> bool:
    """Whether ``message`` may safely lead a replay suffix.

    Anthropic represents tool results as user-role content blocks. Such a carrier belongs to the
    preceding assistant tool use and cannot lead a standalone suffix.
    """
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    return not (
        isinstance(content, list)
        and any(
            isinstance(block, dict)
            and (block.get("type") == "tool_result" or "tool_use_id" in block)
            for block in content
        )
    )


def bounded_replay_history(
    snapshot: List[Dict],
    budget: Optional[int],
) -> Tuple[List[Dict], Optional[str]]:
    """Return the verbatim snapshot when it fits, otherwise its widest safe recent suffix.

    Older turns were eligible for earlier post-turn reviews; the current review needs the newest
    user-led run. Dropping a prefix avoids synthesizing a new message (and therefore preserves the
    stored role sequence). A suffix must begin on a user message so it cannot orphan tool results or
    open on an assistant response. If even the newest user turn exceeds the budget, return no replay;
    the caller skips the automatic review rather than issuing an oversized request.
    """
    from agent.model_metadata import estimate_messages_tokens_rough

    messages = list(snapshot or [])
    if budget is None or budget <= 0:
        return messages, None

    message_costs = [estimate_messages_tokens_rough([message]) for message in messages]
    if sum(message_costs) <= budget:
        return messages, None

    widest_start = None
    suffix_cost = 0
    for start in range(len(messages) - 1, -1, -1):
        suffix_cost += message_costs[start]
        if suffix_cost > budget:
            break
        if _is_complete_user_anchor(messages[start]):
            widest_start = start
    if widest_start is not None:
        return messages[widest_start:], REASON_OVERSIZED
    return [], REASON_OVERSIZED


def session_tag(session_id: Any) -> str:
    """Return a deterministic, non-disclosing session label for logs."""
    return hashlib.sha256(str(session_id).encode()).hexdigest()[:8]
