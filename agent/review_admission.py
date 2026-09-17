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
REASON_ADMISSION_FAILURE = "admission_probe_failed"
REASON_DURABLE_BUSY = "durable_foreground_active"
REASON_DURABLE_FAILURE = "durable_admission_failed"
REASON_REVIEW_SLOT_BUSY = "review_slot_busy"
# Gateway delivery outcomes: a captured candidate is dropped when the terminal delivery was
# not confirmed, or when the turn handed the session to a queued follow-up's drain task.
REASON_DELIVERY_UNCONFIRMED = "delivery_unconfirmed"
REASON_PENDING_HANDOFF = "pending_followup_handoff"
# Cancellation and idle-queue outcomes, so a review that vanished mid-flight is as greppable as
# one that was never admitted.
REASON_LIVE_TURN_CANCELLED = "live_turn_cancelled"
REASON_FOLLOWUP_CANCELLED = "queued_followup_cancelled"
REASON_DEFERRED = "managed_local_deferred"
REASON_PREEMPTED_REQUEUED = "preempted_requeued"
REASON_REQUEUE_CAP = "requeue_cap_exceeded"
REASON_STALE_OWNER = "stale_review_owner"
REASON_DISABLED_WHILE_QUEUED = "disabled_while_queued"
# Liveness escalation for a fenced fork that has not published its request exit.
REASON_CANCEL_UNACKNOWLEDGED = "review_cancel_unacknowledged"
REASON_REVIEW_REVOKED = "review_revoked"
# A foreground turn in another process asked the review's durable lease to yield.
REASON_PREEMPTED_CROSS_PROCESS = "review_preempted_cross_process"
# A user transcript edit (/undo, /retry, an edited prompt, a detached delivery) asked the
# review's durable lease to yield: the edit invalidated its replay basis.
REASON_PREEMPTED_BY_TRANSCRIPT_EDIT = "review_preempted_by_transcript_edit"
# The review's durable row outlived its renewals (dead process, starved refresher): seen by the
# reclaiming foreground, and by the fork's next renewal tick.
REASON_LEASE_EXPIRED_RECLAIMED = "review_lease_expired_reclaimed"
REASON_LEASE_LOST = "review_lease_lost"
# The gateway's post-delivery completion raised before the captured candidate could spawn.
REASON_COMPLETION_ERROR = "review_completion_error"

# Verbatim replay ceiling for one review fork. Well above an ordinary session (so normal learning
# keeps the warm-cache replay) and well below the ~205K incident.
MAX_REPLAY_TOKENS_DEFAULT = 120_000

_lock = threading.RLock()
_live_turns: Dict[Tuple[str, str], Set[int]] = {}
_review_runs: Dict[Tuple[str, str], Any] = {}
# token -> the key it was registered under, so a release is exact whatever profile the releasing
# thread is acting for by then.
_turn_keys: Dict[int, Set[Tuple[str, str]]] = {}
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


def publish_review_run(
    run: Any, session_id: str, profile_key: Optional[str] = None
) -> bool:
    """Publish one prepared review under its canonical owner, without replacing a live run."""
    key = _admission_key(session_id, profile_key)
    with _lock:
        current = _review_runs.get(key)
        if current is not None and not current.request_done.is_set():
            return False
        _review_runs[key] = run
        run._review_owner_key = key
    return True


def remove_review_run(run: Any) -> None:
    """Remove ``run`` only if it still owns its key (ABA-safe)."""
    key = getattr(run, "_review_owner_key", None)
    if key is None:
        return
    with _lock:
        if _review_runs.get(key) is run:
            _review_runs.pop(key, None)


def current_review_run(session_id: str, profile_key: Optional[str] = None) -> Any:
    """Return the current canonical review owner without creating registry state."""
    key = _admission_key(session_id, profile_key)
    with _lock:
        return _review_runs.get(key)


def note_turn_started(session_id: str, profile_key: Optional[str] = None) -> int:
    """Register a live turn on ``session_id``; the token identifies it to the review gate."""
    key = _admission_key(session_id, profile_key)
    with _lock:
        token = next(_tokens)
        _live_turns.setdefault(key, set()).add(token)
        _turn_keys[token] = {key}
    return token


def alias_turn_session(
    token: int, session_id: str, profile_key: Optional[str] = None
) -> bool:
    """Add a session alias to one live token without opening a stale-owner gap."""
    key = _admission_key(session_id, profile_key)
    with _lock:
        keys = _turn_keys.get(token)
        if keys is None:
            return False
        keys.add(key)
        _live_turns.setdefault(key, set()).add(token)
        return True


def note_turn_finished(
    session_id: str, token: int, profile_key: Optional[str] = None
) -> None:
    """Balance :func:`note_turn_started`. Idempotent — a double release is not an error.

    The key recorded at registration wins over the arguments, so a turn that changed profile
    context mid-flight still releases the bucket it actually occupies.
    """
    fallback = _admission_key(session_id, profile_key)
    with _lock:
        keys = _turn_keys.pop(token, None) or {fallback}
        for key in keys:
            live = _live_turns.get(key)
            if live is None:
                continue
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


def _followup_block_reason(agent: Any) -> Optional[str]:
    """Return a body-free follow-up reason, failing safe when the host probe is unhealthy."""
    probe = getattr(agent, "followup_pending_callback", None)
    if not callable(probe):
        return None
    try:
        return REASON_QUEUED_FOLLOWUP if probe() else None
    except Exception:  # noqa: BLE001 — unknown foreground state must block lower-priority work
        logger.warning("Automatic review blocked: %s", REASON_ADMISSION_FAILURE)
        return REASON_ADMISSION_FAILURE


def followup_pending(agent: Any) -> bool:
    """Has the host queued a follow-up that will become the next live turn for this session?

    ``followup_pending_callback`` is installed per turn by the gateway (the only host with a busy
    queue). A probe that raises reads pending: unknown foreground state cannot authorize
    lower-priority provider work.
    """
    return _followup_block_reason(agent) is not None


def foreground_block_reason(
    agent: Any,
    turn_token: Optional[int] = None,
    profile_key: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[str]:
    """Reason the foreground owns this session right now, or None when a review may run.

    ``profile_key`` is the profile the review belongs to (its spawning turn's). Pass it whenever
    the check may run off the turn's thread — the idle-queue dispatcher and the review fork are
    outside the gateway's per-turn profile scope, so "current" is not their answer.
    """
    if other_live_turn(
        session_id
        if session_id is not None
        else getattr(agent, "session_id", None) or "",
        turn_token,
        profile_key,
    ):
        return REASON_LIVE_TURN
    return _followup_block_reason(agent)


def replay_token_budget(task_cfg: Optional[Dict[str, Any]]) -> int:
    """Resolve the operator setting without permitting automatic replay to become unbounded."""
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
    if budget <= 0:
        return MAX_REPLAY_TOKENS_DEFAULT
    return min(budget, MAX_REPLAY_TOKENS_DEFAULT)


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
    """Deterministic, non-disclosing session-only label. Review log lines carry
    :func:`owner_tag` instead, so one hash covers the profile as well as the session."""
    return hashlib.sha256(str(session_id).encode()).hexdigest()[:8]


def owner_tag(profile_key: Any, session_id: Any) -> str:
    """Hash the complete canonical review owner without disclosing either component."""
    identity = f"{profile_key}\0{session_id}"
    return hashlib.sha256(identity.encode()).hexdigest()[:12]
