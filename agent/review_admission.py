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
  ``auxiliary.background_review.max_replay_tokens`` — or past what the fork's first provider
  request can carry under its aggregate input budget once the inherited system prompt, tools[]
  and the review prompt are counted — the fork keeps only the widest recent, user-led suffix
  that fits; if no complete suffix fits, the automatic review is skipped.

Reasons are short, stable slugs — never message bodies — so logs stay greppable and silo-safe.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

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
# A turn nested inside this turn's delivery window (/retry while the reply is on the wire)
# rewound the transcript; the fresher nested candidate replaces this turn's retracted one.
REASON_CANDIDATE_SUPERSEDED = "review_candidate_superseded"
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
# A user transcript rewrite (/undo, /retry, an edited prompt) asked the review's durable lease
# to yield: the rewrite invalidated its replay basis. An append (a detached delegation delivery)
# lands without asking.
REASON_PREEMPTED_BY_TRANSCRIPT_EDIT = "review_preempted_by_transcript_edit"
# A deferred review its durable lease stopped (a yield stamp, or a lost row) is dropped, never
# requeued: the transcript moved on under the captured snapshot.
REASON_DROPPED_AFTER_LEASE_YIELD = "review_dropped_after_lease_yield"
# The review's durable row outlived its renewals (dead process, starved refresher): seen by the
# reclaiming foreground, and by the fork's next renewal tick.
REASON_LEASE_EXPIRED_RECLAIMED = "review_lease_expired_reclaimed"
REASON_LEASE_LOST = "review_lease_lost"
# state.db stayed write-locked until no renewal could land before the review's row expired: the
# fork is stopped before a successor can reclaim the row beside it. Not a yield — the transcript
# did not move — so a deferred review is requeued, not dropped.
REASON_LEASE_RENEWAL_LOCKED = "review_lease_renewal_locked"
# The gateway's post-delivery completion raised before the captured candidate could spawn.
REASON_COMPLETION_ERROR = "review_completion_error"
# The fork's FIRST provider request was refused by its aggregate input budget: zero provider
# calls, no writes (the replay is bounded at spawn so this stays a fail-safe, not the norm).
REASON_INPUT_BUDGET_REFUSED = "review_input_budget_refused"
# The parent's provider client cannot carry Hermes tool calls back (an agent-as-provider shim
# declaring ``SUPPORTS_HERMES_TOOL_CALLS = False``) and the review is not routed elsewhere: the
# fork could write nothing, so it is never spawned.
REASON_PROVIDER_INCAPABLE = "provider_cannot_emit_tool_calls"
# The fixed parts of every request (system prompt, tools[], review prompt, margin) alone exceed
# what one request may carry — a request share of an explicit ``max_input_tokens``, or 75% of
# the window — so no replay fits: a configuration problem, not a big turn (warned, not INFO).
REASON_OVERHEAD_EXCEEDS_BUDGET = "review_overhead_exceeds_budget"

# Verbatim replay ceiling for one review fork. Well above an ordinary session (so normal learning
# keeps the warm-cache replay) and well below the ~205K incident.
MAX_REPLAY_TOKENS_DEFAULT = 120_000
# Slack kept under the aggregate budget for what the rough estimate at spawn cannot see in the
# fork's first request: the tool-whitelist notice appended to the review prompt, ephemeral
# per-request context, cache markers.
REQUEST_OVERHEAD_MARGIN_TOKENS = 2_048
# Provider requests the automatic replay is sized for under the aggregate input budget. The
# budget is charged the FULL prompt on every request — cache reads included — and the replay
# rides every one of them, so a replay sized for the first request alone leaves room for
# exactly one: the fork reads and can never write. Three shares leave a read (the review prompt
# enforces read-before-write), a write and a closing response; the tool results that grow the
# later requests come out of the per-request margin.
REVIEW_REQUEST_SHARES = 3
# The replay the DERIVED aggregate (75% of the window) always funds while one request on the
# window carries it: a few recent exchanges. On a small window the fixed parts alone (the default
# toolset, the review prompt, a gateway system prompt: ~18k tokens) exceed a third of the derived
# aggregate (16,384 at 65,536), which left no replay at all and skipped every automatic review
# there; the aggregate is raised to fund REVIEW_REQUEST_SHARES requests of the fixed parts plus
# this floor instead. Below it only when the window itself carries less. An explicit
# ``max_input_tokens`` is never raised.
REPLAY_FLOOR_TOKENS = 12_288

_lock = threading.RLock()
_live_turns: dict[tuple[str, str], set[int]] = {}
_review_runs: dict[tuple[str, str], Any] = {}
# token -> the key it was registered under, so a release is exact whatever profile the releasing
# thread is acting for by then.
_turn_keys: dict[int, set[tuple[str, str]]] = {}
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
    except Exception:  # an unresolvable home must not break the turn/review path
        logger.debug(
            "Could not resolve the profile key for review admission", exc_info=True
        )
        return ""


def _admission_key(session_id: Any, profile_key: Optional[str]) -> tuple[str, str]:
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
    except Exception:  # health: allow BLE001 -- unknown foreground state must block lower-priority work
        logger.warning("Automatic review blocked: %s", REASON_ADMISSION_FAILURE)
        return REASON_ADMISSION_FAILURE


def followup_pending(agent: Any) -> bool:
    """Has the host queued a follow-up that will become the next live turn for this session?

    ``followup_pending_callback`` is installed per turn by the gateway (the only host with a busy
    queue). A probe that raises reads pending: unknown foreground state cannot authorize
    lower-priority provider work.
    """
    return _followup_block_reason(agent) is not None


def live_turn_block_reason(
    session_id: Any,
    turn_token: Optional[int] = None,
    profile_key: Optional[str] = None,
) -> Optional[str]:
    """``REASON_LIVE_TURN`` while a turn other than ``turn_token`` is live on the session.

    Registry-only: it reads no host state, so a caller may hold :func:`admission_lock` across it.
    ``_BackgroundReviewRun.begin_request`` re-checks with it under that lock, right before it
    publishes a fork.
    """
    if other_live_turn(session_id, turn_token, profile_key):
        return REASON_LIVE_TURN
    return None


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

    Runs the host follow-up probe, which takes gateway state locks: never call this while
    holding :func:`admission_lock` (the gateway fence takes a state lock, then this registry).
    """
    reason = live_turn_block_reason(
        session_id
        if session_id is not None
        else getattr(agent, "session_id", None) or "",
        turn_token,
        profile_key,
    )
    return reason or _followup_block_reason(agent)


def request_overhead_tokens(agent: Any, review_prompt: Optional[str] = None) -> int:
    """Rough tokens the fork's first provider request carries BESIDES the replay.

    The same-model fork inherits the parent's system prompt and tools[] byte-identically
    (prompt-cache parity) and appends the review prompt as its user message; the loop projects
    that whole request against the aggregate input budget before request #1 and refuses it
    outright when it does not fit. ``REQUEST_OVERHEAD_MARGIN_TOKENS`` covers what is only known
    at request time. The estimators are the loop's own, so the two figures agree.
    """
    from agent.model_metadata import (
        _estimate_tools_tokens_rough,
        estimate_messages_tokens_rough,
    )

    fixed: list[dict[str, Any]] = []
    system_prompt = getattr(agent, "_cached_system_prompt", None)
    if isinstance(system_prompt, str) and system_prompt:
        fixed.append({"role": "system", "content": system_prompt})
    if isinstance(review_prompt, str) and review_prompt:
        fixed.append({"role": "user", "content": review_prompt})
    tools = getattr(agent, "tools", None)
    tools_tokens = _estimate_tools_tokens_rough(tools) if isinstance(tools, list) else 0
    return (
        estimate_messages_tokens_rough(fixed)
        + tools_tokens
        + REQUEST_OVERHEAD_MARGIN_TOKENS
    )


@dataclass(frozen=True)
class ReviewBudgets:
    """What one automatic review may send: ``replay`` tokens of verbatim history on every
    request (never below 1) and ``aggregate`` input tokens over the fork's whole loop.
    ``overhead`` is what every request carries besides the replay; ``unfunded`` says the fixed
    parts alone leave no replay room, so the spawn is skipped (``REASON_OVERHEAD_EXCEEDS_BUDGET``)."""

    replay: int
    aggregate: int
    overhead: int
    unfunded: bool


def _context_window(agent: Any) -> Optional[int]:
    """The agent's resolved context window, or None when unknown (the fallback aggregate then
    answers alone)."""
    window = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return None
    return window


def review_budgets(
    task_cfg: Optional[dict[str, Any]],
    agent: Any,
    review_prompt: Optional[str] = None,
) -> ReviewBudgets:
    """Size one automatic review for a read, a write and a closing request.

    Every provider request the fork makes carries the replay plus :func:`request_overhead_tokens`
    (system prompt, tools[], ``review_prompt``, margin) and is charged in full, so the replay is
    one of :data:`REVIEW_REQUEST_SHARES` equal shares of the aggregate input budget net of that
    overhead: a replay sized for the first request alone would leave no second request — a
    review that reads but never writes. It never exceeds the operator ceiling, nor what one
    request carries on the window (75% of it, the headroom the derived aggregate keeps).

    The derived aggregate (75% of the window) is raised — up to the 600k cap — to fund the three
    requests of overhead plus :data:`REPLAY_FLOOR_TOKENS` when its share cannot: on a small
    window the fixed parts alone are wider than a share, and the old share arithmetic left no
    replay and no review at all. An explicit ``max_input_tokens`` is the operator's cap and is
    never raised. When the fixed parts exceed a whole request nothing fits (``unfunded``). The
    fork resolves the same context window as its parent on the same-model path, so the parent
    answers for it at replay-bounding time; the fork re-derives the same aggregate when built.
    """
    from agent.background_review import (
        _REVIEW_INPUT_CONTEXT_FRACTION,
        _REVIEW_MAX_INPUT_TOKENS_CAP,
        _review_input_token_budget,
        explicit_review_input_budget,
    )

    overhead = request_overhead_tokens(agent, review_prompt)
    aggregate = _review_input_token_budget(task_cfg, agent)
    window = _context_window(agent)
    request_fit = (
        None
        if window is None
        else int(window * _REVIEW_INPUT_CONTEXT_FRACTION) - overhead
    )
    if explicit_review_input_budget(task_cfg) is None:
        floor = (
            REPLAY_FLOOR_TOKENS
            if request_fit is None
            else min(REPLAY_FLOOR_TOKENS, request_fit)
        )
        if floor > 0:
            aggregate = min(
                _REVIEW_MAX_INPUT_TOKENS_CAP,
                max(aggregate, REVIEW_REQUEST_SHARES * (overhead + floor)),
            )
    replay = min(
        _replay_ceiling(task_cfg), aggregate // REVIEW_REQUEST_SHARES - overhead
    )
    if request_fit is not None:
        replay = min(replay, request_fit)
    return ReviewBudgets(
        replay=max(1, replay),
        aggregate=aggregate,
        overhead=overhead,
        unfunded=replay < 1,
    )


def replay_token_budget(
    task_cfg: Optional[dict[str, Any]],
    agent: Any = None,
    review_prompt: Optional[str] = None,
) -> int:
    """Resolve the operator setting without permitting automatic replay to become unbounded:
    the ceiling alone without ``agent``, else :func:`review_budgets`'s replay for the spawning
    parent (never below 1)."""
    if agent is None:
        return _replay_ceiling(task_cfg)
    return review_budgets(task_cfg, agent, review_prompt).replay


def _replay_ceiling(task_cfg: Optional[dict[str, Any]]) -> int:
    """``max_replay_tokens`` clamped to the hard ceiling; the default on any invalid value."""
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
    snapshot: list[dict],
    budget: Optional[int],
) -> tuple[list[dict], Optional[str]]:
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
