"""Durable delivery-obligation ledger for gateway final responses (rows in the launch home's store,
written through its coordination repository; owner pid + process-start liveness, bounded retention)
so a crash between finalize and platform ACK cannot lose a response silently. Checkpoints: record_obligation() 'pending' before
any send | mark_attempting() 'attempting' right before the await | mark_delivered() 'delivered'
only on SendResult.success | mark_failed() 'failed' on a definitive rejection. Crash semantics
(never silently resend an ambiguous send): pending = never started, redeliver plainly; attempting
= crashed mid-await, platform MAY have it, redeliver WITH a visible recovered marker; failed =
rejected once, restart is a retry boundary, also marked; delivered = prune. Attempts are capped
and stale rows expire, both -> 'abandoned' (kept briefly, then pruned). Everything is
best-effort: ledger failures must never block a send; callers wrap every call in try/except.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

from gateway.dead_targets import classify_dead_error
from hermes_constants import get_process_hermes_home
from persistence.authority.coordination import ClaimedObligation, OwnerStamp, RetryDue

logger = logging.getLogger(__name__)
_DB_LOCK = threading.Lock()

# Redelivery policy knobs (deliberately not config — the ledger is gated by
# ``gateway.delivery_ledger`` and these only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60

# Visible prefixes for redeliveries that might duplicate an already-received message (crash mid-send /
# post-rejection retry) — honest at-least-once. Runtime recovery uses a distinct marker: no restart
# occurred, but a network rejection's acknowledgement can still have been lost independently.
RECOVERED_MARKER = "♻️ Recovered reply — the gateway restarted during delivery, so this may be a duplicate:\n\n"
RECONNECTED_MARKER = ("♻️ Recovered reply — the messaging platform reconnected after the original "
                      "delivery failed, so this may be a duplicate:\n\n")
# A reply refused by flood control may have gone out as several requests (the adapter chunks long replies,
# and MarkdownV2 escaping alone can push a reply that fits one message into two), and the platform may have
# accepted the first chunk(s) before refusing the rest; the send result does not say which landed. The raw
# length of the stored text says nothing about that, so every flood redelivery carries a marker, and neither
# of the markers above tells the truth here (no restart, no reconnect): the rate limit gets its own.
FLOOD_MARKER = ("♻️ Recovered reply — the messaging platform's rate limit refused the original, so part of "
                "it may already have arrived above:\n\n")

# Errors whose send contract proves the platform never saw the request: retried as soon as the adapter
# is back, no backoff. Every other rejection is retried too (#91653: a 5xx or a transient parse error
# used to strand the reply in ``failed`` until the next restart), but only after a backoff that grows
# with the attempts already spent, so a platform-side outage is not hammered by the redelivery timer.
# A whole-chat death (blocked bot, deleted group, deactivated user) is never retried: the target is gone.
_RUNTIME_RETRYABLE_ERRORS = frozenset({"send_path_degraded"})
# One tier per in-process retry; the last budgeted attempt is left to the boot sweep (retry_not_before).
_RETRY_BACKOFF_SECONDS = (30.0, 120.0)
assert len(_RETRY_BACKOFF_SECONDS) == MAX_ATTEMPTS - 1

# A final send the platform refused with flood control is the other transient case: a 429 means the
# refused request was never accepted, and the platform said how long to wait. Adapters fail such sends
# closed as ``flood_control:<seconds>`` on purpose (#91969) so that this ledger owns the wait instead of
# the send coroutine sleeping through it. Before this the row simply sat in ``failed`` until the next
# restart's sweep, which then redelivered it hours late under the "gateway restarted during delivery"
# marker.
FLOOD_ERROR_PREFIX = "flood_control:"
FLOOD_RETRY_DEFAULT_SECONDS = 60.0
FLOOD_RETRY_CAP_SECONDS = 15 * 60.0
FLOOD_RETRY_SLACK_SECONDS = 2.0

# The canonical prefix above is what the adapters produce for a flood they decide not to sleep. It is
# not the only shape that reaches this ledger. PTB raises ``RetryAfter``, whose own text reads
# "Flood control exceeded. Retry in 185 seconds", and that text is what lands in ``last_error``
# whenever a send fails on a path that has not been normalized, or was written by an older build
# before it was. Such a row has to be recognised too: unrecognised, it is treated as an ordinary
# failure, so no redelivery timer is armed for it and a boot sweep claims it immediately instead of
# adopting it until its deadline, spending the one attempt inside the penalty that caused it.
# Matching requires the "flood control" wording as well as the delay, so an unrelated error that
# merely suggests retrying is never mistaken for a flood.
_RAW_FLOOD_RE = re.compile(r"flood control exceeded.*?retry in\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def _raw_flood_wait(text: str) -> Optional[float]:
    """Seconds asked for by a flood error still carrying the platform's own wording, else ``None``."""
    match = _RAW_FLOOD_RE.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def is_reconnect_only(error: Any) -> bool:
    """True for a row that only an adapter reconnect may retry (no timer, no backoff)."""
    return str(error or "").strip().lower() in _RUNTIME_RETRYABLE_ERRORS


def is_flood_error(error: Any) -> bool:
    """True for a flood refusal: the adapters' fail-closed ``flood_control:<seconds>`` result, or a
    row still carrying the platform's own flood wording (see ``_RAW_FLOOD_RE``)."""
    text = str(error or "").strip().lower()
    return text.startswith(FLOOD_ERROR_PREFIX) or _raw_flood_wait(text) is not None


def flood_wait_seconds(error: Any, default: float = FLOOD_RETRY_DEFAULT_SECONDS) -> float:
    """The wait the platform asked for, read from ``flood_control:<seconds>``; ``default`` when unreadable."""
    text = str(error or "").strip().lower()
    wait = default
    if text.startswith(FLOOD_ERROR_PREFIX):
        try:
            wait = float(text[len(FLOOD_ERROR_PREFIX):].strip())
        except ValueError:
            wait = default
    else:
        # A row still carrying the platform's own flood wording states its delay just as precisely,
        # and the deadline must use it rather than the generic default.
        raw = _raw_flood_wait(text)
        if raw is not None:
            wait = raw
    return wait if wait > 0 else default


def flood_retry_delay(seconds: Any) -> float:
    """How long a redelivery timer sleeps for a wait of ``seconds``: capped so a huge or bogus value cannot
    park the timer for hours (the row's own deadline, not the timer, decides eligibility when it fires),
    plus a little slack so the timer lands after the deadline rather than on it."""
    try:
        wait = float(seconds)
    except (TypeError, ValueError):
        wait = FLOOD_RETRY_DEFAULT_SECONDS
    return min(max(wait, 0.0), FLOOD_RETRY_CAP_SECONDS) + FLOOD_RETRY_SLACK_SECONDS


def _failed_stamp(updated_at: Any) -> float:
    try:
        return float(updated_at or 0.0)
    except (TypeError, ValueError):
        return 0.0


def flood_not_before(updated_at: Any, last_error: Any) -> float:
    """Earliest moment a flood-refused row may be resent: the refusal's timestamp (``mark_failed`` sets
    ``updated_at``) plus the platform's wait. Enforced by the sweeps so neither an early timer nor a
    reconnect sweep spends a redelivery attempt inside the penalty window."""
    return _failed_stamp(updated_at) + flood_wait_seconds(last_error)


def retry_not_before(updated_at: Any, last_error: Any, attempts: Any) -> Optional[float]:
    """Earliest moment a failed row may be resent, or ``None`` for a row the runtime must leave alone:
    a flood refusal keeps the platform's own wait, an allowlisted reconnect error is due at once, a
    whole-chat death is final, and any other rejection backs off by the attempts already spent — but
    never spends the LAST budgeted attempt. An unclassified outage can outlast any timer, and a row
    the timer abandoned would be lost for good; leaving one attempt keeps it recoverable by the boot
    sweep after a restart, which is a real recovery signal."""
    if is_flood_error(last_error):
        return flood_not_before(updated_at, last_error)
    text = str(last_error or "").strip().lower()
    if is_reconnect_only(text):
        return _failed_stamp(updated_at)
    if classify_dead_error(text):
        return None
    spent = int(attempts or 0)
    if spent >= MAX_ATTEMPTS - 1:
        return None
    return _failed_stamp(updated_at) + _RETRY_BACKOFF_SECONDS[spent]


def _scope():
    # Launch home, not get_hermes_home(): a multiplexed gateway records a served profile's replies
    # under that profile's home override, but the boot sweep reads from the launch context, so both
    # must open the one shared store (adapter_profile tells the bots apart). No get_hermes_home()
    # fallback for an unset HERMES_HOME: a default gateway run in the foreground has none.
    from persistence_assembly import profile_scope
    return profile_scope(home=get_process_hermes_home())


def _store():
    """A handle on the ledger's store through the launch home's registered writer; the caller closes it."""
    from persistence_assembly import open_persistence
    return open_persistence(_scope())


def _start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time  # lazy: tests monkeypatch gateway.status
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    return pid, _start_time(pid)


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    current_start = _start_time(pid)
    if current_start is None:
        # Start time unreadable: alive iff the pid exists. Route through the cross-platform probe — on Windows
        # ``os.kill(pid, 0)`` is NOT a no-op (bpo-14484: it maps to ``GenerateConsoleCtrlEvent(0, pid)`` and could
        # Ctrl+C the gateway's own console group). ``_pid_exists`` keeps EPERM-means-alive (pid owned by another user).
        try:
            from gateway.status import _pid_exists
        except Exception:
            if os.name == "nt":
                return False  # never fall back to a raw sig-0 probe on Windows
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback branch
                return True
            except OSError as exc:  # incl. ProcessLookupError; EPERM means the pid exists
                return isinstance(exc, PermissionError)
        try:
            return bool(_pid_exists(pid))
        except Exception:
            return False
    try:
        from gateway.status import start_time_fingerprints_match
        return started_at is None or start_time_fingerprints_match(started_at, current_start)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while distinct threads/topics on one
    chat never collide (session_key carries platform/chat/thread; ``message_ref`` = inbound message id)."""
    return hashlib.sha256(f"{session_key}|{message_ref}|{content}".encode("utf-8", "replace")).hexdigest()[:24]


def record_obligation(*, obligation_id: str, session_key: str, platform: str, chat_id: str,
                      thread_id: Optional[str], content: str, adapter_profile: Optional[str] = None) -> None:
    """Record a final response as owed to the platform (state='pending'); the store prunes old rows
    in the same write."""
    pid, started = _owner_stamp()
    with _DB_LOCK, _store() as store:
        store.coordination.record_delivery_obligation(
            obligation_id=obligation_id, session_key=session_key, platform=platform, chat_id=chat_id,
            thread_id=thread_id, content=content, adapter_profile=adapter_profile,
            owner=OwnerStamp(pid=pid, started_at=started))


def record_crash_left_reply(*, obligation_id: str, session_key: str, platform: str, chat_id: str,
                            thread_id: Optional[str], content: str, since: float,
                            adapter_profile: Optional[str] = None) -> None:
    """Adopt a reply a killed process persisted but never ledgered. Unowned, so this boot's sweep
    claims it, and 'attempting', because a streamed reply may already be on screen: it is
    redelivered once, with the recovered marker. A no-op when the same reply was already ledgered
    since *since* (the turn start), and idempotent across boots that die before their sweep."""
    with _DB_LOCK, _store() as store:
        store.coordination.record_crash_left_reply(
            obligation_id=obligation_id, session_key=session_key, platform=platform, chat_id=chat_id,
            thread_id=thread_id, content=content, since=since, adapter_profile=adapter_profile)


def mark_attempting(obligation_id: str) -> None:
    with _DB_LOCK, _store() as store:
        store.coordination.mark_obligation_attempting(obligation_id=obligation_id)


def mark_delivered(obligation_id: str) -> None:
    with _DB_LOCK, _store() as store:
        store.coordination.mark_obligation_delivered(obligation_id=obligation_id)


def mark_failed(obligation_id: str, error: str = "") -> None:
    with _DB_LOCK, _store() as store:
        store.coordination.mark_obligation_failed(obligation_id=obligation_id, error=error)


def release_runtime_claim(obligation_id: str, error: str = "") -> bool:
    """Return an unsent runtime claim to ``failed`` without spending an attempt.

    Runtime recovery claims before clearing ``resume_pending`` so two reconnect paths cannot send the
    same row; if the flag cannot be cleared no send was attempted and the claim must not consume the
    redelivery budget. Fail-closed to the exact current process instance and ``attempting`` state."""
    pid, started = _owner_stamp()
    if started is None:
        return False
    with _DB_LOCK, _store() as store:
        return store.coordination.release_runtime_obligation_claim(
            obligation_id=obligation_id, error=error, owner=OwnerStamp(pid=pid, started_at=started))


def _claimed_dict(row: ClaimedObligation) -> dict[str, Any]:
    """Claimed-row dict handed back for redelivery. A marked row names its own cause: a flood row gets
    FLOOD_MARKER, a runtime reconnect replay RECONNECTED_MARKER, and a boot-recovered crash keeps the
    runner's restart marker default (no ``marker`` key). ``last_error`` is the row's pre-claim error,
    carried so a runtime claim that is released unsent goes back to ``failed`` with the same error and
    keeps its retry eligibility. An adopted flood row carries ``adopted`` and its ``not_before``."""
    head = {"obligation_id": row.obligation_id, "session_key": row.session_key, "platform": row.platform,
            "chat_id": row.chat_id, "thread_id": row.thread_id, "content": row.content}
    if row.adopted:
        return {**head, "profile": row.profile, "attempts": row.attempts, "adopted": True,
                "not_before": row.not_before}
    return {**head, "needs_marker": row.needs_marker, **({"marker": row.marker} if row.marker else {}),
            "profile": row.profile, **({"runtime_recovery": True} if row.runtime_recovery else {}),
            **({"last_error": row.last_error} if row.last_error else {}), "attempts": row.attempts}


def sweep_recoverable(now: Optional[float] = None, *, deliverable_platforms: Optional[set] = None,
                      deliverable_targets: Optional[set] = None) -> list[dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for redelivery.

    Claiming atomically re-stamps the owner to THIS process, moves the row to 'attempting' and increments
    ``attempts`` (the UPDATE is guarded on the previous owner stamp, so a second gateway racing the same
    sweep cannot double-claim).
    Rows over the attempts cap or stale cutoff become 'abandoned'. ``deliverable_platforms`` restricts
    claiming to platforms the caller can send on this boot: ``attempts`` is the redelivery budget and
    must only be spent on a real send, else a platform that failed to connect burns one attempt per boot
    and hits the cap having never been sent once (the stale cutoff still bounds untouched rows).
    ``deliverable_targets`` further scopes multiplexed gateways by exact ``(platform, adapter_profile)``
    so one connected bot cannot spend another disconnected bot's retry budget.

    A flood-refused row still inside its wait is adopted (owner re-stamped, no attempt spent) and
    returned flagged ``adopted`` with its ``not_before``: the caller clears its session's resume flag
    like any other claimed row, since the answer is in the ledger, but must not send it; the flood
    timer does once the wait has passed. A legacy row without ``adapter_profile`` is normalised to
    ``'default'`` on claim or adoption (the caller only accepts such rows when it is not multiplexed),
    because the runtime sweep matches profiles exactly and could otherwise never claim it."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    with _DB_LOCK, _store() as store:
        # Owners are probed between the owner read and the sweep's write; an owner left out counts as live.
        dead = frozenset(owner for owner in store.coordination.list_obligation_owners()
                         if not _owner_alive(owner.pid, owner.started_at))
        claimed = store.coordination.sweep_recoverable_obligations(
            now=now,
            deliverable_platforms=None if deliverable_platforms is None else frozenset(deliverable_platforms),
            deliverable_targets=None if deliverable_targets is None else frozenset(deliverable_targets),
            owner=OwnerStamp(pid=pid, started_at=started), dead_owners=dead)
    return [_claimed_dict(row) for row in claimed]


def sweep_failed_for_runtime(platform: str, now: Optional[float] = None, *,
                             profile: Optional[str] = None) -> list[dict[str, Any]]:
    """Claim this process's failed rows that are due for another send, for one adapter.

    ``profile`` scopes multiplexed gateways to the bot identity that owned the failed send (``None`` =
    primary/default adapter); unowned rows and rows owned by another process are left for the
    startup/dead-owner sweep. Startup recovery ignores rows owned by a live gateway, so a response
    rejected with ``send_path_degraded`` would stay stranded when only the adapter reconnects; this closes
    that gap without weakening ownership: only rows stamped to this exact process instance, only rows
    past their ``retry_not_before`` deadline, same attempts/staleness bounds, every update guarded by the
    prior owner stamp and ``failed`` state. Claimed rows always carry a marker (the failed send's ack is not safe to
    infer): the reconnect one, or the rate-limit one for a flood-refused row."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    if started is None:  # PID alone cannot distinguish this process from a stale row left after PID
        return []        # reuse; runtime replay is optional, so fail closed (startup recovery remains).
    owner = OwnerStamp(pid=pid, started_at=started)
    with _DB_LOCK, _store() as store:
        # Deadlines are computed between the candidate read and the claim, which takes a row only while
        # it is as the read saw it; a row with no deadline is left alone.
        due = [RetryDue(obligation_id=row.obligation_id, updated_at=row.updated_at, last_error=row.last_error,
                        attempts=row.attempts, not_before=not_before)
               for row in store.coordination.list_runtime_retry_candidates(platform, profile=profile, owner=owner)
               if (not_before := retry_not_before(row.updated_at, row.last_error, row.attempts)) is not None]
        claimed = store.coordination.sweep_failed_obligations_for_runtime(
            platform, now=now, profile=profile, owner=owner, due=due)
    return [_claimed_dict(row) for row in claimed]


def pending_retries(now: Optional[float] = None) -> list[dict[str, Any]]:
    """This process's failed rows that still await redelivery, one entry per adapter identity with the
    earliest deadline (``not_before``). The runner arms one redelivery timer per entry, so a row adopted
    at boot, skipped because its wait had not passed, or rejected again is never stranded. Rows past the
    attempts cap or stale cutoff are left for the sweeps to abandon."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    if started is None:
        return []
    with _DB_LOCK, _store() as store:
        rows = store.coordination.list_owned_failed_obligations(owner=OwnerStamp(pid=pid, started_at=started))
    earliest: dict[tuple, float] = {}
    for row in rows:
        # Reconnect-only rows (a claim released because the adapter was gone) are re-claimed by the
        # reconnect sweep; a timer would claim and release them every tick until the adapter is back.
        if is_reconnect_only(row.last_error):
            continue
        due = retry_not_before(row.updated_at, row.last_error, row.attempts)
        if due is None or row.attempts >= MAX_ATTEMPTS or (now - row.created_at) > STALE_AFTER_SECONDS:
            continue
        key = (row.platform, row.adapter_profile or "default")
        if key not in earliest or due < earliest[key]:
            earliest[key] = due
    return [{"platform": platform, "profile": profile, "not_before": due}
            for (platform, profile), due in sorted(earliest.items())]


def ledger_enabled(config: Optional[dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config
            config = load_config()
        value = (config.get("gateway") or {}).get("delivery_ledger", True)
        return value.strip().lower() not in {"false", "0", "no", "off"} if isinstance(value, str) else bool(value)
    except Exception:
        return True
