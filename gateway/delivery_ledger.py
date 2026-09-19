"""Durable delivery-obligation ledger for gateway final responses (rows in the shared ``state.db``;
WAL, owner pid + process-start liveness, bounded retention) so a crash between finalize and
platform ACK cannot lose a response silently. Checkpoints: record_obligation() 'pending' before
any send | mark_attempting() 'attempting' right before the await | mark_delivered() 'delivered'
only on SendResult.success | mark_failed() 'failed' on a definitive rejection. Crash semantics
(never silently resend an ambiguous send): pending = never started, redeliver plainly; attempting
= crashed mid-await, platform MAY have it, redeliver WITH a visible recovered marker; failed =
rejected once, restart is a retry boundary, also marked; delivered = prune. Attempts are capped
and stale rows expire, both -> 'abandoned' (kept briefly, then pruned). Everything is
best-effort: ledger failures must never block a send; callers wrap every call in try/except.

A row covers the turn's WHOLE owed payload, not just its text: ``content`` plus ``media_manifest``
(a JSON replay list of the attachment sends the turn still owes). ``mark_delivered`` must only
follow a landed text AND every landed attachment — a partial attachment delivery is a FAILURE that
puts the row back in the redelivery queue (full resend with the marker), never a delivered row with
a "media dropped" log line. Rows written before the column existed read back as text-only, and rows
with no attachments hash/record exactly as they did then (see ``compute_obligation_id``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from gateway.dead_targets import classify_dead_error
from hermes_cli.sqlite_util import add_column_if_missing
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
_DB_LOCK = threading.Lock()

# Redelivery policy knobs (deliberately not config — the ledger is gated by
# ``gateway.delivery_ledger`` and these only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

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


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db

    # Shared state.db: SessionDB owns the durable PRAGMA set; this opener keeps the plain-tuple rows
    # and the 10 s busy timeout it always had.
    return open_db(_db_path(), db_label="state.db (delivery_ledger)", busy_timeout_ms=10_000,
                   row_factory=None, initialize=_initialize_schema)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT,
            adapter_profile TEXT,
            media_manifest TEXT
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")}
    if "adapter_profile" not in columns:
        add_column_if_missing(conn, "delivery_obligations", "adapter_profile", "adapter_profile TEXT")
    if "media_manifest" not in columns:
        add_column_if_missing(conn, "delivery_obligations", "media_manifest", "media_manifest TEXT")


def _transaction():
    from hermes_cli.sqlite_util import transaction

    return transaction(_connect())


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
        return started_at is None or int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def _canonical_manifest(media_manifest: Optional[Sequence[Dict[str, Any]]]) -> Optional[str]:
    """Canonical JSON of a media manifest, or ``None`` when there is nothing to replay.

    One string serves both roles — the identity material for ``compute_obligation_id`` and the value
    stored in the row — so re-recording the identical turn is idempotent and a stored manifest can
    never be read back ambiguously.
    """
    if not media_manifest:
        return None
    try:
        return json.dumps(media_manifest, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        # Unreachable from the gateway's own builder (str/bool/list only); dropping the whole
        # manifest here would be the silent-media-loss bug, so say so loudly.
        logger.error("media manifest is not serialisable; recording the row without it", exc_info=True)
        return None


def _parse_media_manifest(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Decode a stored manifest for redelivery. ``[]`` = the row owes no attachments.

    ``None`` = the stored value cannot be replayed (unreadable JSON, or not a list of ``{"op": ...}``
    dicts). Callers MUST treat that as undeliverable and fail the row: sending the text and marking
    the row delivered while the attachments are dropped is exactly the silent loss this column
    exists to close. Which ``op`` values are actually sendable is the adapter's business
    (``BasePlatformAdapter.redeliver_attachments`` fails closed on an unknown one).
    """
    if raw is None or raw == "":
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    if not all(isinstance(entry, dict) and isinstance(entry.get("op"), str) for entry in parsed):
        return None
    return parsed


def compute_obligation_id(session_key: str, message_ref: str, content: str,
                          media_manifest: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    """Stable id: same turn + same content + same attachments re-records idempotently, while distinct
    threads/topics on one chat never collide (session_key carries platform/chat/thread; ``message_ref`` =
    inbound message id).

    The manifest is part of the identity because the row is what gets redelivered: two turns with
    identical text but different attachments are different obligations, and hashing text alone made the
    second ``INSERT OR REPLACE`` over the first's row, taking its media with it. A row with no
    attachments hashes exactly as it always did, so ids written by older builds still line up.
    """
    material = f"{session_key}|{message_ref}|{content}"
    manifest = _canonical_manifest(media_manifest)
    if manifest:
        material = f"{material}|{manifest}"
    return hashlib.sha256(material.encode("utf-8", "replace")).hexdigest()[:24]


def record_obligation(*, obligation_id: str, session_key: str, platform: str, chat_id: str,
                      thread_id: Optional[str], content: str, adapter_profile: Optional[str] = None,
                      media_manifest: Optional[Sequence[Dict[str, Any]]] = None) -> None:
    """Record a final response as owed to the platform (state='pending'), text and attachments alike."""
    now, (pid, started) = time.time(), _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, adapter_profile, media_manifest)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?)""",
            (obligation_id, session_key, platform, str(chat_id), str(thread_id) if thread_id else None,
             content, now, now, pid, started, str(adapter_profile).strip() if adapter_profile else "default",
             _canonical_manifest(media_manifest)))
    _prune()


def mark_attempting(obligation_id: str) -> None:
    _update_state(obligation_id, "attempting")


def mark_delivered(obligation_id: str) -> None:
    _update_state(obligation_id, "delivered")


def mark_failed(obligation_id: str, error: str = "") -> None:
    _update_state(obligation_id, "failed", error=error)


def release_runtime_claim(obligation_id: str, error: str = "") -> bool:
    """Return an unsent runtime claim to ``failed`` without spending an attempt.

    Runtime recovery claims before clearing ``resume_pending`` so two reconnect paths cannot send the
    same row; if the flag cannot be cleared no send was attempted and the claim must not consume the
    redelivery budget. Fail-closed to the exact current process instance and ``attempting`` state."""
    pid, started = _owner_stamp()
    if started is None:
        return False
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state='failed', attempts=CASE
                       WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                   updated_at=?, last_error=?
               WHERE obligation_id=? AND state='attempting'
                 AND owner_pid IS ? AND owner_started_at IS ?""",
            (time.time(), error[:500] if error else None, obligation_id, pid, started))
    return bool(cursor.rowcount)


def _update_state(obligation_id: str, state: str, error: str = "") -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?
               WHERE obligation_id=?""",
            (state, time.time(), error[:500] if error else None, obligation_id))


def _claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts, profile, *,
                 needs_marker: bool, runtime: bool = False, flood: bool = False,
                 last_error: Optional[str] = None,
                 media_manifest: Optional[str] = None) -> Dict[str, Any]:
    """Claimed-row dict handed back for redelivery. A marked row names its own cause: ``flood`` (a reply
    the rate limit refused, possibly after accepting part of it) gets FLOOD_MARKER at boot or at runtime, a
    ``runtime`` reconnect replay gets RECONNECTED_MARKER, and a boot-recovered crash keeps the runner's
    restart marker default. ``last_error`` is the row's pre-claim error, carried so a runtime claim that is
    released unsent goes back to ``failed`` with the same error and keeps its retry eligibility.

    ``media_manifest`` is the row's attachment replay list, or ``None`` when the stored value cannot be
    replayed — the caller must fail such a row rather than redeliver its text alone.
    """
    marker = FLOOD_MARKER if flood else (RECONNECTED_MARKER if runtime else None)
    return {"obligation_id": oid, "session_key": session_key, "platform": platform, "chat_id": chat_id,
            "thread_id": thread_id, "content": content, "needs_marker": needs_marker,
            **({"marker": marker} if needs_marker and marker else {}), "profile": profile,
            **({"runtime_recovery": True} if runtime else {}),
            **({"last_error": last_error} if last_error else {}), "attempts": attempts + 1,
            "media_manifest": _parse_media_manifest(media_manifest)}


def sweep_recoverable(now: Optional[float] = None, *, deliverable_platforms: Optional[set] = None,
                      deliverable_targets: Optional[set] = None) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for redelivery.

    Claiming atomically re-stamps the owner to THIS process and increments ``attempts`` (the UPDATE is
    guarded on the previous owner stamp, so a second gateway racing the same sweep cannot double-claim).
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
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      owner_pid, owner_started_at, adapter_profile, last_error, updated_at,
                      media_manifest
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')"""
        ).fetchall()
        for (oid, session_key, platform, chat_id, thread_id, content, state, attempts, created_at,
             owner_pid, owner_started_at, adapter_profile, last_error, updated_at,
             media_manifest) in rows:
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:  # exhausted -> abandoned
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=? WHERE obligation_id=?""", (now, oid))
                continue
            if ((deliverable_platforms is not None and platform not in deliverable_platforms)
                    or (deliverable_targets is not None and (platform, adapter_profile) not in deliverable_targets)):
                continue  # no adapter this boot — claiming would spend an attempt on a no-op
            flood_row = state == "failed" and is_flood_error(last_error)
            if flood_row and now < flood_not_before(updated_at, last_error):
                # Still inside the platform's wait: adopt the dead owner's row without spending an attempt
                # (state and error kept) so this process's flood timer can claim it once the wait passes.
                cursor = conn.execute(
                    """UPDATE delivery_obligations
                       SET owner_pid=?, owner_started_at=?,
                           adapter_profile=COALESCE(adapter_profile, 'default')
                       WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                    (pid, started, oid, owner_pid, owner_pid))
                if cursor.rowcount:
                    claimed.append({
                        "obligation_id": oid, "session_key": session_key, "platform": platform,
                        "chat_id": chat_id, "thread_id": thread_id, "content": content,
                        "profile": adapter_profile or "default", "attempts": attempts,
                        "adopted": True, "not_before": flood_not_before(updated_at, last_error),
                        "media_manifest": _parse_media_manifest(media_manifest)})
                continue
            # A claimed flood row is resent as a fresh attempt: clear the stale refusal so an interrupted
            # resend is seen as 'attempting' with no error by the next boot and gets the marker.
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, attempts=attempts+1, updated_at=?,
                       adapter_profile=COALESCE(adapter_profile, 'default'),
                       state=CASE WHEN ? THEN 'attempting' ELSE state END,
                       last_error=CASE WHEN ? THEN NULL ELSE last_error END
                   WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                (pid, started, now, 1 if flood_row else 0, 1 if flood_row else 0, oid, owner_pid, owner_pid))
            if cursor.rowcount:
                # pending = never started, redeliver plainly; anything else (crashed mid-await, other
                # rejection, a flood refusal whose earlier chunks the platform may have accepted) carries
                # the marker.
                claimed.append(_claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts,
                                            adapter_profile or "default", needs_marker=state != "pending",
                                            flood=flood_row, media_manifest=media_manifest))
    return claimed


def sweep_failed_for_runtime(platform: str, now: Optional[float] = None, *,
                             profile: Optional[str] = None) -> List[Dict[str, Any]]:
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
    expected_profile = "default" if not profile or profile == "default" else str(profile)
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, attempts, created_at, owner_pid,
                      owner_started_at, last_error, adapter_profile, updated_at,
                      media_manifest
               FROM delivery_obligations
               WHERE state='failed' AND platform=?""", (platform,)).fetchall()
        for (oid, session_key, row_platform, chat_id, thread_id, content, attempts, created_at,
             owner_pid, owner_started_at, last_error, adapter_profile, updated_at,
             media_manifest) in rows:
            # Exact process-start matching prevents PID reuse from stealing work.
            if adapter_profile != expected_profile or owner_pid != pid or owner_started_at != started:
                continue
            due = retry_not_before(updated_at, last_error, attempts)
            if due is None:
                continue
            owner_guard = (now, oid, owner_pid, owner_started_at)
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:  # exhausted -> abandoned
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=?
                       WHERE obligation_id=? AND state='failed'
                         AND owner_pid IS ? AND owner_started_at IS ?""", owner_guard)
                continue
            if now < due:
                continue  # the platform's wait or the backoff has not passed; the timer comes back for it
            # The claim clears the stale error: this is a fresh attempt, and if it is interrupted the next
            # boot must see 'attempting' with no proof of non-delivery, hence the marker.
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET state='attempting', attempts=attempts+1, updated_at=?, last_error=NULL
                   WHERE obligation_id=? AND state='failed'
                     AND owner_pid IS ? AND owner_started_at IS ?""", owner_guard)
            if cursor.rowcount:
                # The failed send's ack may have been lost (reconnect) or its earlier chunks accepted
                # (flood): every runtime redelivery carries a marker. The pre-claim error rides along so a
                # claim released unsent keeps its flood retry eligibility.
                claimed.append(_claimed_row(oid, session_key, row_platform, chat_id, thread_id, content,
                                            attempts, adapter_profile, needs_marker=True, runtime=True,
                                            flood=is_flood_error(last_error), last_error=last_error,
                                            media_manifest=media_manifest))
    return claimed


def pending_retries(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """This process's failed rows that still await redelivery, one entry per adapter identity with the
    earliest deadline (``not_before``). The runner arms one redelivery timer per entry, so a row adopted
    at boot, skipped because its wait had not passed, or rejected again is never stranded. Rows past the
    attempts cap or stale cutoff are left for the sweeps to abandon."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    if started is None:
        return []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT platform, adapter_profile, updated_at, last_error, attempts, created_at
               FROM delivery_obligations
               WHERE state='failed' AND owner_pid IS ? AND owner_started_at IS ?""", (pid, started)).fetchall()
    earliest: Dict[tuple, float] = {}
    for platform, adapter_profile, updated_at, last_error, attempts, created_at in rows:
        # Reconnect-only rows (a claim released because the adapter was gone) are re-claimed by the
        # reconnect sweep; a timer would claim and release them every tick until the adapter is back.
        if is_reconnect_only(last_error):
            continue
        due = retry_not_before(updated_at, last_error, attempts)
        if due is None or attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
            continue
        key = (platform, adapter_profile or "default")
        if key not in earliest or due < earliest[key]:
            earliest[key] = due
    return [{"platform": platform, "profile": profile, "not_before": due}
            for (platform, profile), due in sorted(earliest.items())]


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    try:
        with _transaction() as conn:
            conn.execute(
                """DELETE FROM delivery_obligations
                   WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""", (now - _RETENTION_SECONDS,))
            total = conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[0]
            if total > _MAX_ROWS:
                conn.execute(
                    """DELETE FROM delivery_obligations WHERE obligation_id IN (
                         SELECT obligation_id FROM delivery_obligations
                         ORDER BY CASE state
                                    WHEN 'delivered' THEN 0
                                    WHEN 'abandoned' THEN 1
                                    ELSE 2
                                  END, updated_at ASC
                         LIMIT ?)""", (total - _MAX_ROWS,))
    except Exception:
        logger.debug("delivery ledger prune failed", exc_info=True)


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config
            config = load_config()
        value = (config.get("gateway") or {}).get("delivery_ledger", True)
        return value.strip().lower() not in {"false", "0", "no", "off"} if isinstance(value, str) else bool(value)
    except Exception:
        return True


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import json  # noqa: F401,E402

def debug_rows(limit: int = 20) -> str:
    """Human-readable dump for ad-hoc inspection (sqlite3-free path)."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, state, attempts,
                      created_at, updated_at, last_error
               FROM delivery_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0], "session": r[1], "state": r[2], "attempts": r[3],
                "created_at": r[4], "updated_at": r[5], "last_error": r[6],
            }
            for r in rows
        ],
        indent=2,
    )
# ---- END PLUGIN-COMPAT ----
