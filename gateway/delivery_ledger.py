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
from pathlib import Path
from typing import Any, Dict, List, Optional

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
MAX_CONTENT_BYTES = 1_000_000
MAX_PARTS_PER_OBLIGATION = 128
_MAX_PART_KIND_BYTES = 64
_MAX_REMOTE_RECEIPT_BYTES = 512
_CLAIMED_RESULT_PART_KINDS = frozenset(
    {"text", "image", "voice", "video", "document", "tts"}
)


class DeliveryObligationConflict(RuntimeError):
    """The same durable result identity was presented with different bytes."""


class DeliveryObligationCapacityError(RuntimeError):
    """The bounded ledger cannot safely admit another owed result."""

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

# Runtime replay is fail-closed: only errors whose send contract proves they are transient reconnect
# failures. Permanent rejects (blocked bot, bad auth, missing chat) must not be retried on reconnect.
_RUNTIME_RETRYABLE_ERRORS = frozenset({"send_path_degraded"})

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


def flood_not_before(updated_at: Any, last_error: Any) -> float:
    """Earliest moment a flood-refused row may be resent: the refusal's timestamp (``mark_failed`` sets
    ``updated_at``) plus the platform's wait. Enforced by the sweeps so neither an early timer nor a
    reconnect sweep spends a redelivery attempt inside the penalty window."""
    try:
        stamp = float(updated_at or 0.0)
    except (TypeError, ValueError):
        stamp = 0.0
    return stamp + flood_wait_seconds(last_error)


def _runtime_retryable(last_error: Any) -> bool:
    text = str(last_error or "").strip().lower()
    return text in _RUNTIME_RETRYABLE_ERRORS or is_flood_error(text)


def _db_path():
    return get_hermes_home() / "state.db"


def _connect(home: Optional[Path] = None) -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db

    # Shared state.db: SessionDB owns the durable PRAGMA set; this opener keeps the plain-tuple rows
    # and the 10 s busy timeout it always had.
    path = Path(home) / "state.db" if home is not None else _db_path()
    return open_db(path, db_label="state.db (delivery_ledger)", busy_timeout_ms=10_000,
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
            claim_id TEXT,
            claim_event_id TEXT,
            active_turn_token TEXT,
            raw_content TEXT,
            source_json TEXT,
            message_ref TEXT,
            parts_manifest TEXT,
            parts_total INTEGER
        )"""
    )
    if "adapter_profile" not in {row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")}:
        add_column_if_missing(conn, "delivery_obligations", "adapter_profile", "adapter_profile TEXT")
    for column, column_type in (
        ("claim_id", "TEXT"),
        ("claim_event_id", "TEXT"),
        ("active_turn_token", "TEXT"),
        ("raw_content", "TEXT"),
        ("source_json", "TEXT"),
        ("message_ref", "TEXT"),
        ("parts_manifest", "TEXT"),
        ("parts_total", "INTEGER"),
    ):
        add_column_if_missing(
            conn, "delivery_obligations", column, f"{column} {column_type}"
        )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS
               delivery_obligations_claim_event_uq
           ON delivery_obligations(claim_id, claim_event_id)
           WHERE claim_id IS NOT NULL AND claim_event_id IS NOT NULL"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligation_parts (
            obligation_id TEXT NOT NULL,
            part_id TEXT NOT NULL,
            part_ordinal INTEGER NOT NULL,
            kind TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL,
            last_error TEXT,
            remote_receipt TEXT,
            PRIMARY KEY (obligation_id, part_id),
            UNIQUE (obligation_id, part_ordinal)
        )"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS delivery_obligation_parts_cleanup
           AFTER DELETE ON delivery_obligations
           BEGIN
             DELETE FROM delivery_obligation_parts
              WHERE obligation_id=OLD.obligation_id;
           END"""
    )


def _transaction(home: Optional[Path] = None):
    from hermes_cli.sqlite_util import transaction

    return transaction(_connect(home))


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


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while distinct threads/topics on one
    chat never collide (session_key carries platform/chat/thread; ``message_ref`` = inbound message id)."""
    return hashlib.sha256(f"{session_key}|{message_ref}|{content}".encode("utf-8", "replace")).hexdigest()[:24]


def compute_claimed_result_id(
    session_key: str, claim_id: str, claim_event_id: str
) -> str:
    """Stable, payload-free identity for one completed claimed execution."""
    payload = f"goal-continuation-result|{session_key}|{claim_id}|{claim_event_id}"
    return hashlib.sha256(payload.encode("utf-8", "strict")).hexdigest()[:24]


def compute_claimed_result_part_id(
    obligation_id: str,
    part_ordinal: int,
    kind: str,
    payload_identity: str,
) -> str:
    """Return a stable payload-free identity for one response part."""
    if not obligation_id or not isinstance(part_ordinal, int) or part_ordinal < 0:
        raise ValueError("claimed-result part identity is invalid")
    if kind not in _CLAIMED_RESULT_PART_KINDS:
        raise ValueError("claimed-result part kind is invalid")
    if not isinstance(payload_identity, str):
        raise TypeError("claimed-result part payload identity must be text")
    payload_digest = hashlib.sha256(
        payload_identity.encode("utf-8", "replace")
    ).hexdigest()
    identity = f"{obligation_id}|{part_ordinal}|{kind}|{payload_digest}"
    return hashlib.sha256(identity.encode("ascii", "strict")).hexdigest()[:24]


def _validated_claimed_result_parts(
    parts: List[tuple[str, str]],
) -> tuple[List[tuple[str, str]], str]:
    if not isinstance(parts, (list, tuple)):
        raise TypeError("claimed-result parts must be a sequence")
    if len(parts) > MAX_PARTS_PER_OBLIGATION:
        raise DeliveryObligationCapacityError(
            "claimed result has too many publication parts"
        )
    normalized: List[tuple[str, str]] = []
    seen: set[str] = set()
    for item in parts:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise DeliveryObligationConflict("claimed-result part manifest is invalid")
        part_id, kind = item
        if (
            not isinstance(part_id, str)
            or len(part_id) != 24
            or any(ch not in "0123456789abcdef" for ch in part_id)
        ):
            raise DeliveryObligationConflict("claimed-result part identity is invalid")
        if (
            not isinstance(kind, str)
            or len(kind.encode("utf-8", "strict")) > _MAX_PART_KIND_BYTES
            or kind not in _CLAIMED_RESULT_PART_KINDS
        ):
            raise DeliveryObligationConflict("claimed-result part kind is invalid")
        if part_id in seen:
            raise DeliveryObligationConflict("claimed-result part identity is duplicated")
        seen.add(part_id)
        normalized.append((part_id, kind))
    manifest_bytes = json.dumps(
        normalized, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii", "strict")
    return normalized, hashlib.sha256(manifest_bytes).hexdigest()


def register_claimed_result_parts(
    obligation_id: str,
    parts: List[tuple[str, str]],
    *,
    home: Optional[Path] = None,
) -> None:
    """Atomically freeze the complete response-part manifest before sending."""
    normalized, manifest = _validated_claimed_result_parts(parts)
    pid, started = _owner_stamp()
    if started is None:
        raise DeliveryObligationConflict(
            "claimed-result publication owner is unavailable"
        )
    now = time.time()
    with _DB_LOCK, _transaction(home) as conn:
        row = conn.execute(
            """SELECT state, claim_id, claim_event_id, owner_pid,
                      owner_started_at, parts_manifest, parts_total
               FROM delivery_obligations WHERE obligation_id=?""",
            (obligation_id,),
        ).fetchone()
        if row is None or not row[1] or not row[2]:
            raise DeliveryObligationConflict(
                "claimed-result delivery ownership is unavailable"
            )
        if (row[3], row[4]) != (pid, started):
            raise DeliveryObligationConflict(
                "claimed-result delivery is owned by another process"
            )
        if row[0] not in {"pending", "failed", "attempting", "delivered"}:
            raise DeliveryObligationConflict("claimed-result delivery state is invalid")
        if row[5] is not None:
            existing = conn.execute(
                """SELECT part_id, kind FROM delivery_obligation_parts
                   WHERE obligation_id=? ORDER BY part_ordinal""",
                (obligation_id,),
            ).fetchall()
            if (
                row[5] != manifest
                or row[6] != len(normalized)
                or [tuple(item) for item in existing] != normalized
            ):
                raise DeliveryObligationConflict(
                    "claimed-result part manifest conflicts with durable ownership"
                )
            return
        if row[0] == "delivered":
            raise DeliveryObligationConflict(
                "delivered claimed result has no part manifest"
            )
        for ordinal, (part_id, kind) in enumerate(normalized):
            conn.execute(
                """INSERT INTO delivery_obligation_parts
                   (obligation_id, part_id, part_ordinal, kind, state,
                    attempts, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', 0, ?)""",
                (obligation_id, part_id, ordinal, kind, now),
            )
        conn.execute(
            """UPDATE delivery_obligations
               SET parts_manifest=?, parts_total=?, updated_at=?
               WHERE obligation_id=?""",
            (manifest, len(normalized), now, obligation_id),
        )


def get_claimed_result_parts(
    obligation_id: str,
    *,
    home: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return bounded payload-free part state in stable publication order."""
    with _DB_LOCK, _transaction(home) as conn:
        rows = conn.execute(
            """SELECT part_id, part_ordinal, kind, state, attempts,
                      remote_receipt
               FROM delivery_obligation_parts
               WHERE obligation_id=? ORDER BY part_ordinal
               LIMIT ?""",
            (obligation_id, MAX_PARTS_PER_OBLIGATION + 1),
        ).fetchall()
    if len(rows) > MAX_PARTS_PER_OBLIGATION:
        raise DeliveryObligationCapacityError(
            "claimed-result part state exceeds its bound"
        )
    return [
        {
            "part_id": row[0],
            "ordinal": row[1],
            "kind": row[2],
            "state": row[3],
            "attempts": row[4],
            "remote_receipt": row[5],
        }
        for row in rows
    ]


def prepare_claimed_result_part(
    obligation_id: str,
    part_id: str,
    *,
    home: Optional[Path] = None,
) -> bool:
    """Fence and checkpoint one incomplete response part before its send."""
    pid, started = _owner_stamp()
    if started is None:
        raise DeliveryObligationConflict(
            "claimed-result publication owner is unavailable"
        )
    now = time.time()
    with _DB_LOCK, _transaction(home) as conn:
        owner = conn.execute(
            """SELECT state, claim_id, claim_event_id, owner_pid,
                      owner_started_at, parts_manifest
               FROM delivery_obligations WHERE obligation_id=?""",
            (obligation_id,),
        ).fetchone()
        if owner is None or not owner[1] or not owner[2] or owner[5] is None:
            raise DeliveryObligationConflict(
                "claimed-result part ownership is unavailable"
            )
        if (owner[3], owner[4]) != (pid, started):
            raise DeliveryObligationConflict(
                "claimed-result delivery is owned by another process"
            )
        if owner[0] == "delivered":
            return False
        if owner[0] not in {"pending", "failed", "attempting"}:
            raise DeliveryObligationConflict("claimed-result delivery state is invalid")
        part = conn.execute(
            """SELECT state, attempts FROM delivery_obligation_parts
               WHERE obligation_id=? AND part_id=?""",
            (obligation_id, part_id),
        ).fetchone()
        if part is None:
            raise DeliveryObligationConflict(
                "claimed-result part ownership is unavailable"
            )
        if part[0] == "delivered":
            return False
        if part[0] not in {"pending", "failed", "attempting"}:
            raise DeliveryObligationConflict("claimed-result part state is invalid")
        if part[1] >= MAX_ATTEMPTS + 1:
            raise DeliveryObligationCapacityError(
                "claimed-result part attempts exhausted"
            )
        conn.execute(
            """UPDATE delivery_obligation_parts
               SET state='attempting', attempts=attempts+1,
                   updated_at=?, last_error=NULL
               WHERE obligation_id=? AND part_id=?""",
            (now, obligation_id, part_id),
        )
        conn.execute(
            """UPDATE delivery_obligations
               SET state='attempting', updated_at=?, last_error=NULL
               WHERE obligation_id=?""",
            (now, obligation_id),
        )
    return True


def mark_claimed_result_part_delivered(
    obligation_id: str,
    part_id: str,
    *,
    remote_receipt: Optional[str] = None,
    home: Optional[Path] = None,
) -> bool:
    """Acknowledge one part and finish the obligation only when all are done."""
    pid, started = _owner_stamp()
    if started is None:
        return False
    receipt = str(remote_receipt) if remote_receipt else None
    if receipt is not None and len(receipt.encode("utf-8", "replace")) > _MAX_REMOTE_RECEIPT_BYTES:
        raise DeliveryObligationCapacityError(
            "claimed-result remote receipt exceeds durable limit"
        )
    now = time.time()
    with _DB_LOCK, _transaction(home) as conn:
        owner = conn.execute(
            """SELECT owner_pid, owner_started_at, parts_manifest, parts_total
               FROM delivery_obligations
               WHERE obligation_id=? AND claim_id IS NOT NULL
                 AND claim_event_id IS NOT NULL""",
            (obligation_id,),
        ).fetchone()
        if owner is None or owner[2] is None or (owner[0], owner[1]) != (pid, started):
            return False
        cursor = conn.execute(
            """UPDATE delivery_obligation_parts
               SET state='delivered', updated_at=?, last_error=NULL,
                   remote_receipt=COALESCE(remote_receipt, ?)
               WHERE obligation_id=? AND part_id=?
                 AND state IN ('pending', 'attempting', 'failed', 'delivered')""",
            (now, receipt, obligation_id, part_id),
        )
        if not cursor.rowcount:
            return False
        remaining = conn.execute(
            """SELECT COUNT(*) FROM delivery_obligation_parts
               WHERE obligation_id=? AND state != 'delivered'""",
            (obligation_id,),
        ).fetchone()[0]
        actual_total = conn.execute(
            """SELECT COUNT(*) FROM delivery_obligation_parts
               WHERE obligation_id=?""",
            (obligation_id,),
        ).fetchone()[0]
        terminal = remaining == 0 and actual_total == owner[3]
        conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=NULL
               WHERE obligation_id=?""",
            ("delivered" if terminal else "attempting", now, obligation_id),
        )
    return True


def mark_claimed_result_part_failed(
    obligation_id: str,
    part_id: str,
    error: str = "platform_part_delivery_failed",
    *,
    home: Optional[Path] = None,
) -> bool:
    """Fail one part and the whole response without exposing its payload."""
    pid, started = _owner_stamp()
    if started is None:
        return False
    stored_error = str(error or "platform_part_delivery_failed")[:500]
    now = time.time()
    with _DB_LOCK, _transaction(home) as conn:
        owner = conn.execute(
            """SELECT owner_pid, owner_started_at, parts_manifest
               FROM delivery_obligations
               WHERE obligation_id=? AND claim_id IS NOT NULL
                 AND claim_event_id IS NOT NULL""",
            (obligation_id,),
        ).fetchone()
        if owner is None or owner[2] is None or (owner[0], owner[1]) != (pid, started):
            return False
        cursor = conn.execute(
            """UPDATE delivery_obligation_parts
               SET state='failed', updated_at=?, last_error=?
               WHERE obligation_id=? AND part_id=? AND state != 'delivered'""",
            (now, stored_error, obligation_id, part_id),
        )
        if not cursor.rowcount:
            return False
        conn.execute(
            """UPDATE delivery_obligations
               SET state='failed', updated_at=?, last_error=?
               WHERE obligation_id=?""",
            (now, stored_error, obligation_id),
        )
    return True


def _ensure_insert_capacity(conn: sqlite3.Connection) -> None:
    """Make one terminal-row slot without ever deleting owed output."""
    cutoff = time.time() - _RETENTION_SECONDS
    conn.execute(
        """DELETE FROM delivery_obligations
           WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""",
        (cutoff,),
    )
    total = conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[0]
    needed = max(0, total - _MAX_ROWS + 1)
    if needed:
        conn.execute(
            """DELETE FROM delivery_obligations WHERE obligation_id IN (
                 SELECT obligation_id FROM delivery_obligations
                 WHERE state IN ('delivered', 'abandoned')
                 ORDER BY updated_at ASC LIMIT ?)""",
            (needed,),
        )
        total = conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[0]
    if total >= _MAX_ROWS:
        raise DeliveryObligationCapacityError("delivery obligation capacity exhausted")


def record_claimed_result(
    *,
    session_key: str,
    claim_id: str,
    claim_event_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
    adapter_profile: Optional[str] = None,
    active_turn_token: Optional[str] = None,
    raw_content: Optional[str] = None,
    source_json: Optional[str] = None,
    message_ref: Optional[str] = None,
    home: Optional[Path] = None,
) -> str:
    """Durably own a completed continuation result before claim retirement."""
    if not claim_id or not claim_event_id:
        raise ValueError("claim result identity is required")
    if not isinstance(content, str):
        raise TypeError("claim result content must be text")
    if len(content.encode("utf-8", "replace")) > MAX_CONTENT_BYTES:
        raise DeliveryObligationCapacityError("claim result exceeds durable limit")
    stored_raw_content = content if raw_content is None else raw_content
    if not isinstance(stored_raw_content, str) or len(
        stored_raw_content.encode("utf-8", "replace")
    ) > MAX_CONTENT_BYTES:
        raise DeliveryObligationCapacityError("claim replay payload exceeds durable limit")
    stored_source_json = source_json if source_json else None
    if stored_source_json is not None:
        try:
            source_payload = json.loads(stored_source_json)
        except (TypeError, ValueError) as exc:
            raise DeliveryObligationConflict("claim replay source is invalid") from exc
        if not isinstance(source_payload, dict) or len(
            stored_source_json.encode("utf-8", "replace")
        ) > 64 * 1024:
            raise DeliveryObligationConflict("claim replay source is invalid")
    stored_message_ref = str(message_ref) if message_ref else None
    if stored_message_ref is not None and len(stored_message_ref) > 512:
        raise DeliveryObligationConflict("claim replay message identity is invalid")

    obligation_id = compute_claimed_result_id(session_key, claim_id, claim_event_id)
    now = time.time()
    stored_profile = str(adapter_profile).strip() if adapter_profile else "default"
    stored_thread = str(thread_id) if thread_id else None
    stored_turn_token = str(active_turn_token).strip() if active_turn_token else None
    expected = (
        obligation_id,
        session_key,
        platform,
        str(chat_id),
        stored_thread,
        content,
        claim_id,
        claim_event_id,
        stored_profile,
        stored_turn_token,
        stored_raw_content,
        stored_source_json,
        stored_message_ref,
    )
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction(home) as conn:
        existing = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, claim_id, claim_event_id, adapter_profile,
                      active_turn_token, raw_content, source_json, message_ref
               FROM delivery_obligations
               WHERE claim_id=? AND claim_event_id=?""",
            (claim_id, claim_event_id),
        ).fetchone()
        if existing is not None:
            if tuple(existing) != expected:
                raise DeliveryObligationConflict(
                    "completed result identity conflicts with durable ownership"
                )
            return obligation_id
        _ensure_insert_capacity(conn)
        try:
            conn.execute(
                """INSERT INTO delivery_obligations
                   (obligation_id, session_key, platform, chat_id, thread_id,
                    content, state, attempts, created_at, updated_at,
                    owner_pid, owner_started_at, adapter_profile,
                    claim_id, claim_event_id, active_turn_token, raw_content,
                    source_json, message_ref)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    obligation_id,
                    session_key,
                    platform,
                    str(chat_id),
                    stored_thread,
                    content,
                    now,
                    now,
                    pid,
                    started,
                    stored_profile,
                    claim_id,
                    claim_event_id,
                    stored_turn_token,
                    stored_raw_content,
                    stored_source_json,
                    stored_message_ref,
                ),
            )
        except sqlite3.IntegrityError:
            raced = conn.execute(
                """SELECT obligation_id, session_key, platform, chat_id,
                          thread_id, content, claim_id, claim_event_id,
                          adapter_profile, active_turn_token, raw_content,
                          source_json, message_ref
                   FROM delivery_obligations
                   WHERE claim_id=? AND claim_event_id=?""",
                (claim_id, claim_event_id),
            ).fetchone()
            if raced is None or tuple(raced) != expected:
                raise DeliveryObligationConflict(
                    "completed result identity conflicts with durable ownership"
                )
    return obligation_id


def get_claimed_result(
    claim_id: str,
    claim_event_id: str,
    *,
    home: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return durable publication ownership for one claim event, if present."""
    with _DB_LOCK, _transaction(home) as conn:
        row = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, adapter_profile, active_turn_token,
                      raw_content, source_json, message_ref
               FROM delivery_obligations
               WHERE claim_id=? AND claim_event_id=?""",
            (claim_id, claim_event_id),
        ).fetchone()
    if row is None:
        return None
    return {
        "obligation_id": row[0],
        "session_key": row[1],
        "platform": row[2],
        "chat_id": row[3],
        "thread_id": row[4],
        "content": row[5],
        "state": row[6],
        "attempts": row[7],
        "profile": row[8],
        "active_turn_token": row[9],
        "raw_content": row[10],
        "source_json": row[11],
        "message_ref": row[12],
    }


def completed_active_turn_tokens(
    *, home: Optional[Path] = None
) -> Dict[str, set[str]]:
    """Return bounded turn tokens whose claimed results own publication."""
    with _DB_LOCK, _transaction(home) as conn:
        rows = conn.execute(
            """SELECT session_key, active_turn_token
               FROM delivery_obligations
               WHERE claim_id IS NOT NULL AND claim_event_id IS NOT NULL
                 AND active_turn_token IS NOT NULL
                 AND state != 'abandoned'
               LIMIT ?""",
            (_MAX_ROWS,),
        ).fetchall()
    tokens: Dict[str, set[str]] = {}
    for session_key, token in rows:
        if token:
            tokens.setdefault(str(session_key), set()).add(str(token))
    return tokens


def prepare_claimed_result_delivery(
    obligation_id: str,
    *,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
    adapter_profile: Optional[str],
    home: Optional[Path] = None,
) -> bool:
    """Bind staged output to final visible text before its first send."""
    if not obligation_id or not session_key or not platform or not chat_id:
        raise ValueError("claimed result delivery identity is required")
    if not isinstance(content, str):
        raise TypeError("claimed result delivery content must be text")
    if len(content.encode("utf-8", "replace")) > MAX_CONTENT_BYTES:
        raise DeliveryObligationCapacityError("claim result exceeds durable limit")

    now = time.time()
    stored_profile = str(adapter_profile).strip() if adapter_profile else "default"
    stored_thread = str(thread_id) if thread_id else None
    with _DB_LOCK, _transaction(home) as conn:
        row = conn.execute(
            """SELECT session_key, platform, chat_id, thread_id,
                      adapter_profile, state, claim_id, claim_event_id,
                      owner_pid, owner_started_at
               FROM delivery_obligations WHERE obligation_id=?""",
            (obligation_id,),
        ).fetchone()
        if row is None or not row[6] or not row[7]:
            raise DeliveryObligationConflict(
                "claimed-result delivery ownership is unavailable"
            )
        actual = {
            "session_key": row[0],
            "platform": row[1],
            "chat_id": row[2],
            "thread_id": row[3],
            "adapter_profile": row[4],
        }
        expected = {
            "session_key": session_key,
            "platform": platform,
            "chat_id": str(chat_id),
            "thread_id": stored_thread,
            "adapter_profile": stored_profile,
        }
        if actual != expected:
            raise DeliveryObligationConflict(
                "claimed-result delivery route conflicts with durable ownership"
            )
        if row[5] == "delivered":
            return False
        owner_pid, owner_started_at = _owner_stamp()
        if (row[8], row[9]) != (owner_pid, owner_started_at):
            raise DeliveryObligationConflict(
                "claimed-result delivery is owned by another process"
            )
        if row[5] not in {"pending", "failed", "attempting"}:
            raise DeliveryObligationConflict("claimed-result delivery state is invalid")
        conn.execute(
            """UPDATE delivery_obligations
               SET content=?, state='attempting', updated_at=?, last_error=NULL
               WHERE obligation_id=?""",
            (content, now, obligation_id),
        )
    return True


def record_obligation(*, obligation_id: str, session_key: str, platform: str, chat_id: str,
                      thread_id: Optional[str], content: str, adapter_profile: Optional[str] = None) -> None:
    """Record a final response as owed to the platform (state='pending')."""
    now, (pid, started) = time.time(), _owner_stamp()
    stored_profile = str(adapter_profile).strip() if adapter_profile else "default"
    stored_thread = str(thread_id) if thread_id else None
    expected = (
        session_key,
        platform,
        str(chat_id),
        stored_thread,
        content,
        stored_profile,
    )
    with _DB_LOCK, _transaction() as conn:
        existing = conn.execute(
            """SELECT session_key, platform, chat_id, thread_id, content,
                      adapter_profile, claim_id, claim_event_id
               FROM delivery_obligations WHERE obligation_id=?""",
            (obligation_id,),
        ).fetchone()
        if existing is not None:
            if tuple(existing[:6]) == expected and not existing[6] and not existing[7]:
                return
            raise DeliveryObligationConflict(
                "delivery obligation identity conflicts with existing ownership"
            )
        _ensure_insert_capacity(conn)
        conn.execute(
            """INSERT INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, adapter_profile)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?)""",
            (obligation_id, session_key, platform, str(chat_id), stored_thread,
             content, now, now, pid, started, stored_profile))


def mark_attempting(obligation_id: str) -> None:
    _update_state(obligation_id, "attempting")


def mark_delivered(obligation_id: str) -> None:
    _update_state(obligation_id, "delivered")


def mark_claimed_result_delivered(obligation_id: str) -> bool:
    """Mark one claimed result delivered only for its current process owner."""
    return _update_claimed_result_state(obligation_id, "delivered")


def mark_claimed_result_failed(
    obligation_id: str, error: str = "platform_delivery_failed"
) -> bool:
    """Return one claimed result to failed only for its current owner."""
    return _update_claimed_result_state(obligation_id, "failed", error=error)


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


def _update_claimed_result_state(
    obligation_id: str, state: str, error: str = ""
) -> bool:
    pid, started = _owner_stamp()
    if started is None:
        return False
    with _DB_LOCK, _transaction() as conn:
        completion_guard = ""
        if state == "delivered":
            completion_guard = """
                 AND (
                   parts_manifest IS NULL
                   OR (
                     parts_total = (
                       SELECT COUNT(*) FROM delivery_obligation_parts
                        WHERE delivery_obligation_parts.obligation_id =
                              delivery_obligations.obligation_id
                     )
                     AND parts_total = (
                       SELECT COUNT(*) FROM delivery_obligation_parts
                        WHERE delivery_obligation_parts.obligation_id =
                              delivery_obligations.obligation_id
                          AND delivery_obligation_parts.state = 'delivered'
                     )
                   )
                 )"""
        cursor = conn.execute(
            f"""UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?
               WHERE obligation_id=? AND claim_id IS NOT NULL
                 AND claim_event_id IS NOT NULL
                 AND owner_pid IS ? AND owner_started_at IS ?
                 {completion_guard}""",
            (
                state,
                time.time(),
                error[:500] if error else None,
                obligation_id,
                pid,
                started,
            ),
        )
    return bool(cursor.rowcount)


def _claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts, profile, *,
                 needs_marker: bool, runtime: bool = False, flood: bool = False,
                 last_error: Optional[str] = None, claim_id: Optional[str] = None,
                 claim_event_id: Optional[str] = None, raw_content: Optional[str] = None,
                 source_json: Optional[str] = None, message_ref: Optional[str] = None) -> Dict[str, Any]:
    """Claimed-row dict handed back for redelivery. A marked row names its own cause: ``flood`` (a reply
    the rate limit refused, possibly after accepting part of it) gets FLOOD_MARKER at boot or at runtime, a
    ``runtime`` reconnect replay gets RECONNECTED_MARKER, and a boot-recovered crash keeps the runner's
    restart marker default. ``last_error`` is the row's pre-claim error, carried so a runtime claim that is
    released unsent goes back to ``failed`` with the same error and keeps its retry eligibility."""
    marker = FLOOD_MARKER if flood else (RECONNECTED_MARKER if runtime else None)
    return {"obligation_id": oid, "session_key": session_key, "platform": platform, "chat_id": chat_id,
            "thread_id": thread_id, "content": content, "needs_marker": needs_marker,
            **({"marker": marker} if needs_marker and marker else {}), "profile": profile,
            "claim_id": claim_id, "claim_event_id": claim_event_id,
            "raw_content": raw_content, "source_json": source_json,
            "message_ref": message_ref,
            **({"runtime_recovery": True} if runtime else {}),
            **({"last_error": last_error} if last_error else {}), "attempts": attempts + 1}


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
                      owner_pid, owner_started_at, adapter_profile, claim_id,
                      claim_event_id, raw_content, source_json, message_ref,
                      last_error, updated_at
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')
               LIMIT ?""",
            (_MAX_ROWS + 1,),
        ).fetchall()
        for (oid, session_key, platform, chat_id, thread_id, content, state, attempts, created_at,
             owner_pid, owner_started_at, adapter_profile, claim_id, claim_event_id,
             raw_content, source_json, message_ref, last_error, updated_at) in rows:
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
                        "claim_id": claim_id, "claim_event_id": claim_event_id,
                        "raw_content": raw_content, "source_json": source_json,
                        "message_ref": message_ref,
                        "profile": adapter_profile or "default", "attempts": attempts,
                        "adopted": True, "not_before": flood_not_before(updated_at, last_error)})
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
                                            flood=flood_row, claim_id=claim_id,
                                            claim_event_id=claim_event_id,
                                            raw_content=raw_content, source_json=source_json,
                                            message_ref=message_ref))
    return claimed


def sweep_failed_for_runtime(platform: str, now: Optional[float] = None, *,
                             profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """Claim this process's reconnect-retryable failed rows for one adapter.

    ``profile`` scopes multiplexed gateways to the bot identity that owned the failed send (``None`` =
    primary/default adapter); unowned rows and rows owned by another process are left for the
    startup/dead-owner sweep. Startup recovery ignores rows owned by a live gateway, so a response
    rejected with ``send_path_degraded`` would stay stranded when only the adapter reconnects; this closes
    that gap without weakening ownership: only rows stamped to this exact process instance, only
    allowlisted transient errors, same attempts/staleness bounds, every update guarded by the prior owner
    stamp and ``failed`` state. Claimed rows always carry a marker (the failed send's ack is not safe to
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
                      claim_id, claim_event_id, raw_content, source_json, message_ref
               FROM delivery_obligations
               WHERE state='failed' AND platform=?
               LIMIT ?""", (platform, _MAX_ROWS + 1)).fetchall()
        for (oid, session_key, row_platform, chat_id, thread_id, content, attempts, created_at,
             owner_pid, owner_started_at, last_error, adapter_profile, updated_at,
             claim_id, claim_event_id, raw_content, source_json, message_ref) in rows:
            # Exact process-start matching prevents PID reuse from stealing work.
            if (adapter_profile != expected_profile or owner_pid != pid or owner_started_at != started
                    or not _runtime_retryable(last_error)):
                continue
            owner_guard = (now, oid, owner_pid, owner_started_at)
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:  # exhausted -> abandoned
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=?
                       WHERE obligation_id=? AND state='failed'
                         AND owner_pid IS ? AND owner_started_at IS ?""", owner_guard)
                continue
            if is_flood_error(last_error) and now < flood_not_before(updated_at, last_error):
                continue  # the platform's wait has not passed; the flood timer comes back for it
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
                                            claim_id=claim_id, claim_event_id=claim_event_id,
                                            raw_content=raw_content, source_json=source_json,
                                            message_ref=message_ref))
    return claimed


def pending_flood_retries(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """This process's flood-refused rows that still await redelivery, one entry per adapter identity
    with the earliest deadline (``not_before``). The runner arms one redelivery timer per entry, so a
    row adopted at boot, skipped because its wait had not passed, or refused again is never stranded.
    Rows past the attempts cap or stale cutoff are left for the sweeps to abandon."""
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
        if not is_flood_error(last_error) or attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
            continue
        due = flood_not_before(updated_at, last_error)
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
                         WHERE state IN ('delivered', 'abandoned')
                         ORDER BY updated_at ASC
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
