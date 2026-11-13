"""Durable, exactly-once gateway handoff notice per logical session per owner epoch.

No new leader table: bounded versioned ``state_meta`` entries plus one
transcript notice row committed in the SAME transaction, so a crash commits
all three or none. ``message_uid`` is deterministic from profile identity +
logical session + to-epoch, so resending the same notice id after a lost ACK
is safe and duplicate renders collapse on the client. ``role='session_meta'``
keeps the notice out of the model history by existing projection rules; it is
a display-only system notice, never provider context. A session first seen on
the current epoch gets no notice (it did not survive a transition); a session
that missed several epochs gets ONE summary (last_seen -> current).
"""
from __future__ import annotations

import hashlib
import json
import time

SEEN_PREFIX = "session_handoff_seen_v1:"
RECORD_PREFIX = "session_handoff_v1:"


def _seen_key(logical: str) -> str:
    return SEEN_PREFIX + logical


def _record_key(logical: str, epoch: int) -> str:
    return f"{RECORD_PREFIX}{logical}:{epoch}"


def _meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
    return row[0] if row is not None else None


def _recovery_summary(conn, logical: str) -> dict:
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM session_admissions WHERE target_session_id=? "
        "AND status IN ('queued','unknown') GROUP BY status", (logical,)).fetchall()
    counts = {status: count for status, count in rows}
    return {"queued": int(counts.get("queued", 0)), "unknown": int(counts.get("unknown", 0))}


def _body(from_epoch: int, to_epoch: int, summary: dict) -> str:
    return (
        f"Gateway switching: owner epoch {from_epoch} → {to_epoch}. "
        f"Queued inputs retained: {summary['queued']}. "
        f"Prior runs: {summary['unknown']} unknown (never replayed automatically). "
        "Client-owned tools and approvals may require reattachment."
    )


def ensure_session_handoff(owner, session_id: str) -> dict | None:
    """Return this transition's handoff record, creating it (and its one notice) once.

    ``owner`` is the live session authority: ``.db`` (SessionDB), ``.epoch``,
    ``.profile_id``, ``.logical_owner(session_id)``, ``.physical_target(ref)``.
    """
    db = owner.db
    logical = owner.logical_owner(session_id) or session_id
    from types import SimpleNamespace
    # Production physical_target(ref) reads ref.session_id; the fixture lambda takes any arg.
    physical = owner.physical_target(SimpleNamespace(session_id=session_id))
    epoch = int(owner.epoch)
    with db._read_ctx() as conn:
        seen = _meta(conn, _seen_key(logical))
        if seen is not None and int(seen) == epoch:
            record = _meta(conn, _record_key(logical, epoch))
            return json.loads(record) if record is not None else None

    if seen is None:
        # First observation of this session: stamp without a notice. A session created
        # on the current epoch did not survive a transition.
        def stamp(conn):
            conn.execute("INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)",
                         (_seen_key(logical), str(epoch)))
        db._execute_write(stamp)
        return None

    def write(conn):
        current = _meta(conn, _seen_key(logical))
        if current is None:
            # Seen key vanished mid-flight (retirement raced this write): treat as first sight.
            conn.execute("INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)",
                         (_seen_key(logical), str(epoch)))
            return None
        if int(current) == epoch:
            record = _meta(conn, _record_key(logical, epoch))
            return json.loads(record) if record is not None else None
        from_epoch = int(current)
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM session_admissions WHERE target_session_id=? "
            "AND status IN ('queued','started','unknown') GROUP BY status", (logical,)).fetchall()
        counts = {status: count for status, count in rows}
        # 'started' without a receipt is uncertainty the notice must state, not hide: a crash
        # before recover_session_inputs leaves it unmarked.
        summary = {"queued": int(counts.get("queued", 0)),
                   "unknown": int(counts.get("unknown", 0)) + int(counts.get("started", 0))}
        handoff_id = hashlib.sha256(
            f"{owner.profile_id}:{logical}:{epoch}".encode()).hexdigest()[:32]
        message_uid = f"gateway-handoff-{handoff_id}"
        record = {
            "version": 1,
            "handoff_id": handoff_id,
            "from_epoch": from_epoch,
            "to_epoch": epoch,
            "message_uid": message_uid,
            "recovery_summary": summary,
        }
        message = {
            "role": "session_meta",
            "content": _body(from_epoch, epoch, summary),
            "display_kind": "gateway_handoff",
            "display_metadata": {k: record[k] for k in
                                 ("handoff_id", "from_epoch", "to_epoch", "recovery_summary")},
            "message_uid": message_uid,
            "timestamp": time.time(),
        }
        db._insert_message_rows(conn, physical, [message], prune_checkpoints=False)
        conn.execute("INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)",
                     (_record_key(logical, epoch), json.dumps(record)))
        conn.execute("INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)",
                     (_seen_key(logical), str(epoch)))
        return record

    return db._execute_write(write)
