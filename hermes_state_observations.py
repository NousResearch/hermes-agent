"""Content-free, durable turn/attention proofs for read-only conversation lists.

The conversation key is the native compression-root lease key. Lease identity
and generation are checked in the same transaction as every producer write.
A dead running record is uncertainty, never an invented terminal result.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import sqlite3
import time
import uuid

from hermes_state_sessions import INTERNAL_LISTING_SOURCES
from hermes_state_common import _sql_session_last_active

_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,100}", re.ASCII)
_INTERNAL = frozenset((*INTERNAL_LISTING_SOURCES, "subagent", "cron", "unknown"))
_ROW_SQL = "SELECT id, parent_session_id, source, model_config, end_reason, hidden, profile_name FROM sessions WHERE id = ?"


def validate_observation_ids(session_ids):
    ids = list(session_ids)
    if not ids or len(ids) > 40 or any(not isinstance(s, str) or not _SESSION_ID.fullmatch(s) for s in ids):
        raise ValueError("Provide 1–40 exact ASCII session identifiers")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate session identifiers")
    return ids


def _iso(at):
    return datetime.fromtimestamp(at, timezone.utc).isoformat().replace("+00:00", "Z")


def unknown_session_observation(session_id, profile, *, observed_at=None):
    return {"session_id": session_id, "lineage_tip_id": None, "profile": profile, "source": None,
            "observed_at": observed_at or _iso(time.time()), "provenance": "unavailable", "revision": 0,
            "execution": "unknown", "turn_id": None, "last_result": None,
            "attention": {"kind": "unknown", "request_id": None, "turn_id": None, "opened_at": None}}


def _digest(holder):
    return hashlib.sha256(holder.encode()).hexdigest()


def read_session_observations_at_path(db_path, session_ids, *, profile):
    """Read-only adapter for metadata readers (including external companions).
    A missing/legacy/unreadable store is explicit uncertainty, never bootstrapped.
    """
    from hermes_state import SessionDB
    ids = validate_observation_ids(session_ids)
    try:
        with SessionDB(db_path=db_path, read_only=True) as db:
            return db.read_session_observations(ids, profile=profile)
    except (sqlite3.DatabaseError, OSError, ValueError, TypeError, KeyError):
        return [unknown_session_observation(sid, profile) for sid in ids]


def _live_lease(conn, key, now):
    from hermes_state import _compression_lock_holder_process_is_dead
    row = conn.execute("SELECT holder, acquired_at, expires_at FROM session_turn_leases WHERE conversation_id = ?", (key,)).fetchone()
    if row is None or row["expires_at"] <= now or _compression_lock_holder_process_is_dead(row["holder"]):
        return None
    return row


def _covered(record, lease):
    return lease is not None and record["lease_digest"] == _digest(lease["holder"]) and record["lease_acquired_at"] == lease["acquired_at"]


def _requests(record):
    return json.loads(record["attention_json"])


class SessionObservationsMixin:
    def begin_session_observation(self, session_id, holder, *, attention_covered=True):
        """Mint a new generation only under the currently admitted native lease."""
        turn_id = uuid.uuid4().hex
        def write(conn):
            key = self._session_turn_lease_key_on_conn(conn, session_id)
            now = time.time()
            lease = _live_lease(conn, key, now)
            if lease is None or lease["holder"] != holder:
                return None
            prior = conn.execute("SELECT * FROM session_observations WHERE conversation_id = ?", (key,)).fetchone()
            # Validations are durable, not silently resolved by the next message.
            attention = [r for r in _requests(prior) if r["kind"] == "validation"] if prior else []
            conn.execute("""INSERT INTO session_observations
                (conversation_id, session_id, turn_id, revision, lease_digest, lease_acquired_at,
                 state, result_status, result_at, attention_json, attention_covered, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'running', NULL, NULL, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET session_id=excluded.session_id,
                turn_id=excluded.turn_id, revision=excluded.revision, lease_digest=excluded.lease_digest,
                lease_acquired_at=excluded.lease_acquired_at, state='running', result_status=NULL,
                result_at=NULL, attention_json=excluded.attention_json,
                attention_covered=excluded.attention_covered, updated_at=excluded.updated_at""",
                (key, session_id, turn_id, prior["revision"] + 1 if prior else 1,
                 _digest(holder), lease["acquired_at"], json.dumps(attention), int(bool(attention_covered)), now))
            return turn_id
        return self._execute_write(write)

    def _observation_generation(self, conn, session_id, turn_id, holder=None, *, terminal_allowed=False):
        key = self._session_turn_lease_key_on_conn(conn, session_id)
        row = conn.execute("SELECT * FROM session_observations WHERE conversation_id = ?", (key,)).fetchone()
        if row is None or row["turn_id"] != turn_id:
            return None
        lease = _live_lease(conn, key, time.time())
        if holder is not None and (lease is None or lease["holder"] != holder):
            return None
        if _covered(row, lease) or (terminal_allowed and lease is None and row["state"] == "terminal"):
            return row
        return None

    def finish_session_observation(self, session_id, holder, turn_id, status):
        if status not in {"complete", "interrupted", "error"}:
            raise ValueError("Invalid terminal status")
        def write(conn):
            row = self._observation_generation(conn, session_id, turn_id, holder)
            if row is None or row["state"] != "running":
                return False
            attention = [r for r in _requests(row) if r["kind"] == "validation"]
            now = time.time()
            conn.execute("UPDATE session_observations SET session_id=?, state='terminal', result_status=?, result_at=?, "
                         "attention_json=?, revision=revision+1, updated_at=? WHERE conversation_id=?",
                         (session_id, status, now, json.dumps(attention), now, row["conversation_id"]))
            return True
        return bool(self._execute_write(write))

    def open_session_attention(self, session_id, turn_id, kind, *, request_id=None, holder=None):
        """Declare a content-free request. Native requests require the private holder;
        validation declarations accept an exact current generation, including after completion.
        """
        if kind not in {"question", "approval", "validation"} or (kind != "validation" and holder is None):
            raise ValueError("Native attention requires a lease holder")
        request_id = request_id or uuid.uuid4().hex
        if not isinstance(request_id, str) or not _SESSION_ID.fullmatch(request_id):
            raise ValueError("Invalid request identifier")
        def write(conn):
            row = self._observation_generation(conn, session_id, turn_id, holder, terminal_allowed=kind == "validation")
            if row is None or (kind != "validation" and row["state"] != "running"):
                return None
            requests = _requests(row)
            # One validation per turn; repeated declarations are idempotent.
            prior = next((r for r in requests if r["request_id"] == request_id or
                          (kind == "validation" and r["kind"] == kind and r["turn_id"] == turn_id)), None)
            if prior:
                return prior["request_id"] if prior["turn_id"] == turn_id and prior["kind"] == kind else None
            now = time.time()
            requests.append({"kind": kind, "request_id": request_id, "turn_id": turn_id, "opened_at": _iso(now)})
            conn.execute("UPDATE session_observations SET attention_json=?, revision=revision+1, updated_at=? WHERE conversation_id=?",
                         (json.dumps(requests), now, row["conversation_id"]))
            return request_id
        return self._execute_write(write)

    def resolve_session_attention(self, session_id, turn_id, request_id, *, request_turn_id=None, holder=None):
        """Explicit answer/cancellation, NOT permission to mutate anything. ``turn_id``
        fences the current generation; ``request_turn_id`` targets an older validation.
        """
        def write(conn):
            row = self._observation_generation(conn, session_id, turn_id, holder, terminal_allowed=True)
            if row is None:
                return False
            requests = _requests(row)
            target = next((r for r in requests if r["request_id"] == request_id and r["turn_id"] == (request_turn_id or turn_id)), None)
            if target is None or (target["kind"] != "validation" and holder is None):
                return False
            requests.remove(target)
            conn.execute("UPDATE session_observations SET attention_json=?, revision=revision+1, updated_at=? WHERE conversation_id=?",
                         (json.dumps(requests), time.time(), row["conversation_id"]))
            return True
        return bool(self._execute_write(write))

    def _observation_lineage(self, conn, session_id, profile):
        def read(sid):
            row = conn.execute(_ROW_SQL, (sid,)).fetchone()
            if row is None or row["hidden"] or row["source"] in _INTERNAL or not row["source"] or row["profile_name"] not in (None, profile):
                return None
            try:
                cfg = json.loads(row["model_config"] or "{}")
            except (ValueError, TypeError):
                return None
            if not isinstance(cfg, dict) or cfg.get("_delegate_from") is not None:
                return None
            return dict(row)
        row = read(session_id)
        if row is None:
            return None
        key = self._session_turn_lease_key_on_conn(conn, session_id)
        current = read(key)
        seen = set()
        while current and current["id"] not in seen and len(seen) < 1000:
            seen.add(current["id"])
            if current["end_reason"] != "compression":
                return (key, current) if session_id in seen else None
            children = conn.execute(_ROW_SQL.replace("id = ?", "parent_session_id = ?"), (current["id"],)).fetchall()
            children = [dict(c) for c in children if not self._is_explicit_fork_child_row(dict(c), include_reset=True)]
            # No arbitrary newest-sibling choice: an ambiguous continuation is unknown.
            if len(children) != 1:
                return None
            current = read(children[0]["id"])
        return None

    def read_session_observations(self, session_ids, *, profile):
        ids = validate_observation_ids(session_ids)
        with self._read_ctx() as conn:
            # One SQLite snapshot for identity, lease and proof. This method never migrates.
            conn.execute("BEGIN")
            try:
                has_records = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_observations'").fetchone() is not None
                rows = []
                for sid in ids:
                    now = time.time()
                    out = unknown_session_observation(sid, profile, observed_at=_iso(now))
                    lineage = self._observation_lineage(conn, sid, profile)
                    if lineage is None:
                        rows.append(out)
                        continue
                    key, tip = lineage
                    activity = conn.execute(
                        f"SELECT {_sql_session_last_active()} FROM sessions s WHERE s.id = ?",
                        (tip["id"],),
                    ).fetchone()
                    out.update(lineage_tip_id=tip["id"], source=tip["source"], last_active=activity[0])
                    lease = _live_lease(conn, key, now)
                    record = conn.execute("SELECT * FROM session_observations WHERE conversation_id=?", (key,)).fetchone() if has_records else None
                    if record and (lease is None or _covered(record, lease)):
                        out.update(provenance="native", revision=record["revision"], turn_id=record["turn_id"])
                        out["execution"] = "running" if _covered(record, lease) else "idle" if record["state"] == "terminal" else "unknown"
                        if record["state"] == "terminal":
                            out["last_result"] = {"turn_id": record["turn_id"], "status": record["result_status"], "at": _iso(record["result_at"])}
                        requests = [r for r in _requests(record) if r["kind"] == "validation" or _covered(record, lease)]
                        if requests:
                            out["attention"] = {k: requests[0][k] for k in ("kind", "request_id", "turn_id", "opened_at")}
                        elif out["execution"] != "unknown" and (record["attention_covered"] or record["state"] == "terminal"):
                            out["attention"]["kind"] = "none"
                    elif lease is not None:
                        out.update(provenance="lease", execution="running")
                    rows.append(out)
                return rows
            finally:
                conn.rollback()
