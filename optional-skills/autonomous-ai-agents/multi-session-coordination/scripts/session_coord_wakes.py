#!/usr/bin/env python3
"""Durable wait episodes and wake outbox for session_coord.

This module is deliberately transport-free.  It owns only SQLite state transitions;
Hermes prompt admission is performed by an exact-target consumer using wake-lease and
wake-result receipts.
"""

import hashlib
import json
import os
import sqlite3
import time
import uuid

LEASE_SECONDS = 120.0
TARGET_KINDS = {"desktop", "gateway", "cli", "bot", "subagent_parent"}


SCHEMA = """
CREATE TABLE IF NOT EXISTS wait_episodes(
    episode_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    request_json TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    generation INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL,
    native_session_id TEXT NOT NULL,
    profile_home TEXT NOT NULL,
    transport TEXT NOT NULL,
    target_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'waiting',
    wake_seq INTEGER NOT NULL DEFAULT 0,
    verify_required INTEGER NOT NULL DEFAULT 0,
    verify_details_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    canceled_at REAL,
    cancel_reason TEXT
);
CREATE TABLE IF NOT EXISTS wake_outbox(
    event_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    wake_seq INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL,
    verify_required INTEGER NOT NULL DEFAULT 0,
    prompt TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    lease_owner TEXT,
    lease_expires_at REAL,
    last_error TEXT,
    UNIQUE(episode_id, wake_seq)
);
CREATE TABLE IF NOT EXISTS wake_attempts(
    event_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'leased',
    receipt_json TEXT,
    leased_at REAL NOT NULL,
    result_at REAL,
    PRIMARY KEY(event_id, attempt)
);
CREATE INDEX IF NOT EXISTS idx_wait_episode_session
    ON wait_episodes(session_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_wake_outbox_status
    ON wake_outbox(status, created_at);
CREATE INDEX IF NOT EXISTS idx_wake_attempt_event
    ON wake_attempts(event_id, attempt);
"""


ADDITIVE_COLUMNS = {
    "sessions": [
        ("native_session_id", "TEXT"),
        ("wake_transport", "TEXT"),
        ("wake_target_json", "TEXT"),
        ("profile_home", "TEXT"),
    ],
    "waiters": [
        ("episode_id", "TEXT"),
    ],
}


def clock():
    return time.time()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def ensure_schema(conn):
    """Create only additive v3 tables/columns; never backfill legacy waiters."""
    conn.executescript(SCHEMA)
    for table, columns in ADDITIVE_COLUMNS.items():
        have = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})")  # nosec B608
        }
        for name, declaration in columns:
            if name not in have:
                # Identifiers and declarations are constants above, never user input.
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"  # nosec B608
                )


def canonical_profile_home(path):
    if not path:
        return None
    return os.path.realpath(os.path.expanduser(str(path)))


def parse_registration_target(raw_json, native_session_id, wake_transport, surface,
                              parent_id, env=None):
    """Validate or conservatively derive a Hermes wake target.

    Only HERMES_SESSION_ID and HERMES_HOME are trusted for automatic capture.  A child
    target is never inferred because the parent receiving conversation is not knowable
    from the child board id.
    """
    env = env or os.environ
    native = (native_session_id or env.get("HERMES_SESSION_ID") or "").strip() or None
    transport = (wake_transport or "").strip() or None
    target = None
    if raw_json:
        try:
            target = json.loads(raw_json)
        except (TypeError, ValueError) as exc:
            return None, None, None, None, f"invalid --wake-target-json: {exc}"
        if not isinstance(target, dict):
            return None, None, None, None, "--wake-target-json must be a JSON object"
    elif native and env.get("HERMES_HOME") and not parent_id and surface in TARGET_KINDS:
        target = {
            "profile_home": env["HERMES_HOME"],
            "kind": surface,
            "session_id": native,
        }
        transport = transport or "hermes"

    any_wake = bool(native or transport or target)
    if not any_wake:
        return None, None, None, None, None
    if target is None:
        # Keep trustworthy native identity on the session, but do not pretend it is a
        # routable target.  Yield will report this as a blocker.
        return native, transport, None, None, None

    missing = [key for key in ("profile_home", "kind", "session_id") if not target.get(key)]
    if missing:
        return None, None, None, None, (
            f"--wake-target-json missing required key(s): {', '.join(missing)}"
        )
    if target["kind"] not in TARGET_KINDS:
        return None, None, None, None, (
            f"wake target kind must be one of: {', '.join(sorted(TARGET_KINDS))}"
        )
    if parent_id:
        if target["kind"] != "subagent_parent":
            return None, None, None, None, (
                "child wake target kind must be 'subagent_parent'"
            )
        if not target.get("parent_session_id"):
            return None, None, None, None, (
                "child wake target requires parent_session_id"
            )
        if target["session_id"] != target["parent_session_id"]:
            return None, None, None, None, (
                "child wake target session_id must equal parent_session_id"
            )
    profile_home = canonical_profile_home(target["profile_home"])
    target = dict(target)
    target["profile_home"] = profile_home
    if "profile" in target and (
        not isinstance(target["profile"], str) or not target["profile"]
    ):
        return None, None, None, None, "wake target profile must be a non-empty string"
    # Seed only the known receiver. A child actor is not its parent receiver,
    # and neither a profile label nor unobserved ancestry can be guessed here.
    lineage = target.setdefault("compression_lineage", [target["session_id"]])
    if (
        not isinstance(lineage, list)
        or not lineage
        or any(not isinstance(value, str) or not value for value in lineage)
        or target["session_id"] not in lineage
    ):
        return None, None, None, None, (
            "wake target compression_lineage must contain session_id"
        )
    native = native or str(target["session_id"])
    transport = transport or "hermes"
    return native, transport, target, profile_home, None


def registration_values(raw_json, native_session_id, wake_transport, surface,
                        parent_id, env=None):
    native, transport, target, profile_home, error = parse_registration_target(
        raw_json, native_session_id, wake_transport, surface, parent_id, env
    )
    return {
        "native_session_id": native,
        "wake_transport": transport,
        "wake_target_json": canonical_json(target) if target is not None else None,
        "profile_home": profile_home,
        "error": error,
    }


def validate_checkpoint(path):
    if not path:
        return None, "--checkpoint is required with --yield"
    if not os.path.isabs(path):
        return None, "--checkpoint must be an existing absolute path"
    if not os.path.exists(path):
        return None, f"--checkpoint does not exist: {path}"
    return path, None


def make_request(kind, entries, checkpoint_path):
    """Return the canonical, immutable whole-set request stored on an episode."""
    normalized = []
    for entry in entries:
        normalized.append(
            {
                "resource": entry["resource"],
                "mode": entry.get("mode") or "exclusive",
                "task": entry.get("task"),
                "ttl_min": float(entry["ttl_min"]),
            }
        )
    normalized.sort(
        key=lambda item: (
            item["resource"], item["mode"], item["task"] or "", item["ttl_min"]
        )
    )
    return {
        "kind": kind,
        "resources": normalized,
        "checkpoint_path": checkpoint_path,
    }


def _request_hash(request):
    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


def request_entries(episode):
    request = json.loads(episode["request_json"])
    return request["resources"]


def target_of(episode):
    return json.loads(episode["target_json"])


def _cancel_episode(conn, episode_id, status, reason, at=None):
    at = at or clock()
    conn.execute(
        "UPDATE wait_episodes SET status=?, canceled_at=?, cancel_reason=?, updated_at=? "
        "WHERE episode_id=? AND status='waiting'",
        (status, at, reason, at, episode_id),
    )
    conn.execute(
        "UPDATE waiters SET active=0 WHERE episode_id=?", (episode_id,)
    )
    conn.execute(
        "UPDATE wake_outbox SET status='canceled', updated_at=?, lease_owner=NULL, "
        "lease_expires_at=NULL, last_error=? WHERE episode_id=? "
        "AND status IN ('pending','leased','unknown')",
        (at, reason, episode_id),
    )
    conn.execute(
        "UPDATE wake_attempts SET outcome='canceled', result_at=? WHERE event_id IN "
        "(SELECT event_id FROM wake_outbox WHERE episode_id=?) AND outcome='leased'",
        (at, episode_id),
    )


def cancel_for_session(conn, session_id, reason, episode_id=None, status="canceled"):
    params = [session_id]
    sql = "SELECT episode_id FROM wait_episodes WHERE session_id=? AND status='waiting'"
    if episode_id:
        sql += " AND episode_id=?"
        params.append(episode_id)
    rows = conn.execute(sql, params).fetchall()
    for row in rows:
        _cancel_episode(conn, row["episode_id"], status, reason)
    return [row["episode_id"] for row in rows]


def cancel_for_native(conn, native_session_id, profile_home, reason):
    profile_home = canonical_profile_home(profile_home)
    rows = conn.execute(
        "SELECT DISTINCT e.episode_id FROM wait_episodes e "
        "LEFT JOIN sessions s ON s.id=e.session_id "
        "WHERE e.status='waiting' AND "
        "((e.native_session_id=? AND e.profile_home=?) OR "
        "(s.native_session_id=? AND s.profile_home=?))",
        (native_session_id, profile_home, native_session_id, profile_home),
    ).fetchall()
    for row in rows:
        _cancel_episode(conn, row["episode_id"], "canceled", reason)
    return [row["episode_id"] for row in rows]


def _session_wake_snapshot(conn, session_id):
    return conn.execute(
        "SELECT native_session_id,wake_transport,wake_target_json,profile_home "
        "FROM sessions WHERE id=?",
        (session_id,),
    ).fetchone()


def park_episode(conn, session_id, kind, entries, checkpoint_path, waiter_note,
                 now_fn=clock):
    """Create/reuse one active whole-request episode inside caller transaction."""
    snapshot = _session_wake_snapshot(conn, session_id)
    if not snapshot:
        return None, False, "session is not registered"
    if not (
        snapshot["native_session_id"]
        and snapshot["wake_transport"]
        and snapshot["wake_target_json"]
        and snapshot["profile_home"]
    ):
        return None, False, (
            "automatic wake unavailable: register a trustworthy native session id, "
            "Hermes transport, and exact wake target before using --yield"
        )
    try:
        target = json.loads(snapshot["wake_target_json"])
    except (TypeError, ValueError):
        return None, False, "automatic wake unavailable: stored wake target is invalid"
    if not isinstance(target, dict) or target.get("session_id") is None:
        return None, False, "automatic wake unavailable: stored wake target is incomplete"

    request = make_request(kind, entries, checkpoint_path)
    request_json = canonical_json(request)
    digest = _request_hash(request)
    active = conn.execute(
        "SELECT * FROM wait_episodes WHERE session_id=? AND status='waiting' "
        "ORDER BY generation DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if active and active["request_hash"] == digest:
        for entry in request["resources"]:
            row = conn.execute(
                "SELECT id FROM waiters WHERE episode_id=? AND resource=? AND active=1",
                (active["episode_id"], entry["resource"]),
            ).fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO waiters(session_id,resource,since,note,active,mode,"
                    "last_poll,episode_id) "
                    "VALUES(?,?,?,?,1,?,?,?)",
                    (
                        session_id,
                        entry["resource"],
                        active["created_at"],
                        waiter_note,
                        entry["mode"],
                        active["created_at"],
                        active["episode_id"],
                    ),
                )
        return active, True, None

    if active:
        _cancel_episode(
            conn,
            active["episode_id"],
            "superseded",
            "request changed; replaced by a new yield episode",
            now_fn(),
        )
    # Defensive cleanup if a damaged DB contains multiple open episodes.
    for stale in conn.execute(
        "SELECT episode_id FROM wait_episodes WHERE session_id=? AND status='waiting'",
        (session_id,),
    ).fetchall():
        _cancel_episode(
            conn,
            stale["episode_id"],
            "superseded",
            "reconciled duplicate open episode",
            now_fn(),
        )

    generation = conn.execute(
        "SELECT COALESCE(MAX(generation),0)+1 AS generation FROM wait_episodes "
        "WHERE session_id=?",
        (session_id,),
    ).fetchone()["generation"]
    created = now_fn()
    episode_id = "ep-" + uuid.uuid4().hex[:20]
    conn.execute(
        "INSERT INTO wait_episodes(episode_id,session_id,kind,request_json,request_hash,"
        "generation,checkpoint_path,native_session_id,profile_home,transport,target_json,"
        "status,wake_seq,verify_required,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,'waiting',0,0,?,?)",
        (
            episode_id,
            session_id,
            kind,
            request_json,
            digest,
            generation,
            checkpoint_path,
            str(target["session_id"]),
            snapshot["profile_home"],
            snapshot["wake_transport"],
            snapshot["wake_target_json"],
            created,
            created,
        ),
    )
    for entry in request["resources"]:
        conn.execute(
            "INSERT INTO waiters(session_id,resource,since,note,active,mode,last_poll,episode_id) "
            "VALUES(?,?,?,?,1,?,?,?)",
            (
                session_id,
                entry["resource"],
                created,
                waiter_note,
                entry["mode"],
                created,
                episode_id,
            ),
        )
    return conn.execute(
        "SELECT * FROM wait_episodes WHERE episode_id=?", (episode_id,)
    ).fetchone(), False, None


def _verification_details(episode):
    try:
        details = json.loads(episode["verify_details_json"] or "[]")
    except (TypeError, ValueError):
        details = []
    return details if isinstance(details, list) else []


def verification_details(episode):
    return _verification_details(episode)


def note_expired_holder(conn, claim, overlaps, now_fn=clock):
    """Attach real, just-observed expired-holder evidence without waking partial sets."""
    changed = 0
    for episode in conn.execute(
        "SELECT * FROM wait_episodes WHERE status='waiting'"
    ).fetchall():
        if not any(
            overlaps(entry["resource"], claim["resource"])
            for entry in request_entries(episode)
        ):
            continue
        details = _verification_details(episode)
        marker = {"claim_id": claim["id"]}
        if not any(
            item.get("claim_id") == claim["id"]
            for item in details
            if isinstance(item, dict)
        ):
            marker.update(
                {
                    "resource": claim["resource"],
                    "holder_session_id": claim["session_id"],
                    "task": claim["task"],
                    "observed_expired_at": now_fn(),
                }
            )
            details.append(marker)
        conn.execute(
            "UPDATE wait_episodes SET verify_required=1, verify_details_json=?, "
            "updated_at=? WHERE episode_id=?",
            (canonical_json(details), now_fn(), episode["episode_id"]),
        )
        changed += 1
    return changed


def note_parked_expiry(conn, claim, now_fn=clock):
    rows = conn.execute(
        "SELECT DISTINCT e.* FROM wait_episodes e JOIN waiters w "
        "ON w.episode_id=e.episode_id WHERE w.session_id=? AND w.resource=? "
        "AND e.status='waiting'",
        (claim["session_id"], claim["resource"]),
    ).fetchall()
    for episode in rows:
        details = _verification_details(episode)
        if not any(
            isinstance(item, dict) and item.get("parked_claim_id") == claim["id"]
            for item in details
        ):
            details.append(
                {
                    "parked_claim_id": claim["id"],
                    "resource": claim["resource"],
                    "observed_expired_at": now_fn(),
                }
            )
        conn.execute(
            "UPDATE wait_episodes SET verify_required=1, verify_details_json=?, "
            "updated_at=? WHERE episode_id=?",
            (canonical_json(details), now_fn(), episode["episode_id"]),
        )


def _active_event(conn, episode_id):
    return conn.execute(
        "SELECT * FROM wake_outbox WHERE episode_id=? "
        "AND status IN ('pending','leased','unknown','delivered') "
        "ORDER BY wake_seq DESC LIMIT 1",
        (episode_id,),
    ).fetchone()


def _prompt(event_id, episode, reason):
    warning = ""
    if episode["verify_required"]:
        warning = (
            "\nWARNING: a holder TTL expiry was observed from actual board state. "
            "TTL does not prove that process stopped touching the resource. Inspect "
            "the expired-holder evidence and verify resource state before any mutation."
        )
    target = target_of(episode)
    routing = ""
    if target.get("kind") == "subagent_parent":
        routing = (
            "\nThis is a finished/yielded child checkpoint. The parent conversation "
            "must redispatch continuation for child board session "
            f"{episode['session_id']}; do not revive a same-child process blindly."
        )
    return (
        f"Coordination continuation event {event_id} is ready ({reason}).\n"
        f"Checkpoint: {episode['checkpoint_path']}\n"
        f"Board session: {episode['session_id']}\n"
        "First revalidate and atomically acquire the immutable full request:\n"
        "python3 ~/.hermes/scripts/session_coord.py continue "
        f"--id {episode['session_id']} --event {event_id} --json\n"
        "If it yields again, STOP and do not poll. The board is advisory; this prompt "
        f"does not force-suspend or prove completion.{warning}{routing}"
    )


def expire_stale_leases(conn, now_fn=clock):
    t = now_fn()
    rows = conn.execute(
        "SELECT * FROM wake_outbox WHERE status='leased' AND lease_expires_at < ?",
        (t,),
    ).fetchall()
    for row in rows:
        conn.execute(
            "UPDATE wake_outbox SET status='unknown', updated_at=?, lease_owner=NULL, "
            "lease_expires_at=NULL, last_error='lease expired without result; exact "
            "receiver receipt required before retry' WHERE event_id=? AND status='leased'",
            (t, row["event_id"]),
        )
        conn.execute(
            "UPDATE wake_attempts SET outcome='unknown', result_at=? WHERE event_id=? "
            "AND attempt=? AND outcome='leased'",
            (t, row["event_id"], row["attempt"]),
        )
    return len(rows)


def enqueue_eligible(conn, reason, ready_fn, rank_fn, now_fn=clock):
    """Fairly enqueue ready whole-set episodes inside the caller's write transaction."""
    counts = {
        "enqueued": 0,
        "stale_events_canceled": 0,
        "unknown_leases": expire_stale_leases(conn, now_fn),
        "episodes_checked": 0,
    }
    pending = conn.execute(
        "SELECT o.*,e.status AS episode_status FROM wake_outbox o "
        "JOIN wait_episodes e ON e.episode_id=o.episode_id "
        "WHERE o.status='pending' ORDER BY o.created_at,o.event_id"
    ).fetchall()
    for event in pending:
        episode = conn.execute(
            "SELECT * FROM wait_episodes WHERE episode_id=?", (event["episode_id"],)
        ).fetchone()
        if not episode or episode["status"] != "waiting" or not ready_fn(episode):
            conn.execute(
                "UPDATE wake_outbox SET status='canceled', updated_at=?, "
                "last_error='eligibility changed before lease' WHERE event_id=? "
                "AND status='pending'",
                (now_fn(), event["event_id"]),
            )
            counts["stale_events_canceled"] += 1

    episodes = conn.execute(
        "SELECT * FROM wait_episodes WHERE status='waiting' ORDER BY created_at,episode_id"
    ).fetchall()
    episodes = sorted(
        episodes,
        key=lambda ep: (
            rank_fn(ep["session_id"]), ep["created_at"], ep["episode_id"]
        ),
    )
    for episode in episodes:
        counts["episodes_checked"] += 1
        if _active_event(conn, episode["episode_id"]):
            continue
        if not ready_fn(episode):
            continue
        wake_seq = int(episode["wake_seq"] or 0) + 1
        event_id = "wake-" + uuid.uuid4().hex[:20]
        event_reason = "ttl_expired" if episode["verify_required"] else reason
        created = now_fn()
        prompt = _prompt(event_id, episode, event_reason)
        try:
            conn.execute(
                "INSERT INTO wake_outbox(event_id,episode_id,wake_seq,status,attempt,reason,"
                "verify_required,prompt,created_at,updated_at) "
                "VALUES(?,?,?,'pending',0,?,?,?,?,?)",
                (
                    event_id,
                    episode["episode_id"],
                    wake_seq,
                    event_reason,
                    int(bool(episode["verify_required"])),
                    prompt,
                    created,
                    created,
                ),
            )
        except sqlite3.IntegrityError:
            continue
        conn.execute(
            "UPDATE wait_episodes SET wake_seq=?, updated_at=? WHERE episode_id=?",
            (wake_seq, created, episode["episode_id"]),
        )
        counts["enqueued"] += 1
    return counts


def _event_payload(conn, event):
    episode = conn.execute(
        "SELECT * FROM wait_episodes WHERE episode_id=?", (event["episode_id"],)
    ).fetchone()
    if not episode:
        return None
    return {
        "event_id": event["event_id"],
        "attempt": int(event["attempt"] or 0),
        "episode_id": event["episode_id"],
        "board_session_id": episode["session_id"],
        "native_session_id": episode["native_session_id"],
        "profile_home": episode["profile_home"],
        "transport": episode["transport"],
        "target": target_of(episode),
        "checkpoint_path": episode["checkpoint_path"],
        "reason": event["reason"],
        "verify_required": bool(event["verify_required"]),
        "prompt": event["prompt"],
        "episode_created_at": episode["created_at"],
        "created_at": event["created_at"],
    }


def lease_one(conn, worker_id, transport, profile_home, native_session_id, kind,
              ready_fn, rank_fn, now_fn=clock):
    """Lease at most one exact-target event after live fairness revalidation."""
    expire_stale_leases(conn, now_fn)
    enqueue_eligible(conn, "reconciled", ready_fn, rank_fn, now_fn)
    profile_home = canonical_profile_home(profile_home)
    episodes = conn.execute(
        "SELECT e.*,o.event_id,o.status AS event_status,o.created_at AS event_created "
        "FROM wait_episodes e JOIN wake_outbox o ON o.episode_id=e.episode_id "
        "WHERE e.status='waiting' AND o.status='pending' AND e.transport=? "
        "AND e.profile_home=? AND e.native_session_id=? "
        "ORDER BY o.created_at,o.event_id",
        (transport, profile_home, native_session_id),
    ).fetchall()
    episodes = sorted(
        episodes,
        key=lambda ep: (rank_fn(ep["session_id"]), ep["event_created"], ep["event_id"]),
    )
    for joined in episodes:
        target = target_of(joined)
        if kind and target.get("kind") != kind:
            continue
        episode = conn.execute(
            "SELECT * FROM wait_episodes WHERE episode_id=?", (joined["episode_id"],)
        ).fetchone()
        if not ready_fn(episode):
            conn.execute(
                "UPDATE wake_outbox SET status='canceled', updated_at=?, "
                "last_error='eligibility changed before lease' WHERE event_id=? "
                "AND status='pending'",
                (now_fn(), joined["event_id"]),
            )
            continue
        event = conn.execute(
            "SELECT * FROM wake_outbox WHERE event_id=? AND status='pending'",
            (joined["event_id"],),
        ).fetchone()
        if not event:
            continue
        attempt = int(event["attempt"] or 0) + 1
        t = now_fn()
        updated = conn.execute(
            "UPDATE wake_outbox SET status='leased',attempt=?,lease_owner=?,"
            "lease_expires_at=?,updated_at=? WHERE event_id=? AND status='pending'",
            (attempt, worker_id, t + LEASE_SECONDS, t, event["event_id"]),
        )
        if updated.rowcount != 1:
            continue
        conn.execute(
            "INSERT INTO wake_attempts(event_id,attempt,worker_id,outcome,leased_at) "
            "VALUES(?,?,?,'leased',?)",
            (event["event_id"], attempt, worker_id, t),
        )
        leased = conn.execute(
            "SELECT * FROM wake_outbox WHERE event_id=?", (event["event_id"],)
        ).fetchone()
        return _event_payload(conn, leased)
    return None


def _load_receipt(text):
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        return None, f"invalid --receipt-json: {exc}"
    if not isinstance(value, dict):
        return None, "--receipt-json must be a JSON object"
    return value, None


def _receipt_exact(event_payload, receipt):
    return (
        receipt.get("event_id") == event_payload["event_id"]
        and receipt.get("native_session_id") == event_payload["native_session_id"]
        and canonical_profile_home(receipt.get("profile_home"))
        == event_payload["profile_home"]
    )


def record_result(conn, event_id, attempt, outcome, receipt_json, now_fn=clock):
    event = conn.execute(
        "SELECT * FROM wake_outbox WHERE event_id=?", (event_id,)
    ).fetchone()
    if not event:
        return None, f"no wake event matches '{event_id}'", False
    payload = _event_payload(conn, event)
    receipt, error = _load_receipt(receipt_json)
    if error:
        return None, error, False
    if receipt is None:
        return None, "missing receipt object", False
    attempt_row = conn.execute(
        "SELECT * FROM wake_attempts WHERE event_id=? AND attempt=?",
        (event_id, attempt),
    ).fetchone()
    if not attempt_row:
        return None, "unknown wake attempt", True
    if int(event["attempt"] or 0) != int(attempt):
        return None, "stale wake attempt cannot overwrite newer state", True

    previous = attempt_row["outcome"]
    if previous == outcome and event["status"] == outcome:
        return {
            "event_id": event_id,
            "attempt": attempt,
            "status": outcome,
            "idempotent": True,
        }, None, False
    if event["status"] == "canceled" or previous == "canceled":
        return None, "wake event was canceled", True
    if previous not in ("leased", "unknown"):
        return None, f"wake attempt already has final outcome '{previous}'", True

    exact = _receipt_exact(payload, receipt)
    if outcome in ("not_sent", "canceled"):
        durable = bool(receipt.get("payload_hash")) and (
            receipt.get("payload_hash") == receipt.get("payload_sha256")
        )
        disposition = receipt.get("status") == outcome
        if outcome == "not_sent":
            disposition = disposition and receipt.get("not_sent") is True \
                and receipt.get("submitted") is False
        else:
            disposition = disposition and not any(
                receipt.get(field) is True
                for field in ("admitted", "accepted", "duplicate", "not_sent")
            ) and receipt.get("submitted") is not False
        if not (exact and durable and disposition):
            return None, (
                f"{outcome} requires an exact event/profile/native-session durable "
                f"{outcome.replace('_', '-')} receipt"
            ), False
    if outcome == "delivered":
        admitted = bool(
            receipt.get("admitted")
            or receipt.get("accepted")
            or receipt.get("duplicate")
        )
        durable = bool(
            receipt.get("durable")
            or receipt.get("receipt_id")
            or receipt.get("payload_hash")
        )
        if not (exact and admitted and durable):
            return None, (
                "delivered requires an exact event/profile/native-session durable admission receipt"
            ), False
    if previous == "unknown" and outcome == "unknown":
        return {
            "event_id": event_id,
            "attempt": attempt,
            "status": "unknown",
            "idempotent": True,
        }, None, False

    t = now_fn()
    conn.execute(
        "UPDATE wake_attempts SET outcome=?,receipt_json=?,result_at=? "
        "WHERE event_id=? AND attempt=?",
        (outcome, canonical_json(receipt), t, event_id, attempt),
    )
    if outcome == "canceled":
        _cancel_episode(
            conn,
            event["episode_id"],
            "canceled",
            "receiver canceled wake admission",
            t,
        )
        return {"event_id": event_id, "attempt": attempt, "status": "canceled"}, None, False
    status = "pending" if outcome == "not_sent" else outcome
    conn.execute(
        "UPDATE wake_outbox SET status=?,updated_at=?,lease_owner=NULL,"
        "lease_expires_at=NULL,last_error=? WHERE event_id=?",
        (
            status,
            t,
            ("delivery outcome unknown; exact receiver receipt required"
             if outcome == "unknown" else None),
            event_id,
        ),
    )
    return {"event_id": event_id, "attempt": attempt, "status": status}, None, False


def load_continuation(conn, session_id, event_id):
    event = conn.execute(
        "SELECT * FROM wake_outbox WHERE event_id=?", (event_id,)
    ).fetchone()
    if not event:
        return None, None, f"no wake event matches '{event_id}'"
    episode = conn.execute(
        "SELECT * FROM wait_episodes WHERE episode_id=?", (event["episode_id"],)
    ).fetchone()
    if not episode or episode["session_id"] != session_id:
        return None, None, "wake event does not belong to this board session"
    if episode["status"] != "waiting":
        return None, None, f"wait episode is {episode['status']}"
    if event["status"] != "delivered":
        return None, None, f"wake event is {event['status']}, not durably delivered"
    return episode, event, None


def consume_and_rewait(conn, episode, event, reason, now_fn=clock):
    t = now_fn()
    conn.execute(
        "UPDATE wake_outbox SET status='consumed',updated_at=?,last_error=? "
        "WHERE event_id=? AND status='delivered'",
        (t, reason, event["event_id"]),
    )
    conn.execute(
        "UPDATE wait_episodes SET updated_at=? WHERE episode_id=?",
        (t, episode["episode_id"]),
    )


def mark_acquired(conn, episode, event, now_fn=clock):
    t = now_fn()
    conn.execute(
        "UPDATE wake_outbox SET status='consumed',updated_at=? WHERE event_id=?",
        (t, event["event_id"]),
    )
    conn.execute(
        "UPDATE wait_episodes SET status='acquired',updated_at=? WHERE episode_id=?",
        (t, episode["episode_id"]),
    )
    conn.execute(
        "UPDATE waiters SET active=0 WHERE episode_id=?", (episode["episode_id"],)
    )
    conn.execute(
        "UPDATE wake_outbox SET status='canceled',updated_at=?,last_error='episode acquired' "
        "WHERE episode_id=? AND event_id!=? "
        "AND status IN ('pending','leased','unknown')",
        (t, episode["episode_id"], event["event_id"]),
    )


def pending_for_profile(conn, profile_home):
    """Read-only opted-in episode/outbox metadata; never reads checkpoint contents."""
    profile_home = canonical_profile_home(profile_home)
    episodes = conn.execute(
        "SELECT episode_id,session_id,kind,checkpoint_path,native_session_id,profile_home,"
        "transport,target_json,status,wake_seq,verify_required,created_at,updated_at "
        "FROM wait_episodes WHERE profile_home=? AND status='waiting' "
        "ORDER BY created_at,episode_id",
        (profile_home,),
    ).fetchall()
    event_rows = conn.execute(
        "SELECT o.* FROM wake_outbox o JOIN wait_episodes e ON e.episode_id=o.episode_id "
        "WHERE e.profile_home=? AND e.status='waiting' AND o.status IN "
        "('pending','leased','unknown','delivered') ORDER BY o.created_at,o.event_id",
        (profile_home,),
    ).fetchall()
    return {
        "episodes": [
            {
                "episode_id": row["episode_id"],
                "board_session_id": row["session_id"],
                "kind": row["kind"],
                "checkpoint_path": row["checkpoint_path"],
                "native_session_id": row["native_session_id"],
                "profile_home": row["profile_home"],
                "transport": row["transport"],
                "target": json.loads(row["target_json"]),
                "status": row["status"],
                "wake_seq": row["wake_seq"],
                "verify_required": bool(row["verify_required"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in episodes
        ],
        "events": [
            {"status": row["status"], "event": _event_payload(conn, row)}
            for row in event_rows
        ],
    }
