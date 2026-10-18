"""Durable, fenced ownership for Fred root missions.

Mission rows deliberately live beside Kanban in the shared board database but
are not Kanban tasks: the dispatcher cannot mistake a coordination record for
work to execute. Worker subtasks remain ordinary Kanban tasks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any


class MissionError(RuntimeError):
    """Base class for rejected mission lifecycle operations."""


class StaleOwnerError(MissionError):
    """The caller no longer owns the mission."""


class ContractError(MissionError):
    """The proposed completion evidence does not meet the current contract."""


_ACTIVE_EXECUTION_LEASE: ContextVar[str | None] = ContextVar("fred_mission_execution_lease", default=None)


@contextlib.contextmanager
def _write_txn(conn: sqlite3.Connection):
    """Serialize ownership CAS operations before they read the current generation."""
    if conn.in_transaction:
        name = "mission_" + secrets.token_hex(6)
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield
            conn.execute(f"RELEASE SAVEPOINT {name}")
        except BaseException:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            conn.execute(f"RELEASE SAVEPOINT {name}")
            raise
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def ensure_schema(conn: sqlite3.Connection) -> None:
    if conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' "
        "AND name IN ('missions','mission_history','mission_owner_history','mission_execution_leases','mission_validations')"
    ).fetchone()[0] == 5:
        return
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS missions (
        id TEXT PRIMARY KEY,
        ingress_key TEXT NOT NULL UNIQUE,
        original_request TEXT NOT NULL,
        contract_json TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('active','completed','cancelled')),
        owner_kind TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        owner_generation INTEGER NOT NULL,
        owner_expires INTEGER NOT NULL,
        progress_json TEXT NOT NULL,
        children_json TEXT NOT NULL,
        recovery_json TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        completed_at INTEGER
    );
    CREATE TABLE IF NOT EXISTS mission_history (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        mission_id TEXT NOT NULL REFERENCES missions(id),
        kind TEXT NOT NULL,
        actor_kind TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_mission_history_id_seq
        ON mission_history(mission_id, seq);
    CREATE TABLE IF NOT EXISTS mission_owner_history (
        mission_id TEXT NOT NULL REFERENCES missions(id),
        generation INTEGER NOT NULL,
        owner_kind TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        started_at INTEGER NOT NULL,
        ended_at INTEGER,
        PRIMARY KEY(mission_id,generation)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_mission_one_current_owner
        ON mission_owner_history(mission_id) WHERE ended_at IS NULL;
    CREATE TABLE IF NOT EXISTS mission_execution_leases (
        id TEXT PRIMARY KEY,
        mission_id TEXT NOT NULL REFERENCES missions(id),
        actor_kind TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        generation INTEGER NOT NULL,
        tool_name TEXT NOT NULL,
        pid INTEGER NOT NULL,
        thread_id INTEGER,
        started_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_mission_execution_leases_mission
        ON mission_execution_leases(mission_id);
    CREATE TABLE IF NOT EXISTS mission_validations (
        id TEXT PRIMARY KEY,
        mission_id TEXT NOT NULL REFERENCES missions(id),
        generation INTEGER NOT NULL,
        contract_version INTEGER NOT NULL,
        contract_sha256 TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        reviewer_identity TEXT NOT NULL,
        verifier_id TEXT NOT NULL,
        repository_root TEXT NOT NULL,
        head_sha TEXT NOT NULL,
        artifacts_json TEXT NOT NULL,
        checks_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    """)


def current_execution_lease(mission_id: str) -> str | None:
    """Return this tool invocation's lease for a mission, if it has one."""
    return (_ACTIVE_EXECUTION_LEASE.get() or {}).get(str(mission_id))


@contextlib.contextmanager
def execution_scope(*, session_id: str = "", task_id: str = "", tool_name: str = "tool"):
    """Hold durable leases for all active missions owned by this execution.

    The lease starts before any tool hooks or effectful dispatch and is deleted
    in ``finally``. A hard process death leaves the row behind; handoff then
    fails closed until a separately reviewed recovery clears it.
    """
    actor_kind, actor_id = ("task", str(task_id).strip()) if str(task_id).strip() else (
        "session", str(session_id).strip(),
    )
    if not actor_id:
        raise MissionError("mission execution identity is missing")
    token = _ACTIVE_EXECUTION_LEASE.get()
    if token is not None:
        yield token
        return

    from hermes_cli.kanban_db_connect import connect_closing

    leases: dict[str, str] = {}
    with connect_closing() as conn:
        ensure_schema(conn)
        with _write_txn(conn):
            if not authorize_actor(
                conn, task_id=actor_id if actor_kind == "task" else "",
                session_id=actor_id if actor_kind == "session" else "",
            ):
                raise StaleOwnerError("tool execution identity has been fenced by mission ownership")
            rows = conn.execute("SELECT id,owner_kind,owner_id,owner_generation,children_json FROM missions WHERE status='active'").fetchall()
            for row in rows:
                owned = (row["owner_kind"], row["owner_id"]) == (actor_kind, actor_id)
                child = actor_kind == "task" and actor_id in json.loads(row["children_json"])
                if not (owned or child):
                    continue
                lease_id = "x_" + secrets.token_hex(12)
                now = int(time.time())
                conn.execute(
                    "INSERT INTO mission_execution_leases VALUES (?,?,?,?,?,?,?,?,?)",
                    (lease_id, row["id"], actor_kind, actor_id, row["owner_generation"],
                     str(tool_name)[:100], os.getpid(), threading.get_ident(), now),
                )
                leases[str(row["id"])] = lease_id
    context_token = _ACTIVE_EXECUTION_LEASE.set(leases)
    try:
        yield leases
    finally:
        _ACTIVE_EXECUTION_LEASE.reset(context_token)
        if leases:
            with connect_closing() as conn:
                with _write_txn(conn):
                    conn.executemany(
                        "DELETE FROM mission_execution_leases WHERE id=?",
                        [(lease_id,) for lease_id in leases.values()],
                    )


def _assert_quiescent(conn: sqlite3.Connection, mission_id: str) -> None:
    """Reject lifecycle transitions while any other tool call holds a lease."""
    current = current_execution_lease(mission_id)
    rows = conn.execute(
        "SELECT id,tool_name,pid FROM mission_execution_leases WHERE mission_id=?",
        (mission_id,),
    ).fetchall()
    competing = [row for row in rows if row["id"] != current]
    if competing:
        details = ", ".join(f"{row['tool_name']} (pid {row['pid']})" for row in competing[:4])
        raise MissionError(
            f"mission {mission_id} is not quiescent; active or abandoned execution lease(s): {details}. "
            "Handoff, recovery, and completion are refused until they are reconciled."
        )
    mission = conn.execute("SELECT children_json FROM missions WHERE id=?", (mission_id,)).fetchone()
    child_ids = json.loads(mission["children_json"]) if mission else []
    if child_ids:
        placeholders = ",".join("?" for _ in child_ids)
        running = conn.execute(
            f"SELECT id,status FROM tasks WHERE id IN ({placeholders}) AND status='running'",
            child_ids,
        ).fetchall()
        if running:
            raise MissionError(
                "mission has running native Kanban child task(s): " + ", ".join(row["id"] for row in running)
            )


def _validate_child_ids(conn: sqlite3.Connection, mission_id: str, child_ids: list[str] | None) -> None:
    values = list(dict.fromkeys(str(value).strip() for value in (child_ids or []) if str(value).strip()))
    if not values:
        return
    if mission_id in values:
        raise MissionError("mission id cannot be recorded as a Kanban child task")
    placeholders = ",".join("?" for _ in values)
    found = {str(row[0]) for row in conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders})", values,
    )}
    missing = [value for value in values if value not in found]
    if missing:
        raise MissionError("mission child ids must refer to native Kanban tasks: " + ", ".join(missing))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _owner(kind: str, owner_id: str) -> tuple[str, str]:
    kind, owner_id = str(kind).strip(), str(owner_id).strip()
    if kind not in {"session", "task"} or not owner_id:
        raise ValueError("owner must be a non-empty session or task identity")
    return kind, owner_id


def admit(
    conn: sqlite3.Connection, *, ingress_key: str, request: str,
    contract: dict[str, Any], owner_kind: str = "session", owner_id: str,
) -> tuple[str, bool]:
    """Create one mission per transport delivery key; return ``(id, created)``.

    The key is a stable platform/chat/message identity, not a hash of prose. A
    later intentional request with identical wording is therefore independent.
    """
    ensure_schema(conn)
    ingress_key, request = str(ingress_key).strip(), str(request)
    if not ingress_key or not request.strip():
        raise ValueError("ingress_key and complete request are required")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("a non-empty structured mission contract is required")
    if not isinstance(contract.get("outcome"), str) or not contract["outcome"].strip():
        raise ValueError("contract.outcome is required")
    if not isinstance(contract.get("acceptance"), list) or not contract["acceptance"]:
        raise ValueError("contract.acceptance must contain at least one criterion")
    kind, oid = _owner(owner_kind, owner_id)
    now = int(time.time())
    try:
        with _write_txn(conn):
            row = conn.execute("SELECT id,original_request FROM missions WHERE ingress_key=?", (ingress_key,)).fetchone()
            if row:
                if str(row["original_request"]) != request:
                    raise MissionError("transport delivery identifier was reused for different request content")
                return str(row[0]), False
            mission_id = "m_" + secrets.token_hex(12)
            conn.execute(
                "INSERT INTO missions VALUES (?, ?, ?, ?, 1, 'active', ?, ?, 1, ?, ?, '[]', '{}', ?, ?, NULL)",
                (mission_id, ingress_key, request, _json(contract), kind, oid, now + 300, _json([]), now, now),
            )
            conn.execute(
                "INSERT INTO mission_history(mission_id,kind,actor_kind,actor_id,payload_json,created_at) "
                "VALUES (?,?,?,?,?,?)",
                (mission_id, "admitted", kind, oid,
                 _json({"ingress_key": ingress_key, "request_sha256": hashlib.sha256(request.encode()).hexdigest()}), now),
            )
            conn.execute(
                "INSERT INTO mission_owner_history VALUES (?,?,?,?,?,NULL)",
                (mission_id, 1, kind, oid, now),
            )
    except sqlite3.IntegrityError:
        # The unique ingress key is the final duplicate-delivery fence when
        # two gateway/process surfaces raced past the read-before-write check.
        row = conn.execute("SELECT id FROM missions WHERE ingress_key=?", (ingress_key,)).fetchone()
        if row:
            return str(row[0]), False
        raise
    return mission_id, True


_ACTION = re.compile(
    r"\b(implement|build|create|fix|repair|modify|change|update|migrat|synchron|"
    r"sync|automat|deploy|integrat|audit|investigat|monitor|maintain|refactor|"
    r"test|verify|research)\w*\b", re.IGNORECASE)
_SCOPE = re.compile(
    r"\b(task|mission|project|system|pipeline|workflow|service|repository|repo|x|"
    r"agent|telegram|hermes|kanban|github|persistent|ongoing|durable|automatic|"
    r"continu|implementation|evidence|synchron|source|capture|markdown|knowledge|"
    r"processing|content)\w*\b", re.IGNORECASE)
_IMPLEMENTATION = re.compile(
    r"\b(implement|build|create|fix|repair|modify|change|update|migrat|synchron|"
    r"sync|automat|deploy|integrat|refactor|work)\w*\b", re.IGNORECASE)
_USER_CORRECTION = re.compile(
    r"\b(actually|correction|also|instead|include|exclude|scope|acceptance|"
    r"intended|what i mean|what do you mean|i plan to|should|must|don't|do not|"
    r"please ensure|please keep|please change|please add)\b", re.IGNORECASE)


def is_substantial_objective(text: str) -> bool:
    clean = " ".join(str(text or "").split())
    return bool(clean and not clean.startswith(("/", "!")) and _ACTION.search(clean) and _SCOPE.search(clean))


def ingress_key(*, session_id: str, platform: str, delivery_id: str | None,
                turn_id: str | None = None) -> str:
    """Identity by delivery when available; otherwise isolate this exact turn.

    Content is deliberately excluded: two identical requests are independent
    unless transport proves they are the same delivery.
    """
    if delivery_id:
        source = f"delivery:{platform.casefold()}:{session_id}:{delivery_id}"
    else:
        source = f"turn:{session_id}:{turn_id or uuid.uuid4().hex}"
    return "ingress:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def ensure_turn_mission(
    conn: sqlite3.Connection, *, session_id: str, text: str, task_id: str = "",
    ingress_id: str | None = None,
) -> dict[str, Any] | None:
    """Core turn-admission path; plugin hooks are optional observers, not authority.

    A transport delivery ID, or otherwise this turn's ID, deduplicates admission.
    Similarity never attaches a new session to an existing mission. Short user
    steering in the current owner session is recorded as a correction.
    """
    session_id, text = str(session_id).strip(), str(text or "")
    if not session_id or not text.strip():
        return None
    ensure_schema(conn)
    if ingress_id:
        delivered = conn.execute(
            "SELECT id,original_request,owner_kind,owner_id FROM missions WHERE ingress_key=?",
            (str(ingress_id),),
        ).fetchone()
        if delivered is not None:
            if str(delivered["original_request"]) != text:
                raise MissionError("transport delivery identifier was reused for different request content")
            if (delivered["owner_kind"], delivered["owner_id"]) != ("session", session_id):
                raise MissionError(
                    f"this delivery already belongs to mission {delivered['id']} in another owner; "
                    "explicitly resume it rather than starting a duplicate turn"
                )
            return get(conn, delivered["id"])
    owner_kind, owner_id = ("task", str(task_id).strip()) if str(task_id).strip() else ("session", session_id)
    owned = list_for_owner(conn, owner_kind, owner_id)
    if task_id and len(owned) == 1:
        return owned[0]
    # User steering belongs to the current root contract even when it is
    # phrased as a substantial implementation request. Preserve it before the
    # root-admission classifier gets a chance to mint another mission.
    if len(owned) == 1 and owner_kind == "session" and _USER_CORRECTION.search(text):
        mission = owned[0]
        update_user_message(
            conn, mission_id=mission["id"], owner_id=session_id,
            generation=mission["owner_generation"], message=text,
        )
        return get(conn, mission["id"])
    if is_substantial_objective(text):
        if task_id:
            return None  # child workers cannot mint root missions
        if len(owned) == 1:
            raise MissionError(
                f"Fred session already owns active mission {owned[0]['id']}. "
                "Record this as a correction if it changes that outcome, or complete/cancel the current mission before admitting a new root."
            )
        kind = "implementation" if _IMPLEMENTATION.search(text) else "analysis"
        contract = {
            "kind": kind,
            "outcome": text,
            "acceptance": ["Satisfy the complete original request and all stated constraints."],
            "user_updates": [],
        }
        mission_id, _created = admit(
            conn, ingress_key=ingress_id or ingress_key(
                session_id=session_id, platform="unknown", delivery_id=None,
            ), request=text, contract=contract,
            owner_kind="session", owner_id=session_id,
        )
        mission = get(conn, mission_id)
        if mission["owner_kind"] != "session" or mission["owner_id"] != session_id:
            raise MissionError(
                f"Objective already belongs to mission {mission_id} in another session. "
                "A new session cannot take ownership automatically; explicitly hand off the mission."
            )
        return mission
    if len(owned) == 1 and owner_kind == "session":
        return get(conn, owned[0]["id"])
    return None


def list_active(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    ensure_schema(conn)
    ids = conn.execute("SELECT id FROM missions WHERE status='active' ORDER BY created_at,id")
    return [item for row in ids if (item := get(conn, row[0])) is not None]


def update_user_message(
    conn: sqlite3.Connection, *, mission_id: str, owner_id: str,
    generation: int, message: str,
) -> int:
    """Append verbatim in-session steering and invalidate older completion receipts."""
    ensure_schema(conn)
    now = int(time.time())
    with _write_txn(conn):
        row = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        _assert_owner(row, "session", owner_id, generation)
        contract = json.loads(row["contract_json"])
        updates = list(contract.get("user_updates", []))
        updates.append(message)
        contract["user_updates"] = updates
        acceptance = list(contract.get("acceptance", []))
        acceptance.append("Honor this user correction: " + message)
        contract["acceptance"] = acceptance
        version = int(row["contract_version"]) + 1
        conn.execute(
            "UPDATE missions SET contract_json=?,contract_version=?,updated_at=?,owner_expires=? WHERE id=?",
            (_json(contract), version, now, now + 300, mission_id),
        )
        conn.execute(
            "INSERT INTO mission_history(mission_id,kind,actor_kind,actor_id,payload_json,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (mission_id, "user_update", "session", owner_id,
             _json({"message": message, "contract_version": version}), now),
        )
        return version


def get(conn: sqlite3.Connection, mission_id: str) -> dict[str, Any] | None:
    ensure_schema(conn)
    row = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
    if row is None:
        return None
    item = dict(row)
    for key in ("contract_json", "progress_json", "children_json", "recovery_json"):
        parsed = json.loads(item.pop(key))
        item[key.removesuffix("_json")] = parsed
    return item


def list_for_owner(conn: sqlite3.Connection, owner_kind: str, owner_id: str) -> list[dict[str, Any]]:
    ensure_schema(conn)
    rows = conn.execute(
        "SELECT id FROM missions WHERE status='active' AND owner_kind=? AND owner_id=? ORDER BY created_at,id",
        _owner(owner_kind, owner_id),
    )
    return [item for row in rows if (item := get(conn, row[0])) is not None]


def update_contract(
    conn: sqlite3.Connection, *, mission_id: str, expected_owner_kind: str,
    expected_owner_id: str, expected_generation: int, correction: str,
    contract: dict[str, Any],
) -> int:
    """Append a user correction and atomically replace the current acceptance contract."""
    ensure_schema(conn)
    now = int(time.time())
    with _write_txn(conn):
        row = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        _assert_owner(row, expected_owner_kind, expected_owner_id, expected_generation)
        if not correction.strip() or not isinstance(contract, dict) or not contract:
            raise ValueError("correction and replacement contract are required")
        if not isinstance(contract.get("outcome"), str) or not contract["outcome"].strip():
            raise ValueError("replacement contract.outcome is required")
        if not isinstance(contract.get("acceptance"), list) or not contract["acceptance"]:
            raise ValueError("replacement contract.acceptance must contain at least one criterion")
        previous = json.loads(row["contract_json"])
        contract = dict(contract)
        contract["user_updates"] = [*previous.get("user_updates", []), correction]
        contract["acceptance"] = [
            *contract["acceptance"], "Honor this user correction: " + correction,
        ]
        version = int(row["contract_version"]) + 1
        conn.execute(
            "UPDATE missions SET contract_json=?,contract_version=?,updated_at=?,owner_expires=? WHERE id=?",
            (_json(contract), version, now, now + 300, mission_id),
        )
        conn.execute(
            "INSERT INTO mission_history(mission_id,kind,actor_kind,actor_id,payload_json,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (mission_id, "corrected", expected_owner_kind, expected_owner_id,
             _json({"correction": correction, "contract_version": version}), now),
        )
        return version


def record_progress(
    conn: sqlite3.Connection, *, mission_id: str, owner_kind: str, owner_id: str,
    generation: int, progress: dict[str, Any], child_ids: list[str] | None = None,
    recovery: dict[str, Any] | None = None,
) -> None:
    ensure_schema(conn)
    now = int(time.time())
    with _write_txn(conn):
        row = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        _assert_owner(row, owner_kind, owner_id, generation)
        _validate_child_ids(conn, mission_id, child_ids)
        children = list(dict.fromkeys([*json.loads(row["children_json"]), *(child_ids or [])]))
        conn.execute(
            "UPDATE missions SET progress_json=?,children_json=?,recovery_json=?,updated_at=?,owner_expires=? WHERE id=?",
            (_json(progress), _json(children), _json(recovery or json.loads(row["recovery_json"])), now, now + 300, mission_id),
        )
        conn.execute(
            "INSERT INTO mission_history(mission_id,kind,actor_kind,actor_id,payload_json,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (mission_id, "progress", owner_kind, owner_id,
             _json({"progress": progress, "children": child_ids or [], "recovery": recovery or {}}), now),
        )


def handoff(
    conn: sqlite3.Connection, *, mission_id: str, expected_owner_kind: str,
    expected_owner_id: str, expected_generation: int, new_owner_kind: str,
    new_owner_id: str, progress: dict[str, Any], child_ids: list[str] | None = None,
    recovery: dict[str, Any] | None = None,
) -> int:
    """Atomically fence the old owner and install a new owner.

    No external worker should be started until this commits. A child task is
    merely recorded in ``children`` and never changes root ownership.
    """
    ensure_schema(conn)
    new_kind, new_id = _owner(new_owner_kind, new_owner_id)
    now = int(time.time())
    with _write_txn(conn):
        row = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        _assert_owner(row, expected_owner_kind, expected_owner_id, expected_generation)
        _assert_quiescent(conn, mission_id)
        _validate_child_ids(conn, mission_id, child_ids)
        if (row["owner_kind"], row["owner_id"]) == (new_kind, new_id):
            raise ValueError("handoff target must differ from current owner")
        if new_kind == "task":
            target = conn.execute("SELECT id,status,assignee FROM tasks WHERE id=?", (new_id,)).fetchone()
            if target is None or target["status"] != "triage" or str(target["assignee"] or "").casefold() != "fred":
                raise MissionError("root task handoff target must be a non-runnable triage task assigned to Fred")
            # Both the root-owner change and dispatcher promotion commit in the
            # same SQLite transaction. A dispatcher cannot observe a ready task
            # while the old session still owns the mission.
            promoted = conn.execute(
                "UPDATE tasks SET status='ready',started_at=COALESCE(started_at,?) "
                "WHERE id=? AND status='triage'", (now, new_id),
            ).rowcount
            if promoted != 1:
                raise MissionError("task handoff target changed state before ownership transfer")
        generation = int(row["owner_generation"]) + 1
        children = list(dict.fromkeys([*json.loads(row["children_json"]), *(child_ids or [])]))
        recovery_doc = recovery or json.loads(row["recovery_json"])
        changed = conn.execute(
            "UPDATE missions SET owner_kind=?,owner_id=?,owner_generation=?,owner_expires=?,progress_json=?,"
            "children_json=?,recovery_json=?,updated_at=? WHERE id=? AND owner_generation=? AND status='active'",
            (new_kind, new_id, generation, now + 300, _json(progress), _json(children), _json(recovery_doc), now,
             mission_id, expected_generation),
        ).rowcount
        if changed != 1:
            raise StaleOwnerError("mission ownership changed during handoff")
        closed = conn.execute(
            "UPDATE mission_owner_history SET ended_at=? WHERE mission_id=? AND generation=? AND ended_at IS NULL",
            (now, mission_id, expected_generation),
        ).rowcount
        if closed != 1:
            raise StaleOwnerError("current mission owner record was missing")
        conn.execute(
            "INSERT INTO mission_owner_history VALUES (?,?,?,?,?,NULL)",
            (mission_id, generation, new_kind, new_id, now),
        )
        conn.execute(
            "INSERT INTO mission_history(mission_id,kind,actor_kind,actor_id,payload_json,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (mission_id, "handoff", expected_owner_kind, expected_owner_id,
             _json({"from": [expected_owner_kind, expected_owner_id],
                    "to": [new_kind, new_id], "generation": generation,
                    "progress": progress, "children": child_ids or [], "recovery": recovery_doc}), now),
        )
    return generation


def authorize(conn: sqlite3.Connection, *, owner_kind: str, owner_id: str) -> bool:
    """True only for an active current owner; past owners are fenced by identity."""
    ensure_schema(conn)
    kind, oid = _owner(owner_kind, owner_id)
    return conn.execute(
        "SELECT 1 FROM missions WHERE status='active' AND owner_kind=? AND owner_id=? LIMIT 1",
        (kind, oid),
    ).fetchone() is not None


def authorize_session(conn: sqlite3.Connection, session_id: str) -> bool:
    """Fail closed for a session that handed off an active mission.

    Sessions with no mission history remain ordinary sessions. If one session
    both owns a mission and has an outstanding transferred mission, it is
    conservatively fenced until that mission completes; use separate sessions
    for concurrently coordinated missions.
    """
    ensure_schema(conn)
    session_id = str(session_id).strip()
    if not session_id:
        return True
    stale = conn.execute(
        "SELECT 1 FROM mission_history h JOIN missions m ON m.id=h.mission_id "
        "WHERE h.kind='handoff' AND h.actor_kind='session' AND h.actor_id=? "
        "AND m.status='active' LIMIT 1", (session_id,),
    ).fetchone()
    return stale is None


def authorize_actor(conn: sqlite3.Connection, *, session_id: str = "", task_id: str = "") -> bool:
    """Check both interactive sessions and dispatcher task identities at the tool boundary."""
    ensure_schema(conn)
    checks = []
    if session_id.strip():
        checks.append(("session", session_id.strip()))
    if task_id.strip():
        checks.append(("task", task_id.strip()))
    for kind, identity in checks:
        stale = conn.execute(
            "SELECT 1 FROM mission_owner_history h JOIN missions m ON m.id=h.mission_id "
            "WHERE h.owner_kind=? AND h.owner_id=? AND h.ended_at IS NOT NULL "
            "AND m.status='active' LIMIT 1", (kind, identity),
        ).fetchone()
        if stale is not None:
            return False
    return True


def recover(
    conn: sqlite3.Connection, *, mission_id: str, expected_generation: int,
    new_owner_kind: str, new_owner_id: str, recovery_actor: str,
    now: int | None = None,
) -> int:
    """Explicitly recover an expired owner lease; never auto-take over on session start."""
    ensure_schema(conn)
    new_kind, new_id = _owner(new_owner_kind, new_owner_id)
    timestamp = int(time.time()) if now is None else int(now)
    with _write_txn(conn):
        row = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        if row is None or row["status"] != "active":
            raise StaleOwnerError("mission is missing or no longer active")
        if int(row["owner_generation"]) != int(expected_generation):
            raise StaleOwnerError("mission ownership changed before recovery")
        if int(row["owner_expires"]) > timestamp:
            raise MissionError("mission owner lease has not expired; takeover is not allowed")
        _assert_quiescent(conn, mission_id)
        if row["owner_kind"] == "task":
            task = conn.execute(
                "SELECT status,current_run_id FROM tasks WHERE id=?", (row["owner_id"],),
            ).fetchone()
            if task is not None and task["status"] == "running" and task["current_run_id"]:
                run = conn.execute(
                    "SELECT status,worker_pid,worker_started_at FROM task_runs WHERE id=?",
                    (task["current_run_id"],),
                ).fetchone()
                if run is not None and run["status"] == "running":
                    if not run["worker_pid"]:
                        raise MissionError("mission worker is still being launched; recovery is refused")
                    from hermes_cli.kanban_db_dispatch import _worker_alive
                    if _worker_alive(run["worker_pid"], run["worker_started_at"]):
                        raise MissionError("mission worker is still alive; recovery is refused")
                    # Fence the dead execution task before transferring mission
                    # ownership so the dispatcher cannot revive it as a second root.
                    conn.execute(
                        "UPDATE tasks SET status='blocked',claim_lock=NULL,claim_expires=NULL,"
                        "worker_pid=NULL,worker_started_at=NULL,current_run_id=NULL WHERE id=? AND status='running'",
                        (row["owner_id"],),
                    )
                    conn.execute(
                        "UPDATE task_runs SET status='crashed',ended_at=?,outcome='crashed',"
                        "error='Mission recovery after confirmed worker exit' "
                        "WHERE id=? AND status='running'",
                        (timestamp, task["current_run_id"]),
                    )
        generation = int(row["owner_generation"]) + 1
        changed = conn.execute(
            "UPDATE missions SET owner_kind=?,owner_id=?,owner_generation=?,owner_expires=?,updated_at=? "
            "WHERE id=? AND owner_generation=? AND owner_expires<=? AND status='active'",
            (new_kind, new_id, generation, timestamp + 300, timestamp, mission_id,
             expected_generation, timestamp),
        ).rowcount
        if changed != 1:
            raise StaleOwnerError("mission ownership changed during recovery")
        closed = conn.execute(
            "UPDATE mission_owner_history SET ended_at=? WHERE mission_id=? AND generation=? AND ended_at IS NULL",
            (timestamp, mission_id, expected_generation),
        ).rowcount
        if closed != 1:
            raise StaleOwnerError("current mission owner record was missing")
        conn.execute(
            "INSERT INTO mission_owner_history VALUES (?,?,?,?,?,NULL)",
            (mission_id, generation, new_kind, new_id, timestamp),
        )
        conn.execute(
            "INSERT INTO mission_history(mission_id,kind,actor_kind,actor_id,payload_json,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (mission_id, "recovered", "session", recovery_actor,
             _json({"from": [row["owner_kind"], row["owner_id"]], "to": [new_kind, new_id],
                    "generation": generation}), timestamp),
        )
    return generation


def complete(
    conn: sqlite3.Connection, *, mission_id: str, owner_kind: str, owner_id: str,
    generation: int, contract_version: int, validation_id: str,
) -> None:
    """Close only against a current receipt written by the trusted validator."""
    ensure_schema(conn)
    now = int(time.time())
    with _write_txn(conn):
        row = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        _assert_owner(row, owner_kind, owner_id, generation)
        _assert_quiescent(conn, mission_id)
        if int(row["contract_version"]) != int(contract_version):
            raise ContractError("completion receipt targets a stale contract version")
        contract = json.loads(row["contract_json"])
        contract_sha = hashlib.sha256(_json(contract).encode()).hexdigest()
        receipt = conn.execute("SELECT * FROM mission_validations WHERE id=?", (str(validation_id),)).fetchone()
        if receipt is None:
            raise ContractError("completion requires a trusted validation receipt; model-supplied evidence is not accepted")
        if (receipt["mission_id"], int(receipt["generation"]), int(receipt["contract_version"]),
            receipt["actor_id"], receipt["contract_sha256"]) != (
            mission_id, int(generation), int(contract_version), str(owner_id), contract_sha,
        ):
            raise ContractError("validation receipt is stale or belongs to a different mission, owner, or contract")
        artifacts = json.loads(receipt["artifacts_json"])
        checks = json.loads(receipt["checks_json"])
        check_ids = {item.get("check_id") for item in checks if isinstance(item, dict) and item.get("exit_code") == 0}
        if not {"python-syntax", "ruff"}.issubset(check_ids):
            raise ContractError("validation receipt is missing successful trusted checks")
        root = Path(receipt["repository_root"]).resolve()
        for artifact in artifacts:
            path = (root / artifact["path"]).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ContractError("validation receipt artifact escapes its repository root") from exc
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get("sha256"):
                raise ContractError("validation artifact is missing or stale: " + str(artifact.get("path")))
        child_ids = json.loads(row["children_json"])
        if child_ids:
            placeholders = ",".join("?" for _ in child_ids)
            child_rows = conn.execute(
                f"SELECT id,status FROM tasks WHERE id IN ({placeholders})", child_ids,
            ).fetchall()
            statuses = {str(child["id"]): str(child["status"]) for child in child_rows}
            if set(statuses) != set(child_ids) or any(status not in {"done", "archived"} for status in statuses.values()):
                raise ContractError("every recorded Kanban child task must exist and be done or archived")
        changed = conn.execute(
            "UPDATE missions SET status='completed',completed_at=?,updated_at=? WHERE id=? AND owner_generation=? AND status='active'",
            (now, now, mission_id, generation),
        ).rowcount
        if changed != 1:
            raise StaleOwnerError("mission ownership changed during completion")
        conn.execute(
            "INSERT INTO mission_history(mission_id,kind,actor_kind,actor_id,payload_json,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (mission_id, "completed", owner_kind, owner_id,
             _json({"validation_id": validation_id, "reviewer_identity": receipt["reviewer_identity"],
                    "verifier_id": receipt["verifier_id"], "artifacts": artifacts, "checks": checks}), now),
        )


def _assert_owner(row: sqlite3.Row | None, kind: str, oid: str, generation: int) -> None:
    if row is None or row["status"] != "active":
        raise StaleOwnerError("mission is missing or no longer active")
    if (row["owner_kind"], row["owner_id"], int(row["owner_generation"])) != (
        kind, oid, int(generation),
    ):
        raise StaleOwnerError("caller is not the current mission owner")
