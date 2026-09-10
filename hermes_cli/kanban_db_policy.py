"""Optional board-owned policy, checked before promotion and every native claim.

The operator's command runs under the dispatch lock, never a SQLite transaction.
Its answer is check-time evidence, not a cross-system transaction or reusable permit.
"""
from __future__ import annotations

import contextlib
import json
import math
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@dataclass(frozen=True)
class BoardPolicy:
    db_path: Path
    board: str
    metadata_path: Path
    config: dict | None


def validate_policy(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"command", "timeout_seconds"}:
        raise ValueError("pre_claim requires command and timeout_seconds")
    command, timeout = value["command"], value["timeout_seconds"]
    if (not isinstance(command, list) or not command
            or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in command)
            or not Path(command[0]).is_absolute()):
        raise ValueError("pre_claim command must be fixed argv with an absolute executable")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 60:
        raise ValueError("pre_claim timeout_seconds must be between 1 and 60")
    return value


def board_policy(conn: sqlite3.Connection) -> BoardPolicy | None:
    row = conn.execute("PRAGMA database_list").fetchone()
    if not row or not row[2]:  # In-memory databases have no native board metadata.
        return None
    path = Path(row[2]).resolve()
    if path.name != "kanban.db":
        return None
    parent = path.parent
    if parent.parent.name == "boards" and parent.parent.parent.name == "kanban":
        board = parent.name
        metadata = parent / "board.json"
    else:
        board = kb.DEFAULT_BOARD
        metadata = parent / "kanban" / "boards" / board / "board.json"
    try:
        value = kb.read_board_metadata(board, strict=True, metadata_path=metadata)
        if "pre_claim" not in value:
            return None
        config = validate_policy(value["pre_claim"])
    except (OSError, ValueError, UnicodeError):
        config = None  # A malformed/unreadable existing file must not disable policy.
    return BoardPolicy(path, board, metadata, config)


def _snapshot(conn: sqlite3.Connection, task_id: str):
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None:
        return None
    event = conn.execute(
        "SELECT id, kind, payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    parents = conn.execute(
        "SELECT p.id, p.status FROM tasks p JOIN task_links l ON l.parent_id=p.id "
        "WHERE l.child_id=? ORDER BY p.id", (task_id,),
    ).fetchall()
    return dict(row), dict(event) if event else None, [tuple(p) for p in parents]


def _decision(policy: BoardPolicy, row: dict, phase: str) -> tuple[bool, str]:
    if policy.config is None:
        return False, "pre_claim_invalid_configuration"
    packet = {
        "board": policy.board, "phase": phase, "source_status": row["status"],
        "task": {key: row[key] for key in ("id", "status", "assignee", "project_id", "idempotency_key")},
    }
    try:
        result = subprocess.run(
            policy.config["command"], input=json.dumps(packet).encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=policy.config["timeout_seconds"], check=False, shell=False,
        )
        if result.returncode != 0:
            return False, "pre_claim_command_failed"
        if len(result.stdout) > 4096:
            return False, "pre_claim_invalid_response"
        value = json.loads(result.stdout)
        if (not isinstance(value, dict) or set(value) != {"allow", "reason"}
                or type(value["allow"]) is not bool or not isinstance(value["reason"], str)
                or len(value["reason"].encode("utf-8")) > 512):
            return False, "pre_claim_invalid_response"
        return value["allow"], value["reason"]
    except subprocess.TimeoutExpired:
        return False, "pre_claim_timeout"
    except (OSError, ValueError, UnicodeError):
        return False, "pre_claim_command_error"


def _eligible(conn, snapshot, status, phase, failure_limit):
    if snapshot is None:
        return False
    row = snapshot[0]
    if row["status"] != status or row["claim_lock"] is not None:
        return False
    if phase == "promote" and status == "blocked":
        limit = row["max_retries"] if row["max_retries"] is not None else failure_limit
        return not kb._has_sticky_block(conn, row["id"]) and int(row["consecutive_failures"] or 0) < limit
    return True


def _wait(conn, snapshot, reason):
    row, event, _ = snapshot
    lane = row["status"] if row["status"] in ("ready", "review") else kb._resume_status_from_events(conn, row["id"])
    payload = {"source": "pre_claim", "reason": reason, "source_status": lane}
    if row["status"] == "todo" and event and event["kind"] == "dependency_wait":
        if kb._json_dict(event["payload"]) == payload:
            return
    conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (row["id"],))
    kb._append_event(conn, row["id"], "dependency_wait", payload)


@contextlib.contextmanager
def policy_transaction(conn, task_id, status, *, phase="claim", failure_limit=kb.DEFAULT_FAILURE_LIMIT):
    """Yield permission inside the caller's native write boundary, holding one FD.

    An unavailable lock or caller-owned transaction leaves the task untouched. A
    verified denial records Todo only after snapshot revalidation in our own txn.
    """
    kb._assert_not_delegated_child_mutation()
    policy = board_policy(conn)
    if policy is None:
        with kbc.write_txn(conn):
            yield True
        return
    if conn.in_transaction:
        yield False
        return
    with kbc._dispatch_tick_lock(policy.db_path, reentrant=True, required=True) as held:
        if not held:
            yield False
            return
        before = _snapshot(conn, task_id)
        if not _eligible(conn, before, status, phase, failure_limit):
            yield False
            return
        allow, reason = _decision(policy, before[0], phase)
        with kbc.write_txn(conn):
            if before != _snapshot(conn, task_id) or policy != board_policy(conn):
                yield False
            elif not allow:
                _wait(conn, before, reason)
                yield False
            else:
                yield True


def recompute_ready(conn, policy: BoardPolicy, failure_limit: int) -> int:
    if conn.in_transaction:
        return 0
    promoted = 0
    with kbc._dispatch_tick_lock(policy.db_path, reentrant=True, required=True) as held:
        if not held:
            return 0
        rows = conn.execute(
            "SELECT id, status FROM tasks WHERE status IN ('todo','blocked') "
            "ORDER BY priority DESC, created_at ASC, id ASC",
        ).fetchall()
        for row in rows:
            if board_policy(conn) != policy:
                break
            with policy_transaction(
                conn, row["id"], row["status"], phase="promote", failure_limit=failure_limit,
            ) as allowed:
                if not allowed or not kb._parents_satisfied(conn, row["id"]):
                    continue
                lane = kb._resume_status_from_events(conn, row["id"])
                conn.execute("UPDATE tasks SET status=? WHERE id=?", (lane, row["id"]))
                kb._append_event(conn, row["id"], "promoted", {"status": lane} if lane != "ready" else None)
                promoted += 1
    return promoted
