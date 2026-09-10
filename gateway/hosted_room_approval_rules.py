"""Remember exact approval operations for one hosted group member, never a whole profile."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from typing import Any

from tools.approval_operation import valid_operation_context, valid_operation_key
from gateway.hosted_room_messaging_approvals import MessagingApprovalError


MAX_RULES_PER_MEMBER = 32
MAX_RULES_TOTAL = 1024
AUTO_COMMAND_PREFIX = "approval-rule:"
_SCOPE = ("room_id", "authority_gateway_id", "authority_epoch", "member_id", "profile", "target_digest", "operation_key")
_REQUEST = ("room_id", "authority_gateway_id", "authority_epoch", "member_id", "task_id", "execution_generation", "request_id")


class ApprovalRuleError(MessagingApprovalError):
    """A permission cannot be created or applied in its current scope."""


def _exists(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_room_approval_rules'").fetchone() is not None


def _write_required(conn: sqlite3.Connection) -> None:
    if not conn.in_transaction:
        raise ApprovalRuleError("permission changes require the caller's transaction")


def _ensure(conn: sqlite3.Connection) -> None:
    _write_required(conn)
    conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_approval_rules (
        rule_id TEXT PRIMARY KEY, room_id TEXT NOT NULL, authority_gateway_id TEXT NOT NULL,
        authority_epoch INTEGER NOT NULL, member_id TEXT NOT NULL, profile TEXT NOT NULL,
        target_digest TEXT NOT NULL, operation_key TEXT NOT NULL, command_text TEXT NOT NULL,
        description TEXT NOT NULL, context_text TEXT NOT NULL DEFAULT '',
        state TEXT NOT NULL CHECK(state IN ('pending','active','revoked')),
        generation INTEGER NOT NULL, grant_command_id TEXT NOT NULL,
        created_at REAL NOT NULL, updated_at REAL NOT NULL, last_used_at REAL
    )""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(hosted_room_approval_rules)")}
    if "context_text" not in columns:
        conn.execute("ALTER TABLE hosted_room_approval_rules ADD COLUMN context_text TEXT NOT NULL DEFAULT ''")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_hosted_approval_rule_room
        ON hosted_room_approval_rules(room_id, member_id, state)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_approval_rule_commands (
        command_id TEXT PRIMARY KEY, rule_id TEXT NOT NULL, generation INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('grant','use'))
    )""")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _binding(conn: sqlite3.Connection, pending: Mapping[str, Any]) -> dict[str, Any]:
    room = conn.execute(
        "SELECT members_json, authority_gateway_id, authority_epoch, disbanded_at FROM hosted_rooms WHERE room_id=?",
        (pending["room_id"],),
    ).fetchone()
    if (room is None or room["disbanded_at"] is not None
            or room["authority_gateway_id"] != pending["authority_gateway_id"]
            or room["authority_epoch"] != pending["authority_epoch"]):
        raise ApprovalRuleError("Group Chat authority is no longer current")
    member = next((m for m in json.loads(room["members_json"])
                   if str(m.get("member_id") or m.get("profile")) == pending["member_id"]), None)
    if member is None:
        raise ApprovalRuleError("Bot is no longer in this Group Chat")
    target = member.get("target") or {}
    profile = str(target.get("profile") or member.get("profile") or "")
    if not profile or profile != pending["profile"]:
        raise ApprovalRuleError("Bot execution profile changed")
    target_digest = _digest({
        "profile": profile,
        "target": {key: target.get(key) for key in (
            "kind", "installation_id", "peer_id", "profile", "capability_digest")},
    })
    return {key: pending[key] for key in _SCOPE[:5]} | {"target_digest": target_digest}


def _current_scope(conn: sqlite3.Connection, pending: Mapping[str, Any]) -> dict[str, Any]:
    from gateway.hosted_room_messaging_approvals import _require_observer_lease

    key = (pending.get("approval") or {}).get("remember_key")
    context = (pending.get("approval") or {}).get("remember_context")
    if (not valid_operation_key(key) or not valid_operation_context(context)
            or "remember" not in (pending.get("approval") or {}).get("choices", [])):
        raise ApprovalRuleError("This request cannot be remembered safely")
    if pending.get("observer_generation", "legacy") == "legacy":
        raise ApprovalRuleError("A current approval observer is required")
    _require_observer_lease(conn, pending, now=time.time())
    row = conn.execute(
        "SELECT * FROM hosted_room_pending_approvals WHERE room_id=? AND member_id=?",
        (pending["room_id"], pending["member_id"]),
    ).fetchone()
    if (row is None or any(row[field] != pending[field] for field in _REQUEST)
            or row["remember_key"] != key or row["remember_context"] != context
            or row["profile"] != pending["profile"]):
        raise ApprovalRuleError("Approval changed before the permission was recorded")
    task = conn.execute(
        "SELECT status, execution_generation, cancel_id FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?",
        (pending["room_id"], pending["task_id"]),
    ).fetchone()
    if (task is None or task["status"] != "running" or task["cancel_id"] is not None
            or task["execution_generation"] != pending["execution_generation"]):
        raise ApprovalRuleError("The approval's task is no longer running")
    return _binding(conn, pending) | {"operation_key": key}


def _link(conn: sqlite3.Connection, command_id: str) -> sqlite3.Row | None:
    if not _exists(conn):
        return None
    return conn.execute("SELECT * FROM hosted_room_approval_rule_commands WHERE command_id=?", (command_id,)).fetchone()


def _save_link(conn: sqlite3.Connection, command_id: str, rule: Mapping[str, Any], kind: str) -> None:
    existing = _link(conn, command_id)
    identity = (rule["rule_id"], rule["generation"], kind)
    if existing is not None:
        if tuple(existing[field] for field in ("rule_id", "generation", "kind")) != identity:
            raise ApprovalRuleError("Approval command has a different permission intent")
        return
    conn.execute("INSERT INTO hosted_room_approval_rule_commands VALUES (?, ?, ?, ?)", (command_id, *identity))


def is_remembered_grant(conn: sqlite3.Connection, command_id: str) -> bool:
    row = _link(conn, command_id)
    return row is not None and row["kind"] == "grant"


def stage_rule(conn: sqlite3.Connection, pending: Mapping[str, Any], command_id: str) -> dict[str, Any]:
    """Freeze intent in the same transaction as the original one-time approval."""
    _write_required(conn)
    scope = _current_scope(conn, pending)
    _ensure(conn)
    rule_id = _digest(scope)
    old = conn.execute("SELECT * FROM hosted_room_approval_rules WHERE rule_id=?", (rule_id,)).fetchone()
    if old is not None and old["state"] == "revoked" and old["grant_command_id"] == command_id:
        raise ApprovalRuleError("This remembered permission was removed")
    if old is not None and (old["state"] == "active" or old["grant_command_id"] == command_id):
        _save_link(conn, command_id, old, "grant")
        return dict(old)
    now = time.time()
    prune_rules(conn, now=now)
    per_member = conn.execute(
        "SELECT COUNT(*) FROM hosted_room_approval_rules WHERE room_id=? AND member_id=? AND state!='revoked'",
        (scope["room_id"], scope["member_id"]),
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM hosted_room_approval_rules WHERE state!='revoked'").fetchone()[0]
    if per_member >= MAX_RULES_PER_MEMBER or total >= MAX_RULES_TOTAL:
        raise ApprovalRuleError("Too many remembered approvals. Remove one before adding another.")
    generation = int(old["generation"]) + 1 if old is not None else 1
    approval = pending["approval"]
    conn.execute("""INSERT INTO hosted_room_approval_rules (
        rule_id, room_id, authority_gateway_id, authority_epoch, member_id, profile, target_digest, operation_key,
        command_text, description, context_text, state, generation, grant_command_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
        ON CONFLICT(rule_id) DO UPDATE SET state='pending', generation=excluded.generation,
        grant_command_id=excluded.grant_command_id, command_text=excluded.command_text,
        description=excluded.description, context_text=excluded.context_text, updated_at=excluded.updated_at""",
        (rule_id, *(scope[field] for field in _SCOPE), approval["command"], approval["description"],
         approval["remember_context"], generation, command_id, now, now))
    rule = dict(conn.execute("SELECT * FROM hosted_room_approval_rules WHERE rule_id=?", (rule_id,)).fetchone())
    _save_link(conn, command_id, rule, "grant")
    return rule


def matching_rule(conn: sqlite3.Connection, pending: Mapping[str, Any]) -> dict[str, Any] | None:
    if not _exists(conn):
        return None
    scope = _current_scope(conn, pending)
    row = conn.execute("SELECT * FROM hosted_room_approval_rules WHERE rule_id=? AND state='active'", (_digest(scope),)).fetchone()
    return dict(row) if row is not None else None


def bind_rule_use(conn: sqlite3.Connection, pending: Mapping[str, Any], command_id: str, rule: Mapping[str, Any]) -> None:
    _write_required(conn)
    if not command_id.startswith(AUTO_COMMAND_PREFIX):
        raise ApprovalRuleError("automatic approvals require a reserved command identity")
    current = matching_rule(conn, pending)
    if current is None or any(current[field] != rule[field] for field in ("rule_id", "generation")):
        raise ApprovalRuleError("Remembered permission changed before use")
    _save_link(conn, command_id, current, "use")


def require_live_rule_use(conn: sqlite3.Connection, pending: Mapping[str, Any], command_id: str) -> None:
    """Run before an automatic decision crosses the existing application boundary."""
    link = _link(conn, command_id)
    if link is None and command_id.startswith(AUTO_COMMAND_PREFIX):
        raise ApprovalRuleError("Remembered permission was removed or changed")
    if link is None or link["kind"] != "use":
        return
    current = matching_rule(conn, pending)
    if current is None or any(current[field] != link[field] for field in ("rule_id", "generation")):
        raise ApprovalRuleError("Remembered permission was removed or changed")


def discard_unstarted_use(conn: sqlite3.Connection, command_id: str) -> bool:
    _write_required(conn)
    link = _link(conn, command_id)
    if not command_id.startswith(AUTO_COMMAND_PREFIX) and (link is None or link["kind"] != "use"):
        return False
    return bool(conn.execute("DELETE FROM hosted_room_messaging_approval_commands WHERE command_id=? "
                             "AND state='pending' AND application_started_at IS NULL", (command_id,)).rowcount)


def complete_rule_decision(conn: sqlite3.Connection, command_id: str, result: str) -> None:
    _write_required(conn)
    link = _link(conn, command_id)
    if link is None or result != "Approved once.":
        return
    receipt = conn.execute("SELECT state, choice, result_text, application_started_at "
                           "FROM hosted_room_messaging_approval_commands WHERE command_id=?", (command_id,)).fetchone()
    if (receipt is None or receipt["state"] != "completed" or receipt["choice"] != "once"
            or receipt["result_text"] != result or receipt["application_started_at"] is None):
        return
    rule = conn.execute("SELECT * FROM hosted_room_approval_rules WHERE rule_id=?", (link["rule_id"],)).fetchone()
    if rule is None or rule["generation"] != link["generation"] or rule["state"] == "revoked":
        return
    try:
        current = _binding(conn, rule)
    except ApprovalRuleError:
        return
    if current["target_digest"] != rule["target_digest"]:
        return
    now = time.time()
    if link["kind"] == "grant":
        conn.execute("UPDATE hosted_room_approval_rules SET state='active', updated_at=? "
                     "WHERE rule_id=? AND state='pending' AND grant_command_id=? AND generation=?",
                     (now, rule["rule_id"], command_id, link["generation"]))
    else:
        conn.execute("UPDATE hosted_room_approval_rules SET last_used_at=? WHERE rule_id=? AND generation=?",
                     (now, rule["rule_id"], link["generation"]))


def list_rules(conn: sqlite3.Connection, room_id: str, *, include_pending: bool = False) -> list[dict[str, Any]]:
    if not _exists(conn):
        return []
    rows = conn.execute("SELECT * FROM hosted_room_approval_rules WHERE room_id=? AND "
                        "(state='active' OR (? AND state='pending')) "
                        "ORDER BY member_id, created_at, rule_id LIMIT ?", (room_id, include_pending, MAX_RULES_TOTAL)).fetchall()
    result = []
    for row in rows:
        try:
            if _binding(conn, row)["target_digest"] == row["target_digest"]:
                result.append(dict(row))
        except ApprovalRuleError:
            continue
    return result


def granted_rule(conn: sqlite3.Connection, command_id: str) -> dict[str, Any] | None:
    link = _link(conn, command_id)
    if link is None or link["kind"] != "grant":
        return None
    row = conn.execute("SELECT * FROM hosted_room_approval_rules WHERE rule_id=? AND generation=? AND state='active'",
                       (link["rule_id"], link["generation"])).fetchone()
    if row is None:
        return None
    try:
        return dict(row) if _binding(conn, row)["target_digest"] == row["target_digest"] else None
    except ApprovalRuleError:
        return None


def revoke_rule(conn: sqlite3.Connection, room_id: str, rule_id: str) -> bool:
    """Revoke future/unstarted uses; an already accepted decision may still finish."""
    _write_required(conn)
    if not _exists(conn):
        return False
    changed = conn.execute("UPDATE hosted_room_approval_rules SET state='revoked', generation=generation+1, updated_at=? "
                           "WHERE rule_id=? AND room_id=? AND state!='revoked'", (time.time(), rule_id, room_id))
    if not changed.rowcount:
        return False
    conn.execute("""DELETE FROM hosted_room_messaging_approval_commands WHERE state='pending'
        AND application_started_at IS NULL AND command_id IN (
            SELECT command_id FROM hosted_room_approval_rule_commands WHERE rule_id=? AND kind='use')""", (rule_id,))
    return True


def prune_rules(conn: sqlite3.Connection, *, now: float) -> None:
    """Keep permission payloads and use links inside the existing journal's retention."""
    _write_required(conn)
    if not _exists(conn):
        return
    conn.execute("""UPDATE hosted_room_approval_rules SET state='revoked', generation=generation+1, updated_at=?
        WHERE state!='revoked' AND NOT EXISTS (
            SELECT 1 FROM hosted_rooms AS room WHERE room.room_id=hosted_room_approval_rules.room_id
            AND room.authority_gateway_id=hosted_room_approval_rules.authority_gateway_id
            AND room.authority_epoch=hosted_room_approval_rules.authority_epoch AND room.disbanded_at IS NULL)""", (now,))
    conn.execute("DELETE FROM hosted_room_approval_rule_commands WHERE NOT EXISTS "
                 "(SELECT 1 FROM hosted_room_messaging_approval_commands AS decision "
                 "WHERE decision.command_id=hosted_room_approval_rule_commands.command_id)")
    conn.execute("DELETE FROM hosted_room_approval_rules WHERE state!='active' AND updated_at<?", (now - 7 * 86400,))


def purge_room_rules(conn: sqlite3.Connection, room_ids: tuple[str, ...]) -> None:
    """Purge dependent links before their room-scoped permission payloads."""
    _write_required(conn)
    if not room_ids or not _exists(conn):
        return
    placeholders = ",".join("?" for _ in room_ids)
    conn.execute(f"DELETE FROM hosted_room_approval_rule_commands WHERE rule_id IN "
                 f"(SELECT rule_id FROM hosted_room_approval_rules WHERE room_id IN ({placeholders}))", room_ids)
    conn.execute(f"DELETE FROM hosted_room_approval_rules WHERE room_id IN ({placeholders})", room_ids)
