"""Board-local immutable recipe receipts and atomic native task composition."""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import time

from hermes_cli.kanban_recipes import RecipeError, canonical

# Kept import-light so the native fresh schema can include these tables without
# a facade cycle. Execute statements individually when composing migrations.
RECIPE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recipe_instances (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    recipe_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    definition_digest TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    bindings_json TEXT NOT NULL,
    effective_plan_json TEXT NOT NULL,
    created_by TEXT,
    created_at INTEGER NOT NULL,
    imported INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS recipe_instance_tasks (
    instance_id TEXT NOT NULL REFERENCES recipe_instances(id) ON DELETE RESTRICT,
    node_key TEXT NOT NULL,
    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(id) ON DELETE RESTRICT,
    PRIMARY KEY(instance_id, node_key)
);
"""


def migrate_recipe_tables(conn):
    for statement in RECIPE_SCHEMA_SQL.split(";"):
        if statement.strip():
            conn.execute(statement)


def _digest(value):
    # Each component was validated at its own document boundary. Wrapping
    # inputs in a request envelope must not reduce their allowed nesting depth.
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(conn, row):
    """Missing members are damage, never an invitation to reconstruct a subset."""
    try:
        nodes = json.loads(row["definition_json"])["nodes"]
        expected = {node["key"] for node in nodes}
    except (ValueError, KeyError, TypeError):
        raise RecipeError("INSTANCE_DAMAGED", "", "Stored definition is damaged") from None
    members = conn.execute(
        "SELECT m.node_key,m.task_id,t.id AS live_id FROM recipe_instance_tasks m "
        "LEFT JOIN tasks t ON t.id=m.task_id WHERE m.instance_id=?", (row["id"],),
    ).fetchall()
    if (len(members) != len(nodes) or {m["node_key"] for m in members} != expected
            or any(m["live_id"] is None for m in members)):
        raise RecipeError("INSTANCE_DAMAGED", "", "Instance membership or tasks are missing")
    return {m["node_key"]: m["task_id"] for m in members}


def _result(conn, row, replayed):
    return {"instance_id": row["id"], "replayed": replayed,
            "definition_digest": row["definition_digest"], "request_digest": row["request_digest"],
            "tasks": _mapping(conn, row)}


def _replay(conn, row, prepared, bindings):
    if row["imported"]:
        raise RecipeError("IMPORTED_INSTANCE_REBIND_REQUIRED", "/key",
                          "Imported history cannot replay; export and run with a new key")
    if (row["definition_json"] != canonical(prepared["definition"])
            or row["inputs_json"] != canonical(prepared["inputs"])
            or row["bindings_json"] != canonical(bindings)):
        raise RecipeError("IDEMPOTENCY_CONFLICT", "/key", "Key is reserved for a different invocation")
    return _result(conn, row, True)


def instantiate(conn, prepared, bindings, key, *, board=None, created_by=None, creator_task_id=None):
    """Publish the receipt, tasks, edges and provenance with one durable commit.

    Replay compares explicit identity under the lock *before* resolving any
    ambient state. Removed profiles/projects therefore cannot invalidate history.
    Native storage and permission exceptions remain available to the CLI boundary.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import write_txn
    from hermes_cli import kanban_recipes_bindings as resolver

    if type(key) is not str or re.fullmatch(r"[!-~]{1,128}", key, re.ASCII) is None:
        raise RecipeError("RECIPE_INVALID", "/key", "Key must contain 1-128 printable non-whitespace ASCII characters")
    normalized = resolver.normalize_bindings(prepared, bindings)
    with write_txn(conn):
        row = conn.execute("SELECT * FROM recipe_instances WHERE idempotency_key=?", (key,)).fetchone()
        if row is not None:
            return _replay(conn, row, prepared, normalized)
        from hermes_cli.profiles import profile_exists
        for profile in normalized["roles"].values():
            if not profile_exists(profile):
                raise RecipeError("BINDING_INVALID", "/bindings/roles", "Bound profile is not installed")
        # CLI read-only preflight may freeze local choices before native DB
        # initialization. Direct callers resolve here, after authoritative replay.
        plan = prepared.get("effective_plan")
        if plan is None:
            plan = resolver.resolve_plan(prepared, normalized, board)
        instance_id = "ri_" + secrets.token_hex(16)
        definition_digest = _digest(prepared["definition"])
        request_digest = _digest({"definition_digest": definition_digest, "inputs": prepared["inputs"],
                                  "bindings": normalized, "effective_plan": plan})
        conn.execute(
            "INSERT INTO recipe_instances (id,idempotency_key,recipe_id,schema_version,"
            "definition_digest,request_digest,definition_json,inputs_json,bindings_json,"
            "effective_plan_json,created_by,created_at,imported) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)",
            (instance_id, key, prepared["definition"]["recipe_id"], prepared["definition"]["schema_version"],
             definition_digest, request_digest, canonical(prepared["definition"]), canonical(prepared["inputs"]),
             canonical(normalized), canonical(plan), created_by, int(time.time())),
        )
        nodes = {node["key"]: node for node in plan["nodes"]}
        mapping = {}
        for node_key in prepared["order"]:
            node = nodes[node_key]
            task = node["task"]
            task_id = kb.create_task(
                conn, title=node["title"], body=node["body"], assignee=node["assignee"],
                tenant=node["tenant"], created_by=created_by, creator_task_id=creator_task_id,
                board=board, parents=[mapping[parent] for parent in node["needs"]],
                workspace_kind=task["workspace_kind"], priority=task["priority"], skills=task["skills"],
                max_runtime_seconds=task["max_runtime_seconds"], max_retries=task["max_retries"],
                model_override=task["model_override"], provider_override=task["provider_override"],
                reasoning_effort=task["reasoning_effort"], goal_mode=task["goal_mode"],
                goal_max_turns=task["goal_max_turns"], completion_contract=task["completion_contract"],
                _resolved_recipe_context=plan["project"] or {},
            )
            mapping[node_key] = task_id
            conn.execute("INSERT INTO recipe_instance_tasks (instance_id,node_key,task_id) VALUES (?,?,?)",
                         (instance_id, node_key, task_id))
            kb._append_event(conn, task_id, "recipe_instantiated",
                             {"instance_id": instance_id, "node_key": node_key,
                              "definition_digest": definition_digest, "request_digest": request_digest})
        result = {"instance_id": instance_id, "replayed": False, "definition_digest": definition_digest,
                  "request_digest": request_digest, "tasks": mapping}
    return result


def show_instance(conn, instance_id):
    """Return immutable invocation data alongside current native task states."""
    row = conn.execute("SELECT * FROM recipe_instances WHERE id=?", (instance_id,)).fetchone()
    if row is None:
        raise RecipeError("INSTANCE_DAMAGED", "/instance_id", "Instance does not exist")
    result = _result(conn, row, False)
    for field in ("definition", "inputs", "bindings", "effective_plan"):
        result[field] = json.loads(row[field + "_json"])
    result["current_tasks"] = {
        key: dict(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
        for key, task_id in result["tasks"].items()
    }
    result.update(imported=bool(row["imported"]), created_by=row["created_by"], created_at=row["created_at"])
    return result
