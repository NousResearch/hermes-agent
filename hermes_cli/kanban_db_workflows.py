"""Saved finite workflows persisted by the existing Kanban board.

This module owns workflow definition and controller state only. Tasks, links,
runs, review, and worker execution continue to use the canonical Kanban and
WorkerStore services.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import time
from typing import Any, Iterable, Mapping, Optional


WORKFLOW_CONTRACT_VERSION = "kanban-workflow-v1"
CONTROL_STATES = frozenset({"active", "paused", "cancelling", "cancelled"})
_STEP_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
_MAX_STEPS = 64
_MAX_INPUT_CHARS = 16_000
_UNKNOWN = "Unknown or unavailable workflow reference."


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _text(value: Any, name: str, *, required: bool = False) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{name} is required")
    return result


def _template_ref(template_id: str, version: int) -> str:
    return f"workflow_template:{template_id}@{int(version)}"


def _invocation_ref(invocation_id: str) -> str:
    return f"workflow:{invocation_id}"


def parse_template_ref(reference: Any) -> tuple[str, int]:
    raw = _text(reference, "template_ref", required=True)
    prefix, sep, value = raw.partition(":")
    template_id, marker, version = value.rpartition("@")
    if prefix != "workflow_template" or not sep or not marker or not template_id:
        raise PermissionError(_UNKNOWN)
    try:
        parsed = int(version)
    except ValueError:
        raise PermissionError(_UNKNOWN) from None
    if parsed < 1:
        raise PermissionError(_UNKNOWN)
    return template_id, parsed


def parse_invocation_ref(reference: Any) -> str:
    raw = _text(reference, "workflow_ref", required=True)
    prefix, sep, value = raw.partition(":")
    if prefix != "workflow" or not sep or not value:
        raise PermissionError(_UNKNOWN)
    return value


def _normalized_definition(definition: Any) -> dict[str, Any]:
    if not isinstance(definition, Mapping):
        raise ValueError("definition must be an object")
    name = _text(definition.get("name"), "definition.name", required=True)
    raw_steps = definition.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("definition.steps must be a nonempty list")
    if len(raw_steps) > _MAX_STEPS:
        raise ValueError(f"definition.steps is limited to {_MAX_STEPS} steps")
    steps: list[dict[str, Any]] = []
    keys: set[str] = set()
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            raise ValueError(f"definition.steps[{index}] must be an object")
        key = _text(raw.get("key"), f"definition.steps[{index}].key", required=True)
        if not _STEP_KEY.fullmatch(key):
            raise ValueError(f"invalid workflow step key: {key!r}")
        if key in keys:
            raise ValueError(f"duplicate workflow step key: {key}")
        keys.add(key)
        depends_on = raw.get("depends_on")
        if depends_on is None:
            depends_on = []
        if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
            raise ValueError(f"definition.steps[{index}].depends_on must be a string list")
        if len(depends_on) != len(set(depends_on)):
            raise ValueError(f"workflow step {key} has duplicate dependencies")
        max_corrections = raw.get("max_corrections", 1)
        if isinstance(max_corrections, bool) or not isinstance(max_corrections, int):
            raise ValueError(f"workflow step {key} max_corrections must be an integer")
        if not 0 <= max_corrections <= 8:
            raise ValueError(f"workflow step {key} max_corrections must be between 0 and 8")
        steps.append({
            "key": key,
            "title": _text(raw.get("title"), f"workflow step {key} title", required=True),
            "body": _text(raw.get("body"), f"workflow step {key} body"),
            "profile": _text(raw.get("profile"), f"workflow step {key} profile", required=True),
            "reviewer": _text(raw.get("reviewer"), f"workflow step {key} reviewer"),
            "depends_on": list(depends_on),
            "max_corrections": max_corrections,
        })
    by_key = {step["key"]: step for step in steps}
    for step in steps:
        unknown = [key for key in step["depends_on"] if key not in by_key]
        if unknown:
            raise ValueError(f"workflow step {step['key']} has unknown dependencies: {unknown}")
        if step["key"] in step["depends_on"]:
            raise ValueError(f"workflow step {step['key']} cannot depend on itself")
    _topological_steps(steps)
    return {"name": name, "steps": steps}


def _topological_steps(steps: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ordered = list(steps)
    by_key = {str(step["key"]): step for step in ordered}
    waiting = {key: set(step.get("depends_on") or ()) for key, step in by_key.items()}
    result: list[Mapping[str, Any]] = []
    while waiting:
        ready = [key for key in by_key if key in waiting and not waiting[key]]
        if not ready:
            raise ValueError("workflow definition contains a dependency cycle")
        for key in ready:
            result.append(by_key[key])
            waiting.pop(key)
            for parents in waiting.values():
                parents.discard(key)
    return result


def workflow_schema_present(conn: sqlite3.Connection) -> bool:
    names = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "('workflow_templates','workflow_invocations','workflow_invocation_events')"
    )}
    return names == {"workflow_templates", "workflow_invocations", "workflow_invocation_events"}


def save_template(
    conn: sqlite3.Connection, definition: Mapping[str, Any], *, created_by: str,
    template_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create one immutable version, returning an existing identical head."""
    from hermes_cli.kanban_db import write_txn

    normalized = _normalized_definition(definition)
    encoded = _canonical(normalized)
    content_hash = _digest(normalized)
    template_id = _text(template_id, "template_id") or f"wft_{secrets.token_hex(8)}"
    now = int(time.time())
    with write_txn(conn):
        current = conn.execute(
            "SELECT version,definition_hash FROM workflow_templates "
            "WHERE template_id=? ORDER BY version DESC LIMIT 1", (template_id,),
        ).fetchone()
        if current is not None and current["definition_hash"] == content_hash:
            version = int(current["version"])
        else:
            version = int(current["version"] if current is not None else 0) + 1
            conn.execute(
                "INSERT INTO workflow_templates "
                "(template_id,version,name,definition,definition_hash,created_by,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (template_id, version, normalized["name"], encoded, content_hash, created_by, now),
            )
    return {
        "template_ref": _template_ref(template_id, version),
        "template_id": template_id,
        "version": version,
        "definition_hash": content_hash,
        "name": normalized["name"],
    }


def _load_template(conn: sqlite3.Connection, template_id: str, version: int) -> tuple[sqlite3.Row, dict]:
    row = conn.execute(
        "SELECT * FROM workflow_templates WHERE template_id=? AND version=?",
        (template_id, int(version)),
    ).fetchone()
    if row is None:
        raise PermissionError(_UNKNOWN)
    definition = json.loads(row["definition"])
    if not isinstance(definition, dict) or _digest(definition) != row["definition_hash"]:
        raise RuntimeError("Stored workflow template failed its immutable digest check")
    return row, definition


def invoke_workflow(
    conn: sqlite3.Connection, template_ref: str, *, owner_session_id: str,
    admission_key: str, input_payload: Optional[Mapping[str, Any]], created_by: str,
    board: Optional[str] = None,
) -> dict[str, Any]:
    """Atomically admit an invocation, coordinator, finite steps, and links."""
    from hermes_cli import kanban_db as kb

    template_id, version = parse_template_ref(template_ref)
    admission_key = _text(admission_key, "admission_key", required=True)
    owner_session_id = _text(owner_session_id, "owner_session_id", required=True)
    raw_payload: Any = {} if input_payload is None else input_payload
    if not isinstance(raw_payload, Mapping):
        raise ValueError("input must be an object")
    payload = dict(raw_payload)
    input_encoded = _canonical(payload)
    if len(input_encoded) > _MAX_INPUT_CHARS:
        raise ValueError(f"workflow input is limited to {_MAX_INPUT_CHARS} characters")
    input_hash = _digest(payload)
    now = int(time.time())
    with kb.write_txn(conn):
        template_row, definition = _load_template(conn, template_id, version)
        graph_hash = _digest({
            "contract": WORKFLOW_CONTRACT_VERSION,
            "template": template_row["definition_hash"],
            "input": input_hash,
        })
        existing = conn.execute(
            "SELECT * FROM workflow_invocations WHERE owner_session_id=? AND admission_key=?",
            (owner_session_id, admission_key),
        ).fetchone()
        if existing is not None:
            if (
                existing["template_id"] != template_id
                or int(existing["template_version"]) != version
                or existing["template_hash"] != template_row["definition_hash"]
                or existing["input_hash"] != input_hash
                or existing["graph_hash"] != graph_hash
            ):
                raise ValueError("Workflow admission key already has different immutable content")
            return invocation_detail(conn, existing["id"], owner_session_id=owner_session_id)

        invocation_id = f"wfi_{secrets.token_hex(8)}"
        task_ids: dict[str, str] = {}
        input_context = f"Workflow input: {input_encoded}" if payload else ""
        for step in _topological_steps(definition["steps"]):
            parents = [task_ids[key] for key in step["depends_on"]]
            body = "\n\n".join(part for part in (step.get("body"), input_context) if part)
            task_id = kb.create_task(
                conn,
                title=step["title"],
                body=body or None,
                assignee=step["profile"],
                created_by=created_by,
                parents=parents,
                session_id=owner_session_id,
                board=board,
                execution_mode="parent",
            )
            task_ids[step["key"]] = task_id
            conn.execute(
                "UPDATE tasks SET workflow_template_id=?, workflow_template_version=?, "
                "workflow_invocation_id=?, current_step_key=? WHERE id=?",
                (template_id, version, invocation_id, step["key"], task_id),
            )
        coordinator_id = kb.create_task(
            conn,
            title=f"Workflow: {definition['name']}",
            body=input_context or None,
            assignee=None,
            created_by=created_by,
            parents=list(task_ids.values()),
            session_id=owner_session_id,
            board=board,
            execution_mode="parent",
        )
        conn.execute(
            "UPDATE tasks SET workflow_template_id=?, workflow_template_version=?, "
            "workflow_invocation_id=?, current_step_key='__coordinator__' WHERE id=?",
            (template_id, version, invocation_id, coordinator_id),
        )
        conn.execute(
            "INSERT INTO workflow_invocations "
            "(id,template_id,template_version,template_hash,owner_session_id,admission_key,"
            "input_payload,input_hash,graph_hash,coordinator_task_id,control_state,control_version,"
            "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?, 'active',1,?,?)",
            (
                invocation_id, template_id, version, template_row["definition_hash"],
                owner_session_id, admission_key, input_encoded, input_hash, graph_hash,
                coordinator_id, now, now,
            ),
        )
        mapping = {
            "version": WORKFLOW_CONTRACT_VERSION,
            "workflow_ref": _invocation_ref(invocation_id),
            "template_ref": _template_ref(template_id, version),
            "coordinator_ref": f"task:{coordinator_id}",
            "step_refs": {key: f"task:{value}" for key, value in task_ids.items()},
            "graph_hash": graph_hash,
        }
        kb._append_event(conn, coordinator_id, "workflow_admitted", mapping)
        conn.execute(
            "INSERT INTO workflow_invocation_events "
            "(invocation_id,kind,subject_key,payload,created_at) VALUES (?,?,?,?,?)",
            (invocation_id, "workflow_admitted", "", _canonical(mapping), now),
        )
        return invocation_detail(conn, invocation_id, owner_session_id=owner_session_id)


def _owned_invocation(
    conn: sqlite3.Connection, invocation_id: str, owner_session_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM workflow_invocations WHERE id=? AND owner_session_id=?",
        (invocation_id, owner_session_id),
    ).fetchone()
    if row is None:
        raise PermissionError(_UNKNOWN)
    return row


def invocation_detail(
    conn: sqlite3.Connection, invocation_id: str, *, owner_session_id: str,
) -> dict[str, Any]:
    row = _owned_invocation(conn, invocation_id, owner_session_id)
    tasks = conn.execute(
        "SELECT id,title,assignee,status,current_step_key,current_run_id FROM tasks "
        "WHERE workflow_invocation_id=? ORDER BY created_at,id", (invocation_id,),
    ).fetchall()
    steps = [{
        "task_ref": f"task:{task['id']}",
        "step_key": task["current_step_key"],
        "title": task["title"],
        "profile": task["assignee"],
        "status": task["status"],
        "kanban_run_id": task["current_run_id"],
    } for task in tasks if task["current_step_key"] != "__coordinator__"]
    return {
        "workflow_ref": _invocation_ref(invocation_id),
        "template_ref": _template_ref(row["template_id"], row["template_version"]),
        "coordinator_ref": f"task:{row['coordinator_task_id']}",
        "owner_scope": "originating_session",
        "control_state": row["control_state"],
        "control_version": int(row["control_version"]),
        "graph_hash": row["graph_hash"],
        "steps": steps,
        "completed": row["completed_at"] is not None,
    }


def template_detail(conn: sqlite3.Connection, template_id: str, version: int) -> dict[str, Any]:
    row, definition = _load_template(conn, template_id, version)
    return {
        "template_ref": _template_ref(template_id, version),
        "name": row["name"],
        "version": int(version),
        "definition_hash": row["definition_hash"],
        "definition": definition,
    }


def list_visible(conn: sqlite3.Connection, *, owner_session_id: str) -> dict[str, Any]:
    templates = conn.execute(
        "SELECT template_id,MAX(version) AS version,name FROM workflow_templates "
        "GROUP BY template_id ORDER BY name,template_id"
    ).fetchall()
    invocations = conn.execute(
        "SELECT id FROM workflow_invocations WHERE owner_session_id=? ORDER BY created_at,id",
        (owner_session_id,),
    ).fetchall()
    return {
        "templates": [{
            "template_ref": _template_ref(row["template_id"], row["version"]),
            "name": row["name"],
        } for row in templates],
        "invocations": [
            invocation_detail(conn, row["id"], owner_session_id=owner_session_id)
            for row in invocations
        ],
    }


def set_control(
    conn: sqlite3.Connection, invocation_id: str, *, owner_session_id: str,
    expected_version: int, action: str,
) -> dict[str, Any]:
    from hermes_cli.kanban_db import write_txn

    transitions = {"pause": ("active", "paused"), "resume": ("paused", "active")}
    if action not in transitions:
        raise ValueError("workflow control action must be pause or resume")
    source, target = transitions[action]
    now = int(time.time())
    with write_txn(conn):
        _owned_invocation(conn, invocation_id, owner_session_id)
        changed = conn.execute(
            "UPDATE workflow_invocations SET control_state=?,control_version=control_version+1,"
            "updated_at=? WHERE id=? AND owner_session_id=? AND control_state=? AND control_version=?",
            (target, now, invocation_id, owner_session_id, source, int(expected_version)),
        )
        if changed.rowcount != 1:
            raise RuntimeError("Workflow control version or state changed")
        conn.execute(
            "INSERT INTO workflow_invocation_events "
            "(invocation_id,kind,subject_key,payload,created_at) VALUES (?,?,?,?,?)",
            (invocation_id, f"workflow_{action}d", str(expected_version), _canonical({
                "from": source, "to": target, "expected_version": int(expected_version),
            }), now),
        )
        return invocation_detail(conn, invocation_id, owner_session_id=owner_session_id)


def begin_cancellation(
    conn: sqlite3.Connection, invocation_id: str, *, owner_session_id: str,
    expected_version: int,
) -> dict[str, Any]:
    from hermes_cli.kanban_db import write_txn

    now = int(time.time())
    with write_txn(conn):
        row = _owned_invocation(conn, invocation_id, owner_session_id)
        if row["control_state"] == "cancelling" and int(row["control_version"]) == int(expected_version):
            return invocation_detail(conn, invocation_id, owner_session_id=owner_session_id)
        if row["control_state"] not in {"active", "paused"}:
            raise RuntimeError("Workflow cannot enter cancellation from its current state")
        changed = conn.execute(
            "UPDATE workflow_invocations SET control_state='cancelling',"
            "control_version=control_version+1,updated_at=? "
            "WHERE id=? AND owner_session_id=? AND control_version=?",
            (now, invocation_id, owner_session_id, int(expected_version)),
        )
        if changed.rowcount != 1:
            raise RuntimeError("Workflow control version changed")
        return invocation_detail(conn, invocation_id, owner_session_id=owner_session_id)


def correction_policy(
    conn: sqlite3.Connection, task_id: str, *, owner_session_id: str,
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT workflow_invocation_id,workflow_template_id,workflow_template_version,"
        "current_step_key,session_id FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    if row is None or not row["workflow_invocation_id"]:
        return None
    if row["session_id"] != owner_session_id:
        raise PermissionError(_UNKNOWN)
    _owned_invocation(conn, row["workflow_invocation_id"], owner_session_id)
    _template, definition = _load_template(
        conn, row["workflow_template_id"], int(row["workflow_template_version"]),
    )
    step = next((item for item in definition["steps"] if item["key"] == row["current_step_key"]), None)
    if step is None:
        raise RuntimeError("Workflow task no longer matches its immutable template")
    used = int(conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='changes_requested'",
        (task_id,),
    ).fetchone()[0])
    limit = int(step["max_corrections"])
    return {
        "used": used,
        "limit": limit,
        "allowed": used < limit,
        "reviewer": str(step.get("reviewer") or ""),
    }


def record_acceptance_evidence(
    conn: sqlite3.Connection, task_id: str, *, owner_session_id: str,
    kanban_run_id: int, worker_status: str,
) -> dict[str, Any]:
    """Persist the exact successful reviewer execution before acceptance."""
    from hermes_cli import kanban_db as kb

    if worker_status != "SUCCEEDED":
        raise ValueError("Reviewer success is required before acceptance")
    with kb.write_txn(conn):
        policy = correction_policy(conn, task_id, owner_session_id=owner_session_id)
        if policy is None:
            raise RuntimeError("Task is not an owned workflow step")
        task = conn.execute(
            "SELECT status,current_run_id FROM tasks WHERE id=?", (task_id,),
        ).fetchone()
        if (
            task is None or task["status"] != "running"
            or int(task["current_run_id"] or 0) != int(kanban_run_id)
            or kb.run_claim_source(conn, task_id, int(kanban_run_id)) != "review"
        ):
            raise RuntimeError("Acceptance evidence does not match the active review run")
        attachment = kb.get_execution_attachment(conn, task_id, int(kanban_run_id))
        if attachment is None or attachment.get("role") != "reviewer":
            raise RuntimeError("Acceptance evidence has no exact reviewer execution")
        payload = {
            "kanban_run_id": int(kanban_run_id),
            "worker_ref": attachment["worker_ref"],
            "run_ref": attachment["run_ref"],
            "worker_status": worker_status,
        }
        prior = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
            "AND kind='workflow_acceptance_evidence' ORDER BY id DESC LIMIT 1",
            (task_id, int(kanban_run_id)),
        ).fetchone()
        if prior is not None:
            if json.loads(prior["payload"]) != payload:
                raise RuntimeError("Acceptance evidence changed for the exact review run")
            return payload
        kb._append_event(
            conn, task_id, "workflow_acceptance_evidence", payload,
            run_id=int(kanban_run_id),
        )
        return payload


def completion_allowed(
    conn: sqlite3.Connection, task_id: str, *, expected_run_id: Optional[int],
) -> bool:
    """Require immutable review and recorded worker success for workflow steps."""
    from hermes_cli import kanban_db as kb

    task = conn.execute(
        "SELECT workflow_invocation_id,session_id,status,current_run_id,current_step_key "
        "FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    if task is None or not task["workflow_invocation_id"]:
        return True
    if task["current_step_key"] == "__coordinator__":
        return False
    policy = correction_policy(conn, task_id, owner_session_id=task["session_id"])
    if policy is None or not policy["reviewer"]:
        return True
    run_id = int(task["current_run_id"] or 0)
    if (
        task["status"] != "running" or run_id == 0 or expected_run_id is None
        or run_id != int(expected_run_id)
        or kb.run_claim_source(conn, task_id, run_id) != "review"
    ):
        return False
    attachment = kb.get_execution_attachment(conn, task_id, run_id)
    if attachment is None or attachment.get("role") != "reviewer":
        return False
    evidence = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
        "AND kind='workflow_acceptance_evidence' ORDER BY id DESC LIMIT 1",
        (task_id, run_id),
    ).fetchone()
    value = json.loads(evidence["payload"]) if evidence is not None else {}
    return (
        int(value.get("kanban_run_id") or 0) == run_id
        and value.get("worker_ref") == attachment.get("worker_ref")
        and value.get("run_ref") == attachment.get("run_ref")
        and value.get("worker_status") == "SUCCEEDED"
    )


def finalize_completed(
    conn: sqlite3.Connection, invocation_id: str, *, owner_session_id: str,
) -> bool:
    """Complete only the coordinator after every immutable step is accepted."""
    from hermes_cli import kanban_db as kb

    now = int(time.time())
    with kb.write_txn(conn):
        row = _owned_invocation(conn, invocation_id, owner_session_id)
        if row["control_state"] in {"cancelling", "cancelled"}:
            return False
        remaining = conn.execute(
            "SELECT 1 FROM tasks WHERE workflow_invocation_id=? "
            "AND current_step_key!='__coordinator__' AND status!='done' LIMIT 1",
            (invocation_id,),
        ).fetchone()
        if remaining is not None:
            return False
        coordinator = conn.execute(
            "SELECT status FROM tasks WHERE id=? AND workflow_invocation_id=?",
            (row["coordinator_task_id"], invocation_id),
        ).fetchone()
        if coordinator is None:
            raise RuntimeError("Workflow coordinator is unavailable")
        if coordinator["status"] == "done":
            return True
        changed = conn.execute(
            "UPDATE tasks SET status='done',completed_at=?,claim_lock=NULL,claim_expires=NULL,"
            "worker_pid=NULL WHERE id=? AND status IN ('todo','ready')",
            (now, row["coordinator_task_id"]),
        )
        if changed.rowcount != 1:
            return False
        run_id = kb._synthesize_ended_run(
            conn, row["coordinator_task_id"], outcome="completed",
            summary="All workflow steps were accepted.",
        )
        kb._append_event(
            conn, row["coordinator_task_id"], "completed",
            {"summary": "All workflow steps were accepted.", "workflow_ref": _invocation_ref(invocation_id)},
            run_id=run_id,
        )
        conn.execute(
            "UPDATE workflow_invocations SET completed_at=?,updated_at=? WHERE id=?",
            (now, now, invocation_id),
        )
        conn.execute(
            "INSERT OR IGNORE INTO workflow_invocation_events "
            "(invocation_id,kind,subject_key,payload,created_at) VALUES (?,?,?,?,?)",
            (invocation_id, "workflow_completed", "", None, now),
        )
    kb.recompute_ready(conn)
    return True


def _workflow_execution_attachments(
    conn: sqlite3.Connection, invocation_id: str,
) -> list[dict[str, Any]]:
    """Return every exact execution attachment owned by workflow steps."""
    rows = conn.execute(
        "SELECT t.id AS task_id,e.run_id,e.payload FROM tasks t "
        "JOIN task_events e ON e.task_id=t.id AND e.kind='execution_attached' "
        "WHERE t.workflow_invocation_id=? AND t.current_step_key!='__coordinator__' "
        "ORDER BY t.id,e.run_id,e.id",
        (invocation_id,),
    ).fetchall()
    attachments: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        task_id = str(row["task_id"])
        run_id = int(row["run_id"] or 0)
        key = (task_id, run_id)
        if run_id < 1 or key in seen:
            continue
        value = json.loads(row["payload"] or "{}")
        if not isinstance(value, dict) or not value.get("worker_ref") or not value.get("run_ref"):
            raise RuntimeError("Workflow execution attachment is incomplete")
        seen.add(key)
        attachments.append({
            "task_ref": f"task:{task_id}",
            "kanban_run_id": run_id,
            "worker_ref": value["worker_ref"],
            "run_ref": value["run_ref"],
        })
    return attachments


def _cancel_subject(target: Mapping[str, Any]) -> str:
    return f"{target['task_ref'].partition(':')[2]}:{int(target['kanban_run_id'])}"


def cancellation_targets(
    conn: sqlite3.Connection, invocation_id: str, *, owner_session_id: str,
) -> list[dict[str, Any]]:
    """Persist exact cancel intents before any external worker interruption."""
    from hermes_cli import kanban_db as kb

    now = int(time.time())
    targets: list[dict[str, Any]] = []
    with kb.write_txn(conn):
        row = _owned_invocation(conn, invocation_id, owner_session_id)
        if row["control_state"] != "cancelling":
            raise RuntimeError("Workflow is not cancelling")
        for payload in _workflow_execution_attachments(conn, invocation_id):
            subject = _cancel_subject(payload)
            terminal = conn.execute(
                "SELECT payload FROM workflow_invocation_events "
                "WHERE invocation_id=? AND kind='cancel_terminal' AND subject_key=?",
                (invocation_id, subject),
            ).fetchone()
            if terminal is not None:
                value = json.loads(terminal["payload"] or "{}")
                if int(value.get("kanban_run_id") or 0) != int(payload["kanban_run_id"]):
                    raise RuntimeError("Workflow cancel terminal evidence changed identity")
                continue
            prior = conn.execute(
                "SELECT payload FROM workflow_invocation_events "
                "WHERE invocation_id=? AND kind='cancel_requested' AND subject_key=?",
                (invocation_id, subject),
            ).fetchone()
            is_new = prior is None
            if prior is None:
                conn.execute(
                    "INSERT INTO workflow_invocation_events "
                    "(invocation_id,kind,subject_key,payload,created_at) VALUES (?,?,?,?,?)",
                    (invocation_id, "cancel_requested", subject, _canonical(payload), now),
                )
            elif json.loads(prior["payload"]) != payload:
                raise RuntimeError("Workflow cancel intent no longer matches the exact task execution")
            targets.append({**payload, "request_new": is_new})
    return targets


def record_cancel_terminal(
    conn: sqlite3.Connection, invocation_id: str, *, owner_session_id: str,
    task_id: str, kanban_run_id: int, worker_status: str,
) -> None:
    from hermes_cli.kanban_db import write_txn

    if worker_status not in {"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"}:
        raise ValueError("worker_status is not terminal")
    with write_txn(conn):
        _owned_invocation(conn, invocation_id, owner_session_id)
        subject = f"{task_id}:{int(kanban_run_id)}"
        intent = conn.execute(
            "SELECT payload FROM workflow_invocation_events WHERE invocation_id=? "
            "AND kind='cancel_requested' AND subject_key=?", (invocation_id, subject),
        ).fetchone()
        payload = json.loads(intent["payload"]) if intent is not None else {}
        if int(payload.get("kanban_run_id") or 0) != int(kanban_run_id):
            raise RuntimeError("Terminal evidence does not match the cancel intent")
        conn.execute(
            "INSERT OR IGNORE INTO workflow_invocation_events "
            "(invocation_id,kind,subject_key,payload,created_at) VALUES (?,?,?,?,?)",
            (invocation_id, "cancel_terminal", subject, _canonical({
                "kanban_run_id": int(kanban_run_id), "worker_status": worker_status,
            }), int(time.time())),
        )


def begin_cancel_interrupt(
    conn: sqlite3.Connection, invocation_id: str, *, owner_session_id: str,
    task_id: str, kanban_run_id: int,
) -> bool:
    """Record the send boundary once; return true only to its first caller."""
    from hermes_cli.kanban_db import write_txn

    subject = f"{task_id}:{int(kanban_run_id)}"
    now = int(time.time())
    with write_txn(conn):
        _owned_invocation(conn, invocation_id, owner_session_id)
        intent = conn.execute(
            "SELECT payload FROM workflow_invocation_events WHERE invocation_id=? "
            "AND kind='cancel_requested' AND subject_key=?",
            (invocation_id, subject),
        ).fetchone()
        value = json.loads(intent["payload"]) if intent is not None else {}
        if int(value.get("kanban_run_id") or 0) != int(kanban_run_id):
            raise RuntimeError("Interrupt attempt does not match the cancel intent")
        prior = conn.execute(
            "SELECT 1 FROM workflow_invocation_events WHERE invocation_id=? "
            "AND kind='cancel_interrupt_attempted' AND subject_key=?",
            (invocation_id, subject),
        ).fetchone()
        if prior is not None:
            return False
        conn.execute(
            "INSERT INTO workflow_invocation_events "
            "(invocation_id,kind,subject_key,payload,created_at) VALUES (?,?,?,?,?)",
            (invocation_id, "cancel_interrupt_attempted", subject, _canonical({
                "task_ref": f"task:{task_id}", "kanban_run_id": int(kanban_run_id),
                "worker_ref": value.get("worker_ref"), "run_ref": value.get("run_ref"),
            }), now),
        )
        return True


def finalize_cancelled(
    conn: sqlite3.Connection, invocation_id: str, *, owner_session_id: str,
) -> dict[str, Any]:
    """Atomically sticky-block every unfinished task after terminal evidence."""
    from hermes_cli import kanban_db as kb

    now = int(time.time())
    with kb.write_txn(conn):
        row = _owned_invocation(conn, invocation_id, owner_session_id)
        if row["control_state"] == "cancelled":
            return invocation_detail(conn, invocation_id, owner_session_id=owner_session_id)
        if row["control_state"] != "cancelling":
            raise RuntimeError("Workflow is not cancelling")
        attachments = _workflow_execution_attachments(conn, invocation_id)
        for attachment in attachments:
            terminal = conn.execute(
                "SELECT payload FROM workflow_invocation_events WHERE invocation_id=? "
                "AND kind='cancel_terminal' AND subject_key=?",
                (invocation_id, _cancel_subject(attachment)),
            ).fetchone()
            value = json.loads(terminal["payload"]) if terminal is not None else {}
            if int(value.get("kanban_run_id") or 0) != int(attachment["kanban_run_id"]):
                raise RuntimeError("Workflow cancellation is waiting for exact terminal worker evidence")
        tasks = conn.execute(
            "SELECT id,status,current_run_id,current_step_key FROM tasks "
            "WHERE workflow_invocation_id=? ORDER BY id", (invocation_id,),
        ).fetchall()
        for task in tasks:
            if task["status"] == "done":
                continue
            prior_status = task["status"]
            conn.execute(
                "UPDATE tasks SET status='blocked',claim_lock=NULL,claim_expires=NULL,worker_pid=NULL,"
                "block_kind='needs_input',block_recurrences=1 WHERE id=?",
                (task["id"],),
            )
            run_id = None
            if prior_status == "running":
                run_id = kb._end_run(
                    conn, task["id"], outcome="cancelled", status="blocked",
                    summary="Workflow cancellation confirmed by terminal worker evidence.",
                )
            kb._append_event(conn, task["id"], "blocked", {
                "reason": "Workflow cancelled after exact terminal worker evidence.",
                "kind": "needs_input", "source_status": prior_status,
                "workflow_ref": _invocation_ref(invocation_id),
            }, run_id=run_id)
            kb._append_event(conn, task["id"], "workflow_cancelled", {
                "workflow_ref": _invocation_ref(invocation_id),
            }, run_id=run_id)
        conn.execute(
            "UPDATE workflow_invocations SET control_state='cancelled',"
            "control_version=control_version+1,updated_at=? WHERE id=?",
            (now, invocation_id),
        )
        conn.execute(
            "INSERT OR IGNORE INTO workflow_invocation_events "
            "(invocation_id,kind,subject_key,payload,created_at) VALUES (?,?,?,?,?)",
            (invocation_id, "workflow_cancelled", "", None, now),
        )
        return invocation_detail(conn, invocation_id, owner_session_id=owner_session_id)
