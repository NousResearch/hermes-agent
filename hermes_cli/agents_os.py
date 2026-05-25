"""Local Agents OS control plane for Hermes/Doni.

This module is deliberately local-first and side-effect bounded:
- writes only under HERMES_HOME/agents_os by default;
- optionally mirrors human-readable artifacts into the Doni vault lane;
- never sends network requests, starts servers, edits runtime config, or touches credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import textwrap
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

DEFAULT_VAULT_ROOT = Path(
    "/mnt/d/Obsidian_Vault_v2/Hermes-Agent-Doni/08-OPERATIONS/ACTIVE-WORK/AGENTS-OS-v0"
)
SAFE_WORKFLOWS = {
    "youtube-intake": {
        "kind": "source_intake",
        "requires_approval": False,
        "template": "YouTube/link intake: capture source, transcript path, summary, routing, next action.",
    },
    "research-brief": {
        "kind": "research",
        "requires_approval": False,
        "template": "Research brief: question, sources, findings, confidence, implementation route.",
    },
    "code-task": {
        "kind": "implementation",
        "requires_approval": False,
        "template": "Code task: repo, goal, touched files, tests, verification, rollback notes.",
    },
    "qa-report": {
        "kind": "verification",
        "requires_approval": False,
        "template": "QA report: scope, commands, evidence, defects, pass/fail judgement.",
    },
    "external-action-draft": {
        "kind": "approval_draft",
        "requires_approval": True,
        "template": "External action draft: exact outbound action, destination, payload, risk, approval gate.",
    },
}

SCHEMA_VERSION = "2"
TASK_STATUSES = (
    "new",
    "pending",
    "routed",
    "ready",
    "in_progress",
    "needs_approval",
    "blocked",
    "review",
    "completed",
    "cancelled",
)
VALID_TRANSITIONS = {
    "new": {"routed", "ready", "needs_approval", "blocked", "cancelled"},
    "pending": {"routed", "ready", "needs_approval", "blocked", "in_progress", "cancelled"},
    "routed": {"ready", "needs_approval", "blocked", "cancelled"},
    "ready": {"in_progress", "blocked", "cancelled"},
    "in_progress": {"review", "blocked", "completed", "cancelled"},
    "needs_approval": {"ready", "blocked", "cancelled"},
    "blocked": {"ready", "cancelled"},
    "review": {"completed", "in_progress", "blocked", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}
APPROVAL_RISKS = {
    "external-action",
    "credential-access",
    "runtime-config-change",
    "gateway-restart",
    "deploy",
    "public-publish",
    "destructive-delete",
    "financial-action",
}


@dataclass(frozen=True)
class AgentsOSPaths:
    home: Path
    root: Path
    db: Path
    artifacts: Path
    outbox: Path
    vault_root: Path


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str, fallback: str = "item") -> str:
    chars = []
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_", "/", ":"}:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:80] or fallback


def resolve_paths(args: argparse.Namespace | None = None) -> AgentsOSPaths:
    home = get_hermes_home()
    root = Path(os.environ.get("AGENTS_OS_HOME", home / "agents_os")).expanduser()
    vault_raw = getattr(args, "vault_root", None) if args is not None else None
    vault_root = Path(os.environ.get("AGENTS_OS_VAULT_ROOT", vault_raw or DEFAULT_VAULT_ROOT)).expanduser()
    return AgentsOSPaths(
        home=home,
        root=root,
        db=root / "state.sqlite",
        artifacts=root / "artifacts",
        outbox=root / "outbox",
        vault_root=vault_root,
    )


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('new','pending','routed','ready','in_progress','needs_approval','blocked','review','completed','cancelled')),
    workflow TEXT,
    priority INTEGER NOT NULL DEFAULT 3,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    route TEXT,
    approval_required INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected','cancelled')),
    risk TEXT NOT NULL DEFAULT 'normal',
    task_id TEXT,
    payload TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    task_id TEXT,
    workflow TEXT,
    created_at TEXT NOT NULL,
    run_id TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    task_id TEXT,
    run_id TEXT,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task_id TEXT,
    workflow TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    input TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'local',
    status TEXT NOT NULL DEFAULT 'available',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    requires_approval INTEGER NOT NULL DEFAULT 0,
    template TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routing_rules (
    id TEXT PRIMARY KEY,
    workflow TEXT,
    route TEXT NOT NULL,
    requires_approval INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL DEFAULT 100,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    task_id TEXT,
    run_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS state_snapshots (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority, created_at);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id, created_at);
"""

def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _seed_workflows_and_rules(conn: sqlite3.Connection) -> None:
    now = utc_now()
    for workflow, spec in SAFE_WORKFLOWS.items():
        conn.execute(
            "INSERT OR IGNORE INTO workflows(id,kind,requires_approval,template,created_at) VALUES(?,?,?,?,?)",
            (workflow, spec["kind"], 1 if spec["requires_approval"] else 0, spec["template"], now),
        )
    rules = [
        ("rule-external-action", "external-action-draft", "approval_gate", 1, 1),
        ("rule-code-task", "code-task", "skill:test-driven-development", 0, 10),
        ("rule-youtube-intake", "youtube-intake", "vault:first-intake", 0, 20),
        ("rule-research-brief", "research-brief", "skill:async-researcher", 0, 30),
        ("rule-qa-report", "qa-report", "skill:test-driven-development", 0, 40),
    ]
    for rule in rules:
        conn.execute(
            "INSERT OR IGNORE INTO routing_rules(id,workflow,route,requires_approval,priority,created_at) VALUES(?,?,?,?,?,?)",
            (*rule, now),
        )


def _rebuild_legacy_tasks_table_if_needed(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='tasks'").fetchone()
    if not row or "needs_approval" in (row[0] or ""):
        return
    conn.execute("ALTER TABLE tasks RENAME TO tasks_legacy_v1")
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('new','pending','routed','ready','in_progress','needs_approval','blocked','review','completed','cancelled')),
            workflow TEXT,
            priority INTEGER NOT NULL DEFAULT 3,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            route TEXT,
            approval_required INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes)
        SELECT id,title,status,workflow,priority,created_at,updated_at,notes FROM tasks_legacy_v1
    """)
    conn.execute("DROP TABLE tasks_legacy_v1")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _rebuild_legacy_tasks_table_if_needed(conn)
    task_cols = _table_columns(conn, "tasks")
    if "route" not in task_cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN route TEXT")
    if "approval_required" not in task_cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN approval_required INTEGER NOT NULL DEFAULT 0")
    artifact_cols = _table_columns(conn, "artifacts")
    if "run_id" not in artifact_cols:
        conn.execute("ALTER TABLE artifacts ADD COLUMN run_id TEXT")
    conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)", ("created_at", utc_now()))
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", ("schema_version", SCHEMA_VERSION))
    _seed_workflows_and_rules(conn)


def connect(paths: AgentsOSPaths) -> sqlite3.Connection:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    paths.outbox.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(paths.db)
    conn.row_factory = sqlite3.Row
    _migrate_schema(conn)
    conn.commit()
    return conn


def log_event(conn: sqlite3.Connection, event_type: str, task_id: str | None = None, run_id: str | None = None, payload: dict[str, Any] | None = None) -> str:
    event_id = f"event-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO events(id,task_id,run_id,event_type,payload,created_at) VALUES(?,?,?,?,?,?)",
        (event_id, task_id, run_id, event_type, json.dumps(payload or {}, ensure_ascii=False), utc_now()),
    )
    return event_id


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, title: str, body: str, frontmatter: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = frontmatter or {}
    lines = ["---"]
    for key, value in fm.items():
        if isinstance(value, (list, dict)):
            value_text = json.dumps(value, ensure_ascii=False)
        else:
            value_text = str(value).replace("\n", " ")
        lines.append(f"{key}: {value_text}")
    lines.extend(["---", "", f"# {title}", "", body.rstrip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def init_os(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        manifest = {
            "name": "Agents OS v0",
            "status": "active-local-control-plane",
            "created_at": utc_now(),
            "hermes_home": str(paths.home),
            "agents_os_home": str(paths.root),
            "state_db": str(paths.db),
            "vault_root": str(paths.vault_root),
            "safe_workflows": sorted(SAFE_WORKFLOWS.keys()),
            "boundaries": [
                "local only",
                "no network sends",
                "no runtime config edits",
                "no gateway restart",
                "external actions require approval records",
            ],
        }
        write_json(paths.root / "manifest.json", manifest)
        if not args.no_vault:
            write_markdown(
                paths.vault_root / "00-command-center" / "RUNTIME-CONTROL-PLANE.md",
                "Agents OS v0 — Runtime Control Plane",
                "\n".join(
                    [
                        "Ovo više nije samo markdown plan: lokalni runtime state je SQLite baza.",
                        "",
                        f"- State DB: `{paths.db}`",
                        f"- Local artifacts: `{paths.artifacts}`",
                        f"- Outbox: `{paths.outbox}`",
                        "- CLI: `hermes agents-os ...`",
                        "",
                        "Sigurnosne granice: CLI ne šalje podatke van, ne dira config i ne pokreće servere.",
                    ]
                ),
                {"status": "active", "updated_at": utc_now(), "type": "runtime-control-plane"},
            )
        conn.commit()
    print(f"Agents OS inicijaliziran: {paths.db}")
    return 0


def status(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        counts = {
            "tasks": dict(conn.execute("SELECT status, COUNT(*) c FROM tasks GROUP BY status").fetchall()),
            "approvals": dict(conn.execute("SELECT status, COUNT(*) c FROM approvals GROUP BY status").fetchall()),
            "artifacts": conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
        }
    payload = {
        "status": "ok",
        "agents_os_home": str(paths.root),
        "state_db": str(paths.db),
        "vault_root": str(paths.vault_root),
        "counts": counts,
        "workflows": sorted(SAFE_WORKFLOWS.keys()),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Agents OS: OK")
        print(f"State DB: {paths.db}")
        print(f"Vault root: {paths.vault_root}")
        print(f"Taskovi: {counts['tasks']}")
        print(f"Approvali: {counts['approvals']}")
        print(f"Artefakti: {counts['artifacts']}")
    return 0


def task_add(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    task_id = args.id or f"task-{uuid.uuid4().hex[:8]}"
    now = utc_now()
    spec = SAFE_WORKFLOWS.get(args.workflow or "")
    approval_required = 1 if spec and spec["requires_approval"] else 0
    initial_status = "needs_approval" if approval_required else "pending"
    with connect(paths) as conn:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,approval_required) VALUES(?,?,?,?,?,?,?,?,?)",
            (task_id, args.title, initial_status, args.workflow, args.priority, now, now, args.notes or "", approval_required),
        )
        log_event(conn, "task_created", task_id=task_id, payload={"workflow": args.workflow, "status": initial_status})
        conn.commit()
    print(task_id)
    return 0


def task_list(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    sql = "SELECT * FROM tasks"
    params: list[Any] = []
    if args.status:
        sql += " WHERE status=?"
        params.append(args.status)
    sql += " ORDER BY priority ASC, created_at ASC"
    with connect(paths) as conn:
        rows = [row_to_dict(r) for r in conn.execute(sql, params).fetchall()]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        if not rows:
            print("Nema taskova.")
        for row in rows:
            print(f"{row['id']} [{row['status']}] p{row['priority']} {row['title']} ({row['workflow'] or '-'})")
    return 0


def _pending_approval_count(conn: sqlite3.Connection, task_id: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM approvals WHERE task_id=? AND status='pending'", (task_id,)).fetchone()[0]


def _artifact_count(conn: sqlite3.Connection, task_id: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM artifacts WHERE task_id=?", (task_id,)).fetchone()[0]


def _verification_event_count(conn: sqlite3.Connection, task_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM events WHERE task_id=? AND event_type IN ('verified','closed')",
        (task_id,),
    ).fetchone()[0]


def validate_transition(conn: sqlite3.Connection, task: sqlite3.Row, new_status: str) -> tuple[bool, str]:
    old_status = task["status"]
    if old_status == new_status:
        return True, "unchanged"
    if new_status not in VALID_TRANSITIONS.get(old_status, set()):
        return False, f"Ilegalan prijelaz: {old_status} -> {new_status}"
    if new_status in {"in_progress", "ready", "completed"} and (_pending_approval_count(conn, task["id"]) > 0 or task["approval_required"]):
        return False, "Task ima pending approval gate i ne smije u execution/close"
    if new_status == "completed" and _artifact_count(conn, task["id"]) == 0 and _verification_event_count(conn, task["id"]) == 0:
        return False, "Task ne može biti completed bez artifacta ili verification eventa"
    return True, "ok"


def task_set(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (args.id,)).fetchone()
        if task is None:
            print(f"Task ne postoji: {args.id}", file=sys.stderr)
            return 2
        ok, reason = validate_transition(conn, task, args.status)
        if not ok:
            print(reason, file=sys.stderr)
            return 2
        conn.execute(
            "UPDATE tasks SET status=?, updated_at=? WHERE id=?",
            (args.status, utc_now(), args.id),
        )
        log_event(conn, "status_changed", task_id=args.id, payload={"from": task["status"], "to": args.status})
        conn.commit()
    print(f"{args.id} -> {args.status}")
    return 0


def approval_request(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    approval_id = args.id or f"approval-{uuid.uuid4().hex[:8]}"
    with connect(paths) as conn:
        conn.execute(
            "INSERT INTO approvals(id,title,status,risk,task_id,payload,created_at) VALUES(?,?,?,?,?,?,?)",
            (approval_id, args.title, "pending", args.risk, args.task_id, args.payload or "", utc_now()),
        )
        conn.commit()
    print(approval_id)
    return 0


def approval_list(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    sql = "SELECT * FROM approvals"
    params: list[Any] = []
    if args.status:
        sql += " WHERE status=?"
        params.append(args.status)
    sql += " ORDER BY created_at ASC"
    with connect(paths) as conn:
        rows = [row_to_dict(r) for r in conn.execute(sql, params).fetchall()]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        if not rows:
            print("Nema approval zapisa.")
        for row in rows:
            print(f"{row['id']} [{row['status']}] {row['risk']} {row['title']} task={row['task_id'] or '-'}")
    return 0


def approval_set(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        cur = conn.execute(
            "UPDATE approvals SET status=?, resolved_at=? WHERE id=?",
            (args.status, utc_now(), args.id),
        )
        conn.commit()
    if cur.rowcount == 0:
        print(f"Approval ne postoji: {args.id}", file=sys.stderr)
        return 2
    print(f"{args.id} -> {args.status}")
    return 0


def artifact_create(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    artifact_id = args.id or f"artifact-{uuid.uuid4().hex[:8]}"
    stamp = utc_now().split("T", 1)[0]
    filename = f"{stamp}-{slugify(args.title)}.md"
    target_dir = paths.artifacts / (args.workflow or args.kind)
    target = target_dir / filename
    body = args.body or ""
    write_markdown(
        target,
        args.title,
        body,
        {
            "id": artifact_id,
            "kind": args.kind,
            "workflow": args.workflow or "",
            "task_id": args.task_id or "",
            "created_at": utc_now(),
        },
    )
    with connect(paths) as conn:
        conn.execute(
            "INSERT INTO artifacts(id,kind,title,path,task_id,workflow,created_at) VALUES(?,?,?,?,?,?,?)",
            (artifact_id, args.kind, args.title, str(target), args.task_id, args.workflow, utc_now()),
        )
        conn.commit()
    print(str(target))
    return 0


def artifact_list(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        rows = [row_to_dict(r) for r in conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC").fetchall()]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        if not rows:
            print("Nema artefakata.")
        for row in rows:
            print(f"{row['id']} [{row['kind']}] {row['title']} -> {row['path']}")
    return 0


def workflow_run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    spec = SAFE_WORKFLOWS[args.workflow]
    now = utc_now()
    task_id = args.task_id or f"task-{uuid.uuid4().hex[:8]}"
    artifact_id = f"artifact-{uuid.uuid4().hex[:8]}"
    title = args.title or f"{args.workflow}: {args.input[:80]}"
    stamp = now.split("T", 1)[0]
    target = paths.artifacts / args.workflow / f"{stamp}-{slugify(title)}.md"
    body = textwrap.dedent(
        f"""
        ## Input
        {args.input}

        ## Workflow
        {args.workflow}

        ## Template
        {spec['template']}

        ## Execution state
        - status: draft-created
        - next_local_action: fill artifact with evidence and verification results
        - approval_required: {spec['requires_approval']}
        """
    ).strip()
    write_markdown(
        target,
        title,
        body,
        {
            "id": artifact_id,
            "type": spec["kind"],
            "workflow": args.workflow,
            "task_id": task_id,
            "status": "draft-created",
            "created_at": now,
        },
    )
    approval_id = None
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    initial_status = "needs_approval" if spec["requires_approval"] else "pending"
    with connect(paths) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,approval_required) VALUES(?,?,?,?,?,?,?,?,?)",
            (task_id, title, initial_status, args.workflow, args.priority, now, now, args.input, 1 if spec["requires_approval"] else 0),
        )
        conn.execute(
            "INSERT INTO runs(id,task_id,workflow,status,input,created_at) VALUES(?,?,?,?,?,?)",
            (run_id, task_id, args.workflow, "created", args.input, now),
        )
        conn.execute(
            "INSERT INTO artifacts(id,kind,title,path,task_id,workflow,created_at,run_id) VALUES(?,?,?,?,?,?,?,?)",
            (artifact_id, spec["kind"], title, str(target), task_id, args.workflow, now, run_id),
        )
        log_event(conn, "task_created", task_id=task_id, run_id=run_id, payload={"workflow": args.workflow, "status": initial_status})
        log_event(conn, "artifact_created", task_id=task_id, run_id=run_id, payload={"artifact_id": artifact_id, "path": str(target)})
        if spec["requires_approval"]:
            approval_id = f"approval-{uuid.uuid4().hex[:8]}"
            conn.execute(
                "INSERT INTO approvals(id,title,status,risk,task_id,payload,created_at) VALUES(?,?,?,?,?,?,?)",
                (approval_id, f"Approval required: {title}", "pending", "external-action", task_id, args.input, now),
            )
            log_event(conn, "approval_requested", task_id=task_id, run_id=run_id, payload={"approval_id": approval_id, "risk": "external-action"})
        conn.commit()
    result = {"task_id": task_id, "run_id": run_id, "artifact_id": artifact_id, "artifact_path": str(target), "approval_id": approval_id}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _route_for_task(conn: sqlite3.Connection, task: sqlite3.Row) -> dict[str, Any]:
    rule = None
    if task["workflow"]:
        rule = conn.execute(
            "SELECT * FROM routing_rules WHERE workflow=? ORDER BY priority ASC LIMIT 1",
            (task["workflow"],),
        ).fetchone()
    if rule is None:
        notes = (task["notes"] or "") + " " + task["title"]
        lowered = notes.lower()
        if any(word in lowered for word in ["credential", "token", "api key", "deploy", "publish", "public", "delete", "finance", "gateway restart"]):
            return {"route": "approval_gate", "requires_approval": True, "reason": "risk keyword"}
        return {"route": "doni:direct", "requires_approval": False, "reason": "default safe local route"}
    return {"route": rule["route"], "requires_approval": bool(rule["requires_approval"]), "reason": f"workflow:{task['workflow']}"}


def route_task(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (args.id,)).fetchone()
        if task is None:
            print(f"Task ne postoji: {args.id}", file=sys.stderr)
            return 2
        route = _route_for_task(conn, task)
        pending_approvals = _pending_approval_count(conn, args.id)
        requires_approval = route["requires_approval"] or pending_approvals > 0 or bool(task["approval_required"])
        new_status = "needs_approval" if requires_approval else "ready"
        conn.execute(
            "UPDATE tasks SET status=?, route=?, approval_required=?, updated_at=? WHERE id=?",
            (new_status, route["route"], 1 if requires_approval else 0, utc_now(), args.id),
        )
        log_event(conn, "routed", task_id=args.id, payload={"route": route["route"], "status": new_status, "reason": route["reason"]})
        conn.commit()
    payload = {
        "task_id": args.id,
        "route": route["route"],
        "reason": route["reason"],
        "approval_required": requires_approval,
        "execution_allowed": not requires_approval,
        "new_status": new_status,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{args.id}: {payload['route']} -> {new_status}")
    return 0


def next_task(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        task = conn.execute(
            "SELECT * FROM tasks WHERE status IN ('ready','pending','routed') AND approval_required=0 ORDER BY CASE status WHEN 'ready' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END, priority ASC, created_at ASC LIMIT 1"
        ).fetchone()
        payload = {"task": row_to_dict(task) if task else None}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if not payload["task"]:
            print("Nema sigurnog sljedećeg taska.")
        else:
            t = payload["task"]
            print(f"{t['id']} [{t['status']}] p{t['priority']} {t['title']}")
    return 0


def dashboard(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        tasks = [row_to_dict(r) for r in conn.execute("SELECT * FROM tasks ORDER BY priority ASC, created_at ASC LIMIT 20").fetchall()]
        approvals = [row_to_dict(r) for r in conn.execute("SELECT * FROM approvals WHERE status='pending' ORDER BY created_at ASC LIMIT 20").fetchall()]
        runs = [row_to_dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 10").fetchall()]
        events = [row_to_dict(r) for r in conn.execute("SELECT * FROM events ORDER BY created_at DESC LIMIT 10").fetchall()]
    lines = [
        "# Agents OS Runtime Dashboard",
        "",
        f"Generated: {utc_now()}",
        f"State DB: `{paths.db}`",
        "",
        "## Aktivni taskovi",
    ]
    if not tasks:
        lines.append("- Nema taskova.")
    for task in tasks:
        gate = " — Approval gated" if task.get("approval_required") else ""
        lines.append(f"- `{task['id']}` [{task['status']}] p{task['priority']} {task['title']} route={task.get('route') or '-'}{gate}")
    lines.extend(["", "## Pending approvali"])
    if not approvals:
        lines.append("- Nema pending approvala.")
    for approval in approvals:
        lines.append(f"- `{approval['id']}` risk={approval['risk']} task={approval['task_id'] or '-'} {approval['title']}")
    lines.extend(["", "## Zadnji runovi"])
    if not runs:
        lines.append("- Nema runova.")
    for run in runs:
        lines.append(f"- `{run['id']}` [{run['status']}] workflow={run['workflow']} task={run['task_id'] or '-'}")
    lines.extend(["", "## Zadnji eventi"])
    if not events:
        lines.append("- Nema eventa.")
    for event in events:
        lines.append(f"- `{event['id']}` {event['event_type']} task={event['task_id'] or '-'}")
    text = "\n".join(lines) + "\n"
    target = paths.vault_root / "00-command-center" / "RUNTIME-DASHBOARD.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    if args.markdown:
        print(text)
    else:
        print(str(target))
    return 0


def doctor(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        schema_version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        pending_approvals = conn.execute("SELECT COUNT(*) FROM approvals WHERE status='pending'").fetchone()[0]
    required_tables = ["meta", "tasks", "approvals", "artifacts", "events", "runs", "agents", "workflows", "routing_rules", "reviews", "state_snapshots"]
    checks = {
        "state_db_exists": paths.db.exists(),
        "schema_version": schema_version,
        "required_tables_present": all(t in tables for t in required_tables),
        "tables": sorted(tables),
        "artifacts_dir_exists": paths.artifacts.exists(),
        "outbox_dir_exists": paths.outbox.exists(),
        "vault_root_exists": paths.vault_root.exists(),
        "pending_approvals": pending_approvals,
        "network_side_effects": False,
        "runtime_config_changed": False,
    }
    ok = all(v is True for k, v in checks.items() if k.endswith("_exists") or k == "required_tables_present")
    payload = {"ok": ok, "paths": {"db": str(paths.db), "root": str(paths.root), "vault_root": str(paths.vault_root)}, "checks": checks}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Agents OS doctor: " + ("PASS" if ok else "FAIL"))
        for key, value in checks.items():
            print(f"- {key}: {value}")
    return 0 if ok else 1


def _populate_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--vault-root", help="Optional vault mirror root")
    sub = parser.add_subparsers(dest="agents_os_command")

    p = sub.add_parser("init", help="Initialize local Agents OS state")
    p.add_argument("--no-vault", action="store_true", help="Do not mirror runtime note to vault")
    p.set_defaults(func=init_os)

    p = sub.add_parser("status", help="Show state summary")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=status)

    task = sub.add_parser("task", help="Manage tasks")
    task_sub = task.add_subparsers(dest="task_command")
    p = task_sub.add_parser("add", help="Create task")
    p.add_argument("title")
    p.add_argument("--id")
    p.add_argument("--workflow", choices=sorted(SAFE_WORKFLOWS.keys()))
    p.add_argument("--priority", type=int, default=3)
    p.add_argument("--notes", default="")
    p.set_defaults(func=task_add)
    p = task_sub.add_parser("list", help="List tasks")
    p.add_argument("--status", choices=list(TASK_STATUSES))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=task_list)
    p = task_sub.add_parser("set", help="Set task status")
    p.add_argument("id")
    p.add_argument("status", choices=list(TASK_STATUSES))
    p.set_defaults(func=task_set)

    p = sub.add_parser("route", help="Route a task deterministically")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=route_task)

    p = sub.add_parser("next", help="Show next safe task")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=next_task)

    p = sub.add_parser("dashboard", help="Generate local markdown dashboard")
    p.add_argument("--markdown", action="store_true")
    p.set_defaults(func=dashboard)

    approval = sub.add_parser("approval", help="Manage approval gates")
    approval_sub = approval.add_subparsers(dest="approval_command")
    p = approval_sub.add_parser("request", help="Create approval request")
    p.add_argument("title")
    p.add_argument("--id")
    p.add_argument("--risk", default="normal")
    p.add_argument("--task-id")
    p.add_argument("--payload", default="")
    p.set_defaults(func=approval_request)
    p = approval_sub.add_parser("list", help="List approvals")
    p.add_argument("--status", choices=["pending", "approved", "rejected", "cancelled"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=approval_list)
    p = approval_sub.add_parser("set", help="Resolve approval")
    p.add_argument("id")
    p.add_argument("status", choices=["approved", "rejected", "cancelled"])
    p.set_defaults(func=approval_set)

    artifact = sub.add_parser("artifact", help="Manage artifacts")
    artifact_sub = artifact.add_subparsers(dest="artifact_command")
    p = artifact_sub.add_parser("create", help="Create markdown artifact")
    p.add_argument("title")
    p.add_argument("--id")
    p.add_argument("--kind", default="note")
    p.add_argument("--workflow", choices=sorted(SAFE_WORKFLOWS.keys()))
    p.add_argument("--task-id")
    p.add_argument("--body", default="")
    p.set_defaults(func=artifact_create)
    p = artifact_sub.add_parser("list", help="List artifacts")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=artifact_list)

    p = sub.add_parser("run", help="Create task + artifact from a safe workflow")
    p.add_argument("workflow", choices=sorted(SAFE_WORKFLOWS.keys()))
    p.add_argument("input")
    p.add_argument("--title")
    p.add_argument("--task-id")
    p.add_argument("--priority", type=int, default=3)
    p.set_defaults(func=workflow_run)

    p = sub.add_parser("doctor", help="Validate local Agents OS state")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=doctor)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes agents-os", description="Local Agents OS control plane")
    return _populate_parser(parser)


def register_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "agents-os",
        aliases=["aos"],
        help="Local Agents OS control plane",
        description="Manage the local Agents OS state, tasks, approvals, artifacts, and safe workflows.",
    )
    _populate_parser(parser)
    parser.set_defaults(func=cmd_agents_os)
    return parser


def cmd_agents_os(args: argparse.Namespace) -> int:
    if not hasattr(args, "func") or args.func is cmd_agents_os:
        build_parser().print_help()
        return 0
    return args.func(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
