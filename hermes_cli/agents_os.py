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
import re
import shutil
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

SCHEMA_VERSION = "3"
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
    status TEXT NOT NULL DEFAULT 'created' CHECK(status IN ('created','queued','running','succeeded','failed','blocked','needs_review')),
    input TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'local',
    status TEXT NOT NULL DEFAULT 'available' CHECK(status IN ('available','busy','paused','disabled')),
    capabilities TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    requires_approval INTEGER NOT NULL DEFAULT 0,
    template TEXT NOT NULL DEFAULT '',
    route TEXT NOT NULL DEFAULT 'doni:direct',
    capabilities TEXT NOT NULL DEFAULT '[]',
    allowed_paths TEXT NOT NULL DEFAULT '[]',
    blocked_paths TEXT NOT NULL DEFAULT '[]',
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
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','changes_requested','rejected','cancelled')),
    kind TEXT NOT NULL DEFAULT 'general',
    reviewer TEXT NOT NULL DEFAULT '',
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
        route = "approval_gate" if spec["requires_approval"] else "doni:direct"
        caps = json.dumps([spec["kind"], "code" if workflow in {"code-task", "qa-report"} else spec["kind"]], ensure_ascii=False)
        conn.execute(
            "INSERT OR IGNORE INTO workflows(id,kind,requires_approval,template,route,capabilities,allowed_paths,blocked_paths,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (workflow, spec["kind"], 1 if spec["requires_approval"] else 0, spec["template"], route, caps, "[]", "[]", now),
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
    agent_cols = _table_columns(conn, "agents")
    if "capabilities" not in agent_cols:
        conn.execute("ALTER TABLE agents ADD COLUMN capabilities TEXT NOT NULL DEFAULT '[]'")
    workflow_cols = _table_columns(conn, "workflows")
    for col, default in {
        "route": "'doni:direct'",
        "capabilities": "'[]'",
        "allowed_paths": "'[]'",
        "blocked_paths": "'[]'",
    }.items():
        if col not in workflow_cols:
            conn.execute(f"ALTER TABLE workflows ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}")
    review_cols = _table_columns(conn, "reviews")
    if "kind" not in review_cols:
        conn.execute("ALTER TABLE reviews ADD COLUMN kind TEXT NOT NULL DEFAULT 'general'")
    if "reviewer" not in review_cols:
        conn.execute("ALTER TABLE reviews ADD COLUMN reviewer TEXT NOT NULL DEFAULT ''")
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


def _pick_agent(conn: sqlite3.Connection, task: sqlite3.Row) -> str | None:
    needed: set[str] = set()
    if task["workflow"]:
        wf = conn.execute("SELECT * FROM workflows WHERE id=?", (task["workflow"],)).fetchone()
        if wf:
            try:
                needed = set(json.loads(wf["capabilities"] or "[]"))
            except json.JSONDecodeError:
                needed = set()
    agents = conn.execute("SELECT * FROM agents WHERE status='available' ORDER BY created_at ASC").fetchall()
    for agent in agents:
        try:
            caps = set(json.loads(agent["capabilities"] or "[]"))
        except json.JSONDecodeError:
            caps = set()
        if not needed or needed.intersection(caps):
            return agent["id"]
    return agents[0]["id"] if agents else None


def _route_for_task(conn: sqlite3.Connection, task: sqlite3.Row) -> dict[str, Any]:
    rule = None
    if task["workflow"]:
        rule = conn.execute(
            "SELECT * FROM routing_rules WHERE workflow=? ORDER BY priority ASC LIMIT 1",
            (task["workflow"],),
        ).fetchone()
    assigned_agent = _pick_agent(conn, task)
    if rule is None:
        notes = (task["notes"] or "") + " " + task["title"]
        lowered = notes.lower()
        if any(word in lowered for word in ["credential", "token", "api key", "deploy", "publish", "public", "delete", "finance", "gateway restart"]):
            return {"route": "approval_gate", "requires_approval": True, "reason": "risk keyword", "assigned_agent": None}
        return {"route": "doni:direct", "requires_approval": False, "reason": "default safe local route", "assigned_agent": assigned_agent}
    return {"route": rule["route"], "requires_approval": bool(rule["requires_approval"]), "reason": f"workflow:{task['workflow']}", "assigned_agent": None if rule["requires_approval"] else assigned_agent}


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
        log_event(conn, "routed", task_id=args.id, payload={"route": route["route"], "status": new_status, "reason": route["reason"], "assigned_agent": route.get("assigned_agent")})
        conn.commit()
    payload = {
        "task_id": args.id,
        "route": route["route"],
        "reason": route["reason"],
        "assigned_agent": route.get("assigned_agent"),
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
    payload = {
        "health": {"ok": True, "network_side_effects": False, "runtime_config_changed": False},
        "dashboard_path": str(target),
        "tasks": tasks,
        "approvals": approvals,
        "runs": runs,
        "events": events,
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.markdown:
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
        orphan_artifacts = conn.execute("SELECT COUNT(*) FROM artifacts WHERE task_id IS NOT NULL AND task_id != '' AND task_id NOT IN (SELECT id FROM tasks)").fetchone()[0]
        orphan_events = conn.execute("SELECT COUNT(*) FROM events WHERE task_id IS NOT NULL AND task_id != '' AND task_id NOT IN (SELECT id FROM tasks)").fetchone()[0]
        orphan_runs = conn.execute("SELECT COUNT(*) FROM runs WHERE task_id IS NOT NULL AND task_id != '' AND task_id NOT IN (SELECT id FROM tasks)").fetchone()[0]
    required_tables = ["meta", "tasks", "approvals", "artifacts", "events", "runs", "agents", "workflows", "routing_rules", "reviews", "state_snapshots"]
    orphan_records = orphan_artifacts + orphan_events + orphan_runs
    policy_home_isolated = str(paths.root).startswith(str(paths.home)) and ".hermes-marija" not in str(paths.root) and ".openclaw" not in str(paths.root)
    checks = {
        "state_db_exists": paths.db.exists(),
        "schema_version": schema_version,
        "schema_current": schema_version == SCHEMA_VERSION,
        "required_tables_present": all(t in tables for t in required_tables),
        "tables": sorted(tables),
        "artifacts_dir_exists": paths.artifacts.exists(),
        "outbox_dir_exists": paths.outbox.exists(),
        "vault_root_exists": paths.vault_root.exists(),
        "pending_approvals": pending_approvals,
        "orphan_records": orphan_records,
        "policy_home_isolated": policy_home_isolated,
        "network_side_effects": False,
        "runtime_config_changed": False,
    }
    ok = all(v is True for k, v in checks.items() if k.endswith("_exists") or k in {"required_tables_present", "schema_current", "policy_home_isolated"}) and orphan_records == 0
    payload = {"ok": ok, "paths": {"db": str(paths.db), "root": str(paths.root), "vault_root": str(paths.vault_root)}, "checks": checks}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Agents OS doctor: " + ("PASS" if ok else "FAIL"))
        for key, value in checks.items():
            print(f"- {key}: {value}")
    return 0 if ok else 1


def _snapshot_payload(conn: sqlite3.Connection) -> dict[str, Any]:
    tables = ["meta", "tasks", "approvals", "artifacts", "events", "runs", "agents", "workflows", "routing_rules", "reviews"]
    return {table: [row_to_dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()] for table in tables}


def snapshot_cmd(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        if args.snapshot_command == "create":
            sid = args.id or f"snapshot-{uuid.uuid4().hex[:8]}"
            payload = _snapshot_payload(conn)
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            conn.execute("INSERT INTO state_snapshots(id,label,payload,created_at) VALUES(?,?,?,?)", (sid, args.label, text, utc_now()))
            export_path = paths.root / "snapshots" / f"{sid}.json"
            write_json(export_path, {"id": sid, "label": args.label, "payload": payload})
            log_event(conn, "snapshot_created", payload={"snapshot_id": sid, "export_path": str(export_path)})
            conn.commit()
            print(json.dumps({"snapshot_id": sid, "label": args.label, "export_path": str(export_path)}, ensure_ascii=False, indent=2))
        elif args.snapshot_command == "list":
            rows = [row_to_dict(r) for r in conn.execute("SELECT id,label,created_at FROM state_snapshots ORDER BY created_at DESC").fetchall()]
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        elif args.snapshot_command == "export":
            row = conn.execute("SELECT * FROM state_snapshots WHERE id=?", (args.id,)).fetchone()
            if row is None:
                print(f"Snapshot ne postoji: {args.id}", file=sys.stderr)
                return 2
            target = Path(args.output) if args.output else paths.root / "snapshots" / f"{args.id}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(row["payload"], encoding="utf-8")
            print(json.dumps({"snapshot_id": args.id, "export_path": str(target)}, ensure_ascii=False, indent=2))
        elif args.snapshot_command == "restore":
            row = conn.execute("SELECT * FROM state_snapshots WHERE id=?", (args.id,)).fetchone()
            if row is None:
                print(f"Snapshot ne postoji: {args.id}", file=sys.stderr)
                return 2
            log_event(conn, "snapshot_restore_requested", payload={"snapshot_id": args.id, "mode": "dry-run"})
            conn.commit()
            print(json.dumps({"snapshot_id": args.id, "status": "validated", "mode": "dry-run"}, ensure_ascii=False, indent=2))
    return 0


def agent_add(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    caps = [c.strip() for c in (args.capabilities or "").split(",") if c.strip()]
    with connect(paths) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO agents(id,name,kind,status,capabilities,created_at) VALUES(?,?,?,?,?,?)",
            (args.id, args.name or args.id, args.kind, args.status, json.dumps(caps, ensure_ascii=False), utc_now()),
        )
        log_event(conn, "agent_upserted", payload={"agent_id": args.id, "capabilities": caps})
        conn.commit()
    print(json.dumps({"id": args.id, "name": args.name or args.id, "kind": args.kind, "status": args.status, "capabilities": caps}, ensure_ascii=False, indent=2))
    return 0


def agent_list(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        rows = [row_to_dict(r) for r in conn.execute("SELECT * FROM agents ORDER BY created_at ASC").fetchall()]
    for row in rows:
        row["capabilities"] = json.loads(row.get("capabilities") or "[]")
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            print(f"{row['id']} [{row['status']}] {row['name']} caps={','.join(row['capabilities'])}")
    return 0


def workflow_validate_payload(data: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    for key in ["id", "kind", "template"]:
        if not data.get(key):
            errors.append(f"missing:{key}")
    if not isinstance(data.get("requires_approval", False), bool):
        errors.append("requires_approval_must_be_bool")
    return not errors, errors


def workflow_validate_cmd(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    valid, errors = workflow_validate_payload(data)
    print(json.dumps({"valid": valid, "errors": errors, "workflow_id": data.get("id")}, ensure_ascii=False, indent=2))
    return 0 if valid else 2


def workflow_import_cmd(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    valid, errors = workflow_validate_payload(data)
    if not valid:
        print(json.dumps({"valid": False, "errors": errors}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    now = utc_now()
    with connect(paths) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO workflows(id,kind,requires_approval,template,route,capabilities,allowed_paths,blocked_paths,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (data["id"], data["kind"], 1 if data.get("requires_approval") else 0, data["template"], data.get("route", "doni:direct"), json.dumps(data.get("capabilities", []), ensure_ascii=False), json.dumps(data.get("allowed_paths", []), ensure_ascii=False), json.dumps(data.get("blocked_paths", []), ensure_ascii=False), now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO routing_rules(id,workflow,route,requires_approval,priority,created_at) VALUES(?,?,?,?,?,?)",
            (f"rule-{data['id']}", data["id"], data.get("route", "doni:direct"), 1 if data.get("requires_approval") else 0, 50, now),
        )
        log_event(conn, "workflow_imported", payload={"workflow_id": data["id"]})
        conn.commit()
    print(json.dumps({"workflow_id": data["id"], "valid": True}, ensure_ascii=False, indent=2))
    return 0


def workflow_list_cmd(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        rows = [row_to_dict(r) for r in conn.execute("SELECT * FROM workflows ORDER BY id ASC").fetchall()]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def execute_task(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        task_id = args.id
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None:
            print(json.dumps({"status": "error", "reason": "task_not_found"}, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        if task["approval_required"] or _pending_approval_count(conn, task_id) > 0 or task["status"] == "needs_approval":
            log_event(conn, "execution_blocked", task_id=task_id, payload={"reason": "approval_required"})
            conn.commit()
            print(json.dumps({"task_id": task_id, "status": "blocked", "reason": "approval_required"}, ensure_ascii=False, indent=2))
            return 2
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        now = utc_now()
        log_path = paths.artifacts / "runs" / f"{now.split('T',1)[0]}-{run_id}.md"
        body = f"## Execution log\n\n- task_id: {task_id}\n- workflow: {task['workflow'] or ''}\n- dry_run: {bool(args.dry_run)}\n- status: succeeded\n"
        write_markdown(log_path, f"Run {run_id}", body, {"run_id": run_id, "task_id": task_id, "status": "succeeded", "created_at": now})
        conn.execute("INSERT INTO runs(id,task_id,workflow,status,input,created_at,completed_at) VALUES(?,?,?,?,?,?,?)", (run_id, task_id, task["workflow"] or "manual", "succeeded", task["notes"] or "", now, utc_now()))
        artifact_id = f"artifact-{uuid.uuid4().hex[:8]}"
        conn.execute("INSERT INTO artifacts(id,kind,title,path,task_id,workflow,created_at,run_id) VALUES(?,?,?,?,?,?,?,?)", (artifact_id, "run-log", f"Run log {run_id}", str(log_path), task_id, task["workflow"], now, run_id))
        conn.execute("UPDATE tasks SET status='review', updated_at=? WHERE id=?", (utc_now(), task_id))
        log_event(conn, "executed", task_id=task_id, run_id=run_id, payload={"status": "succeeded", "log_path": str(log_path)})
        conn.commit()
    print(json.dumps({"task_id": task_id, "run_id": run_id, "status": "succeeded", "log_path": str(log_path)}, ensure_ascii=False, indent=2))
    return 0


def execute_next(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        task = conn.execute("SELECT * FROM tasks WHERE status IN ('ready','pending','routed') AND approval_required=0 ORDER BY CASE status WHEN 'ready' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END, priority ASC, created_at ASC LIMIT 1").fetchone()
    if task is None:
        print(json.dumps({"task": None, "status": "empty"}, ensure_ascii=False, indent=2))
        return 0
    args.id = task["id"]
    return execute_task(args)


def review_request(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    rid = args.id or f"review-{uuid.uuid4().hex[:8]}"
    with connect(paths) as conn:
        conn.execute("INSERT INTO reviews(id,task_id,run_id,status,kind,reviewer,notes,created_at) VALUES(?,?,?,?,?,?,?,?)", (rid, args.task_id, args.run_id, "pending", args.kind, args.reviewer or "", args.notes or "", utc_now()))
        log_event(conn, "review_requested", task_id=args.task_id, run_id=args.run_id, payload={"review_id": rid, "kind": args.kind})
        conn.commit()
    print(json.dumps({"review_id": rid, "task_id": args.task_id, "status": "pending", "kind": args.kind}, ensure_ascii=False, indent=2))
    return 0


def review_set(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        row = conn.execute("SELECT * FROM reviews WHERE id=?", (args.id,)).fetchone()
        if row is None:
            print(json.dumps({"status": "error", "reason": "review_not_found"}, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        conn.execute("UPDATE reviews SET status=?, notes=? WHERE id=?", (args.status, args.notes or row["notes"], args.id))
        if args.status == "approved" and row["task_id"]:
            log_event(conn, "verified", task_id=row["task_id"], run_id=row["run_id"], payload={"review_id": args.id})
        log_event(conn, "review_updated", task_id=row["task_id"], run_id=row["run_id"], payload={"review_id": args.id, "status": args.status})
        conn.commit()
    print(json.dumps({"review_id": args.id, "status": args.status}, ensure_ascii=False, indent=2))
    return 0


def leak_scan_text(text: str) -> int:
    return len(re.findall(r"(?i)(api[_-]?key|token|secret|password|credential)\s*[:=]\s*['\"][^'\"]+", text))


def maintenance(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    md_args = argparse.Namespace(vault_root=str(paths.vault_root), markdown=True)
    # Build compact local report without sending anywhere.
    with connect(paths) as conn:
        pending = conn.execute("SELECT COUNT(*) FROM approvals WHERE status='pending'").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM runs WHERE status='failed'").fetchone()[0]
        tasks = conn.execute("SELECT COUNT(*) FROM tasks WHERE status NOT IN ('completed','cancelled')").fetchone()[0]
    report = f"# Agents OS maintenance\n\n- generated: {utc_now()}\n- open_tasks: {tasks}\n- pending_approvals: {pending}\n- failed_runs: {failed}\n- network_side_effects: false\n"
    report_path = paths.vault_root / "99-verification" / f"{utc_now().split('T',1)[0]}-maintenance.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    leaks = leak_scan_text(report)
    payload = {"status": "ok" if failed == 0 and leaks == 0 else "attention", "report_path": str(report_path), "open_tasks": tasks, "pending_approvals": pending, "failed_runs": failed, "credential_like_matches": leaks}
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else str(report_path))
    return 0 if payload["status"] == "ok" else 1


class AgentsOSService:
    """Importable local service adapter for Agents OS without shelling out or starting a server."""

    def __init__(self, paths: AgentsOSPaths | None = None):
        self.paths = paths or resolve_paths(None)

    def status_payload(self) -> dict[str, Any]:
        with connect(self.paths) as conn:
            counts = {
                "tasks": dict(conn.execute("SELECT status, COUNT(*) c FROM tasks GROUP BY status").fetchall()),
                "approvals": dict(conn.execute("SELECT status, COUNT(*) c FROM approvals GROUP BY status").fetchall()),
                "artifacts": conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
            }
            schema_version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        return {"status": "ok", "schema_version": schema_version, "agents_os_home": str(self.paths.root), "state_db": str(self.paths.db), "counts": counts}


def service_status(args: argparse.Namespace) -> int:
    payload = AgentsOSService(resolve_paths(args)).status_payload()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{payload['status']} schema={payload['schema_version']} db={payload['state_db']}")
    return 0


def docs_cmd(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    payload = AgentsOSService(paths).status_payload()
    docs_path = paths.vault_root / "90-docs" / "AGENTS-OS-RUNTIME.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    text = textwrap.dedent(
        f"""
        # Agents OS Runtime

        - generated: {utc_now()}
        - schema_version: {payload['schema_version']}
        - state_db: {payload['state_db']}
        - agents_os_home: {payload['agents_os_home']}

        ## Local API adapter

        `AgentsOSService` is an importable local adapter. It does not start a server,
        send network requests, restart Hermes, or mutate runtime configuration.

        ## Safe closeout commands

        ```bash
        export HERMES_HOME=/home/goran/.hermes-doni-clean
        python -m hermes_cli.agents_os status --json
        python -m hermes_cli.agents_os doctor --json
        python -m hermes_cli.agents_os service status --json
        python -m hermes_cli.agents_os dashboard --json
        python -m hermes_cli.agents_os maintenance --json
        ```
        """
    ).strip() + "\n"
    docs_path.write_text(text, encoding="utf-8")
    result = {"status": "ok", "docs_path": str(docs_path), "schema_version": payload["schema_version"]}
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else str(docs_path))
    return 0


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
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=dashboard)

    snapshot = sub.add_parser("snapshot", help="Manage local state snapshots")
    snapshot_sub = snapshot.add_subparsers(dest="snapshot_command")
    p = snapshot_sub.add_parser("create", help="Create snapshot")
    p.add_argument("label")
    p.add_argument("--id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=snapshot_cmd)
    p = snapshot_sub.add_parser("list", help="List snapshots")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=snapshot_cmd)
    p = snapshot_sub.add_parser("export", help="Export snapshot")
    p.add_argument("id")
    p.add_argument("--output")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=snapshot_cmd)
    p = snapshot_sub.add_parser("restore", help="Validate snapshot restore plan")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=snapshot_cmd)

    agent = sub.add_parser("agent", help="Manage local agents")
    agent_sub = agent.add_subparsers(dest="agent_command")
    p = agent_sub.add_parser("add", help="Register or update agent")
    p.add_argument("id")
    p.add_argument("--name")
    p.add_argument("--kind", default="local")
    p.add_argument("--status", default="available", choices=["available", "busy", "paused", "disabled"])
    p.add_argument("--capabilities", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=agent_add)
    p = agent_sub.add_parser("list", help="List agents")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=agent_list)

    workflow = sub.add_parser("workflow", help="Validate/import workflow plugins")
    workflow_sub = workflow.add_subparsers(dest="workflow_command")
    p = workflow_sub.add_parser("validate", help="Validate workflow JSON")
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=workflow_validate_cmd)
    p = workflow_sub.add_parser("import", help="Import workflow JSON")
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=workflow_import_cmd)
    p = workflow_sub.add_parser("list", help="List workflows")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=workflow_list_cmd)

    p = sub.add_parser("execute", help="Execute a safe local task")
    p.add_argument("id")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=execute_task)

    p = sub.add_parser("execute-next", help="Execute next safe local task")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=execute_next)

    review = sub.add_parser("review", help="Manage review gates")
    review_sub = review.add_subparsers(dest="review_command")
    p = review_sub.add_parser("request", help="Request review")
    p.add_argument("task_id")
    p.add_argument("--run-id")
    p.add_argument("--id")
    p.add_argument("--kind", default="general")
    p.add_argument("--reviewer")
    p.add_argument("--notes", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=review_request)
    p = review_sub.add_parser("set", help="Resolve review")
    p.add_argument("id")
    p.add_argument("status", choices=["approved", "changes_requested", "rejected", "cancelled"])
    p.add_argument("--notes", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=review_set)

    p = sub.add_parser("maintenance", help="Run local maintenance checks")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=maintenance)

    service = sub.add_parser("service", help="Local importable service adapter")
    service_sub = service.add_subparsers(dest="service_command")
    p = service_sub.add_parser("status", help="Show service adapter status payload")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=service_status)

    p = sub.add_parser("docs", help="Generate local Agents OS runtime docs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=docs_cmd)

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
