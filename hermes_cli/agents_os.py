"""Local Agents OS control plane for Hermes.

This module is deliberately local-first and side-effect bounded:
- writes only under HERMES_HOME/agents_os by default;
- optionally mirrors human-readable artifacts into a local vault/mirror lane;
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
from hermes_cli.agents_os_idea_factory import draft_idea, idea_factory_schema

DEFAULT_VAULT_DIRNAME = "vault_mirror"
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
AGENTS_OS_TABLES = {
    "meta",
    "tasks",
    "approvals",
    "artifacts",
    "events",
    "runs",
    "agents",
    "workflows",
    "routing_rules",
    "reviews",
    "state_snapshots",
}
WORKFLOW_TEXT_COLUMNS = {
    "route",
    "capabilities",
    "allowed_paths",
    "blocked_paths",
    "approval_risks",
    "precheck",
    "execute",
    "verify",
    "review",
    "close",
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
    vault_root = Path(os.environ.get("AGENTS_OS_VAULT_ROOT", vault_raw or root / DEFAULT_VAULT_DIRNAME)).expanduser()
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
    route TEXT NOT NULL DEFAULT 'local:direct',
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

def _quote_sql_identifier(identifier: str, *, allowed: set[str] | None = None) -> str:
    """Quote a SQLite identifier after strict allowlist validation.

    SQLite does not support binding table/column names as query parameters, so
    identifier-bearing statements must be built from trusted identifiers only.
    """
    if allowed is not None and identifier not in allowed:
        raise ValueError(f"Unexpected SQL identifier: {identifier!r}")
    if not identifier.replace("_", "").isalnum() or not identifier or identifier[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return '"' + identifier.replace('"', '""') + '"'


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    quoted_table = _quote_sql_identifier(table, allowed=AGENTS_OS_TABLES)
    return {row[1] for row in conn.execute("PRAGMA table_info(" + quoted_table + ")").fetchall()}


def _seed_workflows_and_rules(conn: sqlite3.Connection) -> None:
    now = utc_now()
    for workflow, spec in SAFE_WORKFLOWS.items():
        route = "approval_gate" if spec["requires_approval"] else "local:direct"
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
        "route": "'local:direct'",
        "capabilities": "'[]'",
        "allowed_paths": "'[]'",
        "blocked_paths": "'[]'",
        "approval_risks": "'[]'",
        "precheck": "'[]'",
        "execute": "'[]'",
        "verify": "'[]'",
        "review": "'[]'",
        "close": "'[]'",
    }.items():
        if col not in workflow_cols:
            quoted_col = _quote_sql_identifier(col, allowed=WORKFLOW_TEXT_COLUMNS)
            conn.execute("ALTER TABLE workflows ADD COLUMN " + quoted_col + " TEXT NOT NULL DEFAULT " + default)
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
    return None


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
        return {"route": "local:direct", "requires_approval": False, "reason": "default safe local route", "assigned_agent": assigned_agent}
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
        no_agent = not requires_approval and route.get("assigned_agent") is None
        new_status = "needs_approval" if requires_approval else ("blocked" if no_agent else "ready")
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
        "execution_allowed": not requires_approval and not no_agent,
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
        tasks = [row_to_dict(r) for r in conn.execute("""
            SELECT * FROM tasks
            ORDER BY CASE status
                WHEN 'blocked' THEN 0
                WHEN 'needs_approval' THEN 1
                WHEN 'review' THEN 2
                WHEN 'in_progress' THEN 3
                WHEN 'ready' THEN 4
                WHEN 'pending' THEN 5
                WHEN 'routed' THEN 6
                WHEN 'new' THEN 7
                WHEN 'completed' THEN 8
                ELSE 9
            END, priority ASC, created_at ASC
            LIMIT 20
        """).fetchall()]
        approvals = [row_to_dict(r) for r in conn.execute("SELECT * FROM approvals WHERE status='pending' ORDER BY created_at ASC LIMIT 20").fetchall()]
        runs = [row_to_dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 10").fetchall()]
        execution_task_ids = {
            run.get("task_id")
            for run in runs
            if run.get("task_id") and (run.get("completed_at") or run.get("status") in {"succeeded", "failed", "blocked", "needs_review"})
        }
        for run in runs:
            is_execution = run.get("completed_at") or run.get("status") in {"succeeded", "failed", "blocked", "needs_review"}
            if is_execution:
                run["kind"] = "execution"
            elif run.get("task_id") in execution_task_ids:
                run["kind"] = "draft_superseded"
            else:
                run["kind"] = "draft"
        queue_summary = {
            "open_tasks": sum(1 for task in tasks if task["status"] in {"new", "pending", "routed", "ready", "in_progress", "needs_approval"}),
            "blocked_tasks": sum(1 for task in tasks if task["status"] == "blocked"),
            "review_tasks": sum(1 for task in tasks if task["status"] == "review"),
            "completed_tasks": sum(1 for task in tasks if task["status"] == "completed"),
            "pending_approvals": len(approvals),
            "failed_executions": sum(1 for run in runs if run.get("kind") == "execution" and run.get("status") == "failed"),
            "stale_drafts": sum(1 for run in runs if run.get("kind") == "draft"),
        }
        queue_summary["action_required"] = queue_summary["blocked_tasks"] + queue_summary["review_tasks"] + queue_summary["pending_approvals"] + queue_summary["failed_executions"]
        events = [row_to_dict(r) for r in conn.execute("SELECT * FROM events ORDER BY created_at DESC LIMIT 10").fetchall()]
        agents = [row_to_dict(r) for r in conn.execute("SELECT * FROM agents ORDER BY status ASC, created_at ASC LIMIT 20").fetchall()]
        reviews = [row_to_dict(r) for r in conn.execute("SELECT * FROM reviews ORDER BY created_at DESC LIMIT 20").fetchall()]
        snapshots = [row_to_dict(r) for r in conn.execute("SELECT id,label,created_at FROM state_snapshots ORDER BY created_at DESC LIMIT 10").fetchall()]
        recent_completions = []
        for row in conn.execute("SELECT * FROM events WHERE event_type='task_closed' ORDER BY created_at DESC LIMIT 10").fetchall():
            item = row_to_dict(row)
            try:
                payload = json.loads(item.get("payload") or "{}")
            except json.JSONDecodeError:
                payload = {}
            recent_completions.append({
                "event_id": item["id"],
                "task_id": item.get("task_id"),
                "review_id": payload.get("review_id"),
                "evidence": payload.get("evidence"),
                "created_at": item.get("created_at"),
            })
    lines = [
        "# Agents OS Runtime Dashboard",
        "",
        f"Generated: {utc_now()}",
        f"State DB: `{paths.db}`",
        "",
        "## Queue summary",
    ]
    for key in ["action_required", "open_tasks", "blocked_tasks", "review_tasks", "completed_tasks", "pending_approvals", "failed_executions", "stale_drafts"]:
        lines.append(f"- {key}: {queue_summary[key]}")
    lines.extend([
        "",
        "## Aktivni taskovi",
    ])
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
    lines.extend(["", "## Agent registry"])
    if not agents:
        lines.append("- Nema registriranih agenata.")
    for agent in agents:
        caps = agent.get("capabilities") or "[]"
        lines.append(f"- `{agent['id']}` [{agent['status']}] {agent['name']} kind={agent['kind']} capabilities={caps}")
    lines.extend(["", "## Review gateovi"])
    if not reviews:
        lines.append("- Nema review gateova.")
    for review in reviews:
        lines.append(f"- `{review['id']}` [{review['status']}] kind={review['kind']} task={review['task_id'] or '-'} reviewer={review['reviewer'] or '-'}")
    lines.extend(["", "## Snapshoti"])
    if not snapshots:
        lines.append("- Nema snapshotova.")
    for snapshot in snapshots:
        lines.append(f"- `{snapshot['id']}` {snapshot['label']} created={snapshot['created_at']}")
    lines.extend(["", "## Recent completions"])
    if not recent_completions:
        lines.append("- Nema recent completiona.")
    for completion in recent_completions:
        evidence = completion.get("evidence") or "-"
        review_id = completion.get("review_id") or "-"
        lines.append(f"- task={completion.get('task_id') or '-'} review={review_id} evidence={evidence}")
    lines.extend(["", "## Zadnji runovi"])
    if not runs:
        lines.append("- Nema runova.")
    for run in runs:
        lines.append(f"- `{run['id']}` [{run['status']}] kind={run.get('kind', '-')} workflow={run['workflow']} task={run['task_id'] or '-'}")
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
        "queue_summary": queue_summary,
        "tasks": tasks,
        "approvals": approvals,
        "agents": agents,
        "reviews": reviews,
        "snapshots": snapshots,
        "recent_completions": recent_completions,
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
    root_text = str(paths.root).lower()
    policy_home_isolated = str(paths.root).startswith(str(paths.home)) and not any(marker in root_text for marker in ("separate-profile", "external-runtime", "shared-runtime"))
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
    ok = all(v is True for k, v in checks.items() if (k.endswith("_exists") and k != "vault_root_exists") or k in {"required_tables_present", "schema_current", "policy_home_isolated"}) and orphan_records == 0
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
    payload: dict[str, Any] = {}
    for table in tables:
        quoted_table = _quote_sql_identifier(table, allowed=AGENTS_OS_TABLES)
        payload[table] = [row_to_dict(r) for r in conn.execute("SELECT * FROM " + quoted_table).fetchall()]
    return payload


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


def agent_show(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        row = conn.execute("SELECT * FROM agents WHERE id=?", (args.id,)).fetchone()
    if row is None:
        print(json.dumps({"status": "error", "reason": "agent_not_found", "id": args.id}, ensure_ascii=False, indent=2))
        return 2
    payload = row_to_dict(row)
    payload["capabilities"] = json.loads(payload.get("capabilities") or "[]")
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"{payload['id']} [{payload['status']}] {payload['name']}")
    return 0


def agent_set(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        row = conn.execute("SELECT * FROM agents WHERE id=?", (args.id,)).fetchone()
        if row is None:
            print(json.dumps({"status": "error", "reason": "agent_not_found", "id": args.id}, ensure_ascii=False, indent=2))
            return 2
        conn.execute("UPDATE agents SET status=? WHERE id=?", (args.status, args.id))
        log_event(conn, "agent_updated", payload={"agent_id": args.id, "status": args.status})
        conn.commit()
    payload = {"id": args.id, "status": args.status}
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"{args.id} -> {args.status}")
    return 0


def agent_remove(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        row = conn.execute("SELECT * FROM agents WHERE id=?", (args.id,)).fetchone()
        if row is None:
            print(json.dumps({"removed": False, "reason": "agent_not_found", "id": args.id}, ensure_ascii=False, indent=2))
            return 2
        conn.execute("DELETE FROM agents WHERE id=?", (args.id,))
        log_event(conn, "agent_removed", payload={"agent_id": args.id})
        conn.commit()
    print(json.dumps({"removed": True, "id": args.id}, ensure_ascii=False, indent=2) if args.json else f"removed {args.id}")
    return 0



CREDENTIAL_PATH_MARKERS = (".env", "auth.json", "token", "secret", "credential", "password", "api_key", "apikey")


def _credential_like_path(value: str) -> bool:
    lowered = str(value).lower()
    return any(marker in lowered for marker in CREDENTIAL_PATH_MARKERS)


def workflow_validate_payload(data: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    for key in ["id", "kind", "template"]:
        if not data.get(key):
            errors.append(f"missing:{key}")
    if not isinstance(data.get("requires_approval", False), bool):
        errors.append("requires_approval_must_be_bool")
    for field in ["allowed_paths", "blocked_paths"]:
        values = data.get(field, [])
        if values is None:
            values = []
        if not isinstance(values, list):
            errors.append(f"{field}_must_be_list")
            continue
        if any(_credential_like_path(value) for value in values):
            errors.append(f"credential_path:{field}")
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
            "INSERT OR REPLACE INTO workflows(id,kind,requires_approval,template,route,capabilities,allowed_paths,blocked_paths,approval_risks,precheck,execute,verify,review,close,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                data["id"],
                data["kind"],
                1 if data.get("requires_approval") else 0,
                data["template"],
                data.get("route", "local:direct"),
                json.dumps(data.get("capabilities", []), ensure_ascii=False),
                json.dumps(data.get("allowed_paths", []), ensure_ascii=False),
                json.dumps(data.get("blocked_paths", []), ensure_ascii=False),
                json.dumps(data.get("approval_risks", []), ensure_ascii=False),
                json.dumps(data.get("precheck", []), ensure_ascii=False),
                json.dumps(data.get("execute", []), ensure_ascii=False),
                json.dumps(data.get("verify", []), ensure_ascii=False),
                json.dumps(data.get("review", []), ensure_ascii=False),
                json.dumps(data.get("close", []), ensure_ascii=False),
                now,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO routing_rules(id,workflow,route,requires_approval,priority,created_at) VALUES(?,?,?,?,?,?)",
            (f"rule-{data['id']}", data["id"], data.get("route", "local:direct"), 1 if data.get("requires_approval") else 0, 50, now),
        )
        log_event(conn, "workflow_imported", payload={"workflow_id": data["id"]})
        conn.commit()
    print(json.dumps({"workflow_id": data["id"], "valid": True}, ensure_ascii=False, indent=2))
    return 0


def _decode_workflow_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = row_to_dict(row)
    for key in ["capabilities", "allowed_paths", "blocked_paths", "approval_risks", "precheck", "execute", "verify", "review", "close"]:
        try:
            payload[key] = json.loads(payload.get(key) or "[]")
        except json.JSONDecodeError:
            payload[key] = []
    payload["requires_approval"] = bool(payload.get("requires_approval"))
    return payload


def workflow_list_cmd(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        rows = [_decode_workflow_row(r) for r in conn.execute("SELECT * FROM workflows ORDER BY id ASC").fetchall()]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def workflow_show_cmd(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    with connect(paths) as conn:
        row = conn.execute("SELECT * FROM workflows WHERE id=?", (args.id,)).fetchone()
    if row is None:
        print(json.dumps({"status": "error", "reason": "workflow_not_found", "id": args.id}, ensure_ascii=False, indent=2))
        return 2
    payload = _decode_workflow_row(row)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload["id"])
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
        if args.dry_run:
            log_event(conn, "execution_dry_run", task_id=task_id, run_id=run_id, payload={"status": "dry_run"})
            conn.commit()
            print(json.dumps({"task_id": task_id, "run_id": run_id, "status": "dry_run"}, ensure_ascii=False, indent=2))
            return 0
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


def close_task(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    evidence = (args.evidence or "").strip()
    review_id = args.review_id
    with connect(paths) as conn:
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (args.id,)).fetchone()
        if task is None:
            payload = {"status": "error", "reason": "task_not_found", "task_id": args.id}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 2
        if task["approval_required"] or _pending_approval_count(conn, args.id) > 0 or task["status"] == "needs_approval":
            payload = {"status": "blocked", "reason": "approval_required", "task_id": args.id}
            log_event(conn, "close_blocked", task_id=args.id, payload={"reason": "approval_required"})
            conn.commit()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 2
        approved_review = None
        if review_id:
            approved_review = conn.execute("SELECT * FROM reviews WHERE id=? AND task_id=? AND status='approved'", (review_id, args.id)).fetchone()
            if approved_review is None:
                payload = {"status": "error", "reason": "approved_review_not_found", "task_id": args.id, "review_id": review_id}
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 2
        if not evidence and approved_review is None:
            payload = {"status": "error", "reason": "evidence_or_approved_review_required", "task_id": args.id}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 2
        now = utc_now()
        conn.execute("UPDATE tasks SET status='completed', updated_at=? WHERE id=?", (now, args.id))
        event_payload = {"evidence": evidence or None, "review_id": review_id}
        log_event(conn, "task_closed", task_id=args.id, run_id=approved_review["run_id"] if approved_review else None, payload=event_payload)
        conn.commit()
    result = {"task_id": args.id, "status": "completed", "evidence": evidence or None, "review_id": review_id}
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else f"{args.id} -> completed")
    return 0



def mirror_validate(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    dashboard_path = paths.vault_root / "00-command-center" / "RUNTIME-DASHBOARD.md"
    issues = []
    if not dashboard_path.exists():
        issues.append("missing_dashboard")
    credential_like_matches = 0
    if paths.vault_root.exists():
        for md_path in paths.vault_root.rglob("*.md"):
            try:
                text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            credential_like_matches += leak_scan_text(text)
    if credential_like_matches:
        issues.append("credential_like_markdown")
    payload = {"status": "ok" if not issues else "attention", "dashboard_path": str(dashboard_path), "issues": issues, "credential_like_matches": credential_like_matches}
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload["status"])
    return 0 if not issues else 1


def mirror_rebuild(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    dashboard_path = paths.vault_root / "00-command-center" / "RUNTIME-DASHBOARD.md"
    # Reuse dashboard generator for canonical command-center read-back without leaking its stdout into JSON mode.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        dashboard(argparse.Namespace(vault_root=str(paths.vault_root), json=False, markdown=False))
    payload = {"status": "ok", "dashboard_path": str(dashboard_path)}
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else str(dashboard_path))
    return 0



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

    def _json_from_command(self, func) -> dict[str, Any]:
        import contextlib
        import io
        buf = io.StringIO()
        args = argparse.Namespace(vault_root=str(self.paths.vault_root), json=True, markdown=False)
        with contextlib.redirect_stdout(buf):
            func(args)
        return json.loads(buf.getvalue())

    def doctor_payload(self) -> dict[str, Any]:
        return self._json_from_command(doctor)

    def dashboard_payload(self) -> dict[str, Any]:
        return self._json_from_command(dashboard)

    def maintenance_payload(self) -> dict[str, Any]:
        return self._json_from_command(maintenance)


def service_status(args: argparse.Namespace) -> int:
    payload = AgentsOSService(resolve_paths(args)).status_payload()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{payload['status']} schema={payload['schema_version']} db={payload['state_db']}")
    return 0


def idea_schema_cmd(args: argparse.Namespace) -> int:
    payload = idea_factory_schema()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Idea Factory v0 schema")
        print("input_fields: " + ", ".join(payload["input_fields"]))
        print("output_fields: " + ", ".join(payload["output_fields"]))
    return 0


def idea_draft_cmd(args: argparse.Namespace) -> int:
    source_links = args.source_link or []
    payload = draft_idea(
        args.idea_text,
        context=args.context,
        desired_output=args.desired_output,
        urgency=args.urgency,
        source_links=source_links,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{payload['idea_id']} {payload['classification']} risk={payload['risk_class']} approval_required={payload['approval_required']}")
    return 0


def docs_cmd(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    service = AgentsOSService(paths)
    payload = service.status_payload()
    docs_dir = paths.vault_root / "90-docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_path = docs_dir / "AGENTS-OS-RUNTIME.md"
    command_reference_path = docs_dir / "COMMAND-REFERENCE.md"
    recovery_runbook_path = docs_dir / "RECOVERY-RUNBOOK.md"
    safety_policy_path = docs_dir / "SAFETY-POLICY.md"

    runtime_text = textwrap.dedent(
        f"""
        # Agents OS Runtime

        - generated: {utc_now()}
        - schema_version: {payload['schema_version']}
        - state_db: {payload['state_db']}
        - agents_os_home: {payload['agents_os_home']}
        - vault_root: {paths.vault_root}

        ## Local API adapter

        `AgentsOSService` is an importable local adapter. It does not start a server,
        send network requests, restart Hermes, deploy anything, or mutate runtime configuration.

        Exposed local payload methods:
        - `status_payload()`
        - `doctor_payload()`
        - `dashboard_payload()`
        - `maintenance_payload()`
        """
    ).strip() + "\n"
    command_text = textwrap.dedent(
        """
        # Agents OS Command Reference

        Safe local verification:

        ```bash
        export HERMES_HOME=/path/to/hermes-home
        python -m hermes_cli.agents_os status --json
        python -m hermes_cli.agents_os doctor --json
        python -m hermes_cli.agents_os service status --json
        python -m hermes_cli.agents_os dashboard --json
        python -m hermes_cli.agents_os maintenance --json
        python -m hermes_cli.agents_os mirror validate --json
        python -m hermes_cli.agents_os docs --json
        ```

        Closeout path:
        - create/route/execute safe local task
        - request/set review when needed
        - close with `agents-os close <task-id> --evidence <path>` or approved `--review-id`
        """
    ).strip() + "\n"
    recovery_text = textwrap.dedent(
        f"""
        # Agents OS Recovery Runbook

        1. Set `HERMES_HOME=/path/to/hermes-home`.
        2. Run `python -m hermes_cli.agents_os doctor --json`.
        3. If dashboard mirror is missing, run `python -m hermes_cli.agents_os mirror rebuild --json`.
        4. Re-run `python -m hermes_cli.agents_os mirror validate --json`.
        5. Treat SQLite DB as runtime authority: `{paths.db}`.
        6. Treat vault as mirror/read-back only: `{paths.vault_root}`.
        """
    ).strip() + "\n"
    safety_text = textwrap.dedent(
        """
        # Agents OS Safety Policy

        Hard boundaries:
        - no deploy
        - no gateway restart
        - no web UI, TUI, daemon, or server process without explicit approval
        - no credentials, API keys, auth stores, tokens, `.env`, or secrets in reports/dashboard
        - no writes to other Hermes profiles or external agent runtimes
        - no legacy or unrelated Hermes home as active runtime authority
        - outbound/public/financial/security/destructive work must become approval draft, not execution
        """
    ).strip() + "\n"

    docs_path.write_text(runtime_text, encoding="utf-8")
    command_reference_path.write_text(command_text, encoding="utf-8")
    recovery_runbook_path.write_text(recovery_text, encoding="utf-8")
    safety_policy_path.write_text(safety_text, encoding="utf-8")
    docs = {
        "runtime": str(docs_path),
        "command_reference": str(command_reference_path),
        "recovery_runbook": str(recovery_runbook_path),
        "safety_policy": str(safety_policy_path),
    }
    result = {"status": "ok", "docs_path": str(docs_path), "docs": docs, "schema_version": payload["schema_version"]}
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
    p.add_argument("--workflow")
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

    p = sub.add_parser("tui", help="Open interactive Mission Control TUI")
    p.add_argument("--json", action="store_true", help="Print TUI capability/status payload without starting curses")
    p.add_argument("--script", help="Run deterministic key script and print final frame (test/automation helper)")
    p.add_argument("--width", type=int, default=100, help=argparse.SUPPRESS)
    p.add_argument("--height", type=int, default=30, help=argparse.SUPPRESS)
    from hermes_cli.agents_os_tui import tui_cmd
    p.set_defaults(func=tui_cmd)

    p = sub.add_parser("web", help="Open local-only Agents OS Mission Control dashboard")
    p.add_argument("--host", default="127.0.0.1", help="Local bind host; only 127.0.0.1 or localhost are allowed")
    p.add_argument("--port", type=int, default=18790, help="Local dashboard port (default: 18790)")
    p.add_argument("--open", action="store_true", help="Open dashboard in the default browser")
    p.add_argument("--json", action="store_true", help="Print launcher/status payload without starting a server")
    def _web_cmd(args: argparse.Namespace) -> int:
        from hermes_cli.agents_os_web import web_cmd
        return web_cmd(args)
    p.set_defaults(func=_web_cmd)

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
    p = agent_sub.add_parser("show", help="Show agent")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=agent_show)
    p = agent_sub.add_parser("set", help="Update agent status")
    p.add_argument("id")
    p.add_argument("--status", required=True, choices=["available", "busy", "paused", "disabled"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=agent_set)
    p = agent_sub.add_parser("remove", help="Remove agent")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=agent_remove)

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
    p = workflow_sub.add_parser("show", help="Show workflow")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=workflow_show_cmd)

    p = sub.add_parser("execute", help="Execute a safe local task")
    p.add_argument("id")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=execute_task)

    p = sub.add_parser("execute-next", help="Execute next safe local task")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=execute_next)

    p = sub.add_parser("close", help="Close a task with evidence or approved review")
    p.add_argument("id")
    p.add_argument("--evidence", default="")
    p.add_argument("--review-id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=close_task)

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

    mirror = sub.add_parser("mirror", help="Validate or rebuild vault mirror")
    mirror_sub = mirror.add_subparsers(dest="mirror_command")
    p = mirror_sub.add_parser("validate", help="Validate vault mirror")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=mirror_validate)
    p = mirror_sub.add_parser("rebuild", help="Rebuild vault mirror read-back files")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=mirror_rebuild)

    service = sub.add_parser("service", help="Local importable service adapter")
    service_sub = service.add_subparsers(dest="service_command")
    p = service_sub.add_parser("status", help="Show service adapter status payload")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=service_status)

    idea = sub.add_parser("idea", help="Draft local Idea Factory plans")
    idea_sub = idea.add_subparsers(dest="idea_command")
    p = idea_sub.add_parser("schema", help="Show Idea Factory v0 schema")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=idea_schema_cmd)
    p = idea_sub.add_parser("draft", help="Draft and classify an idea without executing it")
    p.add_argument("idea_text")
    p.add_argument("--context")
    p.add_argument("--desired-output")
    p.add_argument("--urgency", default="normal", choices=["low", "normal", "high"])
    p.add_argument("--source-link", action="append")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=idea_draft_cmd)

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
