#!/usr/bin/env python3
"""Import the legacy local Factory SQLite DB into Agent Core Postgres.

This is an idempotent bootstrap bridge: SQLite remains untouched and Postgres
receives upserted rows in the `factory` schema.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / "runtime" / "agent-core-db" / ".env"
DEFAULT_SQLITE = Path.home() / ".hermes" / "factory" / "factory.db"


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env() -> dict[str, str]:
    out = dict(os.environ)
    out.setdefault("AGENT_DB_ADMIN_USER", "agent_admin")
    out.setdefault("AGENT_DB_NAME", "zeus_agent")
    out.update(load_env_file(ENV_FILE))
    return out


def lit(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def json_lit(value: str | None, default: str) -> str:
    raw = value if value not in (None, "") else default
    try:
        normalized = json.dumps(json.loads(raw), separators=(",", ":"))
    except Exception:
        normalized = default
    return lit(normalized) + "::jsonb"


def run_psql(sql: str, runtime_env: dict[str, str]) -> None:
    cmd = [
        "docker", "exec", "-i", "agent-postgres",
        "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1",
        "-U", runtime_env["AGENT_DB_ADMIN_USER"],
        "-d", runtime_env["AGENT_DB_NAME"],
    ]
    subprocess.run(cmd, input=sql, text=True, cwd=REPO_ROOT, env=runtime_env, check=True)


def rows(con: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    try:
        return list(con.execute(f"SELECT * FROM {table}"))
    except sqlite3.OperationalError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Factory SQLite rows into Agent Core Postgres")
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    args = parser.parse_args()
    if not args.sqlite.exists():
        raise SystemExit(f"SQLite DB not found: {args.sqlite}")

    runtime_env = env()
    con = sqlite3.connect(args.sqlite)
    con.row_factory = sqlite3.Row
    statements: list[str] = ["BEGIN;"]

    for r in rows(con, "factory_agents"):
        statements.append(f"""
INSERT INTO factory.agents(agent_id, profile_name, display_name, role, status, preferred_engine, metadata, created_at, updated_at)
VALUES ({lit(r['agent_id'])}, {lit(r['agent_id'])}, {lit(r['display_name'])}, {lit(r['role'])}, {lit('active' if r['active'] else 'inactive')}, {lit(r['preferred_engine'])},
        jsonb_build_object('toolsets', {json_lit(r['toolsets_json'], '[]')}, 'skills', {json_lit(r['skills_json'], '[]')}, 'greenlight_required', {json_lit(r['greenlight_required_json'], '[]')}),
        {lit(r['created_at'])}::timestamptz, {lit(r['updated_at'])}::timestamptz)
ON CONFLICT (agent_id) DO UPDATE SET display_name=EXCLUDED.display_name, role=EXCLUDED.role, status=EXCLUDED.status, metadata=EXCLUDED.metadata, updated_at=EXCLUDED.updated_at;
""")

    for r in rows(con, "factory_projects"):
        statements.append(f"""
INSERT INTO factory.projects(project_id, name, repo_path, repo_remote, base_branch, status, autonomy_level, methodology, risk_level, human_owner, started_at, updated_at, summary, metadata)
VALUES ({lit(r['project_id'])}, {lit(r['name'])}, {lit(r['repo_path'])}, {lit(r['repo_remote'])}, {lit(r['base_branch'])}, {lit(r['status'])}, {lit(r['autonomy_level'])}, {lit(r['methodology'])}, {lit(r['risk_level'])}, {lit(r['human_owner'])}, {lit(r['started_at'])}::timestamptz, {lit(r['updated_at'])}::timestamptz, {lit(r['summary'])}, {json_lit(r['metadata_json'], '{}')})
ON CONFLICT (project_id) DO UPDATE SET name=EXCLUDED.name, status=EXCLUDED.status, updated_at=EXCLUDED.updated_at, metadata=EXCLUDED.metadata;
""")

    for r in rows(con, "factory_lanes"):
        metadata = json_lit(r["metadata_json"], "{}")
        statements.append(f"""
INSERT INTO factory.lanes(lane_id, project_id, name, methodology, branch, worktree_path, status, metadata, created_at, updated_at)
VALUES ({lit(r['lane_id'])}, {lit(r['project_id'])}, {lit(r['name'])}, {lit(r['methodology'])}, {lit(r['branch'])}, {lit(r['worktree_path'])}, {lit(r['status'])},
        ({metadata} || jsonb_build_object('kanban_board', {lit(r['kanban_board'])})), {lit(r['created_at'])}::timestamptz, {lit(r['updated_at'])}::timestamptz)
ON CONFLICT (lane_id) DO UPDATE SET status=EXCLUDED.status, metadata=EXCLUDED.metadata, updated_at=EXCLUDED.updated_at;
""")

    for r in rows(con, "factory_tasks"):
        statements.append(f"""
INSERT INTO factory.tasks(task_id, project_id, lane_id, kanban_id, title, description, phase, status, owner_profile, reviewer_profile, engine, priority, dependencies, branch, worktree_path, acceptance_criteria, started_at, finished_at, result_summary, evidence_required, evidence_status, risk_level, metadata, created_at, updated_at)
VALUES ({lit(r['task_id'])}, {lit(r['project_id'])}, {lit(r['lane_id'])}, {lit(r['kanban_id'])}, {lit(r['title'])}, {lit(r['description'])}, {lit(r['phase'])}, {lit(r['status'])}, {lit(r['owner_agent_id'])}, {lit(r['reviewer_agent_id'])}, {lit(r['engine'])}, {lit(r['priority'])}, {json_lit(r['dependencies_json'], '[]')}, {lit(r['branch'])}, {lit(r['worktree_path'])}, {json_lit(r['acceptance_criteria_json'], '[]')}, {lit(r['started_at'])}::timestamptz, {lit(r['finished_at'])}::timestamptz, {lit(r['result_summary'])}, {bool(r['evidence_required'])}, {lit(r['evidence_status'])}, {lit(r['risk_level'])}, {json_lit(r['metadata_json'], '{}')}, {lit(r['created_at'])}::timestamptz, {lit(r['updated_at'])}::timestamptz)
ON CONFLICT (task_id) DO UPDATE SET status=EXCLUDED.status, owner_profile=EXCLUDED.owner_profile, reviewer_profile=EXCLUDED.reviewer_profile, result_summary=EXCLUDED.result_summary, metadata=EXCLUDED.metadata, updated_at=EXCLUDED.updated_at;
""")

    for r in rows(con, "factory_events"):
        statements.append(f"""
INSERT INTO factory.events(event_id, project_id, lane_id, task_id, timestamp, actor, event_type, message, metadata)
VALUES ({lit(r['event_id'])}, {lit(r['project_id'])}, {lit(r['lane_id'])}, {lit(r['task_id'])}, {lit(r['created_at'])}::timestamptz, {lit(r['actor'])}, {lit(r['event_type'])}, {lit(r['message'])}, {json_lit(r['metadata_json'], '{}')})
ON CONFLICT (event_id) DO NOTHING;
""")

    for r in rows(con, "factory_gates"):
        statements.append(f"""
INSERT INTO factory.gates(gate_id, project_id, lane_id, task_id, gate_type, status, reviewer, evidence, notes, timestamp)
VALUES ({lit(r['gate_id'])}, {lit(r['project_id'])}, {lit(r['lane_id'])}, {lit(r['task_id'])}, {lit(r['gate_type'])}, {lit(r['status'])}, {lit(r['reviewer'])}, {json_lit(r['evidence_json'], '{}')}, {lit(r['notes'])}, {lit(r['created_at'])}::timestamptz)
ON CONFLICT (gate_id) DO NOTHING;
""")

    for r in rows(con, "factory_artifacts"):
        statements.append(f"""
INSERT INTO factory.artifacts(artifact_id, project_id, lane_id, task_id, artifact_type, path, checksum, created_by, created_at, metadata)
VALUES ({lit(r['artifact_id'])}, {lit(r['project_id'])}, {lit(r['lane_id'])}, {lit(r['task_id'])}, {lit(r['artifact_type'])}, {lit(r['path'])}, {lit(r['checksum'])}, {lit(r['created_by'])}, {lit(r['created_at'])}::timestamptz, {json_lit(r['metadata_json'], '{}')})
ON CONFLICT (artifact_id) DO NOTHING;
""")

    statements.append("""
SELECT setval(pg_get_serial_sequence('factory.events','event_id'), COALESCE((SELECT max(event_id) FROM factory.events), 1), true);
SELECT setval(pg_get_serial_sequence('factory.gates','gate_id'), COALESCE((SELECT max(gate_id) FROM factory.gates), 1), true);
SELECT setval(pg_get_serial_sequence('factory.artifacts','artifact_id'), COALESCE((SELECT max(artifact_id) FROM factory.artifacts), 1), true);
COMMIT;
""")
    run_psql("\n".join(statements), runtime_env)
    print(f"Imported Factory SQLite rows from {args.sqlite} into {runtime_env['AGENT_DB_NAME']}.factory")


if __name__ == "__main__":
    main()
