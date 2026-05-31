"""Small psql-based client for the local Agent Core PostgreSQL runtime.

The Hermes runtime intentionally avoids adding a psycopg dependency here.  For
local single-tenant agents, tools execute psql inside the tracked
``agent-postgres`` container.  Runtime roles and passwords still matter for TCP
clients and future sidecars; docker-exec keeps the Hermes Python environment
stdlib-only.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / "runtime" / "agent-core-db" / ".env"
DEFAULTS = {
    "AGENT_CORE_DB_ENABLED": "true",
    "AGENT_DB_CONTAINER": "agent-postgres",
    "AGENT_DB_NAME": "zeus_agent",
    "AGENT_DB_ADMIN_USER": "agent_admin",
    "AGENT_DB_RUNTIME_USER": "agent_runtime",
    "FACTORY_DB_RUNTIME_USER": "factory_runtime",
    "CRM_DB_RUNTIME_USER": "crm_runtime",
}


def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def runtime_env() -> dict[str, str]:
    env = {**DEFAULTS, **os.environ, **load_env_file()}
    return env


def enabled() -> bool:
    return runtime_env().get("AGENT_CORE_DB_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def quote_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return "'" + str(value).replace("'", "''") + "'"


def quote_jsonb(value: Any) -> str:
    return quote_literal(json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)) + "::jsonb"


def slugify(value: str) -> str:
    import re
    text = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return text or "record"


def psql(sql: str, *, database: str | None = None, user: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = runtime_env()
    database = database or env["AGENT_DB_NAME"]
    user = user or env.get("AGENT_DB_RUNTIME_USER", "agent_runtime")
    cmd = [
        "docker", "exec", "-i", env.get("AGENT_DB_CONTAINER", "agent-postgres"),
        "psql", "-X", "-q", "-t", "-A", "-v", "ON_ERROR_STOP=1", "-U", user, "-d", database,
    ]
    return subprocess.run(cmd, input=sql, text=True, cwd=REPO_ROOT, capture_output=True, check=check)


def json_query(sql: str, *, database: str | None = None, user: str | None = None) -> Any:
    proc = psql(sql, database=database, user=user)
    out = proc.stdout.strip()
    if not out:
        return None
    return json.loads(out)


def rows(select_sql: str, *, database: str | None = None, user: str | None = None) -> list[dict[str, Any]]:
    sql = f"SELECT COALESCE(jsonb_agg(to_jsonb(q)), '[]'::jsonb)::text FROM ({select_sql}) q;"
    data = json_query(sql, database=database, user=user)
    return data or []


def one(select_sql: str, *, database: str | None = None, user: str | None = None) -> dict[str, Any] | None:
    result = rows(f"{select_sql} LIMIT 1", database=database, user=user)
    return result[0] if result else None


def statement_one(statement_sql: str, *, database: str | None = None, user: str | None = None) -> dict[str, Any] | None:
    """Execute a data-changing statement with RETURNING and return one row.

    PostgreSQL does not allow ``INSERT ... RETURNING`` directly inside a FROM
    subquery, so DML statements are wrapped as a writable CTE.
    """
    data = json_query(
        f"WITH q AS ({statement_sql}) SELECT to_jsonb(q)::text FROM q LIMIT 1;",
        database=database,
        user=user,
    )
    return data
