#!/usr/bin/env python3
"""Agent Core DB bootstrap and migration runner.

This script treats PostgreSQL as part of the agent runtime substrate. Modules
bring SQL migrations and the runner applies them into the shared Agent Core DB
or module-specific databases on the same Postgres server.

It intentionally shells into the local Docker Compose Postgres container for
psql so the Hermes Python environment does not need a psycopg dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "runtime" / "agent-core-db" / "docker-compose.agent-core.yml"
ENV_FILE = REPO_ROOT / "runtime" / "agent-core-db" / ".env"
RUNTIME_SECRETS_FILE = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "runtime-secrets.env"

DEFAULTS = {
    "AGENT_DB_ADMIN_USER": "agent_admin",
    "AGENT_DB_NAME": "zeus_agent",
    "AGENT_CALENDAR_DB_NAME": "nettu_calendar",
    "AGENT_CRM_DB_NAME": "zeus_agent",
}

MODULES = {
    "agent_core": {
        "database_env": "AGENT_DB_NAME",
        "migrations": REPO_ROOT / "db" / "agent-core",
    },
    "factory": {
        "database_env": "AGENT_DB_NAME",
        "migrations": REPO_ROOT / "db" / "modules" / "factory",
    },
    "calendar": {
        "database_env": "AGENT_DB_NAME",
        "migrations": REPO_ROOT / "db" / "modules" / "calendar",
    },
    "crm": {
        "database_env": "AGENT_CRM_DB_NAME",
        "migrations": REPO_ROOT / "db" / "modules" / "crm",
    },
}


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


def runtime_env() -> dict[str, str]:
    # Priority: defaults < local .env fallback < process env < Infisical runtime-secrets.env.
    # Infisical is the canonical source of truth for running agents.
    env = dict(DEFAULTS)
    env.update(load_env_file(ENV_FILE))
    env.update(os.environ)
    env.update(load_env_file(RUNTIME_SECRETS_FILE))
    if "AGENT_DB_ADMIN_PASSWORD" not in env:
        raise SystemExit(f"AGENT_DB_ADMIN_PASSWORD is required. Create {ENV_FILE} from .env.example or inject it from runtime secrets.")
    return env


def run(cmd: list[str], *, env: dict[str, str], input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, input=input_text, text=True, env=env, cwd=REPO_ROOT, check=check, capture_output=True)


def compose(env: dict[str, str], args: list[str]) -> subprocess.CompletedProcess[str]:
    env_file = RUNTIME_SECRETS_FILE if RUNTIME_SECRETS_FILE.exists() else ENV_FILE
    return run(["docker", "compose", "--env-file", str(env_file), "-f", str(COMPOSE_FILE), *args], env=env)


def psql(env: dict[str, str], database: str, sql: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    user = env["AGENT_DB_ADMIN_USER"]
    cmd = [
        "docker", "exec", "-i", "agent-postgres",
        "psql", "-X", "-q", "-t", "-A", "-v", "ON_ERROR_STOP=1", "-U", user, "-d", database,
    ]
    return run(cmd, env=env, input_text=sql, check=check)


def psql_file(env: dict[str, str], database: str, path: Path) -> None:
    user = env["AGENT_DB_ADMIN_USER"]
    cmd = [
        "docker", "exec", "-i", "agent-postgres",
        "psql", "-X", "-q", "-t", "-A", "-v", "ON_ERROR_STOP=1", "-U", user, "-d", database,
    ]
    with path.open("r", encoding="utf-8") as fh:
        subprocess.run(cmd, stdin=fh, text=True, env=env, cwd=REPO_ROOT, check=True)


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ensure_database(env: dict[str, str], database: str) -> None:
    exists_sql = f"SELECT 1 FROM pg_database WHERE datname = {quote_literal(database)};"
    proc = psql(env, "postgres", exists_sql)
    if "1" not in proc.stdout:
        psql(env, "postgres", f"CREATE DATABASE {quote_ident(database)};\n")


def ensure_migration_ledger(env: dict[str, str], database: str) -> None:
    psql(env, database, """
CREATE SCHEMA IF NOT EXISTS agent_core;
CREATE TABLE IF NOT EXISTS agent_core.schema_migrations (
  module text NOT NULL,
  version text NOT NULL,
  checksum text NOT NULL,
  applied_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (module, version)
);
""")


def migration_applied(env: dict[str, str], database: str, module: str, version: str) -> bool:
    sql = f"SELECT checksum FROM agent_core.schema_migrations WHERE module = {quote_literal(module)} AND version = {quote_literal(version)};"
    proc = psql(env, database, sql)
    return bool(proc.stdout.strip())


def record_migration(env: dict[str, str], database: str, module: str, version: str, checksum: str) -> None:
    sql = f"""
INSERT INTO agent_core.schema_migrations(module, version, checksum)
VALUES ({quote_literal(module)}, {quote_literal(version)}, {quote_literal(checksum)})
ON CONFLICT (module, version) DO UPDATE SET checksum = EXCLUDED.checksum;
"""
    psql(env, database, sql)


def migration_version(path: Path) -> str:
    return path.name.split("_", 1)[0]


def apply_module(env: dict[str, str], module: str) -> None:
    spec = MODULES[module]
    database = env[spec["database_env"]]
    ensure_database(env, database)
    ensure_migration_ledger(env, database)
    paths = sorted(Path(spec["migrations"]).glob("*.sql"))
    if not paths:
        print(f"{module}: no migrations")
        return
    for path in paths:
        version = migration_version(path)
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if migration_applied(env, database, module, version):
            print(f"{module}:{version} already applied")
            continue
        print(f"{module}:{version} applying {path.relative_to(REPO_ROOT)} -> {database}")
        psql_file(env, database, path)
        record_migration(env, database, module, version, checksum)
        print(f"{module}:{version} applied")


def apply_external_sql_dir(env: dict[str, str], module: str, database: str, directory: Path) -> None:
    ensure_database(env, database)
    ensure_migration_ledger(env, database)
    for path in sorted(directory.glob("*.sql")):
        version = path.stem
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if migration_applied(env, database, module, version):
            print(f"{module}:{version} already applied")
            continue
        print(f"{module}:{version} applying external {path} -> {database}")
        psql_file(env, database, path)
        record_migration(env, database, module, version, checksum)
        print(f"{module}:{version} applied")


def status(env: dict[str, str]) -> None:
    compose_proc = compose(env, ["ps"])
    print(compose_proc.stdout)
    for database in sorted({env["AGENT_DB_NAME"], env["AGENT_CALENDAR_DB_NAME"], env.get("AGENT_CRM_DB_NAME", env["AGENT_DB_NAME"])}):
        ensure_database(env, database)
        ensure_migration_ledger(env, database)
        proc = psql(env, database, "SELECT current_database(), module, version, applied_at FROM agent_core.schema_migrations ORDER BY module, version;")
        print(f"--- {database} migrations ---")
        print(proc.stdout.strip() or "(none)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Agent Core PostgreSQL runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("up", help="Start Agent Core PostgreSQL")
    sub.add_parser("migrate", help="Apply core module migrations")
    sub.add_parser("status", help="Show DB status and applied migrations")
    ext = sub.add_parser("apply-external", help="Apply external SQL directory into a module database")
    ext.add_argument("--module", required=True)
    ext.add_argument("--database", required=True)
    ext.add_argument("--path", required=True, type=Path)
    args = parser.parse_args()

    env = runtime_env()
    if args.command == "up":
        proc = compose(env, ["up", "-d"])
        print(proc.stdout)
        ensure_database(env, env["AGENT_DB_NAME"])
        ensure_database(env, env["AGENT_CALENDAR_DB_NAME"])
        print(f"Agent Core DB ready: {env['AGENT_DB_NAME']} + {env['AGENT_CALENDAR_DB_NAME']} + crm:{env.get('AGENT_CRM_DB_NAME', env['AGENT_DB_NAME'])}")
    elif args.command == "migrate":
        compose(env, ["up", "-d"])
        ensure_database(env, env["AGENT_DB_NAME"])
        ensure_database(env, env["AGENT_CALENDAR_DB_NAME"])
        ensure_database(env, env.get("AGENT_CRM_DB_NAME", env["AGENT_DB_NAME"]))
        for module in ["agent_core", "factory", "calendar", "crm"]:
            apply_module(env, module)
    elif args.command == "status":
        status(env)
    elif args.command == "apply-external":
        apply_external_sql_dir(env, args.module, args.database, args.path)


if __name__ == "__main__":
    main()
