#!/usr/bin/env python3
"""Create/rotate Agent Core DB runtime roles from environment secrets.

Secrets are read from runtime/agent-core-db/.env or process env. This script
never commits secrets; use Infisical/runtime injection as the canonical source.
"""
from __future__ import annotations

import argparse
import os
import secrets
import string
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / "runtime" / "agent-core-db" / ".env"
RUNTIME_SECRETS_FILE = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "runtime-secrets.env"

DEFAULTS = {
    "AGENT_DB_CONTAINER": "agent-postgres",
    "AGENT_DB_ADMIN_USER": "agent_admin",
    "AGENT_DB_NAME": "zeus_agent",
    "AGENT_CALENDAR_DB_NAME": "nettu_calendar",
    "AGENT_CRM_DB_NAME": "zeus_agent",
    "AGENT_DB_RUNTIME_USER": "agent_runtime",
    "FACTORY_DB_RUNTIME_USER": "factory_runtime",
    "CALENDAR_DB_RUNTIME_USER": "calendar_runtime",
    "CRM_DB_RUNTIME_USER": "crm_runtime",
}

SECRET_KEYS = [
    "AGENT_DB_RUNTIME_PASSWORD",
    "FACTORY_DB_RUNTIME_PASSWORD",
    "CALENDAR_DB_RUNTIME_PASSWORD",
    "CRM_DB_RUNTIME_PASSWORD",
]


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


def save_env_file(values: dict[str, str]) -> None:
    existing = load_env_file()
    merged = {**existing, **values}
    lines = [
        "# Local Agent Core DB runtime. Canonical production source: Infisical/runtime env.",
    ]
    for key in [
        "AGENT_CORE_DB_ENABLED", "AGENT_DB_CONTAINER", "AGENT_DB_ADMIN_USER", "AGENT_DB_ADMIN_PASSWORD",
        "AGENT_DB_HOST_BIND", "AGENT_DB_HOST_PORT", "AGENT_DB_NAME", "AGENT_CALENDAR_DB_NAME", "AGENT_CRM_DB_NAME",
        "AGENT_DB_RUNTIME_USER", "AGENT_DB_RUNTIME_PASSWORD", "FACTORY_DB_RUNTIME_USER", "FACTORY_DB_RUNTIME_PASSWORD",
        "CALENDAR_DB_RUNTIME_USER", "CALENDAR_DB_RUNTIME_PASSWORD", "CRM_DB_RUNTIME_USER", "CRM_DB_RUNTIME_PASSWORD",
        "AGENT_DATABASE_URL", "FACTORY_DATABASE_URL", "CALENDAR_DATABASE_URL", "CRM_DATABASE_URL",
    ]:
        if key in merged:
            lines.append(f"{key}={merged[key]}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ENV_FILE.chmod(0o600)




def _fill_passwords_from_urls(env: dict[str, str]) -> None:
    from urllib.parse import urlparse, unquote
    pairs = {
        "AGENT_DB_RUNTIME_PASSWORD": "AGENT_DATABASE_URL",
        "FACTORY_DB_RUNTIME_PASSWORD": "FACTORY_DATABASE_URL",
        "CALENDAR_DB_RUNTIME_PASSWORD": "CALENDAR_DATABASE_URL",
        "CRM_DB_RUNTIME_PASSWORD": "CRM_DATABASE_URL",
    }
    for password_key, url_key in pairs.items():
        if env.get(password_key):
            continue
        url = (env.get(url_key) or "").strip().strip('"').strip("'")
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.password:
            env[password_key] = unquote(parsed.password)

def runtime_env(write_missing: bool = False) -> dict[str, str]:
    # Priority: defaults < local .env fallback < process env < Infisical runtime-secrets.env.
    # Infisical is canonical for operation; local .env only bootstraps dev/offline.
    env = {**DEFAULTS, **load_env_file(ENV_FILE), **os.environ, **load_env_file(RUNTIME_SECRETS_FILE)}
    env.setdefault("AGENT_CORE_DB_ENABLED", "true")
    env.setdefault("AGENT_DB_HOST_BIND", "127.0.0.1")
    env.setdefault("AGENT_DB_HOST_PORT", "55430")
    for key, default_user in [
        ("AGENT_DB_RUNTIME_USER", "agent_runtime"),
        ("FACTORY_DB_RUNTIME_USER", "factory_runtime"),
        ("CALENDAR_DB_RUNTIME_USER", "calendar_runtime"),
        ("CRM_DB_RUNTIME_USER", "crm_runtime"),
    ]:
        env.setdefault(key, default_user)
    _fill_passwords_from_urls(env)
    if write_missing:
        alphabet = string.ascii_letters + string.digits
        for key in SECRET_KEYS:
            env.setdefault(key, "".join(secrets.choice(alphabet) for _ in range(32)))
        host = env["AGENT_DB_HOST_BIND"]
        port = env["AGENT_DB_HOST_PORT"]
        env["AGENT_DATABASE_URL"] = f"postgresql://{env['AGENT_DB_RUNTIME_USER']}:{env['AGENT_DB_RUNTIME_PASSWORD']}@{host}:{port}/{env['AGENT_DB_NAME']}"
        env["FACTORY_DATABASE_URL"] = f"postgresql://{env['FACTORY_DB_RUNTIME_USER']}:{env['FACTORY_DB_RUNTIME_PASSWORD']}@{host}:{port}/{env['AGENT_DB_NAME']}"
        env["CALENDAR_DATABASE_URL"] = f"postgresql://{env['CALENDAR_DB_RUNTIME_USER']}:{env['CALENDAR_DB_RUNTIME_PASSWORD']}@{host}:{port}/{env['AGENT_CALENDAR_DB_NAME']}"
        env["CRM_DATABASE_URL"] = f"postgresql://{env['CRM_DB_RUNTIME_USER']}:{env['CRM_DB_RUNTIME_PASSWORD']}@{host}:{port}/{env.get('AGENT_CRM_DB_NAME', env['AGENT_DB_NAME'])}"
        save_env_file({k: env[k] for k in env if k.startswith(("AGENT_", "FACTORY_", "CALENDAR_", "CRM_"))})
    missing = [key for key in ["AGENT_DB_ADMIN_PASSWORD", *SECRET_KEYS] if not env.get(key)]
    if missing:
        raise SystemExit(f"Missing required secrets: {', '.join(missing)}. Inject them or run with --write-missing-local-env for local dev.")
    return env


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def run_psql(env: dict[str, str], database: str, sql: str) -> str:
    cmd = [
        "docker", "exec", "-i", env["AGENT_DB_CONTAINER"],
        "psql", "-X", "-q", "-t", "-A", "-v", "ON_ERROR_STOP=1",
        "-U", env["AGENT_DB_ADMIN_USER"], "-d", database,
    ]
    proc = subprocess.run(cmd, input=sql, text=True, cwd=REPO_ROOT, env=env, capture_output=True, check=True)
    return proc.stdout


def ensure_login_role(env: dict[str, str], role_key: str, password_key: str) -> None:
    role = env[role_key]
    password = env[password_key]
    sql = f"""
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = {quote_literal(role)}) THEN
    EXECUTE 'CREATE ROLE ' || quote_ident({quote_literal(role)}) || ' LOGIN PASSWORD ' || quote_literal({quote_literal(password)});
  ELSE
    EXECUTE 'ALTER ROLE ' || quote_ident({quote_literal(role)}) || ' LOGIN PASSWORD ' || quote_literal({quote_literal(password)});
  END IF;
END $$;
"""
    run_psql(env, "postgres", sql)


def apply_grants(env: dict[str, str]) -> None:
    # Keep the existing Postgres admin role password in sync with Infisical too.
    ensure_login_role(env, "AGENT_DB_ADMIN_USER", "AGENT_DB_ADMIN_PASSWORD")
    for role_key, password_key in [
        ("AGENT_DB_RUNTIME_USER", "AGENT_DB_RUNTIME_PASSWORD"),
        ("FACTORY_DB_RUNTIME_USER", "FACTORY_DB_RUNTIME_PASSWORD"),
        ("CALENDAR_DB_RUNTIME_USER", "CALENDAR_DB_RUNTIME_PASSWORD"),
        ("CRM_DB_RUNTIME_USER", "CRM_DB_RUNTIME_PASSWORD"),
    ]:
        ensure_login_role(env, role_key, password_key)

    agent_db = env["AGENT_DB_NAME"]
    calendar_db = env["AGENT_CALENDAR_DB_NAME"]
    agent_runtime = quote_ident(env["AGENT_DB_RUNTIME_USER"])
    factory_runtime = quote_ident(env["FACTORY_DB_RUNTIME_USER"])
    calendar_runtime = quote_ident(env["CALENDAR_DB_RUNTIME_USER"])
    crm_runtime = quote_ident(env["CRM_DB_RUNTIME_USER"])

    run_psql(env, agent_db, f"""
GRANT CONNECT ON DATABASE {quote_ident(agent_db)} TO {agent_runtime}, {factory_runtime}, {calendar_runtime}, {crm_runtime};
GRANT USAGE ON SCHEMA agent_core TO {agent_runtime}, {factory_runtime}, {calendar_runtime}, {crm_runtime};
GRANT SELECT ON ALL TABLES IN SCHEMA agent_core TO {agent_runtime}, {factory_runtime}, {calendar_runtime}, {crm_runtime};
GRANT USAGE ON SCHEMA factory TO {agent_runtime}, {factory_runtime};
GRANT SELECT ON ALL TABLES IN SCHEMA factory TO {agent_runtime};
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA factory TO {factory_runtime};
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA factory TO {factory_runtime};
GRANT USAGE ON SCHEMA crm TO {agent_runtime}, {crm_runtime};
GRANT SELECT ON ALL TABLES IN SCHEMA crm TO {agent_runtime};
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA crm TO {crm_runtime};
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA crm TO {crm_runtime};
""")
    run_psql(env, calendar_db, f"""
GRANT CONNECT ON DATABASE {quote_ident(calendar_db)} TO {agent_runtime}, {calendar_runtime};
GRANT USAGE ON SCHEMA public TO {calendar_runtime};
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {calendar_runtime};
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {calendar_runtime};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {calendar_runtime};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {calendar_runtime};
""")


def print_infisical(env: dict[str, str]) -> None:
    keys = [
        "AGENT_CORE_DB_ENABLED", "AGENT_DB_CONTAINER", "AGENT_DB_ADMIN_USER", "AGENT_DB_ADMIN_PASSWORD",
        "AGENT_DB_HOST_BIND", "AGENT_DB_HOST_PORT", "AGENT_DB_NAME", "AGENT_CALENDAR_DB_NAME", "AGENT_CRM_DB_NAME",
        "AGENT_DB_RUNTIME_USER", "AGENT_DB_RUNTIME_PASSWORD", "FACTORY_DB_RUNTIME_USER", "FACTORY_DB_RUNTIME_PASSWORD",
        "CALENDAR_DB_RUNTIME_USER", "CALENDAR_DB_RUNTIME_PASSWORD", "CRM_DB_RUNTIME_USER", "CRM_DB_RUNTIME_PASSWORD",
        "AGENT_DATABASE_URL", "FACTORY_DATABASE_URL", "CALENDAR_DATABASE_URL", "CRM_DATABASE_URL",
    ]
    for key in keys:
        if key in env:
            print(f"{key}={env[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/rotate Agent Core runtime roles")
    parser.add_argument("--write-missing-local-env", action="store_true", help="Generate missing local runtime secrets into runtime/agent-core-db/.env")
    parser.add_argument("--print-infisical", action="store_true", help="Print the env block to copy into Infisical")
    args = parser.parse_args()
    env = runtime_env(write_missing=args.write_missing_local_env)
    apply_grants(env)
    print("Agent Core runtime roles ready: agent_runtime, factory_runtime, calendar_runtime, crm_runtime")
    if args.print_infisical:
        print_infisical(env)


if __name__ == "__main__":
    main()
