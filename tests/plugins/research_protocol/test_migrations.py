"""Migration and packaging gates for Research Protocol PR2."""

from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO_ROOT / "plugins" / "research_protocol" / "migrations"


def test_pr2_migration_set_is_exact_and_versioned():
    assert [path.name for path in sorted(MIGRATIONS.glob("*.sql"))] == [
        "0001_approval_ledger.sql",
        "0002_roles.sql",
    ]


def test_approval_migration_enforces_replay_expiry_and_hash_invariants():
    sql = (MIGRATIONS / "0001_approval_ledger.sql").read_text(encoding="utf-8").lower()

    assert "create schema research_protocol" in sql
    assert "create schema if not exists" not in sql
    assert "create table research_protocol.approvals" in sql
    assert "approval_id" in sql and "primary key" in sql
    assert "scope_sha256" in sql and "plan_sha256" in sql
    assert "scope_json jsonb" in sql
    assert "max_executions > 0" in sql
    assert "consumed_count >= 0" in sql
    assert "consumed_count <= max_executions" in sql
    assert "expires_at > created_at" in sql
    assert "verdict in ('approved', 'denied')" in sql
    assert "check (verdict = 'denied' or expires_at > created_at)" in sql
    assert "function research_protocol.claim_approval" in sql
    assert "security definer" in sql
    assert "set search_path = pg_catalog, research_protocol" in sql
    assert "revoke all on function research_protocol.claim_approval" in sql
    assert "clock_timestamp()" in sql
    assert "current_timestamp" not in sql
    assert "create function research_protocol.store_approval" in sql
    assert "p_verdict text" in sql
    assert "p_verdict not in ('approved', 'denied')" in sql
    assert "p_verdict = 'approved'" in sql
    assert "p_verdict," in sql
    assert "revoke all on function research_protocol.store_approval" in sql
    assert "create table if not exists" not in sql
    assert "create index if not exists" not in sql
    assert "create or replace function" not in sql
    assert "char_length(approval_id) between 22 and 256" in sql
    assert "char_length(surface) between 1 and 128" in sql
    assert "revoke all on" in sql and "from public" in sql


def test_approval_migration_matches_runtime_record_and_claim_columns():
    sql = (MIGRATIONS / "0001_approval_ledger.sql").read_text(encoding="utf-8").lower()

    for declaration in (
        "surface text not null",
        "last_consumed_at timestamptz",
    ):
        assert declaration in sql
    assert "schema_version text not null" not in sql
    assert "\n    consumed_at timestamptz" not in sql
    assert "consumed_count = 0 and last_consumed_at is null" in sql
    assert "consumed_count > 0 and last_consumed_at is not null" in sql


def test_store_approval_recomputes_scope_sha256_from_exact_json():
    sql = (MIGRATIONS / "0001_approval_ledger.sql").read_text(encoding="utf-8").lower()

    assert "create extension" not in sql
    assert "p_scope_json text" in sql
    assert "pg_catalog.sha256(" in sql
    assert "pg_catalog.convert_to(p_scope_json, 'utf8')" in sql
    assert "<> p_scope_sha256" in sql


def test_roles_are_non_login_non_inheriting_and_least_privilege():
    sql = (MIGRATIONS / "0002_roles.sql").read_text(encoding="utf-8").lower()

    roles = (
        "research_protocol_planner_reader",
        "research_protocol_approval_writer",
        "research_protocol_approval_claimant",
    )
    for role in roles:
        assert f"create role {role}" in sql
    assert "if not exists" not in sql
    assert "pg_roles" not in sql
    assert "pg_auth_members" not in sql
    assert "alter role" not in sql
    assert sql.count("nologin noinherit") >= 3
    assert (
        "grant select on research_protocol.approvals to research_protocol_planner_reader"
        in sql
    )
    assert (
        "grant insert on research_protocol.approvals to research_protocol_approval_writer"
        not in sql
    )
    assert "grant select, update on research_protocol.approvals" not in sql
    assert "grant insert (" not in sql
    assert "grant execute on function research_protocol.store_approval" in sql
    assert "to research_protocol_approval_writer" in sql
    assert "grant select (" not in sql
    assert "grant update" not in sql
    assert "grant execute on function research_protocol.claim_approval" in sql
    assert "to research_protocol_approval_claimant" in sql
    for role in roles:
        assert f"revoke all on research_protocol.approvals from {role}" in sql
        assert f"revoke all on schema research_protocol from {role}" in sql
        assert f"nosuperuser nocreatedb nocreaterole noreplication nobypassrls" in sql
    assert "revoke all on function research_protocol.store_approval" in sql
    assert "revoke all on function research_protocol.claim_approval" in sql
    assert "grant delete" not in sql
    assert "grant all" not in sql


def test_research_protocol_extra_is_consistent_with_uv_lock():
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))

    assert pyproject["project"]["optional-dependencies"]["research-protocol"] == [
        "asyncpg==0.31.0"
    ]
    root_package = next(
        package for package in lock["package"] if package["name"] == "hermes-agent"
    )
    assert root_package["optional-dependencies"]["research-protocol"] == [
        {"name": "asyncpg"}
    ]
    assert {
        "name": "asyncpg",
        "marker": "extra == 'research-protocol'",
        "specifier": "==0.31.0",
    } in root_package["metadata"]["requires-dist"]


def test_migrations_are_declared_for_wheel_and_sdist():
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    plugin_data = pyproject["tool"]["setuptools"]["package-data"]["plugins"]
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "research_protocol/**/*.sql" in plugin_data
    assert (
        "recursive-include plugins/research_protocol *.json *.yaml *.sql *.md"
        in manifest
    )


def test_asyncpg_is_isolated_in_research_protocol_extra():
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["optional-dependencies"]["research-protocol"] == [
        "asyncpg==0.31.0"
    ]
