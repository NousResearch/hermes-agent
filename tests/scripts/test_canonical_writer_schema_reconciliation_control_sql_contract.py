from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "scripts/sql/canonical_writer_schema_reconciliation_control_v1.sql"
RETIRE = (
    ROOT
    / "scripts/sql/canonical_writer_schema_reconciliation_control_retire_v1.sql"
)

EXECUTOR = "canonical_brain_schema_reconciler"
CONTROL_SCHEMA = "canonical_brain_reconciliation"
OBSERVER = "observe_missing_discord_routeback_helper_v1"
APPLY = "apply_missing_discord_routeback_helper_v1"
LOCK = "pg_catalog.pg_advisory_xact_lock(4841739663211427921)"
HELPER_SIGNATURE = (
    "canonical_brain._discord_guild_routeback_target_valid(jsonb)"
)
OBSERVER_PROSRC_SHA256 = (
    "47b63aa737d29e1d5b3a54fc824606d91c322a7869118b6f331040e0a3ef96fe"
)
OBSERVER_DEFINITION_SHA256 = (
    "7813ead62d79011f2f2c4e1895405bb35a8edc959e244a14fc22d1ab1be56974"
)
APPLY_PROSRC_SHA256 = (
    "2a28d4700d550bcc8ddc56ea870fc5f669f55a47f9abc7e1993b99b178db1719"
)
APPLY_DEFINITION_SHA256 = (
    "63d6388e50086bf2203bafb7d74291cbec32d04c1f2e05af4f007df4c1e9c8d6"
)


def _text(path: Path) -> str:
    value = path.read_text(encoding="utf-8")
    assert value.endswith("\n")
    return value


def test_install_is_one_fixed_inert_executor_boundary() -> None:
    sql = _text(INSTALL)

    assert sql.count("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;") == 1
    assert sql.count(LOCK) == 3  # bootstrap, observer, apply
    assert sql.index(LOCK) < sql.index("CREATE ROLE " + EXECUTOR)
    assert sql.index(LOCK) < sql.index("LOCK TABLE public.canonical_event_log")
    assert "pg_advisory_xact_lock_shared" not in sql

    assert sql.count("CREATE ROLE " + EXECUTOR) == 1
    for attribute in (
        "NOLOGIN",
        "NOINHERIT",
        "NOSUPERUSER",
        "NOCREATEDB",
        "NOCREATEROLE",
        "NOREPLICATION",
        "NOBYPASSRLS",
        "PASSWORD NULL",
    ):
        assert attribute in sql
    assert "GRANT CONNECT ON DATABASE muncho_canary_brain" in sql
    assert "GRANT USAGE ON SCHEMA " + CONTROL_SCHEMA in sql

    observer_signature = f"{OBSERVER}()"
    apply_signature = f"{APPLY}()"
    assert sql.count("CREATE FUNCTION " + CONTROL_SCHEMA + ".") == 2
    assert observer_signature in sql
    assert apply_signature in sql
    assert sql.count("\nSECURITY DEFINER\n") == 2
    assert sql.count("SET search_path = pg_catalog, pg_temp") == 2
    assert sql.count("PARALLEL UNSAFE") == 2
    assert sql.count("CALLED ON NULL INPUT") == 2
    assert sql.count("REVOKE ALL PRIVILEGES ON FUNCTION " + CONTROL_SCHEMA) == 2
    assert sql.count("TO " + EXECUTOR + ";") >= 4

    # The only dynamic execution is the one constant, reviewed helper DDL.
    assert sql.count("EXECUTE $fixed_helper_definition$") == 1
    assert "EXECUTE format(" not in sql
    assert "EXECUTE pg_catalog.format(" not in sql
    assert sql.count(
        "CREATE FUNCTION canonical_brain."
        "_discord_guild_routeback_target_valid("
    ) == 1
    assert (
        "CREATE OR REPLACE FUNCTION canonical_brain."
        "_discord_guild_routeback_target_valid"
    ) not in sql
    assert "CREATE TABLE" not in sql
    assert "ALTER ROLE" not in sql
    assert "\nDROP OWNED" not in sql
    assert "\nCASCADE" not in sql

    # Receipt-only GUCs bind authorization; none selects SQL or an object.
    for setting in (
        "muncho.schema_reconciliation_plan_sha256",
        "muncho.schema_reconciliation_authorized_intent_sha256",
        "muncho.schema_reconciliation_truth_receipt_sha256",
        "muncho.schema_reconciliation_control_observation_sha256",
    ):
        assert setting in sql
    assert "canonical-writer-schema-reconciliation-control-observation-v1" in sql
    assert "canonical-writer-schema-reconciliation-control-apply-v1" in sql
    assert sql.count("canonical_brain.writer_") > 30
    assert "IN SHARE MODE;" in sql


def test_install_has_exact_bootstrap_and_runtime_principal_separation() -> None:
    sql = _text(INSTALL)

    assert "^muncho_canary_control_[0-9a-f]{16}$" in sql
    assert sql.count("^muncho_canary_reconciler_[0-9a-f]{16}$") == 3
    assert "grantor_name = 'cloudsqladmin'" in sql
    assert "granted_name = 'cloudsqlsuperuser'" in sql
    assert "admin_option IS FALSE" in sql
    assert "inherit_option IS TRUE" in sql
    assert "set_option IS TRUE" in sql
    assert "grantor.rolname = 'cloudsqladmin'" in sql
    assert "membership.admin_option IS TRUE" in sql
    assert "membership.inherit_option IS FALSE" in sql
    assert "membership.set_option IS FALSE" in sql
    assert "REVOKE canonical_brain_migration_owner FROM SESSION_USER" in sql


def test_retire_is_exact_drift_intolerant_and_non_cascading() -> None:
    sql = _text(RETIRE)

    assert sql.count("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;") == 1
    assert sql.count(LOCK) == 1
    assert sql.index(LOCK) < sql.index("LOCK TABLE public.canonical_event_log")
    assert "^muncho_canary_control_[0-9a-f]{16}$" in sql
    assert "grantor_name = 'cloudsqladmin'" in sql
    for digest in (
        OBSERVER_PROSRC_SHA256,
        OBSERVER_DEFINITION_SHA256,
        APPLY_PROSRC_SHA256,
        APPLY_DEFINITION_SHA256,
    ):
        assert sql.count(digest) == 1

    assert sql.count("DROP FUNCTION " + CONTROL_SCHEMA + ".") == 2
    assert "DROP SCHEMA " + CONTROL_SCHEMA + ";" in sql
    assert "DROP ROLE " + EXECUTOR + ";" in sql
    assert "REVOKE CONNECT ON DATABASE muncho_canary_brain" in sql
    assert "\nDROP OWNED" not in sql
    assert "\nCASCADE" not in sql
    assert "IF EXISTS" not in sql
    assert "CREATE OR REPLACE" not in sql
    assert "EXECUTE $" not in sql
    assert "EXECUTE '" not in sql
    assert HELPER_SIGNATURE in sql
    assert "DROP FUNCTION canonical_brain._discord" not in sql


def test_retire_pins_the_pg18_function_configuration() -> None:
    sql = _text(RETIRE)
    expected = (
        "'search_path=pg_catalog, pg_temp', 'TimeZone=UTC',\n"
        "                  'DateStyle=ISO, YMD', 'IntervalStyle=postgres',\n"
        "                  'extra_float_digits=3', 'bytea_output=hex',\n"
        "                  'lock_timeout=15s', 'statement_timeout=5min'"
    )
    assert sql.count(expected) == 2
