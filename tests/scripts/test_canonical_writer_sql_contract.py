from __future__ import annotations

import json
from pathlib import Path
import re

from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    POSTGRES_ROUTINE_BY_OPERATION,
)


ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = ROOT / "scripts" / "sql" / "canonical_writer_v1.sql"
HELPERS_PATH = ROOT / "scripts" / "sql" / "canonical_writer_v1_helpers.json"
SQL = SQL_PATH.read_text(encoding="utf-8")


def _writer_names() -> set[str]:
    return set(
        re.findall(
            r"CREATE OR REPLACE FUNCTION canonical_brain\."
            r"(writer_[a-z_]+)\(request jsonb, runtime jsonb\)",
            SQL,
        )
    )


def _function_definition(name: str) -> str:
    match = re.search(
        rf"CREATE OR REPLACE FUNCTION canonical_brain\.{re.escape(name)}\("
        r".*?\n\$function\$;",
        SQL,
        flags=re.DOTALL,
    )
    assert match, f"missing SQL definition for {name}"
    return match.group(0)


def _table_columns(name: str) -> tuple[str, ...]:
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS canonical_brain\.{re.escape(name)}\s*"
        r"\((.*?)\n\);",
        SQL,
        flags=re.DOTALL,
    )
    assert match, f"missing SQL table definition for {name}"
    return tuple(
        column.group(1)
        for line in match.group(1).splitlines()
        if (column := re.match(r"^    ([a-z_][a-z0-9_]*)\s+", line))
        and column.group(1).upper() not in {"CHECK", "UNIQUE", "CONSTRAINT"}
    )


def _dollar_block(name: str) -> str:
    start = f"DO ${name}$"
    assert start in SQL, f"missing SQL block {name}"
    return SQL.split(start, 1)[1].split(f"${name}$;", 1)[0]


def test_fixed_catalog_defines_exactly_the_backend_seventeen_routines():
    expected = set(POSTGRES_ROUTINE_BY_OPERATION.values())

    assert len(expected) == 17
    assert _writer_names() == expected
    assert SQL.count("-- Fixed public routine ") == 17


def test_private_writer_tables_have_exact_ordered_columns_without_duplicates():
    expected = {
        "writer_routeback_authorizations": (
            "authorization_id", "case_id", "target_ref", "message_summary",
            "source_refs", "content_sha256", "session_key_sha256",
            "capability_epoch_sha256",
            "runtime_platform", "source_thread_id", "idempotency_key",
            "request_sha256", "created_at", "intent_event_id",
        ),
        "writer_routeback_lifecycle_terminals": (
            "lifecycle_id", "case_id", "idempotency_key", "target_ref",
            "message_summary", "source_refs", "outcome", "receipt",
            "blocker_reason", "request_sha256", "session_key_sha256",
            "capability_epoch_sha256", "finalized_at", "terminal_event_id",
        ),
        "writer_routeback_terminals": (
            "authorization_id", "outcome", "receipt", "blocker_reason",
            "request_sha256", "finalized_at", "terminal_event_id",
        ),
        "writer_public_routeback_targets": (
            "channel_id", "target_type", "approved_by", "approved_at", "enabled",
        ),
        "writer_event_provenance": (
            "event_id", "canonical_content_sha256", "origin", "trusted_runtime",
            "appended_at",
        ),
        "writer_capability_grants": (
            "approval_id", "case_id", "plan_id", "plan_revision",
            "session_key_sha256",
            "capability_epoch_sha256", "approved_by_user_id",
            "approval_source_sha256", "command_hashes", "expires_at", "max_uses",
            "request_sha256", "granted_at", "grant_event_id",
        ),
        "writer_capability_revocation_scopes": (
            "scope_type", "session_key_sha256", "capability_epoch_sha256",
            "plan_id", "reason", "revoked_at",
        ),
        "writer_capability_revocations": (
            "approval_id", "reason", "revoked_by_session_sha256", "revoked_at",
        ),
        "writer_capability_consumptions": (
            "consume_id", "approval_id", "command_sha256", "session_key_sha256",
            "capability_epoch_sha256", "idempotency_key", "request_sha256",
            "remaining_uses", "consumed_at", "receipt_event_id", "response",
        ),
    }

    for table, columns in expected.items():
        assert _table_columns(table) == columns
        assert len(columns) == len(set(columns))


def test_every_public_routine_is_security_definer_with_pinned_search_path():
    for name in _writer_names():
        definition = _function_definition(name)
        assert "SECURITY DEFINER" in definition
        assert "SET search_path = pg_catalog, canonical_brain" in definition
        assert "RETURNS jsonb" in definition
        assert re.search(r"\bVOLATILE\b", definition)

    # Mutating routines must never claim PostgreSQL IMMUTABLE volatility.
    assert not re.search(
        r"FUNCTION canonical_brain\.writer_[\s\S]*?\nIMMUTABLE\n",
        SQL,
    )


def test_public_routines_reraise_only_retryable_transaction_aborts():
    retry_clause = (
        "EXCEPTION\n"
        "WHEN serialization_failure OR deadlock_detected THEN\n"
        "    RAISE;\n"
        "WHEN OTHERS THEN\n"
    )

    for name in _writer_names():
        definition = _function_definition(name)
        assert definition.count(retry_clause) == 1
        assert "RETURN canonical_brain._fail('database_failure'" in definition

    assert SQL.count(retry_clause) == 17


def test_owner_and_execute_acl_cover_exactly_public_catalog():
    expected = _writer_names()
    owners = set(
        re.findall(
            r"ALTER FUNCTION canonical_brain\.(writer_[a-z_]+)\(jsonb,jsonb\)\s+"
            r"OWNER TO canonical_brain_migration_owner;",
            SQL,
        )
    )
    grants = set(
        re.findall(
            r"GRANT EXECUTE ON FUNCTION canonical_brain\.(writer_[a-z_]+)"
            r"\(jsonb,jsonb\)\s+TO canonical_brain_writer;",
            SQL,
        )
    )

    assert owners == expected
    assert grants == expected
    retire_acl = _dollar_block("retire_canonical_acl")
    direct_acl = _dollar_block("canonical_direct_acl_contract")
    assert "namespace.nspname = 'canonical_brain'" in retire_acl
    assert "acl.grantee <> namespace.nspowner" in retire_acl
    assert "acl.grantee <> class.relowner" in retire_acl
    assert "acl.grantee <> routine.proowner" in retire_acl
    assert "REVOKE ALL PRIVILEGES ON %s %s FROM %s CASCADE" in retire_acl
    assert "(SELECT * FROM actual EXCEPT SELECT * FROM expected)" in direct_acl
    assert "(SELECT * FROM expected EXCEPT SELECT * FROM actual)" in direct_acl
    assert not re.search(
        r"GRANT EXECUTE ON FUNCTION canonical_brain\._", SQL
    )
    assert not re.search(
        r"GRANT EXECUTE[\s\S]{0,120}\bTO PUBLIC\b", SQL
    )


def test_helper_manifest_is_complete_owned_pinned_and_non_executable():
    manifest = json.loads(HELPERS_PATH.read_text(encoding="utf-8"))
    signatures = manifest["signatures"]
    sql_helpers = set(
        re.findall(
            r"CREATE OR REPLACE FUNCTION canonical_brain\.(_[a-z0-9_]+)\(", SQL
        )
    )

    assert manifest["owner"] == "canonical_brain_migration_owner"
    assert manifest["owner"] == CANONICAL_WRITER_MIGRATION_OWNER
    assert manifest["runtime_execute"] is False
    assert len(signatures) == len(set(signatures)) == 12
    assert tuple(sorted(signatures)) == EXPECTED_HELPER_ROUTINE_SIGNATURES
    assert {item.split("(", 1)[0].split(".", 1)[1] for item in signatures} == sql_helpers
    for helper in sql_helpers:
        definition = _function_definition(helper)
        assert "SET search_path = pg_catalog, canonical_brain" in definition
        assert re.search(
            rf"ALTER FUNCTION canonical_brain\.{re.escape(helper)}\([\s\S]*?\)"
            r"\s+OWNER TO canonical_brain_migration_owner;",
            SQL,
        )


def test_migration_locks_and_fails_closed_on_schema_prerequisites():
    begin_offset = SQL.index("BEGIN;")
    lock_offset = SQL.index(
        "SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);"
    )
    prerequisite_offset = SQL.index("DO $prerequisites$")

    assert begin_offset < lock_offset < prerequisite_offset
    assert "public.canonical_event_log is incompatible" in SQL
    assert "pg_catalog.sha256(bytea)" in SQL
    assert "server_version_num" in SQL
    assert "server_version_num')::integer < 160000" in SQL
    assert "PostgreSQL 16+ SET-only role membership" in SQL
    assert (
        "canonical_brain_migration_owner must be a least-privilege NOLOGIN role"
        in SQL
    )
    assert "canonical_brain_writer must be a least-privilege NOLOGIN role" in SQL
    assert "canonical_brain_writer must not inherit the migration-owner role" in SQL
    assert "at most one direct least-privilege LOGIN member" in SQL
    assert "inheritors(oid, depth, admin_path)" in SQL
    assert "must have no role memberships or inheriting members" in SQL
    assert "('event_id','uuid',1)" in SQL
    assert "('payload','jsonb',14)" in SQL
    assert "= 'PRIMARY KEY (event_id)'" in SQL
    assert "preexisting writer table column contract mismatch" in SQL
    assert "preexisting writer table type contract mismatch" in SQL
    assert "preexisting writer table relation contract mismatch" in SQL
    assert "preexisting writer table active surface forbidden" in SQL
    assert "preexisting writer foreign-key contract mismatch" in SQL
    assert "(SELECT * FROM actual EXCEPT SELECT * FROM expected)" in SQL
    assert "preexisting %.% has untrusted owner %" in SQL


def test_managed_admin_uses_only_approval_bound_transactional_owner_authority():
    temporary = _dollar_block("temporary_owner_membership")
    retirement = _dollar_block("retire_temporary_owner_membership")
    final_contract = _dollar_block("final_owner_membership_contract")

    assert "canonical_writer_migration_scope" in SQL
    assert "canonical_writer_migration_database" in SQL
    assert "canonical_writer_migration_approval_receipt_sha256" in SQL
    assert "isolated_canary_copy" in SQL
    assert "owner_approved_cutover" in SQL
    assert "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE" in temporary
    assert "WITH ADMIN FALSE'" not in temporary
    assert "SET LOCAL ROLE canonical_brain_migration_owner;" in SQL
    assert "GRANT TEMPORARY ON DATABASE %I" in temporary
    assert "RESET ROLE;" in SQL
    assert "REVOKE canonical_brain_migration_owner FROM %I" in retirement
    assert "pg_catalog.pg_auth_members" in final_contract
    assert "has_database_privilege" in final_contract
    assert "'TEMP'" in final_contract
    assert SQL.index("DO $prerequisites$") < SQL.index("DO $temporary_owner_membership$")
    assert SQL.index("DO $temporary_owner_membership$") < SQL.index(
        "DO $owner_schema_create$"
    ) < SQL.index("SET LOCAL ROLE canonical_brain_migration_owner;")
    assert (
        "CREATE SCHEMA canonical_brain\n"
        "            AUTHORIZATION canonical_brain_migration_owner"
    ) in SQL
    assert "ALTER SCHEMA canonical_brain OWNER TO" not in SQL
    assert SQL.index("SET LOCAL ROLE canonical_brain_migration_owner;") < SQL.index(
        "RESET ROLE;"
    )
    assert SQL.index("DO $final_owner_membership_contract$") < SQL.rindex("COMMIT;")


def test_cloudsqladmin_exception_is_exact_and_requires_hba_rejection_receipt():
    assert "_cw_managed_cloudsqladmin_exception" in SQL
    helper = SQL.split(
        "CREATE OR REPLACE FUNCTION pg_temp._cw_managed_cloudsqladmin_exception",
        1,
    )[1].split("$function$;", 1)[0]

    assert "hba_rejection_sha256 ~ '^[0-9a-f]{64}$'" in helper
    assert "datname = 'cloudsqladmin'" in helper
    assert "owner_name = 'cloudsqladmin'" in helper
    assert "rolname = 'cloudsqladmin'" in helper
    assert "rolsuper" in helper
    assert "rolreplication" in helper
    assert "rolbypassrls" in helper
    assert "rolname = 'cloudsqlsuperuser'" in helper
    assert "NOT rolsuper" in helper
    assert "pg_catalog.acldefault('d', database_row.datdba)" in helper
    for privilege in ("'PUBLIC','cloudsqladmin','CONNECT',false",
                      "'PUBLIC','cloudsqladmin','TEMPORARY',false",
                      "'cloudsqladmin','cloudsqladmin','CREATE',false"):
        assert privilege in helper
    assert "SELECT * FROM actual_acl EXCEPT SELECT * FROM expected_acl" in helper
    assert "SELECT * FROM expected_acl EXCEPT SELECT * FROM actual_acl" in helper
    assert SQL.count("pg_temp._cw_managed_cloudsqladmin_exception(") >= 3


def test_canonical_event_log_has_exact_owner_shape_and_exclusive_append_boundary():
    contract = SQL.split("DO $event_log_contract$", 1)[1].split(
        "$event_log_contract$;", 1
    )[0]
    exclusivity = SQL.split("DO $event_log_exclusivity$", 1)[1].split(
        "$event_log_exclusivity$;", 1
    )[0]

    assert "LOCK TABLE public.canonical_event_log IN ACCESS EXCLUSIVE MODE;" in SQL
    assert "relation_record.relkind <> 'r'" in contract
    assert "relation_record.relpersistence" in contract
    assert "relation_record.relispartition" in contract
    assert "relation_record.relreplident" in contract
    assert "relation_record.relrowsecurity" in contract
    assert "relation_record.relforcerowsecurity" in contract
    assert "exact column/type contract mismatch" in contract
    assert "nullability/identity contract mismatch" in contract
    assert "only the exact event_id primary key constraint" in contract
    assert "constraint_row.contype <> 'n'" in contract
    assert "constraint_row.contype NOT IN ('p','u','f','c','n')" in SQL
    assert "constraint_row.connoinherit <>" in SQL
    assert "'{}'::aclitem[]" not in SQL
    assert "FROM ROWS FROM (" in SQL
    for forbidden_surface in (
        "pg_catalog.pg_trigger",
        "pg_catalog.pg_rewrite",
        "pg_catalog.pg_policy",
        "pg_catalog.pg_inherits",
    ):
        assert forbidden_surface in contract
    assert "must already be owned by canonical_brain_migration_owner" in SQL
    assert "ALTER TABLE public.canonical_event_log" not in SQL
    assert "REVOKE ALL ON TABLE public.canonical_event_log FROM PUBLIC;" in SQL
    assert "retire_event_log_writers" in SQL
    retire = _dollar_block("retire_event_log_writers")
    assert "attribute.attacl" in retire
    assert "attribute.attnum > 0" in retire
    assert "direct_acl.grantee <> direct_acl.relowner" in retire
    assert "REVOKE ALL PRIVILEGES (%I) ON TABLE" in retire
    assert "acl.grantee NOT IN (owner_oid)" in exclusivity
    assert "attribute.attacl" in exclusivity
    assert "acl.privilege_type IN (" in exclusivity
    for column_privilege in ("'SELECT'", "'INSERT'", "'UPDATE'", "'REFERENCES'"):
        assert column_privilege in exclusivity
    assert "owner/RLS/ACL exclusivity attestation failed" in exclusivity
    assert "pg_catalog.unnest(index.indkey)" in exclusivity
    assert "pg_catalog.unnest(index.indclass)" in exclusivity


def test_runtime_role_has_only_fixed_routines_and_current_database_connect():
    required = (
        "GRANT USAGE ON SCHEMA canonical_brain TO canonical_brain_writer;",
        "REVOKE ALL ON public.canonical_event_log FROM canonical_brain_writer;",
    )
    for statement in required:
        assert statement in SQL
    assert "public.digest" not in SQL
    assert "pg_catalog.sha256(pg_catalog.convert_to(value, 'UTF8'))" in SQL
    assert "can CONNECT to another database" in SQL
    assert "effective current-database privileges are not CONNECT-only" in SQL
    assert "retains effective public-schema privileges" in SQL
    assert "retains direct table authority" in SQL
    assert "REVOKE ALL PRIVILEGES ON DATABASE %I" not in SQL
    assert "GRANT CONNECT ON DATABASE %I" not in SQL


def test_migration_does_not_mutate_shared_public_or_database_public_acls():
    # Shared-database hardening is a deployment prerequisite.  This artifact
    # retires ACLs only on its own namespace/event table and on the writer role;
    # it must not break unrelated applications by rewriting PUBLIC globally.
    forbidden = (
        "REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM PUBLIC;",
        "REVOKE ALL ON ALL PROCEDURES IN SCHEMA public FROM PUBLIC;",
        "REVOKE USAGE, CREATE ON SCHEMA public FROM PUBLIC;",
        "REVOKE ALL ON SCHEMA public FROM PUBLIC;",
        "REVOKE ALL PRIVILEGES ON DATABASE %I FROM PUBLIC",
        "REVOKE CONNECT ON DATABASE %I FROM PUBLIC",
    )
    for statement in forbidden:
        assert statement not in SQL

    effective_acl = _dollar_block("effective_writer_acl")
    assert "retains effective public-schema privileges" in effective_acl
    assert "retains executable non-system routines outside the fixed catalog" in effective_acl


def test_canonical_acl_cleanup_and_attestation_cover_every_direct_surface():
    retire = _dollar_block("retire_canonical_acl")
    attest = _dollar_block("canonical_direct_acl_contract")
    defaults_retire = _dollar_block("retire_canonical_default_acl")
    defaults_attest = _dollar_block("canonical_default_acl_contract")

    for surface in ("'schema'::text", "'table'", "'sequence'", "'function'", "'procedure'"):
        assert surface in retire
    assert "attribute.attacl" in retire
    assert "attribute.attnum > 0" in retire
    assert "acl.grantee <> class.relowner" in retire

    for surface in ("'schema'", "'table'", "'column'", "'function'", "'procedure'"):
        assert surface in attest
    assert "class.oid = 'public.canonical_event_log'::regclass" in attest
    assert "attribute.attacl" in attest
    assert "acl.grantee <> class.relowner" in attest
    assert re.search(
        r"'canonical_brain_writer',\s*'USAGE', false", attest
    )
    assert re.search(
        r"'canonical_brain_writer',\s*'EXECUTE', false", attest
    )
    assert "canonical direct ACL contract mismatch" in attest

    assert "acl.grantee <> defaults.defaclrole" in defaults_retire
    assert "REVOKE ALL PRIVILEGES ON %s FROM %s CASCADE" in defaults_retire
    assert "acl.grantee <> defaults.defaclrole" in defaults_attest
    assert "canonical default ACL contract retains a non-owner grant" in defaults_attest


def test_rerun_attestation_uses_exact_default_check_and_index_templates():
    contract = _dollar_block("table_contract")
    defaults = contract.split(
        "WITH expected(table_name, column_name, default_expression) AS (", 1
    )[1].split("), actual AS (", 1)[0]
    default_rows = set(
        re.findall(r"\('([^']+)','([^']+)','([^']+)'\)", defaults)
    )
    assert default_rows == {
        ("writer_routeback_authorizations", "created_at", "clock_timestamp()"),
        ("writer_routeback_lifecycle_terminals", "finalized_at", "clock_timestamp()"),
        ("writer_routeback_terminals", "finalized_at", "clock_timestamp()"),
        ("writer_public_routeback_targets", "enabled", "true"),
        ("writer_event_provenance", "appended_at", "clock_timestamp()"),
        ("writer_capability_grants", "granted_at", "clock_timestamp()"),
        ("writer_capability_revocation_scopes", "revoked_at", "clock_timestamp()"),
        ("writer_capability_revocations", "revoked_at", "clock_timestamp()"),
        ("writer_capability_consumptions", "consumed_at", "clock_timestamp()"),
    }

    checks = contract.split(
        "WITH expected_columns(table_name, column_name) AS (", 1
    )[1].split("), template AS (", 1)[0]
    check_rows = set(re.findall(r"\('([^']+)','([^']+)'\)", checks))
    assert check_rows == {
        ("writer_routeback_authorizations", "content_sha256"),
        ("writer_routeback_authorizations", "session_key_sha256"),
        ("writer_routeback_authorizations", "capability_epoch_sha256"),
        ("writer_routeback_authorizations", "idempotency_key"),
        ("writer_routeback_authorizations", "request_sha256"),
        ("writer_routeback_lifecycle_terminals", "idempotency_key"),
        ("writer_routeback_lifecycle_terminals", "outcome"),
        ("writer_routeback_lifecycle_terminals", "request_sha256"),
        ("writer_routeback_lifecycle_terminals", "session_key_sha256"),
        ("writer_routeback_lifecycle_terminals", "capability_epoch_sha256"),
        ("writer_routeback_terminals", "outcome"),
        ("writer_routeback_terminals", "request_sha256"),
        ("writer_public_routeback_targets", "target_type"),
        ("writer_event_provenance", "canonical_content_sha256"),
        ("writer_event_provenance", "trusted_runtime"),
        ("writer_capability_grants", "plan_revision"),
        ("writer_capability_grants", "session_key_sha256"),
        ("writer_capability_grants", "capability_epoch_sha256"),
        ("writer_capability_grants", "approval_source_sha256"),
        ("writer_capability_grants", "command_hashes"),
        ("writer_capability_grants", "max_uses"),
        ("writer_capability_grants", "request_sha256"),
        ("writer_capability_revocation_scopes", "scope_type"),
        ("writer_capability_revocation_scopes", "session_key_sha256"),
        ("writer_capability_revocation_scopes", "capability_epoch_sha256"),
        ("writer_capability_revocations", "revoked_by_session_sha256"),
        ("writer_capability_consumptions", "command_sha256"),
        ("writer_capability_consumptions", "session_key_sha256"),
        ("writer_capability_consumptions", "capability_epoch_sha256"),
        ("writer_capability_consumptions", "idempotency_key"),
        ("writer_capability_consumptions", "request_sha256"),
        ("writer_capability_consumptions", "remaining_uses"),
    }
    assert "CREATE TEMPORARY TABLE canonical_writer_check_contract" in SQL
    assert "pg_catalog.pg_get_constraintdef(constraint_row.oid, false)" in contract
    assert "preexisting writer table CHECK contract mismatch" in contract
    assert "occurrences" in contract
    assert "constraint_row.contype NOT IN ('p','u','f','c','n')" in contract
    assert "preexisting writer constraint index contract mismatch" in contract
    assert "pg_catalog.unnest(index.indkey)" in contract

    index_map = contract.split(
        "WITH index_map(actual_name, template_name) AS (", 1
    )[1].split("), template AS (", 1)[0]
    index_rows = set(
        re.findall(r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", index_map)
    )
    assert index_rows == {
        ("writer_routeback_target_idx", "contract_routeback_target_idx"),
        ("writer_event_provenance_session_idx", "contract_event_provenance_session_idx"),
        ("writer_event_provenance_thread_idx", "contract_event_provenance_thread_idx"),
        ("writer_capability_scope_idx", "contract_capability_scope_idx"),
        ("writer_capability_command_idx", "contract_capability_command_idx"),
        ("writer_capability_use_idx", "contract_capability_use_idx"),
    }
    assert "CREATE TEMPORARY TABLE canonical_writer_index_contract" in SQL
    for property_name in (
        "key_columns",
        "expressions",
        "predicate",
        "operator_classes",
        "collations",
        "options",
    ):
        assert property_name in contract
    assert "preexisting writer table index contract mismatch" in contract
    assert "preexisting writer table index state/owner contract mismatch" in contract
    assert contract.count("SELECT * FROM expected EXCEPT SELECT * FROM actual") >= 4
    assert contract.count("SELECT * FROM actual EXCEPT SELECT * FROM expected") >= 4


def test_canonical_and_private_ledgers_are_append_only_for_runtime_contract():
    assert not re.search(
        r"\b(?:UPDATE|DELETE\s+FROM|TRUNCATE)\s+"
        r"(?:public\.)?canonical_event_log\b",
        SQL,
        flags=re.IGNORECASE,
    )
    assert not re.search(
        r"\b(?:UPDATE|DELETE\s+FROM|TRUNCATE)\s+canonical_brain\.writer_",
        SQL,
        flags=re.IGNORECASE,
    )
    assert "INSERT INTO public.canonical_event_log" in SQL
    assert "INSERT INTO canonical_brain.writer_event_provenance" in SQL
    assert "INSERT INTO canonical_brain.writer_capability_consumptions" in SQL
    assert "INSERT INTO canonical_brain.writer_routeback_terminals" in SQL


def test_only_privileged_writer_inserts_trusted_event_provenance():
    append = _function_definition("_append_event")
    scope = _function_definition("_case_scope_authorized")
    query = _function_definition("writer_case_query")

    assert "writer_event_provenance" in append
    assert append.index("INSERT INTO public.canonical_event_log") < append.index(
        "INSERT INTO canonical_brain.writer_event_provenance"
    )
    assert append.index(
        "canonical event readback mismatch before provenance append"
    ) < append.index("INSERT INTO canonical_brain.writer_event_provenance")
    assert "event_provenance_missing" in append
    assert "preexisting event was not inserted by the privileged writer" in append
    assert "JOIN canonical_brain.writer_event_provenance" in scope
    assert "provenance.trusted_runtime" in scope
    assert "event.source->'observed_session'" not in scope
    assert "JOIN canonical_brain.writer_event_provenance" in query
    assert "provenance.trusted_runtime" in query


def test_event_retry_readback_uses_original_provenance_runtime():
    append = _function_definition("_append_event")
    dedupe = append.split("IF FOUND THEN", 1)[1].split(
        "INSERT INTO public.canonical_event_log", 1
    )[0]

    assert "provenance.trusted_runtime" in dedupe
    assert "INTO provenance_sha, provenance_at, provenance_runtime" in dedupe
    assert "'observed_session', provenance_runtime" in dedupe


def test_case_scope_is_server_observed_and_rechecked_under_case_lock():
    scope = _function_definition("_case_scope_authorized")
    assert "runtime_value->>'owner_authenticated' = 'true'" in scope
    assert "provenance.trusted_runtime->>'session_key_sha256'" in scope
    assert "provenance.trusted_runtime->>'thread_id'" in scope
    assert "event.source->'source_refs'" not in scope
    assert "writer_routeback_authorizations" in scope
    assert "writer_routeback_terminals" in scope
    assert "terminal.outcome = 'sent'" in scope
    assert "writer_public_routeback_targets" in scope
    assert "allowed.enabled" in scope
    assert "allowed.target_type IN ('public_channel', 'public_thread')" in scope
    assert "terminal.outcome = 'blocked'" not in scope

    for name, allow_new in (
        ("writer_event_append_model", "true"),
        ("writer_plan_transition", "true"),
        ("writer_verification_append", "false"),
        ("writer_routeback_claim", "false"),
        ("writer_lease_shadow_record", "false"),
    ):
        definition = _function_definition(name)
        assert f"_case_scope_authorized" in definition
        assert re.search(
            rf"_case_scope_authorized\([^;]*runtime, {allow_new}\)", definition
        )
    for name in ("writer_event_append_model", "writer_plan_transition"):
        definition = _function_definition(name)
        assert definition.index("canonical-case:") < definition.index(
            "_case_scope_authorized"
        )


def test_thread_reads_cannot_use_model_authored_source_refs_as_authority():
    query = _function_definition("writer_case_query")
    route_context = _function_definition("writer_routeback_context")

    assert "thread query must match the exact observed runtime thread" in query
    assert "provenance.trusted_runtime->>'thread_id'" in query
    assert "authorization_row.source_thread_id" in route_context
    assert "authorization_row.source_refs" not in route_context


def test_routeback_claim_is_acl_only_atomic_and_returns_claim_time():
    claim = _function_definition("writer_routeback_claim")
    sent = _function_definition("writer_routeback_finalize_sent")
    blocked = _function_definition("writer_routeback_finalize_blocked")

    assert "writer_public_routeback_targets" in claim
    assert "allowed.enabled" in claim
    assert "allowed.target_type IN ('public_channel', 'public_thread')" in claim
    assert "owner-provisioned public target ACL" in claim
    assert "case-linked" not in claim
    assert "routeback-lifecycle:" in claim
    lifecycle_identity = claim.split("authorization_value :=", 1)[1].split(
        "request_hash :=", 1
    )[0]
    assert "'session'" not in lifecycle_identity
    assert claim.index("pg_advisory_xact_lock") < claim.index(
        "INSERT INTO canonical_brain.writer_routeback_authorizations"
    )
    assert claim.count("'claimed_at'") >= 2
    assert "source_thread_id" in claim
    pending_dedupe = claim.split(
        "SELECT * INTO existing_record", 1
    )[1].split("IF NOT EXISTS (", 1)[0]
    assert "terminal_record.authorization_id IS NULL" in pending_dedupe
    for exact_scope_binding in (
        "existing_record.session_key_sha256 IS DISTINCT FROM session_value",
        "existing_record.capability_epoch_sha256 IS DISTINCT FROM epoch_value",
        "existing_record.runtime_platform IS DISTINCT FROM runtime_platform_value",
        "existing_record.source_thread_id IS DISTINCT FROM source_thread_value",
    ):
        assert exact_scope_binding in pending_dedupe
    assert pending_dedupe.index("terminal_record.authorization_id IS NULL") < (
        pending_dedupe.index("RETURN canonical_brain._ok")
    )
    assert "pending route-back authorization belongs to another exact runtime scope" in (
        pending_dedupe
    )
    assert "writer_public_routeback_targets" in pending_dedupe
    assert "allowed.enabled" in pending_dedupe
    assert "allowed.target_type IN ('public_channel', 'public_thread')" in (
        pending_dedupe
    )
    assert pending_dedupe.index("'scope_mismatch'") < pending_dedupe.index(
        "'target_not_approved'"
    ) < pending_dedupe.index("RETURN canonical_brain._ok")
    for definition in (claim, sent, blocked):
        assert "_contains_forbidden_dm_ref" in definition
    dm_helper = _function_definition("_contains_forbidden_dm_ref")
    for forbidden in (
        "'dm'",
        "'direct_message'",
        "'private_dm'",
        "'user_dm'",
        "'private'",
        "'group'",
        "'group_dm'",
        "'private_channel'",
        "'private_thread'",
    ):
        assert forbidden in dm_helper
    for field in ("platform", "adapter_receipt", "receipt_readback_verified"):
        assert f"receipt_value->'{field}'" in sent or f"receipt_value->>'{field}'" in sent
    assert "receipt_value->>'platform' <> 'discord'" in sent
    assert "receipt_value->'adapter_receipt' <> 'true'::jsonb" in sent
    assert "receipt_value->'receipt_readback_verified' <> 'true'::jsonb" in sent
    assert "receipt_value->>'content_sha256' <> authorization_record.content_sha256" in sent
    assert "receipt_value->>'channel_id' <> target_id" in sent
    assert "route_back.sent" in sent
    assert "route_back.blocked" in blocked
    assert "'partial_receipt', receipt_value" in blocked
    assert "receipt_readback_verified" in blocked
    assert (
        "receipt_value->'receipt_readback_verified'\n"
        "                   IS DISTINCT FROM 'true'::jsonb"
    ) in blocked
    assert "'route_back_sent_receipt_persistence_failed'" in blocked
    assert "receipt_value->>'channel_id' <> target_id" in blocked
    assert (
        "receipt_value->>'content_sha256'\n"
        "               <> authorization_record.content_sha256"
    ) in blocked
    assert "'outbound_delivery_uncertain'" in blocked
    assert "'accepted_unverified'" not in blocked


def test_routeback_blocked_routine_has_typed_preclaim_without_send_authorization():
    blocked = _function_definition("writer_routeback_finalize_blocked")
    preclaim = blocked.split("IF preclaim_value THEN", 1)[1].split(
        "\n    IF NOT canonical_brain._runtime_valid(runtime)", 1
    )[0]

    assert "preclaim_value boolean := request->'preclaim' = 'true'::jsonb" in blocked
    assert "IF preclaim_value THEN" in blocked
    assert "request->'preclaim' <> 'true'::jsonb" in blocked
    assert "request->>'outcome' <> 'blocked'" in blocked
    assert "request->'receipt' <> '{}'::jsonb" in blocked
    assert "_case_scope_authorized(case_value, runtime, true)" in blocked
    assert "'route_back.blocked'" in blocked
    assert "'preclaim', true" in blocked
    assert "'delivery_state', 'not_attempted'" in blocked
    assert "'outbound', false" in blocked
    assert "'routeback_preclaim_blocked'" in blocked
    assert "INSERT INTO canonical_brain.writer_routeback_authorizations" not in preclaim
    assert "INSERT INTO canonical_brain.writer_routeback_terminals" not in preclaim
    assert "INSERT INTO canonical_brain.writer_routeback_lifecycle_terminals" in preclaim
    assert "'authorization_id'" not in preclaim
    assert "writer_routeback_finalize_preclaim" not in _writer_names()


def test_routeback_restart_recovery_is_exact_lane_and_append_only():
    recover = _function_definition("writer_routeback_recover")
    sent = _function_definition("writer_routeback_finalize_sent")
    blocked = _function_definition("writer_routeback_finalize_blocked")

    assert "recovery_value NOT IN ('edge_evidence', 'edge_no_record')" in recover
    assert "_case_scope_authorized(case_value, runtime, false)" in recover
    assert "authorization_record.session_key_sha256 <> session_value" in recover
    assert "authorization_record.runtime_platform <> platform_value" in recover
    assert "authorization_record.source_thread_id <> source_thread_value" in recover
    assert "recovery_value = 'edge_no_record'" in recover
    assert "writer_public_routeback_targets" in recover
    assert "allowed.enabled" in recover
    assert "'recovered_epoch_sha256', epoch_value" in recover
    assert "INSERT INTO canonical_brain.writer_" not in recover
    assert "UPDATE canonical_brain.writer_" not in recover
    assert "writer_capability_grants" not in recover
    for finalizer in (sent, blocked):
        assert "authorization_record.runtime_platform <> runtime->>'platform'" in finalizer
        assert "authorization_record.source_thread_id <> COALESCE(" in finalizer
        assert "_case_scope_authorized(" in finalizer
        assert "authorization_record.capability_epoch_sha256" not in finalizer


def test_plan_and_capability_transactions_are_exact_and_non_replenishing():
    plan = _function_definition("writer_plan_transition")
    plan_head = _function_definition("_plan_head")
    grant = _function_definition("writer_capability_grant")
    consume = _function_definition("writer_capability_consume")

    assert "canonical-plan:" in plan
    assert "plan_cas_conflict" in plan
    assert "verification_receipt_missing" in plan
    assert "ON CONFLICT (approval_id) DO NOTHING" in plan
    for reason in ("'plan_superseded'", "'plan_revision_advanced'", "'plan_' || state_value"):
        assert reason in plan
    assert plan_head.count("FROM public.canonical_event_log AS event") == plan_head.count(
        "JOIN canonical_brain.writer_event_provenance AS provenance"
    )
    assert "verification_provenance" in plan
    revocation_inserts = re.findall(
        r"INSERT INTO canonical_brain\.writer_capability_revocations\s*"
        r"\((.*?)\)\s*SELECT",
        plan,
        flags=re.DOTALL,
    )
    assert len(revocation_inserts) == 3
    for columns in revocation_inserts:
        assert re.sub(r"\s+", "", columns) == (
            "approval_id,reason,revoked_by_session_sha256,revoked_at"
        )

    assert "runtime->>'owner_authenticated' <> 'true'" in grant
    assert "INTERVAL '8 hours'" in grant
    assert "^([1-9][0-9]{0,2}|1000)$" in grant
    immutable_hash = grant.split("request_hash :=", 1)[1].split(
        "SELECT * INTO existing_record", 1
    )[0]
    for exact_binding in (
        "'plan_revision', plan_revision_value",
        "'approved_by_user_id', user_value",
        "'expires_at', pg_catalog.to_char(",
        "'session_key_sha256', session_value",
        "'capability_epoch_sha256', epoch_value",
        "'command_hashes', request->'command_hashes'",
    ):
        assert exact_binding in immutable_hash
    assert "AT TIME ZONE 'UTC'" in immutable_hash
    assert "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'" in immutable_hash
    assert "writer_capability_consumptions" in grant
    assert "writer_capability_revocations" in grant
    assert "authority_active" in grant
    assert "current_state" in grant
    assert "'capability_epoch_sha256', epoch_value" in immutable_hash

    assert "capability-consume:" in consume
    assert "writer_capability_consumptions" in consume
    assert "_plan_head" in consume
    assert "remaining_value := grant_record.max_uses - used_count - 1" in consume
    response = consume.split("response_value :=", 1)[1]
    for key in ("'approval_id'", "'case_id'", "'plan_id'", "'remaining_uses'"):
        assert key in response


def test_capabilities_are_bound_to_exact_runtime_epoch_across_all_sql_paths():
    runtime = _function_definition("_runtime_valid")
    grant = _function_definition("writer_capability_grant")
    consume = _function_definition("writer_capability_consume")
    revoke = _function_definition("writer_capability_revoke")
    revoke_session = _function_definition("writer_capability_revoke_session")

    assert "'capability_epoch_sha256'" in runtime
    assert "runtime->>'capability_epoch_sha256' ~ '^[0-9a-f]{64}$'" in runtime
    assert SQL.count("capability_epoch_sha256 text NOT NULL") == 5
    assert (
        "UNIQUE (session_key_sha256, capability_epoch_sha256, idempotency_key)"
        in SQL
    )

    for definition in (grant, consume, revoke, revoke_session):
        assert (
            "epoch_value text := COALESCE("
            "runtime->>'capability_epoch_sha256', '')"
        ) in definition
        assert "epoch_value !~ '^[0-9a-f]{64}$'" in definition
        assert "'capability_epoch_sha256'" in definition
        assert "'capability-scope:' || session_value || ':' || epoch_value" in definition

    assert "grant_row.capability_epoch_sha256 = epoch_value" in consume
    assert "consumption.capability_epoch_sha256 = epoch_value" in consume
    assert "'capability-consume:' || session_value || ':' || epoch_value" in consume
    assert "capability_epoch_sha256, idempotency_key" in consume
    assert "grant_row.capability_epoch_sha256 = epoch_value" in revoke
    assert "grant_row.capability_epoch_sha256 = epoch_value" in revoke_session
    assert "'authority_active', true" in grant
    assert "'authority_active', current_state = 'granted'" in grant
    assert "writer_capability_revocation_scopes" in grant
    assert "'scope_type', 'session'" in revoke_session
    assert "'scope_revoked', true" in revoke_session


def test_retired_session_epoch_fence_precedes_every_initiating_mutation_lock():
    initiating = {
        "writer_event_append_model": "canonical-case:",
        "writer_plan_transition": "canonical-case:",
        "writer_verification_append": "_case_scope_authorized",
        "writer_routeback_claim": "_case_scope_authorized",
    }
    for name, next_authority_boundary in initiating.items():
        definition = _function_definition(name)
        assert (
            "session_value text := COALESCE("
            "runtime->>'session_key_sha256', '')"
        ) in definition
        assert (
            "epoch_value text := COALESCE("
            "runtime->>'capability_epoch_sha256', '')"
        ) in definition
        assert "session_value !~ '^[0-9a-f]{64}$'" in definition
        assert "epoch_value !~ '^[0-9a-f]{64}$'" in definition
        assert definition.count("'capability-scope:'") == 1
        assert definition.count(
            "FROM canonical_brain.writer_capability_revocation_scopes AS scope"
        ) == 1
        scope_lock = definition.index("'capability-scope:'")
        tombstone_check = definition.index(
            "FROM canonical_brain.writer_capability_revocation_scopes AS scope"
        )
        retired_failure = definition.index("'session_epoch_retired'")
        next_boundary = definition.index(next_authority_boundary, retired_failure)
        assert scope_lock < tombstone_check < retired_failure < next_boundary
        assert "scope.scope_type = 'session'" in definition
        assert "scope.session_key_sha256 = session_value" in definition
        assert "scope.capability_epoch_sha256 = epoch_value" in definition

    blocked = _function_definition("writer_routeback_finalize_blocked")
    preclaim_branch = blocked.index("IF preclaim_value THEN")
    scope_lock = blocked.index("'capability-scope:'")
    tombstone_check = blocked.index(
        "FROM canonical_brain.writer_capability_revocation_scopes AS scope"
    )
    case_lock = blocked.index("'canonical-case:'", tombstone_check)
    assert preclaim_branch < scope_lock < tombstone_check < case_lock
    assert blocked.count("'capability-scope:'") == 1
    assert blocked.count("'session_epoch_retired'") == 1


def test_claimed_routeback_terminal_truth_is_not_blocked_by_epoch_retirement():
    sent = _function_definition("writer_routeback_finalize_sent")
    blocked = _function_definition("writer_routeback_finalize_blocked")

    assert "writer_capability_revocation_scopes" not in sent
    assert "'capability-scope:'" not in sent
    for definition in (sent, blocked):
        assert (
            "authorization_record.session_key_sha256 "
            "<> runtime->>'session_key_sha256'"
        ) in definition
        assert "authorization_record.capability_epoch_sha256" not in definition
        assert "authorization_record.runtime_platform <> runtime->>'platform'" in definition
        assert "authorization_record.source_thread_id <> COALESCE(" in definition
        assert "_case_scope_authorized(" in definition

    # The blocked routine has one fence only, in its preclaim initiation branch;
    # the existing-claim path remains exact-claimant bound but tombstone-free.
    assert blocked.count("writer_capability_revocation_scopes") == 1
    assert blocked.count("'capability-scope:'") == 1


def test_capability_revoke_receipts_bind_the_exact_sorted_revocation_set():
    plan_revoke = _function_definition("writer_capability_revoke")
    session_revoke = _function_definition("writer_capability_revoke_session")

    for definition, event_type, receipt_error in (
        (
            plan_revoke,
            "approval.capability.revoked",
            "capability revocation receipt append failed",
        ),
        (
            session_revoke,
            "approval.capability.session_revoked",
            "session revocation receipt append failed",
        ),
    ):
        assert "pg_catalog.array_agg(approval_id ORDER BY approval_id)" in definition
        assert "revocation_set_sha256 := canonical_brain._sha256_json(" in definition
        assert "pg_catalog.to_jsonb(revoked_ids)" in definition
        assert "IF inserted_now > 0 THEN" in definition
        assert f"'{event_type}'" in definition
        assert "'revocation_set_sha256', revocation_set_sha256" in definition
        assert "|| revocation_set_sha256" in definition
        assert receipt_error in definition

    assert "'plan_id', plan_value" in plan_revoke
    assert "'capability_epoch_sha256', epoch_value" in plan_revoke
    assert "'session_key_sha256', session_value" in session_revoke
    assert "'capability_epoch_sha256', epoch_value" in session_revoke


def test_verification_and_completion_are_exact_plan_criterion_receipts():
    verification = _function_definition("writer_verification_append")
    completion = _function_definition("writer_plan_transition")

    assert verification.index("canonical-plan:") < verification.index("_plan_head")
    assert "head_plan->>'state' <> 'active'" in verification
    assert "verification_value->'criterion_ids'" in verification
    assert "head_plan->'success_criteria'" in verification
    assert "verification criterion_ids must be a unique subset" in verification
    assert "verification_criteria_invalid" in verification

    assert completion.index("canonical-plan:") < completion.index(
        "IF state_value = 'completed'"
    )
    assert "previous_plan_id <> plan_id_value" in completion
    assert "previous_revision <> revision_value - 1" in completion
    assert "previous_plan->>'state' <> 'active'" in completion
    assert (
        "plan_value->'success_criteria'\n"
        "              IS DISTINCT FROM previous_plan->'success_criteria'"
    ) in completion
    assert "verification_provenance" in completion
    assert "verification_event.payload->'verification'->>'plan_id'" in completion
    assert "= plan_id_value" in completion
    assert "verification_event.payload->'verification'->>'plan_revision'" in completion
    assert "= previous_revision::text" in completion
    assert "verification_event.payload->'verification'->>'outcome'" in completion
    assert "verification_criteria_uncovered" in completion
    assert "covered.criterion_id = required_criterion.value->>'id'" in completion


def test_plan_and_verification_retries_dedupe_before_current_head_checks():
    plan = _function_definition("writer_plan_transition")
    verification = _function_definition("writer_verification_append")

    assert "transition_identity := request->>'idempotency_key'" in plan
    assert plan.index("transition_event_id :=") < plan.index("_plan_head")
    assert plan.index("IF EXISTS (") < plan.index("_plan_head")
    assert "RETURN canonical_brain._append_event(" in plan.split(
        "transition_event_id :=", 1
    )[1].split("head_result :=", 1)[0]
    assert verification.index("verification_event_id :=") < verification.index(
        "_plan_head"
    )
    assert "RETURN canonical_brain._append_event(" in verification.split(
        "verification_event_id :=", 1
    )[1].split("head_result :=", 1)[0]


def test_idempotency_keys_are_bounded_by_utf8_bytes_not_characters():
    assert "pg_catalog.octet_length" in SQL
    assert SQL.count("NOT BETWEEN 1 AND 256") >= 4
    assert "BETWEEN 1 AND 512" not in SQL
    assert not re.search(r"(?<!octet_)length\(idempotency_value\)", SQL)
    assert not re.search(
        r"(?<!octet_)length\(COALESCE\(request->>'idempotency_key'", SQL
    )


def test_resume_bundle_support_is_bounded_explicit_and_no_unsafe_uuid_cast():
    query = _function_definition("writer_case_query")
    for key in (
        "'events'",
        "'support_events'",
        "'truncated'",
        "'candidate_cases_truncated'",
        "'support_incomplete_reasons'",
        "'missing_verification_event_ids'",
        "'view'",
    ):
        assert key in query
    assert "plan_lineage_count > 16" in query
    assert "LIMIT 16" in query
    assert "LIMIT 500" in query
    assert "cumulative_bytes <= 384000" in query
    assert "cumulative_bytes <= 600000" in query
    assert "support_byte_budget_exceeded" in query
    assert "referenced_verification_id_invalid" in query
    assert "legacy_events_quarantined" in query
    for event_type in (
        "task.plan.updated",
        "task.verification.recorded",
        "approval.capability.recorded",
        "capability.check.recorded",
        "route_back.sent",
        "route_back.blocked",
        "lease.shadow.recorded",
    ):
        assert event_type in query
    assert "CASE\n                        WHEN required.event_id ~" in query
    assert not re.search(
        r"required\.event_id\s*~[^\n]+\n\s*AND\s+[^\n]*required\.event_id::uuid",
        query,
    )


def test_projection_read_is_case_scoped_or_internal_and_strictly_paginated():
    projection = _function_definition("writer_projection_read_events")
    assert "case_value = '' AND runtime->>'service_internal' <> 'true'" in projection
    assert "_case_scope_authorized(case_value, runtime, false)" in projection
    assert "limit_value > 500" in projection
    assert "(event.occurred_at, event.event_id) > (cursor_at, cursor_id)" in projection
    assert "LIMIT limit_value + 1" in projection
    assert "cumulative_bytes <= 984000" in projection
    assert "event_exceeds_projection_budget" in projection
    assert "cursor_did_not_advance" in projection
    assert "'events'" in projection
    assert "'has_more'" in projection
    assert "'next_after_event_id'" in projection
    assert "writer_event_provenance" in projection


def test_model_append_reserves_the_complete_privileged_writer_event_catalog():
    model_append = _function_definition("writer_event_append_model")
    reserved_clause = model_append.split("IF event_type_value IN (", 1)[1].split(
        ") THEN", 1
    )[0]
    assert set(re.findall(r"'([^']+)'", reserved_clause)) == {
        "task.plan.updated",
        "task.verification.recorded",
        "route_back.intent.created",
        "route_back.sent",
        "route_back.blocked",
        "approval.capability.recorded",
        "approval.capability.revoked",
        "approval.capability.session_revoked",
        "capability.check.recorded",
        "lease.shadow.recorded",
    }
    assert "privileged_event_forbidden" in model_append
    assert "event type requires its fixed privileged routine" in model_append


def test_sql_contains_no_free_text_business_classifier_or_raw_sql_operation():
    model_append = _function_definition("writer_event_append_model")

    # This exact event-name deny-list is a mechanical privilege boundary, not
    # semantic routing.  It prevents the generic append routine from forging
    # receipts that only fixed writer routines may create.
    assert "privileged_event_forbidden" in model_append
    assert not re.search(r"\bILIKE\b", SQL, flags=re.IGNORECASE)
    assert not re.search(
        r"(?:summary|message_summary)[^\n]*(?:LIKE|~\*?)",
        SQL,
        flags=re.IGNORECASE,
    )
    assert not re.search(r"writer_(?:execute|raw|sql|query_sql)", SQL)
