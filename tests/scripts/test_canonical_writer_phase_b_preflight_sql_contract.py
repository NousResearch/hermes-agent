from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    ROOT
    / "scripts"
    / "sql"
    / "canonical_writer_foundation_phase_b_preflight_v1.sql"
)
SQL = ARTIFACT_PATH.read_text(encoding="utf-8")
EXECUTABLE_SQL = re.sub(r"--[^\n]*", "", SQL)


def test_phase_b_preflight_is_fixed_writer_read_only_and_nonterminal():
    assert (
        "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;"
        in EXECUTABLE_SQL
    )
    assert "SESSION_USER <> 'muncho_canary_writer_login'" in EXECUTABLE_SQL
    assert "CURRENT_USER <> SESSION_USER" in EXECUTABLE_SQL
    assert "'muncho_canary_brain'" in EXECUTABLE_SQL
    assert "'ai_platform_brain'" in EXECUTABLE_SQL
    assert (
        "'muncho-canonical-writer-foundation-phase-b-db-preflight.v1'"
        in EXECUTABLE_SQL
    )
    assert "'preflight', true" in EXECUTABLE_SQL
    assert "'terminal', false" in EXECUTABLE_SQL
    assert "'secret_material_recorded', false" in EXECUTABLE_SQL
    assert "receipt.value::text AS unsigned_receipt_jsonb_text" in EXECUTABLE_SQL
    assert "'unsigned_receipt_jsonb_text'" in EXECUTABLE_SQL
    assert "'receipt_sha256'" in EXECUTABLE_SQL
    assert re.search(
        r"pg_catalog\.sha256\(\s*pg_catalog\.convert_to\(\s*"
        r"receipt\.unsigned_receipt_jsonb_text,\s*'UTF8'\s*\)\s*\)",
        EXECUTABLE_SQL,
    )

    assert not re.search(
        r"\b(?:CREATE|ALTER|DROP|GRANT|REVOKE|INSERT|UPDATE|DELETE|TRUNCATE|"
        r"COPY|CALL)\b",
        EXECUTABLE_SQL,
        flags=re.IGNORECASE,
    )


def test_phase_b_preflight_uses_catalog_oids_not_privilege_sensitive_names():
    assert "pg_catalog.pg_class" in EXECUTABLE_SQL
    assert "pg_catalog.pg_attribute" in EXECUTABLE_SQL
    assert "pg_catalog.pg_proc" in EXECUTABLE_SQL
    assert "pg_catalog.pg_namespace" in EXECUTABLE_SQL
    assert "pg_catalog.pg_auth_members" in EXECUTABLE_SQL
    assert "pg_catalog.pg_database" in EXECUTABLE_SQL
    assert "pg_catalog.pg_constraint" in EXECUTABLE_SQL
    assert "pg_catalog.pg_index" in EXECUTABLE_SQL
    assert "pg_catalog.pg_trigger" in EXECUTABLE_SQL
    assert "pg_catalog.pg_rewrite" in EXECUTABLE_SQL
    assert "pg_catalog.pg_policy" in EXECUTABLE_SQL
    assert "pg_catalog.pg_inherits" in EXECUTABLE_SQL
    assert "pg_catalog.aclexplode" in EXECUTABLE_SQL

    lowered = EXECUTABLE_SQL.lower()
    assert "to_reg" not in lowered
    assert "::regclass" not in lowered
    assert "::regprocedure" not in lowered
    assert "has_schema_privilege" not in lowered
    assert "has_table_privilege" not in lowered
    assert "has_function_privilege" not in lowered
    assert "pg_authid" not in lowered
    assert "rolpassword" not in lowered
    assert not re.search(r"\b(?:from|join)\s+(?:public|canonical_brain)\.", lowered)

    assert re.search(
        r"has_database_privilege\(\s*managed\.oid,\s*database\.oid,\s*"
        r"'CONNECT'",
        EXECUTABLE_SQL,
    )
    assert re.search(
        r"has_database_privilege\(\s*managed\.oid,\s*database\.oid,\s*"
        r"'TEMPORARY'",
        EXECUTABLE_SQL,
    )


def test_phase_b_preflight_observes_complete_recovery_and_truth_surface():
    for required in (
        "'bootstrap_role_absent'",
        "'bootstrap_login_absent'",
        "'temporary_admin_roles'",
        "AS granted_role",
        "AS member_role",
        "AS grantor",
        "membership.admin_option",
        "membership.inherit_option",
        "membership.set_option",
        "'event_log'",
        "'canonical_event_log'",
        "'namespaces'",
        "'namespace_oid'",
        "'writer_ping'",
        "'implementation_sha256'",
        "'configuration_count'",
        "'configuration_is_exact'",
        "'legacy_archive'",
        "'canonical_event_log_legacy_v1'",
        "'owner_superuser'",
        "'owner_create_database'",
        "'owner_create_role'",
        "'owner_replication'",
        "'owner_bypass_row_security'",
        "'owner_connection_limit'",
        "'owner_validity_is_unbounded'",
        "'owner_configuration_is_empty'",
        "'target_database'",
        "'other_connectable_databases'",
        "'effective_public_connect'",
        "'effective_public_temporary'",
        "'managed_cloudsqladmin'",
        "'database_privileges'",
        "'direct_acl'",
        "'acl_is_null'",
        "'relation_acl_is_null'",
        "'relation_acl'",
        "'row_security'",
        "'force_row_security'",
        "'is_partition'",
        "'relation_kind'",
        "'persistence'",
        "'columns'",
        "'constraints'",
        "'indexes'",
        "'user_triggers'",
        "'rules'",
        "'policies'",
        "'inheritance'",
        "'constraint_oids'",
    ):
        assert required in EXECUTABLE_SQL

    assert "database.datallowconn" in EXECUTABLE_SQL
    assert "database.name <> pg_catalog.current_database()" in EXECUTABLE_SQL
    assert "relation.oid::text" in EXECUTABLE_SQL
    assert "relation.owner_oid::text" in EXECUTABLE_SQL
    assert "attribute.attnum::integer AS position" in EXECUTABLE_SQL
    assert "attribute.atthasdef AS has_default" in EXECUTABLE_SQL
    assert "attribute.attidentity::text AS identity" in EXECUTABLE_SQL
    assert "attribute.attgenerated::text AS generated" in EXECUTABLE_SQL
    assert "attribute.attacl IS NULL AS acl_is_null" in EXECUTABLE_SQL
    assert "relation.relrowsecurity" in EXECUTABLE_SQL
    assert "relation.relforcerowsecurity" in EXECUTABLE_SQL
    assert "relation.relispartition" in EXECUTABLE_SQL
    assert "index_row.indisprimary" in EXECUTABLE_SQL
    assert "index_row.indisvalid" in EXECUTABLE_SQL
    assert "index_row.indisready" in EXECUTABLE_SQL
    assert "index_row.indislive" in EXECUTABLE_SQL
    assert "constraint_row.contype <> 'n'" in EXECUTABLE_SQL
    assert "NOT trigger_row.tgisinternal" in EXECUTABLE_SQL
    assert "rewrite_row.ev_class = relation.oid" in EXECUTABLE_SQL
    assert "pg_catalog.to_jsonb(routine.proconfig)" not in EXECUTABLE_SQL
    assert re.search(
        r"pg_catalog\.pg_get_expr\(\s*default_row\.adbin,",
        EXECUTABLE_SQL,
    )
    assert "pg_catalog.pg_get_constraintdef(constraint_row.oid" in EXECUTABLE_SQL

    # Potentially sensitive or unstable expressions are only represented by
    # booleans and SHA-256 digests in the emitted receipt.
    assert "'default_expression'," not in EXECUTABLE_SQL
    assert "'definition'," not in EXECUTABLE_SQL
    assert "'predicate'," not in EXECUTABLE_SQL
    assert "'expressions'," not in EXECUTABLE_SQL
