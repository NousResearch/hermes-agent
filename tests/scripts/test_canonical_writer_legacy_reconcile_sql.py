from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = ROOT / "scripts" / "sql" / "canonical_writer_legacy_reconcile_v1.sql"
SQL = SQL_PATH.read_text(encoding="utf-8")


def _block(name: str) -> str:
    marker = f"DO ${name}$"
    assert marker in SQL
    return SQL.split(marker, 1)[1].split(f"${name}$;", 1)[0]


def test_reconciliation_is_explicitly_isolated_and_refuses_production_name():
    prerequisites = _block("reconcile_prerequisites")

    assert "isolated_canary_copy" in prerequisites
    assert "canonical_writer_reconcile_database" in prerequisites
    assert "database_value IS DISTINCT FROM 'muncho_canary_brain'" in prerequisites
    assert "pg_catalog.current_database() <> 'muncho_canary_brain'" in prerequisites
    assert "pg_catalog.current_database() = 'ai_platform_brain'" in prerequisites
    assert "canonical_writer_reconcile_server_identity_sha256" in prerequisites
    assert "canonical_writer_reconcile_approval_receipt_sha256" in prerequisites
    assert "server_version_num')::integer / 10000 <> 18" in prerequisites


def test_reconciliation_pins_the_exact_nineteen_column_legacy_contract():
    contract = _block("legacy_relation_contract")
    expected = (
        "event_id", "schema_version", "event_type", "occurred_at", "case_id",
        "source", "actor", "subject", "evidence", "decision", "status",
        "next_action", "safety", "payload", "inserted_at", "idempotency_key",
        "source_spool", "spool_line_number", "raw_event_sha256",
    )
    values = re.findall(r"^\s+\('([a-z0-9_]+)'", contract, flags=re.MULTILINE)

    assert tuple(values) == expected
    assert "index_count <> 5" in contract
    assert "distinct_index_key_count <> 5" in contract
    assert "PRIMARY KEY (event_id)" in contract
    assert "triggers/rules/policies/inheritance contract drifted" in contract
    assert "table or column ACL contract drifted" in contract


def test_original_relation_is_moved_intact_and_public_contract_is_exact_fourteen():
    migration = _block("perform_first_reconciliation")

    assert "ALTER TABLE public.canonical_event_log\n        SET SCHEMA" in migration
    assert "RENAME TO canonical_event_log_legacy_v1" in migration
    assert "CREATE TABLE public.canonical_event_log" in migration
    assert "OWNER TO canonical_brain_migration_owner" in migration
    assert migration.count("INSERT INTO public.canonical_event_log") == 1
    assert "DROP TABLE" not in SQL
    assert "DROP COLUMN" not in SQL
    assert "DELETE FROM" not in SQL
    assert "UPDATE public.canonical_event_log" not in SQL
    assert "CASCADE" not in SQL


def test_receipts_bind_count_both_hash_domains_cutoff_and_owner_approval():
    observation = _block("observe_legacy_snapshot")
    migration = _block("perform_first_reconciliation")
    final_contract = _block("reconciled_contract")

    assert "canonical-writer-legacy-reconcile-v1:canonical14" in observation
    assert "canonical-writer-legacy-reconcile-v1:extended19" in observation
    assert "event_id::text || ':' || canonical14_row_sha256" in observation
    assert "event_id::text || ':' || extended19_row_sha256" in observation
    for column in (
        "source_row_count", "canonical14_sha256", "extended19_sha256",
        "occurred_at_cutoff", "inserted_at_cutoff",
        "approval_receipt_sha256", "server_identity_sha256",
        "postgres_version_num",
    ):
        assert column in migration
        assert column in final_contract


def test_reconciliation_is_atomic_idempotent_and_never_promotes_legacy_truth():
    prerequisites = _block("reconcile_prerequisites")
    migration = _block("perform_first_reconciliation")
    final_contract = _block("reconciled_contract")

    assert SQL.startswith("-- Canonical Writer legacy-event reconciliation v1.")
    assert "BEGIN;" in SQL
    assert SQL.rstrip().endswith("COMMIT;")
    assert "pg_advisory_xact_lock(4841739663211427921)" in SQL
    assert "LOCK TABLE public.canonical_event_log IN ACCESS EXCLUSIVE MODE" in SQL
    assert "public_column_count = 19 AND archive_column_count IS NULL" in prerequisites
    assert "public_column_count = 14 AND archive_column_count = 19" in prerequisites
    assert "RETURN;" in migration
    assert "INSERT INTO canonical_brain.writer_event_provenance" not in SQL
    assert "legacy events must not be auto-promoted into writer provenance" in final_contract
    assert "EXCEPT" in final_contract


def test_reconciled_public_relation_is_closed_to_runtime_table_access():
    final_contract = _block("reconciled_contract")

    assert "target_index_count <> 1" in final_contract
    assert "reconciled public exact primary index contract drifted" in final_contract
    assert "reconciled public table or column ACL contract drifted" in final_contract
    assert "canonical_brain_writer', 'canonical_brain_legacy_quarantine', 'USAGE'" in final_contract


def test_managed_admin_owner_authority_is_transaction_scoped_and_retired():
    temporary = _block("temporary_reconcile_owner_membership")
    final_contract = _block("final_reconcile_membership_contract")

    assert "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE" in temporary
    assert SQL.index("DO $temporary_reconcile_owner_membership$") < SQL.index(
        "DO $perform_first_reconciliation$"
    )
    assert SQL.index("GRANT CREATE ON SCHEMA public") < SQL.index(
        "OWNER TO canonical_brain_migration_owner"
    )
    assert "REVOKE CREATE ON SCHEMA public" in SQL
    assert "SET LOCAL ROLE canonical_brain_migration_owner;" in SQL
    assert "RESET ROLE;" in SQL
    assert "REVOKE canonical_brain_migration_owner FROM %I" in SQL
    assert "temporary reconciliation membership row survived" in final_contract
    assert "temporary reconciliation public CREATE survived" in final_contract
    assert "temporary reconciliation archive USAGE survived" in final_contract
    assert "temporary reconciliation legacy SELECT survived" in final_contract
    assert SQL.index("SET LOCAL ROLE canonical_brain_migration_owner;") < SQL.index(
        "RESET ROLE;"
    )
    assert SQL.index("DO $final_reconcile_membership_contract$") < SQL.rindex(
        "COMMIT;"
    )
