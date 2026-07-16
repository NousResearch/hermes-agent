"""One-purpose PostgreSQL boundary for Canonical Brain schema reconciliation.

This is not a general migration or SQL execution interface.  It accepts one
release-authored reconciliation plan, one exact target contract, and one
ephemeral Cloud SQL administrator session.  The only mutable SQL it can
execute is the exact ordered segments whose byte-for-byte concatenation and
digest equal the SQL sealed into that plan.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Protocol

from gateway.canonical_writer_db import (
    CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    ManagedCloudSQLAdminHBAReceipt,
    PostgresProtocolError,
    QueryResult,
    WriterDBConfig,
    _open_postgres_session,
    _require_command,
    _rollback_quietly,
    _validate_active_managed_hba_receipt,
)
from gateway.canonical_writer_schema_reconciliation import (
    CANONICAL_TRUTH_LOCK_SQL,
    CANONICAL_TRUTH_RELATIONS,
    DATABASE,
    CanonicalQuarantineAnchorReceipt,
    CanonicalRelationTruthReceipt,
    CanonicalTruthReceipt,
    SchemaContract,
    SchemaReconciliationError,
    SchemaReconciliationPlan,
    _ExactApiAssignedReconciliationMembershipProjection,
    _old_contract_value,
    _sha256_json,
    _target_policy,
    collect_schema_contract,
)


WRITER_LOGIN = "muncho_canary_writer_login"
_TEMPORARY_ADMIN = re.compile(r"^muncho_canary_admin_[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
POST_DELETE_TERMINAL_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-post-delete-terminal.v1"
)
_POST_DELETE_TRUTH_LIMITATION = (
    "writer_principal_has_no_direct_canonical_data_read_and_no_fixed_"
    "security_definer_full_truth_export"
)
_POST_DELETE_WRITER_PING_REQUEST_ID = (
    "schema-reconciliation-post-delete-terminal-v1"
)
_POST_DELETE_WRITER_PING_RESPONSE = {
    "ok": True,
    "result": {
        "service": "canonical_writer",
        "protocol": "v1",
        "database_identity": "canonical_brain_migration_owner",
        "request_id": _POST_DELETE_WRITER_PING_REQUEST_ID,
    },
}
_POST_DELETE_WRITER_PING_SQL = (
    "SELECT canonical_brain.writer_ping('{}'::jsonb, "
    "'{\"request_id\":\""
    + _POST_DELETE_WRITER_PING_REQUEST_ID
    + "\"}'::jsonb)::text AS writer_ping_response"
)
_MUTATION_BODY_START = (
    "CREATE OR REPLACE FUNCTION canonical_brain._deterministic_uuid(value text)"
)
_MUTATION_CLOSE_START = (
    "DO $reconcile_discord_routeback_helper_authority_close$"
)

_AUTHORITY_OPEN_RECEIPT_SQL = r"""
WITH temporary_login AS (
    SELECT oid, rolname, rolcanlogin, rolinherit, rolsuper, rolcreatedb,
           rolcreaterole, rolreplication, rolbypassrls, rolconnlimit,
           rolvaliduntil, rolconfig
      FROM pg_catalog.pg_roles
     WHERE rolname = SESSION_USER
), role_graph AS (
    SELECT membership.roleid, membership.member, membership.grantor,
           membership.admin_option, membership.inherit_option,
           membership.set_option, granted.rolname AS granted_name,
           member.rolname AS member_name, grantor.rolname AS grantor_name
      FROM pg_catalog.pg_auth_members AS membership
      JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
      JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
      JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = membership.grantor
     WHERE member.rolname = SESSION_USER
        OR granted.rolname = SESSION_USER
        OR grantor.rolname = SESSION_USER
        OR member.rolname = 'canonical_brain_migration_owner'
        OR granted.rolname = 'canonical_brain_migration_owner'
        OR grantor.rolname = 'canonical_brain_migration_owner'
)
SELECT CURRENT_USER = SESSION_USER
           AS current_user_is_session_user,
       SESSION_USER ~ '^muncho_canary_admin_[0-9a-f]{16}$'
           AS session_user_is_temporary_login,
       temporary_login.rolcanlogin AND temporary_login.rolinherit
           AND NOT temporary_login.rolsuper
           AND NOT temporary_login.rolcreatedb
           AND NOT temporary_login.rolcreaterole
           AND NOT temporary_login.rolreplication
           AND NOT temporary_login.rolbypassrls
           AND temporary_login.rolconnlimit = -1
           AND temporary_login.rolvaliduntil IS NULL
           AND temporary_login.rolconfig IS NULL
           AS temporary_login_attributes_exact,
       NOT EXISTS (SELECT 1 FROM pg_catalog.pg_event_trigger)
           AS event_trigger_inventory_empty,
       pg_catalog.has_language_privilege(SESSION_USER, 'plpgsql', 'USAGE')
           AS plpgsql_usage_present,
       (
           pg_catalog.current_setting('max_prepared_transactions')::integer = 0
           OR NOT EXISTS (
               SELECT 1 FROM pg_catalog.pg_prepared_xacts AS prepared
                WHERE prepared.database = pg_catalog.current_database()
                  AND prepared.owner = SESSION_USER
           )
       ) AS temporary_login_has_no_prepared_transaction,
       (
           SELECT pg_catalog.count(*) = 2
                  AND pg_catalog.count(DISTINCT roleid) = 2
                  AND COALESCE(pg_catalog.bool_and(
                      member_name = SESSION_USER
                      AND granted_name IN (
                          'canonical_brain_writer',
                          'canonical_brain_migration_owner'
                      )
                      AND grantor_name = 'cloudsqladmin'
                      AND admin_option IS FALSE
                      AND inherit_option IS TRUE
                      AND set_option IS FALSE
                  ), false)
             FROM role_graph
       ) AS api_role_graph_exact,
       pg_catalog.pg_has_role(
           SESSION_USER, 'canonical_brain_migration_owner', 'MEMBER'
       ) AS migration_owner_member,
       pg_catalog.pg_has_role(
           SESSION_USER, 'canonical_brain_migration_owner', 'USAGE'
       ) AS migration_owner_inherited,
       NOT pg_catalog.pg_has_role(
           SESSION_USER, 'canonical_brain_migration_owner', 'SET'
       ) AS migration_owner_not_settable,
       pg_catalog.pg_has_role(
           SESSION_USER, 'canonical_brain_writer', 'MEMBER'
       ) AND pg_catalog.pg_has_role(
           SESSION_USER, 'canonical_brain_writer', 'USAGE'
       ) AS writer_member_and_inherited,
       NOT pg_catalog.pg_has_role(
           SESSION_USER, 'canonical_brain_writer', 'SET'
       ) AS writer_not_settable,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_stat_activity AS activity
            WHERE activity.datname = pg_catalog.current_database()
              AND activity.backend_type = 'client backend'
              AND activity.pid <> pg_catalog.pg_backend_pid()
       ) AS no_foreign_database_client_sessions,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_shdepend AS dependency
            WHERE dependency.refclassid =
                  'pg_catalog.pg_authid'::pg_catalog.regclass
              AND dependency.refobjid = temporary_login.oid
       ) AS temporary_login_has_zero_shared_dependencies
  FROM temporary_login
""".strip()
_AUTHORITY_OPEN_RECEIPT_COLUMNS = (
    "current_user_is_session_user",
    "session_user_is_temporary_login",
    "temporary_login_attributes_exact",
    "event_trigger_inventory_empty",
    "plpgsql_usage_present",
    "temporary_login_has_no_prepared_transaction",
    "api_role_graph_exact",
    "migration_owner_member",
    "migration_owner_inherited",
    "migration_owner_not_settable",
    "writer_member_and_inherited",
    "writer_not_settable",
    "no_foreign_database_client_sessions",
    "temporary_login_has_zero_shared_dependencies",
)

_AUTHORITY_CLOSE_RECEIPT_SQL = r"""
WITH temporary_login AS (
    SELECT oid FROM pg_catalog.pg_roles WHERE rolname = SESSION_USER
), role_graph AS (
    SELECT membership.roleid, membership.member, membership.grantor,
           membership.admin_option, membership.inherit_option,
           membership.set_option, granted.rolname AS granted_name,
           member.rolname AS member_name, grantor.rolname AS grantor_name
      FROM pg_catalog.pg_auth_members AS membership
      JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
      JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
      JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = membership.grantor
     WHERE member.rolname = SESSION_USER
        OR granted.rolname = SESSION_USER
        OR grantor.rolname = SESSION_USER
        OR member.rolname = 'canonical_brain_migration_owner'
        OR granted.rolname = 'canonical_brain_migration_owner'
        OR grantor.rolname = 'canonical_brain_migration_owner'
)
SELECT CURRENT_USER = SESSION_USER
           AS current_user_is_session_user,
       SESSION_USER ~ '^muncho_canary_admin_[0-9a-f]{16}$'
           AS session_user_is_temporary_login,
       NOT EXISTS (SELECT 1 FROM pg_catalog.pg_event_trigger)
           AS event_trigger_inventory_empty,
       pg_catalog.has_language_privilege(SESSION_USER, 'plpgsql', 'USAGE')
           AS plpgsql_usage_present,
       (
           pg_catalog.current_setting('max_prepared_transactions')::integer = 0
           OR NOT EXISTS (
               SELECT 1 FROM pg_catalog.pg_prepared_xacts AS prepared
                WHERE prepared.database = pg_catalog.current_database()
                  AND prepared.owner = SESSION_USER
           )
       ) AS temporary_login_has_no_prepared_transaction,
       (
           SELECT pg_catalog.count(*) = 2
                  AND pg_catalog.count(DISTINCT roleid) = 2
                  AND COALESCE(pg_catalog.bool_and(
                      member_name = SESSION_USER
                      AND granted_name IN (
                          'canonical_brain_writer',
                          'canonical_brain_migration_owner'
                      )
                      AND grantor_name = 'cloudsqladmin'
                      AND admin_option IS FALSE
                      AND inherit_option IS TRUE
                      AND set_option IS FALSE
                  ), false)
             FROM role_graph
       ) AS api_role_graph_exact,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_stat_activity AS activity
            WHERE activity.datname = pg_catalog.current_database()
              AND activity.backend_type = 'client backend'
              AND activity.pid <> pg_catalog.pg_backend_pid()
       ) AS no_foreign_database_client_sessions,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_shdepend AS dependency
            WHERE dependency.refclassid =
                  'pg_catalog.pg_authid'::pg_catalog.regclass
              AND dependency.refobjid = temporary_login.oid
       ) AS temporary_login_has_zero_shared_dependencies
  FROM temporary_login
""".strip()
_AUTHORITY_CLOSE_RECEIPT_COLUMNS = (
    "current_user_is_session_user",
    "session_user_is_temporary_login",
    "event_trigger_inventory_empty",
    "plpgsql_usage_present",
    "temporary_login_has_no_prepared_transaction",
    "api_role_graph_exact",
    "no_foreign_database_client_sessions",
    "temporary_login_has_zero_shared_dependencies",
)

_CANONICAL_DATA_RELATIONS = CANONICAL_TRUTH_RELATIONS
_CANONICAL_DATA_LOCK_SQL = CANONICAL_TRUTH_LOCK_SQL
_CANONICAL_DATA_PRIMARY_KEYS = (
    ("public.canonical_event_log", "row_value.event_id", "row_value.event_id"),
    ("canonical_brain.writer_capability_consumptions", "row_value.consume_id", "row_value.consume_id"),
    ("canonical_brain.writer_capability_grants", 'row_value.approval_id COLLATE "C"', "row_value.approval_id"),
    (
        "canonical_brain.writer_capability_revocation_scopes",
        'row_value.scope_type COLLATE "C", row_value.session_key_sha256 COLLATE "C", row_value.capability_epoch_sha256 COLLATE "C", row_value.plan_id COLLATE "C"',
        "row_value.scope_type, row_value.session_key_sha256, row_value.capability_epoch_sha256, row_value.plan_id",
    ),
    ("canonical_brain.writer_capability_revocations", 'row_value.approval_id COLLATE "C"', "row_value.approval_id"),
    ("canonical_brain.writer_event_provenance", "row_value.event_id", "row_value.event_id"),
    ("canonical_brain.writer_public_routeback_targets", 'row_value.channel_id COLLATE "C"', "row_value.channel_id"),
    ("canonical_brain.writer_routeback_authorizations", 'row_value.authorization_id COLLATE "C"', "row_value.authorization_id"),
    ("canonical_brain.writer_routeback_lifecycle_terminals", 'row_value.lifecycle_id COLLATE "C"', "row_value.lifecycle_id"),
    ("canonical_brain.writer_routeback_terminals", 'row_value.authorization_id COLLATE "C"', "row_value.authorization_id"),
)
if tuple(item[0] for item in _CANONICAL_DATA_PRIMARY_KEYS) != _CANONICAL_DATA_RELATIONS:
    raise RuntimeError("canonical reconciliation relation manifest drifted")
_CANONICAL_DATA_ROW_UNION_SQL = " UNION ALL ".join(
    "SELECT " + str(ordinal) + "::integer AS ordinal, '" + relation
    + "'::text AS relation_name, ((ordered.row_ordinal - 1) / 4096)::bigint AS chunk_ordinal, "
    "ordered.row_ordinal, ordered.primary_key_json, ordered.row_json FROM (SELECT "
    "pg_catalog.row_number() OVER (ORDER BY " + order_sql
    + ") AS row_ordinal, pg_catalog.jsonb_build_array(" + primary_key_sql
    + ")::text AS primary_key_json, pg_catalog.to_jsonb(row_value)::text AS row_json FROM "
    + relation + " AS row_value) AS ordered"
    for ordinal, (relation, order_sql, primary_key_sql) in enumerate(_CANONICAL_DATA_PRIMARY_KEYS)
)
_CANONICAL_RELATION_VALUES_SQL = ", ".join(
    "(" + str(ordinal) + ", '" + relation + "')"
    for ordinal, relation in enumerate(_CANONICAL_DATA_RELATIONS)
)
_CANONICAL_TRUTH_SQL = (
    "WITH relation_names(ordinal, relation_name) AS (VALUES "
    + _CANONICAL_RELATION_VALUES_SQL
    + "), canonical_rows AS ("
    + _CANONICAL_DATA_ROW_UNION_SQL
    + "), row_receipts AS (SELECT ordinal, relation_name, chunk_ordinal, "
    "row_ordinal, primary_key_json, row_json, pg_catalog.encode("
    "pg_catalog.sha256(pg_catalog.convert_to(relation_name || E'\\n' || "
    "primary_key_json || E'\\n' || row_json, 'UTF8')), 'hex') AS row_sha FROM "
    "canonical_rows), chunk_receipts AS (SELECT ordinal, relation_name, "
    "chunk_ordinal, pg_catalog.count(*) AS chunk_row_count, "
    "pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to("
    "'canonical-writer-schema-reconcile-v2:chunk:' || relation_name || ':' || "
    "chunk_ordinal::text || E'\\n' || pg_catalog.string_agg(primary_key_json || "
    "':' || row_sha, E'\\n' ORDER BY row_ordinal), 'UTF8')), 'hex') AS "
    "chunk_sha256 FROM row_receipts GROUP BY ordinal, relation_name, "
    "chunk_ordinal), relation_receipts AS (SELECT relation_names.ordinal, "
    "relation_names.relation_name, COALESCE(pg_catalog.sum("
    "chunk_receipts.chunk_row_count), 0)::bigint AS row_count, "
    "pg_catalog.count(chunk_receipts.chunk_ordinal)::integer AS chunk_count, "
    "pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to("
    "'canonical-writer-schema-reconcile-v2:relation:' || "
    "relation_names.relation_name || E'\\n' || COALESCE(pg_catalog.string_agg("
    "chunk_receipts.chunk_ordinal::text || ':' || "
    "chunk_receipts.chunk_row_count::text || ':' || "
    "chunk_receipts.chunk_sha256, E'\\n' ORDER BY "
    "chunk_receipts.chunk_ordinal), ''), 'UTF8')), 'hex') AS "
    "chunk_manifest_sha256 FROM relation_names LEFT JOIN chunk_receipts ON "
    "chunk_receipts.ordinal = relation_names.ordinal AND "
    "chunk_receipts.relation_name = relation_names.relation_name GROUP BY "
    "relation_names.ordinal, relation_names.relation_name), "
    "event_row_receipts AS (SELECT event.event_id, pg_catalog.encode("
    "pg_catalog.sha256(pg_catalog.convert_to(pg_catalog.jsonb_build_object("
    "'event_id', pg_catalog.to_jsonb(event)->'event_id', 'schema_version', "
    "pg_catalog.to_jsonb(event)->'schema_version', 'event_type', "
    "pg_catalog.to_jsonb(event)->'event_type', 'occurred_at', "
    "pg_catalog.to_jsonb(event)->'occurred_at', 'case_id', "
    "pg_catalog.to_jsonb(event)->'case_id', 'source', "
    "pg_catalog.to_jsonb(event)->'source', 'actor', "
    "pg_catalog.to_jsonb(event)->'actor', 'subject', "
    "pg_catalog.to_jsonb(event)->'subject', 'evidence', "
    "pg_catalog.to_jsonb(event)->'evidence', 'decision', "
    "pg_catalog.to_jsonb(event)->'decision', 'status', "
    "pg_catalog.to_jsonb(event)->'status', 'next_action', "
    "pg_catalog.to_jsonb(event)->'next_action', 'safety', "
    "pg_catalog.to_jsonb(event)->'safety', 'payload', "
    "pg_catalog.to_jsonb(event)->'payload')::text, 'UTF8')), 'hex') AS row_sha FROM "
    "public.canonical_event_log AS event), event_receipt AS (SELECT "
    "pg_catalog.count(*)::text AS row_count, pg_catalog.encode("
    "pg_catalog.sha256(pg_catalog.convert_to("
    "'canonical-writer-legacy-reconcile-v1:canonical14' || E'\\n' || "
    "COALESCE(pg_catalog.string_agg(event_id::text || ':' || row_sha, E'\\n' "
    "ORDER BY event_id), ''), 'UTF8')), 'hex') AS canonical14_sha256 FROM "
    "event_row_receipts) SELECT event_receipt.row_count, "
    "event_receipt.canonical14_sha256, pg_catalog.jsonb_agg("
    "pg_catalog.jsonb_build_object('relation', relation_receipts.relation_name, "
    "'row_count', relation_receipts.row_count, 'chunk_count', "
    "relation_receipts.chunk_count, 'chunk_manifest_sha256', "
    "relation_receipts.chunk_manifest_sha256) ORDER BY "
    "relation_receipts.ordinal)::text "
    "AS relation_receipts FROM event_receipt CROSS JOIN relation_receipts "
    "GROUP BY event_receipt.row_count, event_receipt.canonical14_sha256"
)

_QUARANTINE_ANCHOR_SQL = r"""
WITH quarantine_namespace AS (
    SELECT namespace.oid, namespace.nspowner,
           pg_catalog.pg_get_userbyid(namespace.nspowner) AS owner_name
      FROM pg_catalog.pg_namespace AS namespace
     WHERE namespace.nspname = 'canonical_brain_legacy_quarantine'
), quarantine_data_relations AS (
    SELECT class.oid, class.relname, class.relowner, class.relkind,
           class.relpersistence, class.relispartition, class.reltablespace,
           class.relrowsecurity, class.relforcerowsecurity, class.reloptions,
           pg_catalog.pg_get_userbyid(class.relowner) AS owner_name
      FROM pg_catalog.pg_class AS class
      JOIN quarantine_namespace AS namespace
        ON namespace.oid = class.relnamespace
     WHERE class.relkind IN ('r', 'p', 'v', 'm', 'f', 'S', 'c')
), quarantine_relations AS (
    SELECT *
      FROM quarantine_data_relations AS relation
     WHERE relation.relkind = 'r'
       AND relation.relname IN (
           'canonical_event_log_legacy_v1', 'reconciliation_receipts'
       )
       AND relation.relpersistence = 'p'
       AND relation.relispartition IS FALSE
       AND relation.reltablespace = 0
       AND relation.relrowsecurity IS FALSE
       AND relation.relforcerowsecurity IS FALSE
       AND relation.reloptions IS NULL
       AND relation.owner_name = 'postgres'
), actual_schema_acl AS (
    SELECT acl.grantor, acl.grantee, acl.privilege_type, acl.is_grantable
      FROM quarantine_namespace AS namespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          (SELECT catalog_namespace.nspacl
             FROM pg_catalog.pg_namespace AS catalog_namespace
            WHERE catalog_namespace.oid = namespace.oid),
          pg_catalog.acldefault('n', namespace.nspowner)
       )) AS acl
), expected_schema_acl AS (
    SELECT namespace.nspowner AS grantor, namespace.nspowner AS grantee,
           privilege.name AS privilege_type, false AS is_grantable
      FROM quarantine_namespace AS namespace
      CROSS JOIN (VALUES ('CREATE'::text), ('USAGE'::text)) AS privilege(name)
), schema_acl_digest AS (
    SELECT pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               'canonical-writer-schema-reconcile-v3:quarantine-schema-acl'
               || E'\n' || COALESCE(pg_catalog.string_agg(
                   acl.grantor::text || ':' || acl.grantee::text || ':'
                   || acl.privilege_type || ':' || acl.is_grantable::text,
                   E'\n' ORDER BY acl.grantor, acl.grantee,
                   acl.privilege_type, acl.is_grantable
               ), ''), 'UTF8')), 'hex') AS acl_sha256
      FROM actual_schema_acl AS acl
), actual_relation_acl AS (
    SELECT relation.relname, acl.grantor, acl.grantee, acl.privilege_type,
           acl.is_grantable
      FROM quarantine_relations AS relation
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          (SELECT class.relacl FROM pg_catalog.pg_class AS class
            WHERE class.oid = relation.oid),
          pg_catalog.acldefault('r', relation.relowner)
      )) AS acl
), expected_relation_acl AS (
    SELECT relation.relname, relation.relowner AS grantor,
           relation.relowner AS grantee, privilege.name AS privilege_type,
           false AS is_grantable
      FROM quarantine_relations AS relation
      CROSS JOIN (VALUES
          ('SELECT'::text), ('INSERT'::text), ('UPDATE'::text),
          ('DELETE'::text), ('TRUNCATE'::text), ('REFERENCES'::text),
          ('TRIGGER'::text), ('MAINTAIN'::text)
      ) AS privilege(name)
), relation_acl_digests AS (
    SELECT relation.relname,
           pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               'canonical-writer-schema-reconcile-v3:quarantine-table-acl:'
               || relation.relname || E'\n' || COALESCE(pg_catalog.string_agg(
                   acl.grantor::text || ':' || acl.grantee::text || ':'
                   || acl.privilege_type || ':' || acl.is_grantable::text,
                   E'\n' ORDER BY acl.grantor, acl.grantee,
                   acl.privilege_type, acl.is_grantable
               ), ''), 'UTF8')), 'hex') AS acl_sha256
      FROM quarantine_relations AS relation
      LEFT JOIN actual_relation_acl AS acl
        ON acl.relname = relation.relname
     GROUP BY relation.relname
), anchor_receipts AS (
    SELECT 0::integer AS ordinal,
           'schema:canonical_brain_legacy_quarantine:postgres:owner-only'::text
               AS anchor,
           namespace.oid::bigint AS object_oid,
           namespace.owner_name AS owner,
           'n'::text AS kind, ''::text AS persistence,
           digest.acl_sha256
      FROM quarantine_namespace AS namespace
      CROSS JOIN schema_acl_digest AS digest
    UNION ALL
    SELECT CASE relation.relname
               WHEN 'canonical_event_log_legacy_v1' THEN 1 ELSE 2
           END AS ordinal,
           CASE relation.relname
               WHEN 'canonical_event_log_legacy_v1' THEN
                   'table:canonical_brain_legacy_quarantine.'
                   'canonical_event_log_legacy_v1:postgres:r:p:owner-only'
               ELSE
                   'table:canonical_brain_legacy_quarantine.'
                   'reconciliation_receipts:postgres:r:p:owner-only'
           END AS anchor,
           relation.oid::bigint AS object_oid,
           relation.owner_name AS owner,
           relation.relkind::text AS kind,
           relation.relpersistence::text AS persistence,
           digest.acl_sha256
      FROM quarantine_relations AS relation
      JOIN relation_acl_digests AS digest USING (relname)
)
SELECT (SELECT pg_catalog.count(*) = 1
          AND pg_catalog.bool_and(owner_name = 'postgres')
          FROM quarantine_namespace)
       AND NOT EXISTS (
           (SELECT * FROM actual_schema_acl
            EXCEPT SELECT * FROM expected_schema_acl)
           UNION ALL
           (SELECT * FROM expected_schema_acl
            EXCEPT SELECT * FROM actual_schema_acl)
       ) AS quarantine_schema_identity_acl_exact,
       (SELECT pg_catalog.count(*) = 2
          AND pg_catalog.bool_and(
              relname IN (
                  'canonical_event_log_legacy_v1',
                  'reconciliation_receipts'
              )
              AND relkind = 'r' AND relpersistence = 'p'
          )
          FROM quarantine_data_relations)
       AND (SELECT pg_catalog.count(*) = 2 FROM quarantine_relations)
       AND NOT EXISTS (
           (SELECT * FROM actual_relation_acl
            EXCEPT SELECT * FROM expected_relation_acl)
           UNION ALL
           (SELECT * FROM expected_relation_acl
            EXCEPT SELECT * FROM actual_relation_acl)
       )
       AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_attribute AS attribute
           CROSS JOIN LATERAL pg_catalog.aclexplode(attribute.attacl) AS acl
            WHERE attribute.attrelid IN (
                SELECT relation.oid FROM quarantine_relations AS relation
            )
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
       ) AS quarantine_relations_identity_acl_exact,
       NOT pg_catalog.has_schema_privilege(
           SESSION_USER, 'canonical_brain_legacy_quarantine', 'CREATE'
       )
       AND NOT pg_catalog.has_schema_privilege(
           SESSION_USER, 'canonical_brain_legacy_quarantine', 'USAGE'
       )
       AND NOT EXISTS (
           SELECT 1 FROM quarantine_relations AS relation
           CROSS JOIN (VALUES
               ('SELECT'::text), ('INSERT'::text), ('UPDATE'::text),
               ('DELETE'::text), ('TRUNCATE'::text), ('REFERENCES'::text),
               ('TRIGGER'::text), ('MAINTAIN'::text)
           ) AS privilege(name)
            WHERE pg_catalog.has_table_privilege(
                SESSION_USER, relation.oid, privilege.name
            )
       )
       AND NOT EXISTS (
           SELECT 1 FROM quarantine_relations AS relation
           CROSS JOIN (VALUES
               ('SELECT'::text), ('INSERT'::text),
               ('UPDATE'::text), ('REFERENCES'::text)
           ) AS privilege(name)
            WHERE pg_catalog.has_any_column_privilege(
                SESSION_USER, relation.oid, privilege.name
            )
       ) AS temporary_login_quarantine_unreachable
       , COALESCE((
           SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
               'anchor', anchor_receipts.anchor,
               'object_oid', anchor_receipts.object_oid,
               'owner', anchor_receipts.owner,
               'kind', anchor_receipts.kind,
               'persistence', anchor_receipts.persistence,
               'acl_sha256', anchor_receipts.acl_sha256
           ) ORDER BY anchor_receipts.ordinal)::text
             FROM anchor_receipts
       ), '[]') AS quarantine_anchor_receipts
""".strip()
_QUARANTINE_ANCHOR_COLUMNS = (
    "quarantine_schema_identity_acl_exact",
    "quarantine_relations_identity_acl_exact",
    "temporary_login_quarantine_unreachable",
    "quarantine_anchor_receipts",
)


class _Session(Protocol):
    username: str
    tls_peer_certificate_sha256: str

    def query(self, sql: str, *, maximum_rows: int) -> QueryResult: ...

    def close(self) -> None: ...


SessionFactory = Callable[[WriterDBConfig], _Session]


@dataclass(frozen=True)
class _SealedMutationSegments:
    authority_open: str
    body: str
    authority_close: str

    @property
    def mutation_sql(self) -> str:
        return (
            self.authority_open
            + "\n\n"
            + self.body
            + "\n\n"
            + self.authority_close
        )


def _split_sealed_mutation_sql(
    plan: SchemaReconciliationPlan,
) -> _SealedMutationSegments:
    """Split only the one reviewed plan without changing any SQL byte."""

    if not isinstance(plan, SchemaReconciliationPlan):
        raise SchemaReconciliationError(
            "schema_reconciliation_database_sql_not_sealed"
        )
    mutation = plan.mutation_sql
    if (
        not isinstance(mutation, str)
        or not mutation.endswith("\n")
        or not hmac.compare_digest(
            hashlib.sha256(mutation.encode("utf-8")).hexdigest(),
            str(plan.value.get("mutation_sql_sha256", "")),
        )
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_database_sql_not_sealed"
        )
    body_boundary = "\n\n" + _MUTATION_BODY_START
    close_boundary = "\n\n" + _MUTATION_CLOSE_START
    if mutation.count(body_boundary) != 2 or mutation.count(close_boundary) != 1:
        raise SchemaReconciliationError(
            "schema_reconciliation_database_sql_split_invalid"
        )
    authority_open, body_and_close = mutation.split(body_boundary, 1)
    body_and_close = _MUTATION_BODY_START + body_and_close
    body, authority_close = body_and_close.split(close_boundary, 1)
    authority_close = _MUTATION_CLOSE_START + authority_close
    segments = _SealedMutationSegments(
        authority_open=authority_open,
        body=body,
        authority_close=authority_close,
    )
    if (
        segments.mutation_sql != mutation
        or not authority_open.startswith(
            "SET LOCAL search_path = pg_catalog;\n\n"
            "DO $reconcile_discord_routeback_helper_authority_open$"
        )
        or not authority_open.endswith(
            "$reconcile_discord_routeback_helper_authority_open$;"
        )
        or body.count(_MUTATION_BODY_START) != 2
        or body.count(
            "CREATE OR REPLACE FUNCTION "
            "canonical_brain._deterministic_uuid(value text)"
        )
        != 2
        or body.count(
            "CREATE FUNCTION canonical_brain."
            "_discord_guild_routeback_target_valid("
        )
        != 1
        or not body.endswith(
            "$reconcile_discord_routeback_helper_terminal_validation$;"
        )
        or not authority_close.startswith(_MUTATION_CLOSE_START)
        or not authority_close.endswith(
            "$reconcile_discord_routeback_helper_authority_close$;\n"
        )
        or "SET ROLE" in mutation
        or "RESET ROLE" in mutation
        or "GRANT canonical_brain_migration_owner" in mutation
        or "REVOKE canonical_brain_migration_owner" in mutation
        or not hmac.compare_digest(
            hashlib.sha256(segments.mutation_sql.encode("utf-8")).hexdigest(),
            str(plan.value["mutation_sql_sha256"]),
        )
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_database_sql_split_invalid"
        )
    return segments


def _require_boolean_receipt(
    result: QueryResult,
    *,
    columns: tuple[str, ...],
    code: str,
) -> None:
    if (
        result.command_tag.upper() != "SELECT 1"
        or result.columns != columns
        or result.rows != (tuple("t" for _ in columns),)
    ):
        raise PostgresProtocolError(code)


def _require_void_lock(result: QueryResult, *, expected_column: str) -> None:
    if (
        result.command_tag.upper() != "SELECT 1"
        or result.columns != (expected_column,)
        or len(result.rows) != 1
        or len(result.rows[0]) != 1
    ):
        raise PostgresProtocolError("schema_reconciliation_advisory_lock_failed")


def _require_unlock(result: QueryResult) -> None:
    if (
        result.command_tag.upper() != "SELECT 1"
        or result.columns != ("pg_advisory_unlock",)
        or result.rows != (("t",),)
    ):
        raise PostgresProtocolError("schema_reconciliation_advisory_unlock_failed")


@dataclass(frozen=True)
class PostDeleteTerminalReceipt:
    release_revision: str
    plan_sha256: str
    database: str
    writer_login: str
    temporary_login: str
    temporary_login_sha256: str
    target_contract_sha256: str
    observed_contract_sha256: str
    writer_session_identity_exact: bool
    temporary_login_absent: bool
    temporary_login_inventory_empty: bool
    migration_owner_memberships_absent: bool
    writer_authority_exact: bool
    writer_ping_verified: bool
    writer_ping_request_id: str
    writer_ping_response_sha256: str
    fresh_writer_session_closed: bool
    tls_peer_certificate_sha256: str
    managed_hba_receipt_sha256: str
    pre_delete_canonical_truth_receipt_sha256: str
    canonical_truth_observed: bool
    canonical_truth_limitation: str
    observed_at_unix: int

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(value, str)
                for value in (
                    self.release_revision,
                    self.plan_sha256,
                    self.database,
                    self.writer_login,
                    self.temporary_login,
                    self.temporary_login_sha256,
                    self.target_contract_sha256,
                    self.observed_contract_sha256,
                    self.writer_ping_request_id,
                    self.writer_ping_response_sha256,
                    self.tls_peer_certificate_sha256,
                    self.managed_hba_receipt_sha256,
                    self.pre_delete_canonical_truth_receipt_sha256,
                    self.canonical_truth_limitation,
                )
            )
            or not re.fullmatch(r"[0-9a-f]{40}", self.release_revision)
            or _SHA256.fullmatch(self.plan_sha256) is None
            or self.database != DATABASE
            or self.writer_login != WRITER_LOGIN
            or _TEMPORARY_ADMIN.fullmatch(self.temporary_login) is None
            or self.temporary_login_sha256
            != hashlib.sha256(self.temporary_login.encode("utf-8")).hexdigest()
            or _SHA256.fullmatch(self.target_contract_sha256) is None
            or self.observed_contract_sha256 != self.target_contract_sha256
            or any(
                value is not True
                for value in (
                    self.writer_session_identity_exact,
                    self.temporary_login_absent,
                    self.temporary_login_inventory_empty,
                    self.migration_owner_memberships_absent,
                    self.writer_authority_exact,
                    self.writer_ping_verified,
                    self.fresh_writer_session_closed,
                )
            )
            or self.writer_ping_request_id
            != _POST_DELETE_WRITER_PING_REQUEST_ID
            or self.writer_ping_response_sha256
            != _sha256_json(_POST_DELETE_WRITER_PING_RESPONSE)
            or _SHA256.fullmatch(self.tls_peer_certificate_sha256) is None
            or _SHA256.fullmatch(self.managed_hba_receipt_sha256) is None
            or _SHA256.fullmatch(
                self.pre_delete_canonical_truth_receipt_sha256
            )
            is None
            or self.canonical_truth_observed is not False
            or self.canonical_truth_limitation != _POST_DELETE_TRUTH_LIMITATION
            or type(self.observed_at_unix) is not int
            or self.observed_at_unix < 0
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_post_delete_terminal_invalid"
            )

    @property
    def value(self) -> Mapping[str, Any]:
        unsigned = {
            "schema": POST_DELETE_TERMINAL_RECEIPT_SCHEMA,
            "ok": True,
            "release_revision": self.release_revision,
            "plan_sha256": self.plan_sha256,
            "database": self.database,
            "writer_login": self.writer_login,
            "temporary_login": self.temporary_login,
            "temporary_login_sha256": self.temporary_login_sha256,
            "target_contract_sha256": self.target_contract_sha256,
            "observed_contract_sha256": self.observed_contract_sha256,
            "writer_session_identity_exact": self.writer_session_identity_exact,
            "temporary_login_absent": self.temporary_login_absent,
            "temporary_login_inventory_empty": (
                self.temporary_login_inventory_empty
            ),
            "migration_owner_memberships_absent": (
                self.migration_owner_memberships_absent
            ),
            "writer_authority_exact": self.writer_authority_exact,
            "writer_ping_verified": self.writer_ping_verified,
            "writer_ping_request_id": self.writer_ping_request_id,
            "writer_ping_response_sha256": self.writer_ping_response_sha256,
            "fresh_writer_session_closed": self.fresh_writer_session_closed,
            "tls_peer_certificate_sha256": self.tls_peer_certificate_sha256,
            "managed_hba_receipt_sha256": self.managed_hba_receipt_sha256,
            "pre_delete_canonical_truth_receipt_sha256": (
                self.pre_delete_canonical_truth_receipt_sha256
            ),
            "canonical_truth_observed": self.canonical_truth_observed,
            "canonical_truth_limitation": self.canonical_truth_limitation,
            "observed_at_unix": self.observed_at_unix,
        }
        return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


_POST_DELETE_TERMINAL_FIELDS = frozenset(
    {
        "schema",
        "ok",
        "release_revision",
        "plan_sha256",
        "database",
        "writer_login",
        "temporary_login",
        "temporary_login_sha256",
        "target_contract_sha256",
        "observed_contract_sha256",
        "writer_session_identity_exact",
        "temporary_login_absent",
        "temporary_login_inventory_empty",
        "migration_owner_memberships_absent",
        "writer_authority_exact",
        "writer_ping_verified",
        "writer_ping_request_id",
        "writer_ping_response_sha256",
        "fresh_writer_session_closed",
        "tls_peer_certificate_sha256",
        "managed_hba_receipt_sha256",
        "pre_delete_canonical_truth_receipt_sha256",
        "canonical_truth_observed",
        "canonical_truth_limitation",
        "observed_at_unix",
        "receipt_sha256",
    }
)


def parse_post_delete_terminal_receipt(
    value: Mapping[str, Any],
) -> PostDeleteTerminalReceipt:
    """Purely validate the exact fields, booleans, limitation, and digest."""
    if (
        not isinstance(value, Mapping)
        or set(value) != _POST_DELETE_TERMINAL_FIELDS
        or value.get("schema") != POST_DELETE_TERMINAL_RECEIPT_SCHEMA
        or value.get("ok") is not True
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_invalid"
        )
    receipt = PostDeleteTerminalReceipt(
        release_revision=value.get("release_revision"),
        plan_sha256=value.get("plan_sha256"),
        database=value.get("database"),
        writer_login=value.get("writer_login"),
        temporary_login=value.get("temporary_login"),
        temporary_login_sha256=value.get("temporary_login_sha256"),
        target_contract_sha256=value.get("target_contract_sha256"),
        observed_contract_sha256=value.get("observed_contract_sha256"),
        writer_session_identity_exact=value.get("writer_session_identity_exact"),
        temporary_login_absent=value.get("temporary_login_absent"),
        temporary_login_inventory_empty=value.get(
            "temporary_login_inventory_empty"
        ),
        migration_owner_memberships_absent=value.get(
            "migration_owner_memberships_absent"
        ),
        writer_authority_exact=value.get("writer_authority_exact"),
        writer_ping_verified=value.get("writer_ping_verified"),
        writer_ping_request_id=value.get("writer_ping_request_id"),
        writer_ping_response_sha256=value.get("writer_ping_response_sha256"),
        fresh_writer_session_closed=value.get("fresh_writer_session_closed"),
        tls_peer_certificate_sha256=value.get(
            "tls_peer_certificate_sha256"
        ),
        managed_hba_receipt_sha256=value.get("managed_hba_receipt_sha256"),
        pre_delete_canonical_truth_receipt_sha256=value.get(
            "pre_delete_canonical_truth_receipt_sha256"
        ),
        canonical_truth_observed=value.get("canonical_truth_observed"),
        canonical_truth_limitation=value.get("canonical_truth_limitation"),
        observed_at_unix=value.get("observed_at_unix"),
    )
    if dict(value) != receipt.value:
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_invalid"
        )
    return receipt


def validate_post_delete_terminal_receipt(
    value: Mapping[str, Any],
    *,
    plan: SchemaReconciliationPlan,
    target: SchemaContract,
    temporary_login: str,
    managed_hba_receipt: ManagedCloudSQLAdminHBAReceipt,
    pre_delete_canonical_truth: CanonicalTruthReceipt,
) -> PostDeleteTerminalReceipt:
    """Purely bind a structurally valid receipt to live domain evidence."""

    if (
        not isinstance(plan, SchemaReconciliationPlan)
        or not isinstance(target, SchemaContract)
        or target.sha256 != plan.value.get("target_contract_sha256")
        or not isinstance(managed_hba_receipt, ManagedCloudSQLAdminHBAReceipt)
        or not isinstance(pre_delete_canonical_truth, CanonicalTruthReceipt)
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_invalid"
        )
    receipt = parse_post_delete_terminal_receipt(value)
    if (
        receipt.release_revision != plan.revision
        or receipt.plan_sha256 != plan.sha256
        or receipt.temporary_login != temporary_login
        or receipt.target_contract_sha256 != target.sha256
        or receipt.managed_hba_receipt_sha256 != managed_hba_receipt.sha256
        or receipt.pre_delete_canonical_truth_receipt_sha256
        != pre_delete_canonical_truth.sha256
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_invalid"
        )
    return receipt


def _post_delete_authority_absence_sql(temporary_login: str) -> str:
    if _TEMPORARY_ADMIN.fullmatch(temporary_login) is None:
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_invalid"
        )
    escaped = temporary_login.replace("'", "''")
    return (
        "SELECT CURRENT_USER = 'muncho_canary_writer_login' AND SESSION_USER "
        "= 'muncho_canary_writer_login' AS writer_session_identity_exact, "
        "NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '"
        + escaped
        + "') AS temporary_login_absent, NOT EXISTS (SELECT 1 FROM "
        "pg_catalog.pg_roles WHERE rolname ~ "
        "'^muncho_canary_admin_[0-9a-f]{16}$') AS "
        "temporary_login_inventory_empty, NOT EXISTS (SELECT 1 FROM "
        "pg_catalog.pg_auth_members AS membership JOIN pg_catalog.pg_roles AS "
        "owner ON owner.oid = membership.roleid OR owner.oid = "
        "membership.member OR owner.oid = membership.grantor WHERE "
        "owner.rolname = 'canonical_brain_migration_owner') AS "
        "migration_owner_memberships_absent, NOT EXISTS (SELECT 1 FROM "
        "pg_catalog.pg_stat_activity AS activity WHERE activity.datname = "
        "pg_catalog.current_database() AND activity.backend_type = 'client "
        "backend' AND activity.pid <> pg_catalog.pg_backend_pid()) AS "
        "no_foreign_database_client_sessions"
    )


_POST_DELETE_AUTHORITY_COLUMNS = (
    "writer_session_identity_exact",
    "temporary_login_absent",
    "temporary_login_inventory_empty",
    "migration_owner_memberships_absent",
    "no_foreign_database_client_sessions",
)


def _require_exact_writer_ping(result: QueryResult) -> str:
    if (
        result.command_tag.upper() != "SELECT 1"
        or result.columns != ("writer_ping_response",)
        or len(result.rows) != 1
        or len(result.rows[0]) != 1
        or not isinstance(result.rows[0][0], str)
    ):
        raise PostgresProtocolError(
            "schema_reconciliation_post_delete_writer_ping_invalid"
        )
    try:
        response = json.loads(result.rows[0][0])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PostgresProtocolError(
            "schema_reconciliation_post_delete_writer_ping_invalid"
        ) from exc
    if response != _POST_DELETE_WRITER_PING_RESPONSE:
        raise PostgresProtocolError(
            "schema_reconciliation_post_delete_writer_ping_invalid"
        )
    return _sha256_json(response)


def collect_post_delete_terminal_receipt(
    *,
    plan: SchemaReconciliationPlan,
    target: SchemaContract,
    temporary_login: str,
    writer_config: WriterDBConfig,
    managed_hba_receipt: ManagedCloudSQLAdminHBAReceipt,
    pre_delete_canonical_truth: CanonicalTruthReceipt,
    observed_at_unix: int | None = None,
    _session_factory: SessionFactory | None = None,
) -> PostDeleteTerminalReceipt:
    """Use one fresh writer session to prove Cloud-login deletion terminally.

    The writer deliberately cannot re-read all canonical data relations.  The
    receipt therefore binds the already-observed privileged truth receipt and
    explicitly records that the fresh writer observation proves authority
    absence plus the exact target contract, not a second full-data digest.
    """

    observed_at = int(time.time()) if observed_at_unix is None else observed_at_unix
    if (
        not isinstance(plan, SchemaReconciliationPlan)
        or not isinstance(target, SchemaContract)
        or not target.is_target
        or target.sha256 != plan.value.get("target_contract_sha256")
        or _TEMPORARY_ADMIN.fullmatch(temporary_login) is None
        or not isinstance(writer_config, WriterDBConfig)
        or writer_config.user != WRITER_LOGIN
        or writer_config.database != DATABASE
        or not isinstance(pre_delete_canonical_truth, CanonicalTruthReceipt)
        or type(observed_at) is not int
        or observed_at < 0
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_invalid"
        )
    try:
        _validate_active_managed_hba_receipt(
            managed_hba_receipt,
            managed_hba_receipt,
            config=writer_config,
            now_unix=observed_at,
            require_expected_fresh=False,
        )
    except BaseException as exc:
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_hba_invalid"
        ) from exc
    session_factory = _session_factory or _open_postgres_session
    session: _Session | None = None
    transaction_open = False
    observed_contract: SchemaContract | None = None
    writer_ping_response_sha256 = ""
    tls_peer = ""
    try:
        session = session_factory(writer_config)
        tls_peer = getattr(session, "tls_peer_certificate_sha256", "")
        if (
            getattr(session, "username", None) != WRITER_LOGIN
            or not isinstance(tls_peer, str)
            or not hmac.compare_digest(
                tls_peer,
                managed_hba_receipt.server_certificate_sha256,
            )
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_post_delete_terminal_session_invalid"
            )
        _require_command(
            session,
            "BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY",
            "BEGIN",
        )
        transaction_open = True
        for statement in (
            "SET LOCAL TimeZone = 'UTC'",
            "SET LOCAL DateStyle = 'ISO, YMD'",
            "SET LOCAL search_path = pg_catalog",
            "SET LOCAL lock_timeout = '15s'",
            "SET LOCAL statement_timeout = '2min'",
        ):
            _require_command(session, statement, "SET")
        lock = session.query(
            "SELECT pg_catalog.pg_advisory_xact_lock_shared("
            + str(CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY)
            + ")",
            maximum_rows=1,
        )
        _require_void_lock(lock, expected_column="pg_advisory_xact_lock_shared")
        authority = session.query(
            _post_delete_authority_absence_sql(temporary_login),
            maximum_rows=1,
        )
        _require_boolean_receipt(
            authority,
            columns=_POST_DELETE_AUTHORITY_COLUMNS,
            code="schema_reconciliation_post_delete_authority_present",
        )
        observed_contract = collect_schema_contract(
            session,
            config=writer_config,
            policy=_target_policy(target.attestation),
            managed_hba_receipt=managed_hba_receipt,
            subject_user=WRITER_LOGIN,
            allow_missing_helper=False,
        )
        if observed_contract.sha256 != target.sha256:
            raise SchemaReconciliationError(
                "schema_reconciliation_post_delete_contract_invalid"
            )
        writer_ping_response_sha256 = _require_exact_writer_ping(
            session.query(_POST_DELETE_WRITER_PING_SQL, maximum_rows=1)
        )
        _require_command(session, "COMMIT", "COMMIT")
        transaction_open = False
    except BaseException:
        if session is not None and transaction_open:
            _rollback_quietly(session)
            transaction_open = False
        raise
    finally:
        if session is not None:
            session.close()
    if observed_contract is None:
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_invalid"
        )
    receipt = PostDeleteTerminalReceipt(
        release_revision=plan.revision,
        plan_sha256=plan.sha256,
        database=DATABASE,
        writer_login=WRITER_LOGIN,
        temporary_login=temporary_login,
        temporary_login_sha256=hashlib.sha256(
            temporary_login.encode("utf-8")
        ).hexdigest(),
        target_contract_sha256=target.sha256,
        observed_contract_sha256=observed_contract.sha256,
        writer_session_identity_exact=True,
        temporary_login_absent=True,
        temporary_login_inventory_empty=True,
        migration_owner_memberships_absent=True,
        writer_authority_exact=True,
        writer_ping_verified=True,
        writer_ping_request_id=_POST_DELETE_WRITER_PING_REQUEST_ID,
        writer_ping_response_sha256=writer_ping_response_sha256,
        fresh_writer_session_closed=True,
        tls_peer_certificate_sha256=tls_peer,
        managed_hba_receipt_sha256=managed_hba_receipt.sha256,
        pre_delete_canonical_truth_receipt_sha256=(
            pre_delete_canonical_truth.sha256
        ),
        canonical_truth_observed=False,
        canonical_truth_limitation=_POST_DELETE_TRUTH_LIMITATION,
        observed_at_unix=observed_at,
    )
    return validate_post_delete_terminal_receipt(
        receipt.value,
        plan=plan,
        target=target,
        temporary_login=temporary_login,
        managed_hba_receipt=managed_hba_receipt,
        pre_delete_canonical_truth=pre_delete_canonical_truth,
    )


class _PostgresSchemaReconciliationTransaction:
    def __init__(
        self,
        *,
        session: _Session,
        plan: SchemaReconciliationPlan,
        target: SchemaContract,
        writer_config: WriterDBConfig,
        managed_hba_receipt: ManagedCloudSQLAdminHBAReceipt,
        mutation_segments: _SealedMutationSegments,
        owner_membership_projection: (
            _ExactApiAssignedReconciliationMembershipProjection
        ),
    ) -> None:
        self._session = session
        self._plan = plan
        self._target = target
        self._writer_config = writer_config
        self._managed_hba_receipt = managed_hba_receipt
        self._mutation_segments = mutation_segments
        self._owner_membership_projection = owner_membership_projection
        self._policy = _target_policy(target.attestation)
        self._active = True
        self._truth_locked = False
        self._mutation_attempted = False
        self._mutation_body_complete = False
        self._authority_closed = False
        self._protocol_usable = True

    @property
    def protocol_usable(self) -> bool:
        return self._protocol_usable

    @property
    def truth_locked(self) -> bool:
        return self._truth_locked

    @property
    def mutation_body_complete(self) -> bool:
        return self._mutation_body_complete

    @property
    def authority_closed(self) -> bool:
        return self._authority_closed

    def _require_active(self) -> None:
        if not self._active:
            raise SchemaReconciliationError(
                "schema_reconciliation_database_scope_inactive"
            )

    def _require_truth_lock(self) -> None:
        self._require_active()
        if not self._truth_locked:
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_not_locked"
            )

    def _query(self, sql: str, *, maximum_rows: int) -> QueryResult:
        try:
            return self._session.query(sql, maximum_rows=maximum_rows)
        except BaseException:
            self._protocol_usable = False
            raise

    def lock_canonical_truth(self) -> None:
        self._require_active()
        if self._truth_locked:
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_lock_repeated"
            )
        result = self._query(
            _CANONICAL_DATA_LOCK_SQL,
            maximum_rows=0,
        )
        if result.command_tag.upper() != "LOCK TABLE" or result.columns or result.rows:
            raise PostgresProtocolError(
                "schema_reconciliation_canonical_truth_lock_failed"
            )
        lock = self._query(
            "SELECT pg_catalog.pg_advisory_xact_lock("
            + str(CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY)
            + ")",
            maximum_rows=1,
        )
        _require_void_lock(lock, expected_column="pg_advisory_xact_lock")
        self._truth_locked = True

    def observe_contract(self) -> SchemaContract:
        self._require_truth_lock()
        try:
            return collect_schema_contract(
                self._session,
                config=self._writer_config,
                policy=self._policy,
                managed_hba_receipt=self._managed_hba_receipt,
                subject_user=WRITER_LOGIN,
                allow_missing_helper=True,
                owner_membership_projection=self._owner_membership_projection,
            )
        except BaseException:
            # Closing the connection remains a safe rollback even when the
            # wire client has already consumed a complete result.
            self._protocol_usable = False
            raise

    def observe_canonical_truth(self) -> CanonicalTruthReceipt:
        self._require_truth_lock()
        result = self._query(_CANONICAL_TRUTH_SQL, maximum_rows=1)
        if (
            result.command_tag.upper() != "SELECT 1"
            or result.columns
            != ("row_count", "canonical14_sha256", "relation_receipts")
            or len(result.rows) != 1
            or len(result.rows[0]) != 3
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )
        count_text, digest, relations_json = result.rows[0]
        if (
            not isinstance(count_text, str)
            or not count_text.isdigit()
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or not isinstance(relations_json, str)
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )
        try:
            relations_value = json.loads(relations_json)
            if not isinstance(relations_value, list):
                raise ValueError
            relations = tuple(
                CanonicalRelationTruthReceipt.from_mapping(item)
                for item in relations_value
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            ) from exc
        quarantine = self._query(_QUARANTINE_ANCHOR_SQL, maximum_rows=1)
        if (
            quarantine.command_tag.upper() != "SELECT 1"
            or quarantine.columns != _QUARANTINE_ANCHOR_COLUMNS
            or len(quarantine.rows) != 1
            or len(quarantine.rows[0]) != len(_QUARANTINE_ANCHOR_COLUMNS)
            or quarantine.rows[0][:3] != ("t", "t", "t")
            or not isinstance(quarantine.rows[0][3], str)
        ):
            raise PostgresProtocolError(
                "schema_reconciliation_quarantine_anchor_invalid"
            )
        try:
            quarantine_value = json.loads(quarantine.rows[0][3])
            if not isinstance(quarantine_value, list):
                raise ValueError
            quarantine_receipts = tuple(
                CanonicalQuarantineAnchorReceipt.from_mapping(item)
                for item in quarantine_value
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SchemaReconciliationError(
                "schema_reconciliation_quarantine_anchor_invalid"
            ) from exc
        return CanonicalTruthReceipt(
            row_count=int(count_text),
            canonical14_sha256=digest,
            relation_receipts=relations,
            quarantine_anchor_receipts=quarantine_receipts,
        )

    def execute_sql(self, sql: str) -> None:
        self._require_truth_lock()
        if self._mutation_attempted:
            raise SchemaReconciliationError(
                "schema_reconciliation_database_apply_repeated"
            )
        if (
            not isinstance(sql, str)
            or sql != self._plan.mutation_sql
            or hashlib.sha256(sql.encode("utf-8")).hexdigest()
            != self._plan.value["mutation_sql_sha256"]
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_database_sql_not_sealed"
            )
        self._mutation_attempted = True
        result = self._query(self._mutation_segments.body, maximum_rows=0)
        if result.command_tag.upper() != "DO" or result.columns or result.rows:
            raise PostgresProtocolError(
                "schema_reconciliation_database_apply_unconfirmed"
            )
        self._mutation_body_complete = True

    def close_authority(self) -> None:
        self._require_active()
        if not self._truth_locked or self._authority_closed:
            raise SchemaReconciliationError(
                "schema_reconciliation_database_authority_close_invalid"
            )
        result = self._query(
            self._mutation_segments.authority_close,
            maximum_rows=0,
        )
        if result.command_tag.upper() != "DO" or result.columns or result.rows:
            raise PostgresProtocolError(
                "schema_reconciliation_database_authority_close_unconfirmed"
            )
        receipt = self._query(_AUTHORITY_CLOSE_RECEIPT_SQL, maximum_rows=1)
        _require_boolean_receipt(
            receipt,
            columns=_AUTHORITY_CLOSE_RECEIPT_COLUMNS,
            code="schema_reconciliation_database_authority_survived",
        )
        self._authority_closed = True

    def invalidate(self) -> None:
        self._active = False


class PostgresSchemaReconciliationDatabase:
    """Exact release boundary used by the reconciliation executor.

    Mutation-required execution runs all three sealed SQL segments.  The
    temporary login arrives with two exact non-settable API memberships; the
    sealed open and close segments only prove that unchanged authority and the
    restored trampoline identity.  An exact-target/replay transaction commits
    no helper DDL and no role mutation.
    """

    def __init__(
        self,
        *,
        plan: SchemaReconciliationPlan,
        target: SchemaContract,
        admin_config: WriterDBConfig,
        writer_config: WriterDBConfig,
        managed_hba_receipt: ManagedCloudSQLAdminHBAReceipt,
        _session_factory: SessionFactory | None = None,
    ) -> None:
        if (
            not isinstance(plan, SchemaReconciliationPlan)
            or not isinstance(target, SchemaContract)
            or not target.is_target
            or target.sha256 != plan.value["target_contract_sha256"]
            or _sha256_json(_old_contract_value(target))
            != plan.value["expected_old_contract_sha256"]
            or plan.value["canonical_truth_lock"]
            != _CANONICAL_DATA_LOCK_SQL
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_database_plan_binding_invalid"
            )
        coordinates = (
            admin_config.host,
            admin_config.tls_server_name,
            admin_config.port,
            admin_config.database,
            admin_config.ca_file,
        )
        writer_coordinates = (
            writer_config.host,
            writer_config.tls_server_name,
            writer_config.port,
            writer_config.database,
            writer_config.ca_file,
        )
        if (
            coordinates != writer_coordinates
            or admin_config.database != DATABASE
            or writer_config.database != DATABASE
            or writer_config.user != WRITER_LOGIN
            or _TEMPORARY_ADMIN.fullmatch(admin_config.user) is None
            or not isinstance(
                managed_hba_receipt,
                ManagedCloudSQLAdminHBAReceipt,
            )
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_database_authority_invalid"
            )
        try:
            _validate_active_managed_hba_receipt(
                managed_hba_receipt,
                managed_hba_receipt,
                config=writer_config,
                now_unix=int(time.time()),
                require_expected_fresh=False,
            )
        except BaseException as exc:
            raise SchemaReconciliationError(
                "schema_reconciliation_database_hba_receipt_invalid"
            ) from exc
        self._plan = plan
        self._target = target
        self._admin_config = admin_config
        self._writer_config = writer_config
        self._managed_hba_receipt = managed_hba_receipt
        self._owner_membership_projection = (
            _ExactApiAssignedReconciliationMembershipProjection(
                owner_role="canonical_brain_migration_owner",
                writer_role="canonical_brain_writer",
                session_user=admin_config.user,
            )
        )
        self._session_factory = _session_factory or _open_postgres_session
        self._mutation_segments = _split_sealed_mutation_sql(plan)
        self._scope_lock = threading.Lock()

    @contextlib.contextmanager
    def transaction(
        self,
        *,
        advisory_lock_key: int,
    ) -> Iterator[_PostgresSchemaReconciliationTransaction]:
        if (
            type(advisory_lock_key) is not int
            or advisory_lock_key != CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
            or advisory_lock_key != self._plan.value["advisory_lock_key"]
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_database_lock_key_invalid"
            )
        if not self._scope_lock.acquire(blocking=False):
            raise SchemaReconciliationError(
                "schema_reconciliation_database_scope_busy"
            )
        session: _Session | None = None
        scope: _PostgresSchemaReconciliationTransaction | None = None
        transaction_open = False
        session_lock_acquired = False
        committed = False
        try:
            session = self._session_factory(self._admin_config)
            if (
                getattr(session, "username", None) != self._admin_config.user
                or not isinstance(
                    getattr(session, "tls_peer_certificate_sha256", None),
                    str,
                )
                or not hmac.compare_digest(
                    session.tls_peer_certificate_sha256,
                    self._managed_hba_receipt.server_certificate_sha256,
                )
            ):
                raise SchemaReconciliationError(
                    "schema_reconciliation_database_session_identity_invalid"
                )
            lock = session.query(
                "SELECT pg_catalog.pg_advisory_lock("
                + str(advisory_lock_key)
                + ")",
                maximum_rows=1,
            )
            _require_void_lock(lock, expected_column="pg_advisory_lock")
            session_lock_acquired = True
            _require_command(
                session,
                "BEGIN ISOLATION LEVEL SERIALIZABLE",
                "BEGIN",
            )
            transaction_open = True
            for statement in (
                "SET LOCAL TimeZone = 'UTC'",
                "SET LOCAL DateStyle = 'ISO, YMD'",
                "SET LOCAL search_path = pg_catalog",
                "SET LOCAL lock_timeout = '15s'",
                "SET LOCAL statement_timeout = '2min'",
            ):
                _require_command(session, statement, "SET")
            authority_open = session.query(
                self._mutation_segments.authority_open,
                maximum_rows=0,
            )
            if (
                authority_open.command_tag.upper() != "DO"
                or authority_open.columns
                or authority_open.rows
            ):
                raise PostgresProtocolError(
                    "schema_reconciliation_database_authority_open_unconfirmed"
                )
            authority_receipt = session.query(
                _AUTHORITY_OPEN_RECEIPT_SQL,
                maximum_rows=1,
            )
            _require_boolean_receipt(
                authority_receipt,
                columns=_AUTHORITY_OPEN_RECEIPT_COLUMNS,
                code="schema_reconciliation_database_authority_open_invalid",
            )
            scope = _PostgresSchemaReconciliationTransaction(
                session=session,
                plan=self._plan,
                target=self._target,
                writer_config=self._writer_config,
                managed_hba_receipt=self._managed_hba_receipt,
                mutation_segments=self._mutation_segments,
                owner_membership_projection=self._owner_membership_projection,
            )
            yield scope
            if not scope.truth_locked:
                raise SchemaReconciliationError(
                    "schema_reconciliation_canonical_truth_not_locked"
                )
            # The caller performs its post-apply contract and canonical-truth
            # observations before leaving the context.  The exact close
            # segment then proves the API-assigned role graph is unchanged and
            # the temporary trampoline has been restored before COMMIT.  Cloud
            # API deletion of the temporary login is an outer terminal step.
            scope.close_authority()
            if not scope.authority_closed:
                raise SchemaReconciliationError(
                    "schema_reconciliation_database_authority_survived"
                )
            scope.invalidate()
            _require_command(session, "COMMIT", "COMMIT")
            transaction_open = False
            committed = True
            unlock = session.query(
                "SELECT pg_catalog.pg_advisory_unlock("
                + str(advisory_lock_key)
                + ")",
                maximum_rows=1,
            )
            _require_unlock(unlock)
            session_lock_acquired = False
        except BaseException:
            if scope is not None:
                scope.invalidate()
            if (
                session is not None
                and transaction_open
                and (scope is None or scope.protocol_usable)
            ):
                _rollback_quietly(session)
                transaction_open = False
            if session is not None and session_lock_acquired and not transaction_open:
                try:
                    unlock = session.query(
                        "SELECT pg_catalog.pg_advisory_unlock("
                        + str(advisory_lock_key)
                        + ")",
                        maximum_rows=1,
                    )
                    _require_unlock(unlock)
                    session_lock_acquired = False
                except BaseException:
                    pass
            raise
        finally:
            if scope is not None:
                scope.invalidate()
            if session is not None:
                session.close()
            self._scope_lock.release()
        if not committed or session_lock_acquired:
            raise SchemaReconciliationError(
                "schema_reconciliation_database_transaction_incomplete"
            )


__all__ = [
    "POST_DELETE_TERMINAL_RECEIPT_SCHEMA",
    "PostDeleteTerminalReceipt",
    "PostgresSchemaReconciliationDatabase",
    "WRITER_LOGIN",
    "collect_post_delete_terminal_receipt",
    "parse_post_delete_terminal_receipt",
    "validate_post_delete_terminal_receipt",
]
