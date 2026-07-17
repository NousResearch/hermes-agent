"""One-purpose PostgreSQL boundary for Canonical Brain schema reconciliation.

This is not a general migration or SQL execution interface.  It accepts one
release-authored reconciliation plan, one exact target contract, and one
ephemeral Cloud SQL executor session.  The executor owns nothing and can call
only the two fixed zero-argument owner-owned control routines.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import inspect
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
    DATABASE,
    CanonicalTruthReceipt,
    SchemaContract,
    SchemaReconciliationError,
    SchemaReconciliationPlan,
    _old_contract_value,
    _sha256_json,
    _target_policy,
    collect_schema_contract,
)
from gateway.canonical_writer_schema_reconciliation_control import (
    APPLY_CALL_SQL,
    APPLY_DEFINITION_SHA256,
    APPLY_PROSRC_SHA256,
    AUTHORIZED_INTENT_GUC,
    OBSERVER_CALL_SQL,
    OBSERVER_DEFINITION_SHA256,
    OBSERVER_PROSRC_SHA256,
    PLAN_GUC,
    TRUTH_RECEIPT_GUC,
    ControlApplyReceipt,
    ControlObservation,
    parse_control_apply_receipt,
    parse_control_observation,
    set_local_hash_sql,
)


WRITER_LOGIN = "muncho_canary_writer_login"
_TEMPORARY_EXECUTOR = re.compile(
    r"^muncho_canary_reconciler_[0-9a-f]{16}$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
POST_DELETE_TERMINAL_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-post-delete-terminal.v2"
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
_ROLE_GRAPH_STABILIZATION_ATTEMPTS = 12
_ROLE_GRAPH_STABILIZATION_INTERVAL_SECONDS = 5.0

# P0 fixed inert-executor authority contract.
_AUTHORITY_OPEN_RECEIPT_SQL = r"""
WITH RECURSIVE temporary_login AS (
    SELECT * FROM pg_catalog.pg_roles WHERE rolname = SESSION_USER
), executor AS (
    SELECT * FROM pg_catalog.pg_roles
     WHERE rolname = 'canonical_brain_schema_reconciler'
), owner_role AS (
    SELECT * FROM pg_catalog.pg_roles
     WHERE rolname = 'canonical_brain_migration_owner'
), writer_role AS (
    SELECT * FROM pg_catalog.pg_roles
     WHERE rolname = 'canonical_brain_writer'
), role_graph AS (
    SELECT membership.*, granted.rolname AS granted_name,
           member.rolname AS member_name, grantor.rolname AS grantor_name
      FROM pg_catalog.pg_auth_members AS membership
      JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
      JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
      JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = membership.grantor
     WHERE member.rolname IN (
               SESSION_USER, 'canonical_brain_schema_reconciler'
           )
        OR granted.rolname IN (
               SESSION_USER, 'canonical_brain_schema_reconciler'
           )
        OR grantor.rolname IN (
               SESSION_USER, 'canonical_brain_schema_reconciler'
           )
), forward_role_closure(roleid) AS (
    SELECT membership.roleid
      FROM pg_catalog.pg_auth_members AS membership
      JOIN temporary_login ON temporary_login.oid = membership.member
    UNION
    SELECT membership.roleid
      FROM pg_catalog.pg_auth_members AS membership
      JOIN forward_role_closure AS reachable
        ON reachable.roleid = membership.member
), control_namespace AS (
    SELECT namespace.* FROM pg_catalog.pg_namespace AS namespace
     WHERE namespace.nspname = 'canonical_brain_reconciliation'
), control_routines AS (
    SELECT routine.*, role.rolname AS owner_name,
           namespace.nspname AS namespace_name,
           language.lanname AS language_name
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
      JOIN pg_catalog.pg_roles AS role ON role.oid = routine.proowner
     JOIN pg_catalog.pg_language AS language
        ON language.oid = routine.prolang
     WHERE namespace.nspname = 'canonical_brain_reconciliation'
), routeback_helper_name_inventory AS (
    SELECT routine.oid, routine.prokind, routine.pronargs,
           routine.proargtypes
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
     WHERE namespace.nspname = 'canonical_brain'
       AND routine.proname = '_discord_guild_routeback_target_valid'
), executor_shared_dependencies AS (
    SELECT dependency.dbid, dependency.classid, dependency.objid,
           dependency.objsubid, dependency.deptype
      FROM pg_catalog.pg_shdepend AS dependency
      JOIN executor ON executor.oid = dependency.refobjid
     WHERE dependency.refclassid = 'pg_catalog.pg_authid'::pg_catalog.regclass
), managed_database AS (
    SELECT database.*, pg_catalog.pg_get_userbyid(database.datdba) AS owner_name
      FROM pg_catalog.pg_database AS database
     WHERE database.datname = 'cloudsqladmin'
), managed_actual_database_acl AS (
    SELECT CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                ELSE pg_catalog.pg_get_userbyid(acl.grantee) END AS grantee,
           pg_catalog.pg_get_userbyid(acl.grantor) AS grantor,
           acl.privilege_type, acl.is_grantable
      FROM managed_database AS database
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          database.datacl,
          pg_catalog.acldefault('d', database.datdba)
      )) AS acl
), managed_expected_database_acl(
    grantee, grantor, privilege_type, is_grantable
) AS (
    VALUES
      ('PUBLIC'::text, 'cloudsqladmin'::text, 'CONNECT'::text, false),
      ('PUBLIC'::text, 'cloudsqladmin'::text, 'TEMPORARY'::text, false),
      ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'CREATE'::text, false),
      ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'CONNECT'::text, false),
      ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'TEMPORARY'::text, false)
), managed_cloudsqladmin_exception AS (
    SELECT (SELECT pg_catalog.count(*) = 1 AND COALESCE(
                       pg_catalog.bool_and(
                           datallowconn AND NOT datistemplate
                           AND owner_name = 'cloudsqladmin'
                       ), false
                   ) FROM managed_database)
       AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'cloudsqladmin'
               AND rolcanlogin AND rolsuper AND rolcreatedb AND rolcreaterole
               AND rolreplication AND rolbypassrls
       )
       AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'cloudsqlsuperuser'
               AND rolcanlogin AND NOT rolsuper
               AND rolcreatedb AND rolcreaterole
               AND NOT rolreplication AND NOT rolbypassrls
       )
       AND NOT EXISTS (
            (SELECT * FROM managed_actual_database_acl
             EXCEPT SELECT * FROM managed_expected_database_acl)
            UNION ALL
            (SELECT * FROM managed_expected_database_acl
             EXCEPT SELECT * FROM managed_actual_database_acl)
       ) AS exact
), actual_control_schema_acl AS (
    SELECT acl.grantor, acl.grantee, acl.privilege_type, acl.is_grantable
      FROM control_namespace AS namespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          namespace.nspacl,
          pg_catalog.acldefault('n', namespace.nspowner)
      )) AS acl
), expected_control_schema_acl AS (
    SELECT owner_role.oid AS grantor, owner_role.oid AS grantee,
           privilege.name AS privilege_type, false AS is_grantable
      FROM owner_role
      CROSS JOIN (VALUES ('CREATE'::text), ('USAGE'::text)) AS privilege(name)
    UNION ALL
    SELECT owner_role.oid, executor.oid, 'USAGE'::text, false
      FROM owner_role CROSS JOIN executor
), actual_control_routine_acl AS (
    SELECT routine.oid AS routine_oid, acl.grantor, acl.grantee,
           acl.privilege_type, acl.is_grantable
      FROM control_routines AS routine
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          routine.proacl,
          pg_catalog.acldefault('f', routine.proowner)
      )) AS acl
), expected_control_routine_acl AS (
    SELECT routine.oid AS routine_oid, owner_role.oid AS grantor,
           grantee.oid AS grantee, 'EXECUTE'::text AS privilege_type,
           false AS is_grantable
      FROM control_routines AS routine
      CROSS JOIN owner_role
      CROSS JOIN executor
      CROSS JOIN LATERAL (VALUES (owner_role.oid), (executor.oid)) AS grantee(oid)
)
SELECT CURRENT_USER = SESSION_USER AS current_user_is_session_user,
       pg_catalog.current_database() = 'muncho_canary_brain'
           AS database_is_exact,
       pg_catalog.current_setting('server_version_num')::integer / 10000 = 18
           AS postgresql_major_is_exact,
       (SELECT pg_catalog.pg_get_userbyid(database.datdba)
          FROM pg_catalog.pg_database AS database
         WHERE database.datname = pg_catalog.current_database())
           = 'cloudsqlsuperuser' AS database_owner_is_exact,
       SESSION_USER ~ '^muncho_canary_reconciler_[0-9a-f]{16}$'
           AS session_user_is_temporary_executor,
       (SELECT pg_catalog.count(*) = 1 FROM pg_catalog.pg_roles
         WHERE rolname ~ '^muncho_canary_reconciler_[0-9a-f]{16}$')
           AS temporary_executor_inventory_exact,
       temporary_login.rolcanlogin AND temporary_login.rolinherit
           AND NOT temporary_login.rolsuper
           AND NOT temporary_login.rolcreatedb
           AND NOT temporary_login.rolcreaterole
           AND NOT temporary_login.rolreplication
           AND NOT temporary_login.rolbypassrls
           AND temporary_login.rolconnlimit = -1
           AND temporary_login.rolvaliduntil IS NULL
           AND temporary_login.rolconfig IS NULL
           AS temporary_executor_attributes_exact,
       executor.oid IS NOT NULL AND NOT executor.rolcanlogin
           AND NOT executor.rolinherit AND NOT executor.rolsuper
           AND NOT executor.rolcreatedb AND NOT executor.rolcreaterole
           AND NOT executor.rolreplication AND NOT executor.rolbypassrls
           AND executor.rolconnlimit = -1
           AND executor.rolvaliduntil IS NULL AND executor.rolconfig IS NULL
           AS executor_role_attributes_exact,
       owner_role.oid IS NOT NULL AND NOT owner_role.rolcanlogin
           AND NOT owner_role.rolinherit AND NOT owner_role.rolsuper
           AND NOT owner_role.rolcreatedb AND NOT owner_role.rolcreaterole
           AND NOT owner_role.rolreplication AND NOT owner_role.rolbypassrls
           AND owner_role.rolconnlimit = -1
           AND owner_role.rolvaliduntil IS NULL
           AND owner_role.rolconfig IS NULL
           AS migration_owner_attributes_exact,
       writer_role.oid IS NOT NULL AND NOT writer_role.rolcanlogin
           AND writer_role.rolinherit AND NOT writer_role.rolsuper
           AND NOT writer_role.rolcreatedb AND NOT writer_role.rolcreaterole
           AND NOT writer_role.rolreplication AND NOT writer_role.rolbypassrls
           AND writer_role.rolconnlimit = -1
           AND writer_role.rolvaliduntil IS NULL
           AND writer_role.rolconfig IS NULL
           AS writer_role_attributes_exact,
       NOT EXISTS (SELECT 1 FROM pg_catalog.pg_event_trigger)
           AS event_trigger_inventory_empty,
       pg_catalog.current_setting('max_prepared_transactions')::integer = 0
           AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_prepared_xacts)
           AS prepared_transactions_disabled_and_empty,
       (SELECT pg_catalog.count(*) = 1 AND COALESCE(pg_catalog.bool_and(
                   member_name = SESSION_USER
                   AND granted_name = 'canonical_brain_schema_reconciler'
                   AND grantor_name = 'cloudsqladmin'
                   AND admin_option IS FALSE
                   AND inherit_option IS TRUE
                   AND set_option IS TRUE
               ), false) FROM role_graph)
           AS provider_executor_edge_exact,
       (SELECT pg_catalog.count(DISTINCT reachable.rolname) = 1
               AND COALESCE(pg_catalog.bool_and(
                   reachable.rolname = 'canonical_brain_schema_reconciler'
               ), false)
          FROM forward_role_closure AS closure
          JOIN pg_catalog.pg_roles AS reachable
            ON reachable.oid = closure.roleid)
           AS recursive_authority_closure_exact,
       NOT EXISTS (
           SELECT 1 FROM forward_role_closure AS closure
           JOIN pg_catalog.pg_roles AS reachable
             ON reachable.oid = closure.roleid
          WHERE reachable.rolname IN (
              'canonical_brain_migration_owner',
              'canonical_brain_writer',
              'cloudsqlsuperuser', 'cloudsqladmin', 'postgres'
          )
       ) AS privileged_roles_unreachable,
       NOT pg_catalog.pg_has_role(
           SESSION_USER, 'canonical_brain_migration_owner', 'MEMBER'
       ) AND NOT pg_catalog.pg_has_role(
           SESSION_USER, 'canonical_brain_writer', 'MEMBER'
       ) AND NOT pg_catalog.pg_has_role(
           SESSION_USER, 'cloudsqlsuperuser', 'MEMBER'
       ) AS old_owner_writer_system_path_rejected,
       (SELECT pg_catalog.count(*) = 0
               OR (pg_catalog.count(*) = 1
                   AND COALESCE(pg_catalog.bool_and(
                       prokind = 'f' AND pronargs = 1
                       AND proargtypes[0] = (
                           SELECT argument_type.oid
                             FROM pg_catalog.pg_type AS argument_type
                             JOIN pg_catalog.pg_namespace AS type_namespace
                               ON type_namespace.oid = argument_type.typnamespace
                            WHERE type_namespace.nspname = 'pg_catalog'
                              AND argument_type.typname = 'jsonb'
                       )
                   ), false))
          FROM routeback_helper_name_inventory)
           AS routeback_helper_name_inventory_bounded,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_shdepend AS dependency
            WHERE dependency.refclassid =
                  'pg_catalog.pg_authid'::pg_catalog.regclass
              AND dependency.refobjid = executor.oid
              AND dependency.deptype = 'o'
       ) AS executor_owns_nothing_clusterwide,
       (SELECT pg_catalog.count(*) = 4 AND COALESCE(pg_catalog.bool_and(
            deptype = 'a' AND objsubid = 0 AND (
                (dbid = 0
                 AND classid = 'pg_catalog.pg_database'::pg_catalog.regclass
                 AND objid = (SELECT oid FROM pg_catalog.pg_database
                               WHERE datname = pg_catalog.current_database()))
                OR (dbid = (SELECT oid FROM pg_catalog.pg_database
                             WHERE datname = pg_catalog.current_database())
                    AND classid = 'pg_catalog.pg_namespace'::pg_catalog.regclass
                    AND objid = (SELECT oid FROM control_namespace))
                OR (dbid = (SELECT oid FROM pg_catalog.pg_database
                             WHERE datname = pg_catalog.current_database())
                    AND classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
                    AND objid IN (SELECT oid FROM control_routines))
            )
       ), false) FROM executor_shared_dependencies)
           AS executor_acl_dependencies_exact,
       (SELECT pg_catalog.count(*) = 4
               AND COALESCE(pg_catalog.string_agg(
                   database.datname::text,
                   ',' ORDER BY database.datname::text
               ), '') =
                   'cloudsqladmin,muncho_canary_brain,postgres,template1'
          FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn)
           AS connectable_database_inventory_exact,
       (SELECT pg_catalog.count(*) = 3
               AND COALESCE(pg_catalog.string_agg(
                   database.datname::text,
                   ',' ORDER BY database.datname::text
               ), '') = 'cloudsqladmin,muncho_canary_brain,postgres'
         FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn AND NOT database.datistemplate)
           AS connectable_non_template_database_inventory_exact,
       (SELECT pg_catalog.count(*) = 1 AND COALESCE(pg_catalog.bool_and(
                   database.datname = pg_catalog.current_database()
                   AND acl.privilege_type = 'CONNECT'
                   AND acl.is_grantable IS FALSE
                   AND pg_catalog.pg_get_userbyid(acl.grantor)
                       = 'cloudsqlsuperuser'
               ), false)
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
              database.datacl,
              pg_catalog.acldefault('d', database.datdba)
          )) AS acl
         WHERE acl.grantee = executor.oid)
           AS executor_database_acl_exact,
       pg_catalog.has_database_privilege(
           executor.oid, pg_catalog.current_database(), 'CONNECT'
       ) AND NOT pg_catalog.has_database_privilege(
           executor.oid, pg_catalog.current_database(), 'CREATE'
       ) AND NOT pg_catalog.has_database_privilege(
           executor.oid, pg_catalog.current_database(), 'TEMPORARY'
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_database AS database
            WHERE database.datallowconn
              AND database.datname <> pg_catalog.current_database()
              AND (
                   pg_catalog.pg_get_userbyid(database.datdba) = executor.rolname
                   OR pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'CONNECT'
                   )
                   OR pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'CREATE'
                   )
                   OR pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'TEMPORARY'
                   )
              )
              AND NOT (
                   database.datname = 'cloudsqladmin'
                   AND (SELECT exact FROM managed_cloudsqladmin_exception)
                   AND pg_catalog.pg_get_userbyid(database.datdba)
                       <> executor.rolname
                   AND pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'CONNECT'
                   )
                   AND NOT pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'CREATE'
                   )
                   AND pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'TEMPORARY'
                   )
              )
       ) AS executor_database_effective_privileges_exact,
       (SELECT pg_catalog.count(*) = 1 AND COALESCE(pg_catalog.bool_and(
                   namespace.nspname = 'canonical_brain_reconciliation'
                   AND acl.privilege_type = 'USAGE'
                   AND acl.is_grantable IS FALSE
                   AND pg_catalog.pg_get_userbyid(acl.grantor)
                       = 'canonical_brain_migration_owner'
               ), false)
          FROM pg_catalog.pg_namespace AS namespace
          CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
              namespace.nspacl,
              pg_catalog.acldefault('n', namespace.nspowner)
          )) AS acl
         WHERE acl.grantee = executor.oid)
           AS executor_schema_acl_exact,
       NOT EXISTS (
           (SELECT * FROM actual_control_schema_acl
            EXCEPT SELECT * FROM expected_control_schema_acl)
           UNION ALL
           (SELECT * FROM expected_control_schema_acl
            EXCEPT SELECT * FROM actual_control_schema_acl)
       ) AS control_schema_acl_surface_exact,
       (SELECT pg_catalog.count(*) = 2 AND COALESCE(pg_catalog.bool_and(
                   acl.privilege_type = 'EXECUTE'
                   AND acl.is_grantable IS FALSE
                   AND pg_catalog.pg_get_userbyid(acl.grantor)
                       = 'canonical_brain_migration_owner'
               ), false)
          FROM control_routines AS routine
          CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
              routine.proacl,
              pg_catalog.acldefault('f', routine.proowner)
          )) AS acl
         WHERE acl.grantee = executor.oid)
           AS executor_routine_acl_exact,
       (SELECT pg_catalog.count(*) = 2 AND COALESCE(pg_catalog.bool_and(
                   owner_name = 'canonical_brain_migration_owner'
                   AND prokind = 'f' AND prosecdef
                   AND provolatile = 'v' AND proparallel = 'u'
                   AND NOT proleakproof AND NOT proisstrict AND proretset
                   AND language_name = 'plpgsql'
                   AND pg_catalog.oidvectortypes(proargtypes) = ''
                   AND proconfig = ARRAY[
                       'search_path=pg_catalog, pg_temp',
                       'TimeZone=UTC',
                       'DateStyle=ISO, YMD',
                       'IntervalStyle=postgres',
                       'extra_float_digits=3',
                       'bytea_output=hex',
                       'lock_timeout=15s',
                       'statement_timeout=5min'
                   ]::text[]
                   AND pg_catalog.encode(pg_catalog.sha256(
                       pg_catalog.convert_to(prosrc, 'UTF8')
                   ), 'hex') = CASE proname
                       WHEN 'observe_missing_discord_routeback_helper_v1'
                           THEN '__OBSERVER_PROSRC_SHA256__'
                       WHEN 'apply_missing_discord_routeback_helper_v1'
                           THEN '__APPLY_PROSRC_SHA256__'
                       ELSE ''
                   END
                   AND pg_catalog.encode(pg_catalog.sha256(
                       pg_catalog.convert_to(
                           pg_catalog.pg_get_functiondef(oid), 'UTF8'
                       )
                   ), 'hex') = CASE proname
                       WHEN 'observe_missing_discord_routeback_helper_v1'
                           THEN '__OBSERVER_DEFINITION_SHA256__'
                       WHEN 'apply_missing_discord_routeback_helper_v1'
                           THEN '__APPLY_DEFINITION_SHA256__'
                       ELSE ''
                   END
               ), false) FROM control_routines)
           AS control_routine_attributes_exact,
       (SELECT pg_catalog.count(*) = 2 FROM control_routines)
           AND (SELECT pg_catalog.count(*) = 1 FROM control_namespace)
           AND (SELECT pg_catalog.pg_get_userbyid(nspowner)
                  FROM control_namespace)
               = 'canonical_brain_migration_owner'
           AS control_inventory_and_ownership_exact,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_class AS relation_entry
            WHERE relation_entry.relnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_type AS type_entry
            WHERE type_entry.typnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_constraint AS constraint_entry
            WHERE constraint_entry.connamespace =
                  (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_operator AS operator_entry
            WHERE operator_entry.oprnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_collation AS collation_entry
            WHERE collation_entry.collnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_conversion AS conversion_entry
            WHERE conversion_entry.connamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_opclass AS operator_class
            WHERE operator_class.opcnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_opfamily AS operator_family
            WHERE operator_family.opfnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_statistic_ext AS statistic_entry
            WHERE statistic_entry.stxnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_ts_config AS config_entry
            WHERE config_entry.cfgnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_ts_dict AS dictionary_entry
            WHERE dictionary_entry.dictnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_ts_parser AS parser_entry
            WHERE parser_entry.prsnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_ts_template AS template_entry
            WHERE template_entry.tmplnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_extension AS extension_entry
            WHERE extension_entry.extnamespace =
                  (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_default_acl AS default_acl_entry
            WHERE default_acl_entry.defaclnamespace =
                  (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1
             FROM pg_catalog.pg_publication_namespace AS publication_entry
            WHERE publication_entry.pnnspid =
                  (SELECT oid FROM control_namespace)
       ) AS control_namespace_other_object_inventory_empty,
       NOT EXISTS (
           (SELECT * FROM actual_control_routine_acl
            EXCEPT SELECT * FROM expected_control_routine_acl)
           UNION ALL
           (SELECT * FROM expected_control_routine_acl
            EXCEPT SELECT * FROM actual_control_routine_acl)
       ) AS control_routine_acl_surface_exact,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_stat_activity AS activity
            WHERE activity.datname = pg_catalog.current_database()
              AND activity.backend_type = 'client backend'
              AND activity.pid <> pg_catalog.pg_backend_pid()
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_stat_activity AS activity
            WHERE activity.usesysid = temporary_login.oid
              AND activity.pid <> pg_catalog.pg_backend_pid()
       ) AS no_foreign_database_client_sessions,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_shdepend AS dependency
            WHERE dependency.refclassid =
                  'pg_catalog.pg_authid'::pg_catalog.regclass
              AND dependency.refobjid = temporary_login.oid
       ) AS temporary_executor_has_zero_shared_dependencies
  FROM temporary_login
  CROSS JOIN executor
  CROSS JOIN owner_role
  CROSS JOIN writer_role
""".replace(
    "__OBSERVER_PROSRC_SHA256__", OBSERVER_PROSRC_SHA256
).replace(
    "__OBSERVER_DEFINITION_SHA256__", OBSERVER_DEFINITION_SHA256
).replace(
    "__APPLY_PROSRC_SHA256__", APPLY_PROSRC_SHA256
).replace(
    "__APPLY_DEFINITION_SHA256__", APPLY_DEFINITION_SHA256
).strip()
_AUTHORITY_OPEN_RECEIPT_COLUMNS = (
    "current_user_is_session_user",
    "database_is_exact",
    "postgresql_major_is_exact",
    "database_owner_is_exact",
    "session_user_is_temporary_executor",
    "temporary_executor_inventory_exact",
    "temporary_executor_attributes_exact",
    "executor_role_attributes_exact",
    "migration_owner_attributes_exact",
    "writer_role_attributes_exact",
    "event_trigger_inventory_empty",
    "prepared_transactions_disabled_and_empty",
    "provider_executor_edge_exact",
    "recursive_authority_closure_exact",
    "privileged_roles_unreachable",
    "old_owner_writer_system_path_rejected",
    "routeback_helper_name_inventory_bounded",
    "executor_owns_nothing_clusterwide",
    "executor_acl_dependencies_exact",
    "connectable_database_inventory_exact",
    "connectable_non_template_database_inventory_exact",
    "executor_database_acl_exact",
    "executor_database_effective_privileges_exact",
    "executor_schema_acl_exact",
    "control_schema_acl_surface_exact",
    "executor_routine_acl_exact",
    "control_routine_attributes_exact",
    "control_inventory_and_ownership_exact",
    "control_namespace_other_object_inventory_empty",
    "control_routine_acl_surface_exact",
    "no_foreign_database_client_sessions",
    "temporary_executor_has_zero_shared_dependencies",
)
_AUTHORITY_PREFLIGHT_FAILURE_CODES = {
    name: "schema_reconciliation_authority_" + name
    for name in _AUTHORITY_OPEN_RECEIPT_COLUMNS
}
_ROLE_GRAPH_EDGE_PRESENCE_INVARIANTS = frozenset(
    {"provider_executor_edge_exact"}
)
_ROLE_GRAPH_MISSING_EDGE_INVARIANTS = frozenset(
    {
        "provider_executor_edge_exact",
        "recursive_authority_closure_exact",
    }
)

_CANONICAL_DATA_LOCK_SQL = CANONICAL_TRUTH_LOCK_SQL


class _Session(Protocol):
    username: str
    tls_peer_certificate_sha256: str

    def query(self, sql: str, *, maximum_rows: int) -> QueryResult: ...

    def close(self) -> None: ...


SessionFactory = Callable[[WriterDBConfig], _Session]


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


def _authority_preflight_failures(result: QueryResult) -> tuple[str, ...]:
    """Return only fixed invariant names after validating the receipt shape."""

    if (
        result.command_tag.upper() != "SELECT 1"
        or result.columns != _AUTHORITY_OPEN_RECEIPT_COLUMNS
        or len(result.rows) != 1
        or len(result.rows[0]) != len(_AUTHORITY_OPEN_RECEIPT_COLUMNS)
        or any(value not in {"t", "f"} for value in result.rows[0])
        or set(_AUTHORITY_PREFLIGHT_FAILURE_CODES)
        != set(_AUTHORITY_OPEN_RECEIPT_COLUMNS)
    ):
        raise PostgresProtocolError(
            "schema_reconciliation_database_authority_preflight_invalid"
        )
    return tuple(
        column
        for column, value in zip(
            _AUTHORITY_OPEN_RECEIPT_COLUMNS,
            result.rows[0],
            strict=True,
        )
        if value == "f"
    )


def _raise_authority_preflight_failure(failures: tuple[str, ...]) -> None:
    if failures:
        raise SchemaReconciliationError(
            _AUTHORITY_PREFLIGHT_FAILURE_CODES[failures[0]]
        )


def _require_authority_preflight_receipt(result: QueryResult) -> None:
    """Name the first failed fixed invariant without reflecting DB text."""

    _raise_authority_preflight_failure(
        _authority_preflight_failures(result)
    )


def _is_missing_role_graph_stabilization_candidate(
    failures: tuple[str, ...],
) -> bool:
    failed = frozenset(failures)
    return (
        bool(failed & _ROLE_GRAPH_EDGE_PRESENCE_INVARIANTS)
        and failed <= _ROLE_GRAPH_MISSING_EDGE_INVARIANTS
    )


def _require_stable_authority_preflight(
    session: _Session,
    *,
    sleep: Callable[[float], None],
) -> None:
    """Bound fresh snapshots only while expected API role edges are absent."""

    for attempt in range(_ROLE_GRAPH_STABILIZATION_ATTEMPTS):
        receipt = session.query(
            _AUTHORITY_OPEN_RECEIPT_SQL,
            maximum_rows=1,
        )
        failures = _authority_preflight_failures(receipt)
        if not failures:
            return
        if not _is_missing_role_graph_stabilization_candidate(failures):
            _raise_authority_preflight_failure(failures)
        if attempt + 1 == _ROLE_GRAPH_STABILIZATION_ATTEMPTS:
            raise SchemaReconciliationError(
                "schema_reconciliation_authority_role_graph_stabilization_timeout"
            )
        sleep(_ROLE_GRAPH_STABILIZATION_INTERVAL_SECONDS)
    raise SchemaReconciliationError(
        "schema_reconciliation_authority_role_graph_stabilization_timeout"
    )


def _require_unchanged_authority_preflight_receipt(
    result: QueryResult,
) -> None:
    try:
        _require_authority_preflight_receipt(result)
    except SchemaReconciliationError:
        raise SchemaReconciliationError(
            "schema_reconciliation_authority_preflight_changed"
        ) from None


def _require_void_lock(result: QueryResult, *, expected_column: str) -> None:
    if (
        result.command_tag.upper() != "SELECT 1"
        or result.columns != (expected_column,)
        or len(result.rows) != 1
        or len(result.rows[0]) != 1
    ):
        raise PostgresProtocolError("schema_reconciliation_advisory_lock_failed")


def _require_unlock(
    result: QueryResult,
    *,
    expected_column: str = "pg_advisory_unlock",
) -> None:
    if (
        result.command_tag.upper() != "SELECT 1"
        or result.columns != (expected_column,)
        or result.rows != (("t",),)
    ):
        raise PostgresProtocolError("schema_reconciliation_advisory_unlock_failed")


@dataclass(frozen=True)
class PostDeleteTerminalReceipt:
    release_revision: str
    plan_sha256: str
    database: str
    writer_login: str
    temporary_executor_login: str
    temporary_executor_login_sha256: str
    control_foundation_contract_sha256: str
    target_contract_sha256: str
    observed_contract_sha256: str
    writer_session_identity_exact: bool
    temporary_executor_absent: bool
    temporary_executor_inventory_empty: bool
    prepared_transactions_disabled_and_empty: bool
    persistent_executor_role_attributes_exact: bool
    persistent_executor_memberships_empty: bool
    persistent_executor_owns_zero_objects_clusterwide: bool
    persistent_executor_acl_dependencies_exact: bool
    connectable_database_inventory_exact: bool
    connectable_non_template_database_inventory_exact: bool
    persistent_executor_database_acl_exact: bool
    persistent_executor_database_effective_privileges_exact: bool
    routeback_helper_name_inventory_exact: bool
    control_schema_identity_acl_exact: bool
    control_namespace_other_object_inventory_empty: bool
    control_routine_identity_acl_exact: bool
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
                    self.temporary_executor_login,
                    self.temporary_executor_login_sha256,
                    self.control_foundation_contract_sha256,
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
            or _TEMPORARY_EXECUTOR.fullmatch(self.temporary_executor_login)
            is None
            or self.temporary_executor_login_sha256
            != hashlib.sha256(
                self.temporary_executor_login.encode("utf-8")
            ).hexdigest()
            or _SHA256.fullmatch(self.control_foundation_contract_sha256)
            is None
            or _SHA256.fullmatch(self.target_contract_sha256) is None
            or self.observed_contract_sha256 != self.target_contract_sha256
            or any(
                value is not True
                for value in (
                    self.writer_session_identity_exact,
                    self.temporary_executor_absent,
                    self.temporary_executor_inventory_empty,
                    self.prepared_transactions_disabled_and_empty,
                    self.persistent_executor_role_attributes_exact,
                    self.persistent_executor_memberships_empty,
                    self.persistent_executor_owns_zero_objects_clusterwide,
                    self.persistent_executor_acl_dependencies_exact,
                    self.connectable_database_inventory_exact,
                    self.connectable_non_template_database_inventory_exact,
                    self.persistent_executor_database_acl_exact,
                    self.persistent_executor_database_effective_privileges_exact,
                    self.routeback_helper_name_inventory_exact,
                    self.control_schema_identity_acl_exact,
                    self.control_namespace_other_object_inventory_empty,
                    self.control_routine_identity_acl_exact,
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
            "temporary_executor_login": self.temporary_executor_login,
            "temporary_executor_login_sha256": (
                self.temporary_executor_login_sha256
            ),
            "control_foundation_contract_sha256": (
                self.control_foundation_contract_sha256
            ),
            "target_contract_sha256": self.target_contract_sha256,
            "observed_contract_sha256": self.observed_contract_sha256,
            "writer_session_identity_exact": self.writer_session_identity_exact,
            "temporary_executor_absent": self.temporary_executor_absent,
            "temporary_executor_inventory_empty": (
                self.temporary_executor_inventory_empty
            ),
            "prepared_transactions_disabled_and_empty": (
                self.prepared_transactions_disabled_and_empty
            ),
            "persistent_executor_role_attributes_exact": (
                self.persistent_executor_role_attributes_exact
            ),
            "persistent_executor_memberships_empty": (
                self.persistent_executor_memberships_empty
            ),
            "persistent_executor_owns_zero_objects_clusterwide": (
                self.persistent_executor_owns_zero_objects_clusterwide
            ),
            "persistent_executor_acl_dependencies_exact": (
                self.persistent_executor_acl_dependencies_exact
            ),
            "connectable_database_inventory_exact": (
                self.connectable_database_inventory_exact
            ),
            "connectable_non_template_database_inventory_exact": (
                self.connectable_non_template_database_inventory_exact
            ),
            "persistent_executor_database_acl_exact": (
                self.persistent_executor_database_acl_exact
            ),
            "persistent_executor_database_effective_privileges_exact": (
                self.persistent_executor_database_effective_privileges_exact
            ),
            "routeback_helper_name_inventory_exact": (
                self.routeback_helper_name_inventory_exact
            ),
            "control_schema_identity_acl_exact": (
                self.control_schema_identity_acl_exact
            ),
            "control_namespace_other_object_inventory_empty": (
                self.control_namespace_other_object_inventory_empty
            ),
            "control_routine_identity_acl_exact": (
                self.control_routine_identity_acl_exact
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
        "temporary_executor_login",
        "temporary_executor_login_sha256",
        "control_foundation_contract_sha256",
        "target_contract_sha256",
        "observed_contract_sha256",
        "writer_session_identity_exact",
        "temporary_executor_absent",
        "temporary_executor_inventory_empty",
        "prepared_transactions_disabled_and_empty",
        "persistent_executor_role_attributes_exact",
        "persistent_executor_memberships_empty",
        "persistent_executor_owns_zero_objects_clusterwide",
        "persistent_executor_acl_dependencies_exact",
        "connectable_database_inventory_exact",
        "connectable_non_template_database_inventory_exact",
        "persistent_executor_database_acl_exact",
        "persistent_executor_database_effective_privileges_exact",
        "routeback_helper_name_inventory_exact",
        "control_schema_identity_acl_exact",
        "control_namespace_other_object_inventory_empty",
        "control_routine_identity_acl_exact",
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
        temporary_executor_login=value.get("temporary_executor_login"),
        temporary_executor_login_sha256=value.get(
            "temporary_executor_login_sha256"
        ),
        control_foundation_contract_sha256=value.get(
            "control_foundation_contract_sha256"
        ),
        target_contract_sha256=value.get("target_contract_sha256"),
        observed_contract_sha256=value.get("observed_contract_sha256"),
        writer_session_identity_exact=value.get("writer_session_identity_exact"),
        temporary_executor_absent=value.get("temporary_executor_absent"),
        temporary_executor_inventory_empty=value.get(
            "temporary_executor_inventory_empty"
        ),
        prepared_transactions_disabled_and_empty=value.get(
            "prepared_transactions_disabled_and_empty"
        ),
        persistent_executor_role_attributes_exact=value.get(
            "persistent_executor_role_attributes_exact"
        ),
        persistent_executor_memberships_empty=value.get(
            "persistent_executor_memberships_empty"
        ),
        persistent_executor_owns_zero_objects_clusterwide=value.get(
            "persistent_executor_owns_zero_objects_clusterwide"
        ),
        persistent_executor_acl_dependencies_exact=value.get(
            "persistent_executor_acl_dependencies_exact"
        ),
        connectable_database_inventory_exact=value.get(
            "connectable_database_inventory_exact"
        ),
        connectable_non_template_database_inventory_exact=value.get(
            "connectable_non_template_database_inventory_exact"
        ),
        persistent_executor_database_acl_exact=value.get(
            "persistent_executor_database_acl_exact"
        ),
        persistent_executor_database_effective_privileges_exact=value.get(
            "persistent_executor_database_effective_privileges_exact"
        ),
        routeback_helper_name_inventory_exact=value.get(
            "routeback_helper_name_inventory_exact"
        ),
        control_schema_identity_acl_exact=value.get(
            "control_schema_identity_acl_exact"
        ),
        control_namespace_other_object_inventory_empty=value.get(
            "control_namespace_other_object_inventory_empty"
        ),
        control_routine_identity_acl_exact=value.get(
            "control_routine_identity_acl_exact"
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
    temporary_executor_login: str,
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
        or receipt.temporary_executor_login != temporary_executor_login
        or receipt.control_foundation_contract_sha256
        != plan.value.get("control_foundation_contract_sha256")
        or receipt.target_contract_sha256 != target.sha256
        or receipt.managed_hba_receipt_sha256 != managed_hba_receipt.sha256
        or receipt.pre_delete_canonical_truth_receipt_sha256
        != pre_delete_canonical_truth.sha256
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_invalid"
        )
    return receipt


def _post_delete_authority_absence_sql(
    temporary_executor_login: str,
) -> str:
    if _TEMPORARY_EXECUTOR.fullmatch(temporary_executor_login) is None:
        raise SchemaReconciliationError(
            "schema_reconciliation_post_delete_terminal_invalid"
        )
    escaped = temporary_executor_login.replace("'", "''")
    return r"""
WITH executor AS (
    SELECT * FROM pg_catalog.pg_roles
     WHERE rolname = 'canonical_brain_schema_reconciler'
), owner_role AS (
    SELECT * FROM pg_catalog.pg_roles
     WHERE rolname = 'canonical_brain_migration_owner'
), control_namespace AS (
    SELECT * FROM pg_catalog.pg_namespace
     WHERE nspname = 'canonical_brain_reconciliation'
), control_routines AS (
    SELECT routine.*, owner.rolname AS owner_name,
           language.lanname AS language_name
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
      JOIN pg_catalog.pg_roles AS owner ON owner.oid = routine.proowner
     JOIN pg_catalog.pg_language AS language ON language.oid = routine.prolang
     WHERE namespace.nspname = 'canonical_brain_reconciliation'
), routeback_helper_name_inventory AS (
    SELECT routine.oid, routine.prokind, routine.pronargs,
           routine.proargtypes
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
     WHERE namespace.nspname = 'canonical_brain'
       AND routine.proname = '_discord_guild_routeback_target_valid'
), executor_shared_dependencies AS (
    SELECT dependency.dbid, dependency.classid, dependency.objid,
           dependency.objsubid, dependency.deptype
      FROM pg_catalog.pg_shdepend AS dependency
      JOIN executor ON executor.oid = dependency.refobjid
     WHERE dependency.refclassid = 'pg_catalog.pg_authid'::pg_catalog.regclass
), managed_database AS (
    SELECT database.*, pg_catalog.pg_get_userbyid(database.datdba) AS owner_name
      FROM pg_catalog.pg_database AS database
     WHERE database.datname = 'cloudsqladmin'
), managed_actual_database_acl AS (
    SELECT CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                ELSE pg_catalog.pg_get_userbyid(acl.grantee) END AS grantee,
           pg_catalog.pg_get_userbyid(acl.grantor) AS grantor,
           acl.privilege_type, acl.is_grantable
      FROM managed_database AS database
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          database.datacl,
          pg_catalog.acldefault('d', database.datdba)
      )) AS acl
), managed_expected_database_acl(
    grantee, grantor, privilege_type, is_grantable
) AS (
    VALUES
      ('PUBLIC'::text, 'cloudsqladmin'::text, 'CONNECT'::text, false),
      ('PUBLIC'::text, 'cloudsqladmin'::text, 'TEMPORARY'::text, false),
      ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'CREATE'::text, false),
      ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'CONNECT'::text, false),
      ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'TEMPORARY'::text, false)
), managed_cloudsqladmin_exception AS (
    SELECT (SELECT pg_catalog.count(*) = 1 AND COALESCE(
                       pg_catalog.bool_and(
                           datallowconn AND NOT datistemplate
                           AND owner_name = 'cloudsqladmin'
                       ), false
                   ) FROM managed_database)
       AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'cloudsqladmin'
               AND rolcanlogin AND rolsuper AND rolcreatedb AND rolcreaterole
               AND rolreplication AND rolbypassrls
       )
       AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'cloudsqlsuperuser'
               AND rolcanlogin AND NOT rolsuper
               AND rolcreatedb AND rolcreaterole
               AND NOT rolreplication AND NOT rolbypassrls
       )
       AND NOT EXISTS (
            (SELECT * FROM managed_actual_database_acl
             EXCEPT SELECT * FROM managed_expected_database_acl)
            UNION ALL
            (SELECT * FROM managed_expected_database_acl
             EXCEPT SELECT * FROM managed_actual_database_acl)
       ) AS exact
), actual_schema_acl AS (
    SELECT acl.grantor, acl.grantee, acl.privilege_type, acl.is_grantable
      FROM control_namespace AS namespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          namespace.nspacl,
          pg_catalog.acldefault('n', namespace.nspowner)
      )) AS acl
), expected_schema_acl AS (
    SELECT owner_role.oid AS grantor, owner_role.oid AS grantee,
           privilege.name AS privilege_type, false AS is_grantable
      FROM owner_role
      CROSS JOIN (VALUES ('CREATE'::text), ('USAGE'::text)) AS privilege(name)
    UNION ALL
    SELECT owner_role.oid, executor.oid, 'USAGE'::text, false
      FROM owner_role CROSS JOIN executor
), actual_routine_acl AS (
    SELECT routine.oid AS routine_oid, acl.grantor, acl.grantee,
           acl.privilege_type, acl.is_grantable
      FROM control_routines AS routine
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          routine.proacl,
          pg_catalog.acldefault('f', routine.proowner)
      )) AS acl
), expected_routine_acl AS (
    SELECT routine.oid AS routine_oid, owner_role.oid AS grantor,
           grantee.oid AS grantee, 'EXECUTE'::text AS privilege_type,
           false AS is_grantable
      FROM control_routines AS routine
      CROSS JOIN owner_role
      CROSS JOIN executor
      CROSS JOIN LATERAL (VALUES (owner_role.oid), (executor.oid)) AS grantee(oid)
)
SELECT CURRENT_USER = 'muncho_canary_writer_login'
           AND SESSION_USER = 'muncho_canary_writer_login'
           AS writer_session_identity_exact,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '__TEMP_LOGIN__'
       ) AS temporary_executor_absent,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_roles
            WHERE rolname ~ '^muncho_canary_reconciler_[0-9a-f]{16}$'
       ) AS temporary_executor_inventory_empty,
       pg_catalog.current_setting('max_prepared_transactions')::integer = 0
           AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_prepared_xacts)
           AS prepared_transactions_disabled_and_empty,
       executor.oid IS NOT NULL AND NOT executor.rolcanlogin
           AND NOT executor.rolinherit AND NOT executor.rolsuper
           AND NOT executor.rolcreatedb AND NOT executor.rolcreaterole
           AND NOT executor.rolreplication AND NOT executor.rolbypassrls
           AND executor.rolconnlimit = -1
           AND executor.rolvaliduntil IS NULL AND executor.rolconfig IS NULL
           AS persistent_executor_role_attributes_exact,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_auth_members AS membership
            WHERE membership.roleid = executor.oid
               OR membership.member = executor.oid
               OR membership.grantor = executor.oid
       ) AS persistent_executor_memberships_empty,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_shdepend AS dependency
            WHERE dependency.refclassid =
                  'pg_catalog.pg_authid'::pg_catalog.regclass
              AND dependency.refobjid = executor.oid
              AND dependency.deptype = 'o'
       ) AS persistent_executor_owns_zero_objects_clusterwide,
       (SELECT pg_catalog.count(*) = 4 AND COALESCE(pg_catalog.bool_and(
            deptype = 'a' AND objsubid = 0 AND (
                (dbid = 0
                 AND classid = 'pg_catalog.pg_database'::pg_catalog.regclass
                 AND objid = (SELECT oid FROM pg_catalog.pg_database
                               WHERE datname = pg_catalog.current_database()))
                OR (dbid = (SELECT oid FROM pg_catalog.pg_database
                             WHERE datname = pg_catalog.current_database())
                    AND classid = 'pg_catalog.pg_namespace'::pg_catalog.regclass
                    AND objid = (SELECT oid FROM control_namespace))
                OR (dbid = (SELECT oid FROM pg_catalog.pg_database
                             WHERE datname = pg_catalog.current_database())
                    AND classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
                    AND objid IN (SELECT oid FROM control_routines))
            )
       ), false) FROM executor_shared_dependencies)
           AS persistent_executor_acl_dependencies_exact,
       (SELECT pg_catalog.count(*) = 4
               AND COALESCE(pg_catalog.string_agg(
                   database.datname::text,
                   ',' ORDER BY database.datname::text
               ), '') =
                   'cloudsqladmin,muncho_canary_brain,postgres,template1'
          FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn)
           AS connectable_database_inventory_exact,
       (SELECT pg_catalog.count(*) = 3
               AND COALESCE(pg_catalog.string_agg(
                   database.datname::text,
                   ',' ORDER BY database.datname::text
               ), '') = 'cloudsqladmin,muncho_canary_brain,postgres'
         FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn AND NOT database.datistemplate)
           AS connectable_non_template_database_inventory_exact,
       (SELECT pg_catalog.count(*) = 1 AND COALESCE(pg_catalog.bool_and(
                   database.datname = pg_catalog.current_database()
                   AND acl.privilege_type = 'CONNECT'
                   AND acl.is_grantable IS FALSE
                   AND pg_catalog.pg_get_userbyid(acl.grantor)
                       = 'cloudsqlsuperuser'
               ), false)
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
              database.datacl,
              pg_catalog.acldefault('d', database.datdba)
          )) AS acl
         WHERE acl.grantee = executor.oid)
           AS persistent_executor_database_acl_exact,
       pg_catalog.has_database_privilege(
           executor.oid, pg_catalog.current_database(), 'CONNECT'
       ) AND NOT pg_catalog.has_database_privilege(
           executor.oid, pg_catalog.current_database(), 'CREATE'
       ) AND NOT pg_catalog.has_database_privilege(
           executor.oid, pg_catalog.current_database(), 'TEMPORARY'
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_database AS database
            WHERE database.datallowconn
              AND database.datname <> pg_catalog.current_database()
              AND (
                   pg_catalog.pg_get_userbyid(database.datdba) = executor.rolname
                   OR pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'CONNECT'
                   )
                   OR pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'CREATE'
                   )
                   OR pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'TEMPORARY'
                   )
              )
              AND NOT (
                   database.datname = 'cloudsqladmin'
                   AND (SELECT exact FROM managed_cloudsqladmin_exception)
                   AND pg_catalog.pg_get_userbyid(database.datdba)
                       <> executor.rolname
                   AND pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'CONNECT'
                   )
                   AND NOT pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'CREATE'
                   )
                   AND pg_catalog.has_database_privilege(
                       executor.oid, database.oid, 'TEMPORARY'
                   )
              )
       ) AS persistent_executor_database_effective_privileges_exact,
       (SELECT pg_catalog.count(*) = 1
          AND pg_catalog.bool_and(
              prokind = 'f' AND pronargs = 1
              AND proargtypes[0] = (
                  SELECT argument_type.oid
                    FROM pg_catalog.pg_type AS argument_type
                    JOIN pg_catalog.pg_namespace AS type_namespace
                      ON type_namespace.oid = argument_type.typnamespace
                   WHERE type_namespace.nspname = 'pg_catalog'
                     AND argument_type.typname = 'jsonb'
              )
          ) FROM routeback_helper_name_inventory)
           AS routeback_helper_name_inventory_exact,
       (SELECT pg_catalog.count(*) = 1 FROM control_namespace)
           AND (SELECT pg_catalog.pg_get_userbyid(nspowner)
                  FROM control_namespace)
               = 'canonical_brain_migration_owner'
           AND NOT EXISTS (
               (SELECT * FROM actual_schema_acl
                EXCEPT SELECT * FROM expected_schema_acl)
               UNION ALL
               (SELECT * FROM expected_schema_acl
                EXCEPT SELECT * FROM actual_schema_acl)
           ) AS control_schema_identity_acl_exact,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_class AS relation_entry
            WHERE relation_entry.relnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_type AS type_entry
            WHERE type_entry.typnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_constraint AS constraint_entry
            WHERE constraint_entry.connamespace =
                  (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_operator AS operator_entry
            WHERE operator_entry.oprnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_collation AS collation_entry
            WHERE collation_entry.collnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_conversion AS conversion_entry
            WHERE conversion_entry.connamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_opclass AS operator_class
            WHERE operator_class.opcnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_opfamily AS operator_family
            WHERE operator_family.opfnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_statistic_ext AS statistic_entry
            WHERE statistic_entry.stxnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_ts_config AS config_entry
            WHERE config_entry.cfgnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_ts_dict AS dictionary_entry
            WHERE dictionary_entry.dictnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_ts_parser AS parser_entry
            WHERE parser_entry.prsnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_ts_template AS template_entry
            WHERE template_entry.tmplnamespace = (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_extension AS extension_entry
            WHERE extension_entry.extnamespace =
                  (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_default_acl AS default_acl_entry
            WHERE default_acl_entry.defaclnamespace =
                  (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1
             FROM pg_catalog.pg_publication_namespace AS publication_entry
            WHERE publication_entry.pnnspid =
                  (SELECT oid FROM control_namespace)
       ) AND NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_event_trigger
       ) AS control_namespace_other_object_inventory_empty,
       (SELECT pg_catalog.count(*) = 2 AND COALESCE(pg_catalog.bool_and(
                   owner_name = 'canonical_brain_migration_owner'
                   AND prokind = 'f' AND prosecdef
                   AND provolatile = 'v' AND proparallel = 'u'
                   AND NOT proleakproof AND NOT proisstrict AND proretset
                   AND language_name = 'plpgsql'
                   AND pg_catalog.oidvectortypes(proargtypes) = ''
                   AND proconfig = ARRAY[
                       'search_path=pg_catalog, pg_temp', 'TimeZone=UTC',
                       'DateStyle=ISO, YMD', 'IntervalStyle=postgres',
                       'extra_float_digits=3', 'bytea_output=hex',
                       'lock_timeout=15s', 'statement_timeout=5min'
                   ]::text[]
                   AND pg_catalog.encode(pg_catalog.sha256(
                       pg_catalog.convert_to(prosrc, 'UTF8')
                   ), 'hex') = CASE proname
                       WHEN 'observe_missing_discord_routeback_helper_v1'
                           THEN '__OBSERVER_PROSRC_SHA256__'
                       WHEN 'apply_missing_discord_routeback_helper_v1'
                           THEN '__APPLY_PROSRC_SHA256__'
                       ELSE ''
                   END
                   AND pg_catalog.encode(pg_catalog.sha256(
                       pg_catalog.convert_to(
                           pg_catalog.pg_get_functiondef(oid), 'UTF8'
                       )
                   ), 'hex') = CASE proname
                       WHEN 'observe_missing_discord_routeback_helper_v1'
                           THEN '__OBSERVER_DEFINITION_SHA256__'
                       WHEN 'apply_missing_discord_routeback_helper_v1'
                           THEN '__APPLY_DEFINITION_SHA256__'
                       ELSE ''
                   END
               ), false) FROM control_routines)
           AND NOT EXISTS (
               (SELECT * FROM actual_routine_acl
                EXCEPT SELECT * FROM expected_routine_acl)
               UNION ALL
               (SELECT * FROM expected_routine_acl
                EXCEPT SELECT * FROM actual_routine_acl)
           ) AS control_routine_identity_acl_exact,
       NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_stat_activity AS activity
            WHERE activity.datname = pg_catalog.current_database()
              AND activity.backend_type = 'client backend'
              AND activity.pid <> pg_catalog.pg_backend_pid()
       ) AS no_foreign_database_client_sessions
  FROM executor
""".replace(
        "__TEMP_LOGIN__", escaped
    ).replace(
        "__OBSERVER_PROSRC_SHA256__", OBSERVER_PROSRC_SHA256
    ).replace(
        "__OBSERVER_DEFINITION_SHA256__", OBSERVER_DEFINITION_SHA256
    ).replace(
        "__APPLY_PROSRC_SHA256__", APPLY_PROSRC_SHA256
    ).replace(
        "__APPLY_DEFINITION_SHA256__", APPLY_DEFINITION_SHA256
    ).strip()


_POST_DELETE_AUTHORITY_COLUMNS = (
    "writer_session_identity_exact",
    "temporary_executor_absent",
    "temporary_executor_inventory_empty",
    "prepared_transactions_disabled_and_empty",
    "persistent_executor_role_attributes_exact",
    "persistent_executor_memberships_empty",
    "persistent_executor_owns_zero_objects_clusterwide",
    "persistent_executor_acl_dependencies_exact",
    "connectable_database_inventory_exact",
    "connectable_non_template_database_inventory_exact",
    "persistent_executor_database_acl_exact",
    "persistent_executor_database_effective_privileges_exact",
    "routeback_helper_name_inventory_exact",
    "control_schema_identity_acl_exact",
    "control_namespace_other_object_inventory_empty",
    "control_routine_identity_acl_exact",
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
    temporary_executor_login: str,
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
        or _TEMPORARY_EXECUTOR.fullmatch(temporary_executor_login) is None
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
    session_lock_acquired = False
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
        for statement in (
            "SET lock_timeout = '15s'",
            "SET statement_timeout = '2min'",
        ):
            _require_command(session, statement, "SET")
        lock = session.query(
            "SELECT pg_catalog.pg_advisory_lock_shared("
            + str(CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY)
            + ")",
            maximum_rows=1,
        )
        _require_void_lock(lock, expected_column="pg_advisory_lock_shared")
        session_lock_acquired = True
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
        authority = session.query(
            _post_delete_authority_absence_sql(temporary_executor_login),
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
        unlock = session.query(
            "SELECT pg_catalog.pg_advisory_unlock_shared("
            + str(CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY)
            + ")",
            maximum_rows=1,
        )
        _require_unlock(
            unlock,
            expected_column="pg_advisory_unlock_shared",
        )
        session_lock_acquired = False
    except BaseException:
        if session is not None and transaction_open:
            _rollback_quietly(session)
            transaction_open = False
        if session is not None and session_lock_acquired:
            try:
                unlock = session.query(
                    "SELECT pg_catalog.pg_advisory_unlock_shared("
                    + str(CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY)
                    + ")",
                    maximum_rows=1,
                )
                _require_unlock(
                    unlock,
                    expected_column="pg_advisory_unlock_shared",
                )
                session_lock_acquired = False
            except BaseException:
                pass
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
        temporary_executor_login=temporary_executor_login,
        temporary_executor_login_sha256=hashlib.sha256(
            temporary_executor_login.encode("utf-8")
        ).hexdigest(),
        control_foundation_contract_sha256=plan.value[
            "control_foundation_contract_sha256"
        ],
        target_contract_sha256=target.sha256,
        observed_contract_sha256=observed_contract.sha256,
        writer_session_identity_exact=True,
        temporary_executor_absent=True,
        temporary_executor_inventory_empty=True,
        prepared_transactions_disabled_and_empty=True,
        persistent_executor_role_attributes_exact=True,
        persistent_executor_memberships_empty=True,
        persistent_executor_owns_zero_objects_clusterwide=True,
        persistent_executor_acl_dependencies_exact=True,
        connectable_database_inventory_exact=True,
        connectable_non_template_database_inventory_exact=True,
        persistent_executor_database_acl_exact=True,
        persistent_executor_database_effective_privileges_exact=True,
        routeback_helper_name_inventory_exact=True,
        control_schema_identity_acl_exact=True,
        control_namespace_other_object_inventory_empty=True,
        control_routine_identity_acl_exact=True,
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
        temporary_executor_login=temporary_executor_login,
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
    ) -> None:
        self._session = session
        self._plan = plan
        self._target = target
        self._writer_config = writer_config
        self._managed_hba_receipt = managed_hba_receipt
        self._policy = _target_policy(target.attestation)
        self._active = True
        self._truth_locked = False
        self._initial_observation: ControlObservation | None = None
        self._initial_truth_returned = False
        self._apply_receipt: ControlApplyReceipt | None = None
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
        # The fixed SECURITY DEFINER observer acquires the exclusive xact
        # advisory lock first and then the canonical SHARE table locks.  The
        # inert executor never receives direct table privileges.
        observation = parse_control_observation(
            self._query(OBSERVER_CALL_SQL, maximum_rows=1)
        )
        self._initial_observation = observation
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
            )
        except BaseException:
            # Closing the connection remains a safe rollback even when the
            # wire client has already consumed a complete result.
            self._protocol_usable = False
            raise

    def observe_canonical_truth(self) -> CanonicalTruthReceipt:
        self._require_truth_lock()
        if self._initial_observation is None:
            raise SchemaReconciliationError(
                "schema_reconciliation_control_observation_invalid"
            )
        if not self._initial_truth_returned:
            self._initial_truth_returned = True
            return self._initial_observation.truth
        return parse_control_observation(
            self._query(OBSERVER_CALL_SQL, maximum_rows=1)
        ).truth

    def apply_missing_helper(
        self,
        *,
        authorized_intent_sha256: str,
    ) -> None:
        self._require_truth_lock()
        if self._mutation_attempted:
            raise SchemaReconciliationError(
                "schema_reconciliation_database_apply_repeated"
            )
        if (
            not isinstance(authorized_intent_sha256, str)
            or _SHA256.fullmatch(authorized_intent_sha256) is None
            or self._initial_observation is None
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_control_binding_invalid"
            )
        self._mutation_attempted = True
        truth_sha256 = self._initial_observation.truth.sha256
        for name, value in (
            (
                PLAN_GUC,
                self._plan.sha256,
            ),
            (
                AUTHORIZED_INTENT_GUC,
                authorized_intent_sha256,
            ),
            (
                TRUTH_RECEIPT_GUC,
                truth_sha256,
            ),
        ):
            _require_command(
                self._session,
                set_local_hash_sql(name, value),
                "SET",
            )
        receipt = parse_control_apply_receipt(
            self._query(APPLY_CALL_SQL, maximum_rows=1),
            plan_sha256=self._plan.sha256,
            authorized_intent_sha256=authorized_intent_sha256,
            canonical_truth_receipt_sha256=truth_sha256,
            observation_sha256=(
                self._initial_observation.observation_sha256
            ),
        )
        if receipt.applied is not True:
            raise SchemaReconciliationError(
                "schema_reconciliation_control_apply_not_performed"
            )
        self._apply_receipt = receipt
        self._mutation_body_complete = True

    def close_authority(self) -> None:
        self._require_active()
        if not self._truth_locked or self._authority_closed:
            raise SchemaReconciliationError(
                "schema_reconciliation_database_authority_close_invalid"
            )
        _require_unchanged_authority_preflight_receipt(
            self._query(_AUTHORITY_OPEN_RECEIPT_SQL, maximum_rows=1)
        )
        self._authority_closed = True

    def invalidate(self) -> None:
        self._active = False


class PostgresSchemaReconciliationDatabase:
    """Exact release boundary used by the reconciliation executor.

    The temporary login receives only the inert executor role.  Mutation is
    the fixed zero-argument owner-owned apply routine; no SQL text, action,
    object name, or identifier crosses this boundary.
    """

    def __init__(
        self,
        *,
        plan: SchemaReconciliationPlan,
        target: SchemaContract,
        executor_config: WriterDBConfig,
        writer_config: WriterDBConfig,
        managed_hba_receipt: ManagedCloudSQLAdminHBAReceipt,
        executor_managed_hba_receipt: ManagedCloudSQLAdminHBAReceipt,
        pre_begin_admission: Callable[[], None],
        _session_factory: SessionFactory | None = None,
        _stabilization_sleep: Callable[[float], None] = time.sleep,
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
            executor_config.host,
            executor_config.tls_server_name,
            executor_config.port,
            executor_config.database,
            executor_config.ca_file,
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
            or executor_config.database != DATABASE
            or writer_config.database != DATABASE
            or writer_config.user != WRITER_LOGIN
            or _TEMPORARY_EXECUTOR.fullmatch(executor_config.user) is None
            or not isinstance(
                managed_hba_receipt,
                ManagedCloudSQLAdminHBAReceipt,
            )
            or not isinstance(
                executor_managed_hba_receipt,
                ManagedCloudSQLAdminHBAReceipt,
            )
            or not callable(pre_begin_admission)
            or not callable(_stabilization_sleep)
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
            _validate_active_managed_hba_receipt(
                executor_managed_hba_receipt,
                executor_managed_hba_receipt,
                config=executor_config,
                now_unix=int(time.time()),
                require_expected_fresh=False,
            )
            if not hmac.compare_digest(
                managed_hba_receipt.server_certificate_sha256,
                executor_managed_hba_receipt.server_certificate_sha256,
            ):
                raise SchemaReconciliationError(
                    "schema_reconciliation_database_hba_receipt_invalid"
                )
        except BaseException as exc:
            raise SchemaReconciliationError(
                "schema_reconciliation_database_hba_receipt_invalid"
            ) from exc
        self._plan = plan
        self._target = target
        self._executor_config = executor_config
        self._writer_config = writer_config
        self._managed_hba_receipt = managed_hba_receipt
        self._executor_managed_hba_receipt = executor_managed_hba_receipt
        self._pre_begin_admission = pre_begin_admission
        self._session_factory = _session_factory or _open_postgres_session
        self._stabilization_sleep = _stabilization_sleep
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
            session = self._session_factory(self._executor_config)
            if (
                getattr(session, "username", None)
                != self._executor_config.user
                or not isinstance(
                    getattr(session, "tls_peer_certificate_sha256", None),
                    str,
                )
                or not hmac.compare_digest(
                    session.tls_peer_certificate_sha256,
                    self._executor_managed_hba_receipt
                    .server_certificate_sha256,
                )
            ):
                raise SchemaReconciliationError(
                    "schema_reconciliation_database_session_identity_invalid"
                )
            for statement in (
                "SET lock_timeout = '15s'",
                "SET statement_timeout = '2min'",
            ):
                _require_command(session, statement, "SET")
            lock = session.query(
                "SELECT pg_catalog.pg_advisory_lock("
                + str(advisory_lock_key)
                + ")",
                maximum_rows=1,
            )
            _require_void_lock(lock, expected_column="pg_advisory_lock")
            session_lock_acquired = True
            _require_stable_authority_preflight(
                session,
                sleep=self._stabilization_sleep,
            )
            admission_result = self._pre_begin_admission()
            if admission_result is not None:
                if inspect.isawaitable(admission_result):
                    close = getattr(admission_result, "close", None)
                    if callable(close):
                        close()
                raise SchemaReconciliationError(
                    "schema_reconciliation_database_admission_invalid"
                )
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
            authority_preflight = session.query(
                _AUTHORITY_OPEN_RECEIPT_SQL,
                maximum_rows=1,
            )
            _require_authority_preflight_receipt(authority_preflight)
            scope = _PostgresSchemaReconciliationTransaction(
                session=session,
                plan=self._plan,
                target=self._target,
                writer_config=self._writer_config,
                managed_hba_receipt=self._managed_hba_receipt,
            )
            yield scope
            if not scope.truth_locked:
                raise SchemaReconciliationError(
                    "schema_reconciliation_canonical_truth_not_locked"
                )
            # The caller performs its post-apply contract and canonical-truth
            # observations before leaving the context.  The final authority
            # receipt proves that the inert role graph and fixed control
            # surface remained exact before COMMIT.  Cloud API deletion of the
            # temporary login remains an outer terminal step.
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
