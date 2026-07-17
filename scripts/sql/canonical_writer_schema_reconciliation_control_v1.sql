-- P0 fixed PostgreSQL control boundary for the one missing Discord helper.
--
-- This is a one-time, owner-approved bootstrap artifact.  It deliberately
-- creates an inert NOLOGIN executor role: the role owns nothing and receives
-- only CONNECT, control-schema USAGE, and EXECUTE on two zero-argument
-- SECURITY DEFINER routines.  The routines accept no SQL, identifiers,
-- actions, or semantic choices from their caller.

BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET LOCAL TimeZone = 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL IntervalStyle = 'postgres';
SET LOCAL extra_float_digits = 3;
SET LOCAL bytea_output = 'hex';
SET LOCAL search_path = pg_catalog, pg_temp;
SET LOCAL createrole_self_grant = '';
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '5min';

-- Lock order is part of the protocol: deployment lock first, tables second.
SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

DO $control_bootstrap_preflight$
BEGIN
    IF CURRENT_USER <> SESSION_USER
       OR pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_setting('server_version_num')::integer / 10000 <> 18
       OR SESSION_USER !~ '^muncho_canary_control_[0-9a-f]{16}$'
       OR (
           SELECT pg_catalog.count(*) FROM pg_catalog.pg_roles
            WHERE rolname ~ '^muncho_canary_control_[0-9a-f]{16}$'
       ) <> 1
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_roles
            WHERE rolname ~ '^muncho_canary_reconciler_[0-9a-f]{16}$'
       )
       OR EXISTS (SELECT 1 FROM pg_catalog.pg_event_trigger)
       OR pg_catalog.current_setting('createrole_self_grant') <> ''
       OR NOT (
           WITH RECURSIVE bootstrap AS (
               SELECT * FROM pg_catalog.pg_roles
                WHERE rolname = SESSION_USER
           ), relevant_edges AS (
               SELECT membership.*, granted.rolname AS granted_name,
                      member.rolname AS member_name,
                      grantor.rolname AS grantor_name
                 FROM pg_catalog.pg_auth_members AS membership
                 JOIN pg_catalog.pg_roles AS granted
                   ON granted.oid = membership.roleid
                 JOIN pg_catalog.pg_roles AS member
                   ON member.oid = membership.member
                 JOIN pg_catalog.pg_roles AS grantor
                   ON grantor.oid = membership.grantor
                WHERE member.rolname = SESSION_USER
                   OR granted.rolname = SESSION_USER
                   OR grantor.rolname = SESSION_USER
           ), forward_role_closure(roleid) AS (
               SELECT membership.roleid
                 FROM pg_catalog.pg_auth_members AS membership
                 JOIN bootstrap ON bootstrap.oid = membership.member
               UNION
               SELECT membership.roleid
                 FROM pg_catalog.pg_auth_members AS membership
                 JOIN forward_role_closure AS reachable
                   ON reachable.roleid = membership.member
           )
           SELECT (SELECT pg_catalog.count(*) = 1 FROM bootstrap)
              AND (SELECT pg_catalog.bool_and(
                       rolcanlogin AND rolinherit AND NOT rolsuper
                       AND rolcreatedb AND rolcreaterole
                       AND NOT rolreplication AND NOT rolbypassrls
                       AND rolconnlimit = -1 AND rolvaliduntil IS NULL
                       AND rolconfig IS NULL
                   ) FROM bootstrap)
              AND (SELECT pg_catalog.count(*) = 1
                          AND pg_catalog.bool_and(
                              granted_name = 'cloudsqlsuperuser'
                              AND member_name = SESSION_USER
                              AND grantor_name = 'cloudsqladmin'
                              AND admin_option IS FALSE
                              AND inherit_option IS TRUE
                              AND set_option IS TRUE
                          ) FROM relevant_edges)
              AND (SELECT pg_catalog.count(DISTINCT role.rolname) = 1
                          AND pg_catalog.bool_and(
                              role.rolname = 'cloudsqlsuperuser'
                          )
                     FROM forward_role_closure AS closure
                     JOIN pg_catalog.pg_roles AS role
                       ON role.oid = closure.roleid)
       )
       OR pg_catalog.current_setting(
              'max_prepared_transactions'
          )::integer <> 0
       OR EXISTS (SELECT 1 FROM pg_catalog.pg_prepared_xacts)
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_stat_activity AS activity
            WHERE activity.backend_type = 'client backend'
              AND activity.pid <> pg_catalog.pg_backend_pid()
              AND (
                  activity.datname = pg_catalog.current_database()
                  OR activity.usename = SESSION_USER
              )
       )
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.pg_roles AS bootstrap
             JOIN pg_catalog.pg_shdepend AS dependency
               ON dependency.refclassid =
                  'pg_catalog.pg_authid'::pg_catalog.regclass
              AND dependency.refobjid = bootstrap.oid
            WHERE bootstrap.rolname = SESSION_USER
       )
       OR NOT EXISTS (
           SELECT 1
             FROM pg_catalog.pg_database AS database
            WHERE database.datname = pg_catalog.current_database()
              AND pg_catalog.pg_get_userbyid(database.datdba)
                  = 'cloudsqlsuperuser'
       )
       OR NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_roles AS owner
            WHERE owner.rolname = 'canonical_brain_migration_owner'
              AND NOT owner.rolcanlogin
              AND NOT owner.rolinherit
              AND NOT owner.rolsuper
              AND NOT owner.rolcreatedb
              AND NOT owner.rolcreaterole
              AND NOT owner.rolreplication
              AND NOT owner.rolbypassrls
              AND owner.rolconnlimit = -1
              AND owner.rolvaliduntil IS NULL
              AND owner.rolconfig IS NULL
       )
       OR pg_catalog.to_regrole('canonical_brain_schema_reconciler') IS NOT NULL
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_namespace
            WHERE nspname = 'canonical_brain_reconciliation'
       )
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.pg_proc AS routine
             JOIN pg_catalog.pg_namespace AS namespace
               ON namespace.oid = routine.pronamespace
            WHERE namespace.nspname = 'canonical_brain'
              AND routine.proname = '_discord_guild_routeback_target_valid'
       )
    THEN
        RAISE EXCEPTION 'schema reconciliation control bootstrap preflight failed';
    END IF;
END
$control_bootstrap_preflight$;

CREATE ROLE canonical_brain_schema_reconciler
    NOLOGIN
    NOINHERIT
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOREPLICATION
    NOBYPASSRLS
    CONNECTION LIMIT -1
    PASSWORD NULL;

-- PostgreSQL 18 gives a CREATEROLE creator one implicit ADMIN membership.
-- Its grantor is the provider bootstrap role, so the member cannot revoke it.
-- The terminal check admits only that exact edge; the outer Cloud deletion
-- must prove that it disappeared before any normal reconciliation session.

SET LOCAL ROLE cloudsqlsuperuser;

-- The broad bootstrap identity starts without migration-owner membership.
-- The edge exists only inside this transaction and is revoked before COMMIT.
GRANT canonical_brain_migration_owner TO SESSION_USER
    WITH ADMIN FALSE, INHERIT FALSE, SET TRUE;
GRANT CONNECT ON DATABASE muncho_canary_brain
    TO canonical_brain_schema_reconciler;
CREATE SCHEMA canonical_brain_reconciliation
    AUTHORIZATION canonical_brain_migration_owner;

RESET ROLE;
SET LOCAL ROLE canonical_brain_migration_owner;

LOCK TABLE public.canonical_event_log,
    canonical_brain.writer_capability_consumptions,
    canonical_brain.writer_capability_grants,
    canonical_brain.writer_capability_revocation_scopes,
    canonical_brain.writer_capability_revocations,
    canonical_brain.writer_event_provenance,
    canonical_brain.writer_public_routeback_targets,
    canonical_brain.writer_routeback_authorizations,
    canonical_brain.writer_routeback_lifecycle_terminals,
    canonical_brain.writer_routeback_terminals
IN SHARE MODE;

REVOKE ALL ON SCHEMA canonical_brain_reconciliation FROM PUBLIC;
GRANT USAGE ON SCHEMA canonical_brain_reconciliation
    TO canonical_brain_schema_reconciler;

CREATE FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1()
RETURNS TABLE(
    row_count text,
    canonical14_sha256 text,
    relation_receipts jsonb,
    quarantine_anchor_receipts jsonb,
    observation_sha256 text
)
LANGUAGE plpgsql
VOLATILE
CALLED ON NULL INPUT
SECURITY DEFINER
PARALLEL UNSAFE
NOT LEAKPROOF
SET search_path = pg_catalog, pg_temp
SET TimeZone = 'UTC'
SET DateStyle = 'ISO, YMD'
SET IntervalStyle = 'postgres'
SET extra_float_digits = 3
SET bytea_output = 'hex'
SET lock_timeout = '15s'
SET statement_timeout = '5min'
AS $control_observer$
DECLARE
    observed_row_count text;
    observed_canonical14_sha256 text;
    observed_relation_receipts jsonb;
    observed_quarantine_anchor_receipts jsonb;
    observed_observation_sha256 text;
BEGIN
    -- This must remain the first database operation in the routine.
    PERFORM pg_catalog.pg_advisory_xact_lock(4841739663211427921);

    IF pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_setting('server_version_num')::integer / 10000 <> 18
       OR pg_catalog.current_setting('transaction_isolation') <> 'serializable'
       OR CURRENT_USER <> 'canonical_brain_migration_owner'
       OR SESSION_USER !~ '^muncho_canary_reconciler_[0-9a-f]{16}$'
    THEN
        RAISE EXCEPTION 'schema reconciliation control observer admission failed';
    END IF;

    LOCK TABLE public.canonical_event_log,
        canonical_brain.writer_capability_consumptions,
        canonical_brain.writer_capability_grants,
        canonical_brain.writer_capability_revocation_scopes,
        canonical_brain.writer_capability_revocations,
        canonical_brain.writer_event_provenance,
        canonical_brain.writer_public_routeback_targets,
        canonical_brain.writer_routeback_authorizations,
        canonical_brain.writer_routeback_lifecycle_terminals,
        canonical_brain.writer_routeback_terminals
    IN SHARE MODE;

    WITH relation_names(ordinal, relation_name) AS (
        VALUES
          (0, 'public.canonical_event_log'::text),
          (1, 'canonical_brain.writer_capability_consumptions'::text),
          (2, 'canonical_brain.writer_capability_grants'::text),
          (3, 'canonical_brain.writer_capability_revocation_scopes'::text),
          (4, 'canonical_brain.writer_capability_revocations'::text),
          (5, 'canonical_brain.writer_event_provenance'::text),
          (6, 'canonical_brain.writer_public_routeback_targets'::text),
          (7, 'canonical_brain.writer_routeback_authorizations'::text),
          (8, 'canonical_brain.writer_routeback_lifecycle_terminals'::text),
          (9, 'canonical_brain.writer_routeback_terminals'::text)
    ), canonical_rows AS (
        SELECT 0::integer AS ordinal,
               'public.canonical_event_log'::text AS relation_name,
               ((ordered.row_ordinal - 1) / 4096)::bigint AS chunk_ordinal,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (ORDER BY row_value.event_id)
                         AS row_ordinal,
                     pg_catalog.jsonb_build_array(row_value.event_id)::text
                         AS primary_key_json,
                     pg_catalog.to_jsonb(row_value)::text AS row_json
                FROM public.canonical_event_log AS row_value
          ) AS ordered
        UNION ALL
        SELECT 1, 'canonical_brain.writer_capability_consumptions',
               ((ordered.row_ordinal - 1) / 4096)::bigint,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (ORDER BY row_value.consume_id),
                     pg_catalog.jsonb_build_array(row_value.consume_id)::text,
                     pg_catalog.to_jsonb(row_value)::text
                FROM canonical_brain.writer_capability_consumptions AS row_value
          ) AS ordered(row_ordinal, primary_key_json, row_json)
        UNION ALL
        SELECT 2, 'canonical_brain.writer_capability_grants',
               ((ordered.row_ordinal - 1) / 4096)::bigint,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (
                         ORDER BY row_value.approval_id COLLATE "C"
                     ), pg_catalog.jsonb_build_array(row_value.approval_id)::text,
                     pg_catalog.to_jsonb(row_value)::text
                FROM canonical_brain.writer_capability_grants AS row_value
          ) AS ordered(row_ordinal, primary_key_json, row_json)
        UNION ALL
        SELECT 3, 'canonical_brain.writer_capability_revocation_scopes',
               ((ordered.row_ordinal - 1) / 4096)::bigint,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (ORDER BY
                         row_value.scope_type COLLATE "C",
                         row_value.session_key_sha256 COLLATE "C",
                         row_value.capability_epoch_sha256 COLLATE "C",
                         row_value.plan_id COLLATE "C"),
                     pg_catalog.jsonb_build_array(
                         row_value.scope_type,
                         row_value.session_key_sha256,
                         row_value.capability_epoch_sha256,
                         row_value.plan_id
                     )::text,
                     pg_catalog.to_jsonb(row_value)::text
                FROM canonical_brain.writer_capability_revocation_scopes
                     AS row_value
          ) AS ordered(row_ordinal, primary_key_json, row_json)
        UNION ALL
        SELECT 4, 'canonical_brain.writer_capability_revocations',
               ((ordered.row_ordinal - 1) / 4096)::bigint,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (
                         ORDER BY row_value.approval_id COLLATE "C"
                     ), pg_catalog.jsonb_build_array(row_value.approval_id)::text,
                     pg_catalog.to_jsonb(row_value)::text
                FROM canonical_brain.writer_capability_revocations AS row_value
          ) AS ordered(row_ordinal, primary_key_json, row_json)
        UNION ALL
        SELECT 5, 'canonical_brain.writer_event_provenance',
               ((ordered.row_ordinal - 1) / 4096)::bigint,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (ORDER BY row_value.event_id),
                     pg_catalog.jsonb_build_array(row_value.event_id)::text,
                     pg_catalog.to_jsonb(row_value)::text
                FROM canonical_brain.writer_event_provenance AS row_value
          ) AS ordered(row_ordinal, primary_key_json, row_json)
        UNION ALL
        SELECT 6, 'canonical_brain.writer_public_routeback_targets',
               ((ordered.row_ordinal - 1) / 4096)::bigint,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (
                         ORDER BY row_value.channel_id COLLATE "C"
                     ), pg_catalog.jsonb_build_array(row_value.channel_id)::text,
                     pg_catalog.to_jsonb(row_value)::text
                FROM canonical_brain.writer_public_routeback_targets AS row_value
          ) AS ordered(row_ordinal, primary_key_json, row_json)
        UNION ALL
        SELECT 7, 'canonical_brain.writer_routeback_authorizations',
               ((ordered.row_ordinal - 1) / 4096)::bigint,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (
                         ORDER BY row_value.authorization_id COLLATE "C"
                     ),
                     pg_catalog.jsonb_build_array(row_value.authorization_id)::text,
                     pg_catalog.to_jsonb(row_value)::text
                FROM canonical_brain.writer_routeback_authorizations AS row_value
          ) AS ordered(row_ordinal, primary_key_json, row_json)
        UNION ALL
        SELECT 8, 'canonical_brain.writer_routeback_lifecycle_terminals',
               ((ordered.row_ordinal - 1) / 4096)::bigint,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (
                         ORDER BY row_value.lifecycle_id COLLATE "C"
                     ), pg_catalog.jsonb_build_array(row_value.lifecycle_id)::text,
                     pg_catalog.to_jsonb(row_value)::text
                FROM canonical_brain.writer_routeback_lifecycle_terminals AS row_value
          ) AS ordered(row_ordinal, primary_key_json, row_json)
        UNION ALL
        SELECT 9, 'canonical_brain.writer_routeback_terminals',
               ((ordered.row_ordinal - 1) / 4096)::bigint,
               ordered.row_ordinal, ordered.primary_key_json, ordered.row_json
          FROM (
              SELECT pg_catalog.row_number() OVER (
                         ORDER BY row_value.authorization_id COLLATE "C"
                     ),
                     pg_catalog.jsonb_build_array(row_value.authorization_id)::text,
                     pg_catalog.to_jsonb(row_value)::text
                FROM canonical_brain.writer_routeback_terminals AS row_value
          ) AS ordered(row_ordinal, primary_key_json, row_json)
    ), row_receipts AS (
        SELECT ordinal, relation_name, chunk_ordinal, row_ordinal,
               primary_key_json,
               pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   relation_name || E'\n' || primary_key_json || E'\n' || row_json,
                   'UTF8'
               )), 'hex') AS row_sha
          FROM canonical_rows
    ), chunk_receipts AS (
        SELECT ordinal, relation_name, chunk_ordinal,
               pg_catalog.count(*) AS chunk_row_count,
               pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   'canonical-writer-schema-reconcile-v2:chunk:'
                   || relation_name || ':' || chunk_ordinal::text || E'\n'
                   || pg_catalog.string_agg(
                       primary_key_json || ':' || row_sha,
                       E'\n' ORDER BY row_ordinal
                   ), 'UTF8'
               )), 'hex') AS chunk_sha256
          FROM row_receipts
         GROUP BY ordinal, relation_name, chunk_ordinal
    ), receipts AS (
        SELECT names.ordinal, names.relation_name,
               COALESCE(pg_catalog.sum(chunks.chunk_row_count), 0)::bigint
                   AS receipt_row_count,
               pg_catalog.count(chunks.chunk_ordinal)::integer AS chunk_count,
               pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   'canonical-writer-schema-reconcile-v2:relation:'
                   || names.relation_name || E'\n'
                   || COALESCE(pg_catalog.string_agg(
                       chunks.chunk_ordinal::text || ':'
                       || chunks.chunk_row_count::text || ':'
                       || chunks.chunk_sha256,
                       E'\n' ORDER BY chunks.chunk_ordinal
                   ), ''), 'UTF8'
               )), 'hex') AS chunk_manifest_sha256
          FROM relation_names AS names
          LEFT JOIN chunk_receipts AS chunks
            ON chunks.ordinal = names.ordinal
           AND chunks.relation_name = names.relation_name
         GROUP BY names.ordinal, names.relation_name
    ), event_row_receipts AS (
        SELECT event.event_id,
               pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   pg_catalog.jsonb_build_object(
                       'event_id', pg_catalog.to_jsonb(event)->'event_id',
                       'schema_version', pg_catalog.to_jsonb(event)->'schema_version',
                       'event_type', pg_catalog.to_jsonb(event)->'event_type',
                       'occurred_at', pg_catalog.to_jsonb(event)->'occurred_at',
                       'case_id', pg_catalog.to_jsonb(event)->'case_id',
                       'source', pg_catalog.to_jsonb(event)->'source',
                       'actor', pg_catalog.to_jsonb(event)->'actor',
                       'subject', pg_catalog.to_jsonb(event)->'subject',
                       'evidence', pg_catalog.to_jsonb(event)->'evidence',
                       'decision', pg_catalog.to_jsonb(event)->'decision',
                       'status', pg_catalog.to_jsonb(event)->'status',
                       'next_action', pg_catalog.to_jsonb(event)->'next_action',
                       'safety', pg_catalog.to_jsonb(event)->'safety',
                       'payload', pg_catalog.to_jsonb(event)->'payload'
                   )::text, 'UTF8'
               )), 'hex') AS row_sha
          FROM public.canonical_event_log AS event
    ), event_receipt AS (
        SELECT pg_catalog.count(*)::text AS event_row_count,
               pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   'canonical-writer-legacy-reconcile-v1:canonical14' || E'\n'
                   || COALESCE(pg_catalog.string_agg(
                       event_id::text || ':' || row_sha,
                       E'\n' ORDER BY event_id
                   ), ''), 'UTF8'
               )), 'hex') AS event_sha256
          FROM event_row_receipts
    )
    SELECT event.event_row_count, event.event_sha256,
           pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
               'relation', receipts.relation_name,
               'row_count', receipts.receipt_row_count,
               'chunk_count', receipts.chunk_count,
               'chunk_manifest_sha256', receipts.chunk_manifest_sha256
           ) ORDER BY receipts.ordinal)
      INTO STRICT observed_row_count, observed_canonical14_sha256,
                  observed_relation_receipts
      FROM event_receipt AS event CROSS JOIN receipts
     GROUP BY event.event_row_count, event.event_sha256;

    WITH quarantine_namespace AS (
        SELECT namespace.oid, namespace.nspowner,
               pg_catalog.pg_get_userbyid(namespace.nspowner) AS owner_name
          FROM pg_catalog.pg_namespace AS namespace
         WHERE namespace.nspname = 'canonical_brain_legacy_quarantine'
    ), quarantine_objects AS (
        SELECT class.oid, class.relname, class.relowner, class.relkind,
               class.relpersistence, class.relispartition, class.reltablespace,
               class.relrowsecurity, class.relforcerowsecurity, class.reloptions,
               pg_catalog.pg_get_userbyid(class.relowner) AS owner_name
          FROM pg_catalog.pg_class AS class
          JOIN quarantine_namespace AS namespace
            ON namespace.oid = class.relnamespace
         WHERE class.relkind IN ('r', 'p', 'v', 'm', 'f', 'S', 'c')
    ), quarantine_relations AS (
        SELECT * FROM quarantine_objects
         WHERE relkind = 'r'
           AND relname IN (
               'canonical_event_log_legacy_v1', 'reconciliation_receipts'
           )
    ), schema_acl AS (
        SELECT acl.grantor, acl.grantee, acl.privilege_type, acl.is_grantable
          FROM quarantine_namespace AS namespace
          CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
              (SELECT nspacl FROM pg_catalog.pg_namespace
                WHERE oid = namespace.oid),
              pg_catalog.acldefault('n', namespace.nspowner)
          )) AS acl
    ), table_acl AS (
        SELECT relation.relname, acl.grantor, acl.grantee,
               acl.privilege_type, acl.is_grantable
          FROM quarantine_relations AS relation
          CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
              (SELECT relacl FROM pg_catalog.pg_class
                WHERE oid = relation.oid),
              pg_catalog.acldefault('r', relation.relowner)
          )) AS acl
    ), schema_acl_digest AS (
        SELECT pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   'canonical-writer-schema-reconcile-v3:quarantine-schema-acl'
                   || E'\n' || COALESCE(pg_catalog.string_agg(
                       acl.grantor::text || ':' || acl.grantee::text || ':'
                       || acl.privilege_type || ':' || acl.is_grantable::text,
                       E'\n' ORDER BY acl.grantor, acl.grantee,
                       acl.privilege_type, acl.is_grantable
                   ), ''), 'UTF8'
               )), 'hex') AS acl_sha256
          FROM schema_acl AS acl
    ), table_acl_digest AS (
        SELECT relation.relname,
               pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   'canonical-writer-schema-reconcile-v3:quarantine-table-acl:'
                   || relation.relname || E'\n'
                   || COALESCE(pg_catalog.string_agg(
                       acl.grantor::text || ':' || acl.grantee::text || ':'
                       || acl.privilege_type || ':' || acl.is_grantable::text,
                       E'\n' ORDER BY acl.grantor, acl.grantee,
                       acl.privilege_type, acl.is_grantable
                   ), ''), 'UTF8'
               )), 'hex') AS acl_sha256
          FROM quarantine_relations AS relation
          LEFT JOIN table_acl AS acl USING (relname)
         GROUP BY relation.relname
    ), anchors AS (
        SELECT 0::integer AS ordinal,
               'schema:canonical_brain_legacy_quarantine:postgres:owner-only'
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
               END,
               CASE relation.relname
                   WHEN 'canonical_event_log_legacy_v1' THEN
                       'table:canonical_brain_legacy_quarantine.'
                       'canonical_event_log_legacy_v1:postgres:r:p:owner-only'
                   ELSE
                       'table:canonical_brain_legacy_quarantine.'
                       'reconciliation_receipts:postgres:r:p:owner-only'
               END,
               relation.oid::bigint, relation.owner_name,
               relation.relkind::text, relation.relpersistence::text,
               digest.acl_sha256
          FROM quarantine_relations AS relation
          JOIN table_acl_digest AS digest USING (relname)
    )
    SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
               'anchor', anchors.anchor,
               'object_oid', anchors.object_oid,
               'owner', anchors.owner,
               'kind', anchors.kind,
               'persistence', anchors.persistence,
               'acl_sha256', anchors.acl_sha256
           ) ORDER BY anchors.ordinal)
      INTO STRICT observed_quarantine_anchor_receipts
      FROM anchors;

    IF pg_catalog.jsonb_array_length(observed_relation_receipts) <> 10
       OR pg_catalog.jsonb_array_length(observed_quarantine_anchor_receipts) <> 3
       OR observed_quarantine_anchor_receipts->0->>'owner' <> 'postgres'
       OR observed_quarantine_anchor_receipts->1->>'owner' <> 'postgres'
       OR observed_quarantine_anchor_receipts->2->>'owner' <> 'postgres'
       OR observed_quarantine_anchor_receipts->0->>'kind' <> 'n'
       OR observed_quarantine_anchor_receipts->1->>'kind' <> 'r'
       OR observed_quarantine_anchor_receipts->2->>'kind' <> 'r'
       OR observed_quarantine_anchor_receipts->1->>'persistence' <> 'p'
       OR observed_quarantine_anchor_receipts->2->>'persistence' <> 'p'
    THEN
        RAISE EXCEPTION 'schema reconciliation quarantine anchor drifted';
    END IF;

    observed_observation_sha256 := pg_catalog.encode(pg_catalog.sha256(
        pg_catalog.convert_to(
            'canonical-writer-schema-reconciliation-control-observation-v1'
            || E'\n' || observed_row_count
            || E'\n' || observed_canonical14_sha256
            || E'\n' || observed_relation_receipts::text
            || E'\n' || observed_quarantine_anchor_receipts::text,
            'UTF8'
        )
    ), 'hex');

    PERFORM pg_catalog.set_config(
        'muncho.schema_reconciliation_control_observation_sha256',
        observed_observation_sha256,
        true
    );

    RETURN QUERY SELECT observed_row_count,
                        observed_canonical14_sha256,
                        observed_relation_receipts,
                        observed_quarantine_anchor_receipts,
                        observed_observation_sha256;
END
$control_observer$;

CREATE FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1()
RETURNS TABLE(
    applied boolean,
    plan_sha256 text,
    authorized_intent_sha256 text,
    canonical_truth_receipt_sha256 text,
    observation_sha256 text,
    helper_definition_sha256 text,
    receipt_sha256 text
)
LANGUAGE plpgsql
VOLATILE
CALLED ON NULL INPUT
SECURITY DEFINER
PARALLEL UNSAFE
NOT LEAKPROOF
SET search_path = pg_catalog, pg_temp
SET TimeZone = 'UTC'
SET DateStyle = 'ISO, YMD'
SET IntervalStyle = 'postgres'
SET extra_float_digits = 3
SET bytea_output = 'hex'
SET lock_timeout = '15s'
SET statement_timeout = '5min'
AS $control_apply$
DECLARE
    before_observation record;
    after_observation record;
    bound_plan_sha256 text;
    bound_authorized_intent_sha256 text;
    bound_truth_receipt_sha256 text;
    bound_observation_sha256 text;
    observed_helper_definition_sha256 text;
    helper_is_exact boolean := false;
    helper_name_count bigint := 0;
    did_apply boolean := false;
    observed_receipt_sha256 text;
BEGIN
    -- This must remain the first database operation in the routine.
    PERFORM pg_catalog.pg_advisory_xact_lock(4841739663211427921);

    IF pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_setting('server_version_num')::integer / 10000 <> 18
       OR pg_catalog.current_setting('transaction_isolation') <> 'serializable'
       OR CURRENT_USER <> 'canonical_brain_migration_owner'
       OR SESSION_USER !~ '^muncho_canary_reconciler_[0-9a-f]{16}$'
    THEN
        RAISE EXCEPTION 'schema reconciliation fixed apply admission failed';
    END IF;

    SELECT * INTO STRICT before_observation
      FROM canonical_brain_reconciliation.
           observe_missing_discord_routeback_helper_v1();

    bound_plan_sha256 := pg_catalog.current_setting(
        'muncho.schema_reconciliation_plan_sha256', true
    );
    bound_authorized_intent_sha256 := pg_catalog.current_setting(
        'muncho.schema_reconciliation_authorized_intent_sha256', true
    );
    bound_truth_receipt_sha256 := pg_catalog.current_setting(
        'muncho.schema_reconciliation_truth_receipt_sha256', true
    );
    bound_observation_sha256 := pg_catalog.current_setting(
        'muncho.schema_reconciliation_control_observation_sha256', true
    );

    IF bound_plan_sha256 !~ '^[0-9a-f]{64}$'
       OR bound_authorized_intent_sha256 !~ '^[0-9a-f]{64}$'
       OR bound_truth_receipt_sha256 !~ '^[0-9a-f]{64}$'
       OR bound_observation_sha256 !~ '^[0-9a-f]{64}$'
       OR bound_observation_sha256 <> before_observation.observation_sha256
    THEN
        RAISE EXCEPTION 'schema reconciliation fixed apply binding failed';
    END IF;

    SELECT pg_catalog.count(*) INTO STRICT helper_name_count
      FROM pg_catalog.pg_proc AS helper
      JOIN pg_catalog.pg_namespace AS helper_namespace
        ON helper_namespace.oid = helper.pronamespace
     WHERE helper_namespace.nspname = 'canonical_brain'
       AND helper.proname = '_discord_guild_routeback_target_valid';

    IF helper_name_count > 1
       OR (
           helper_name_count = 1
           AND pg_catalog.to_regprocedure(
               'canonical_brain._discord_guild_routeback_target_valid(jsonb)'
           ) IS NULL
       )
    THEN
        RAISE EXCEPTION 'schema reconciliation target helper overload drifted';
    END IF;

    IF pg_catalog.to_regprocedure(
        'canonical_brain._discord_guild_routeback_target_valid(jsonb)'
    ) IS NOT NULL THEN
        SELECT pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   pg_catalog.pg_get_functiondef(routine.oid), 'UTF8'
               )), 'hex'),
               owner.rolname = 'canonical_brain_migration_owner'
               AND namespace.nspname = 'canonical_brain'
               AND routine.prokind = 'f'
               AND routine.prosecdef IS FALSE
               AND routine.provolatile = 'i'
               AND routine.proparallel = 'u'
               AND routine.proleakproof IS FALSE
               AND routine.proisstrict IS FALSE
               AND routine.proretset IS FALSE
               AND language.lanname = 'sql'
               AND pg_catalog.oidvectortypes(routine.proargtypes) = 'jsonb'
               AND pg_catalog.format_type(routine.prorettype, NULL) = 'boolean'
               AND routine.proconfig = ARRAY[
                   'search_path=pg_catalog, canonical_brain'
               ]::text[]
               AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   routine.prosrc, 'UTF8'
               )), 'hex') =
                   'e82ee5b2240d61c1e7c60d76ec87729d9d87e134d4b2083d5cd7b447f5ef093c'
               AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   pg_catalog.pg_get_functiondef(routine.oid), 'UTF8'
               )), 'hex') =
                   '2fc8e5334784ae791398033c8b460ccb170324f4c8cc1b3d0387f907e9cf7737'
               AND (
                   SELECT pg_catalog.count(*) = 1
                          AND pg_catalog.bool_and(
                              acl.grantee = routine.proowner
                              AND acl.grantor = routine.proowner
                              AND acl.privilege_type = 'EXECUTE'
                              AND acl.is_grantable IS FALSE
                          )
                     FROM pg_catalog.aclexplode(COALESCE(
                         routine.proacl,
                         pg_catalog.acldefault('f', routine.proowner)
                     )) AS acl
               )
          INTO STRICT observed_helper_definition_sha256, helper_is_exact
          FROM pg_catalog.pg_proc AS routine
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = routine.pronamespace
          JOIN pg_catalog.pg_roles AS owner ON owner.oid = routine.proowner
          JOIN pg_catalog.pg_language AS language ON language.oid = routine.prolang
         WHERE routine.oid = pg_catalog.to_regprocedure(
             'canonical_brain._discord_guild_routeback_target_valid(jsonb)'
         );
        IF NOT helper_is_exact THEN
            RAISE EXCEPTION 'schema reconciliation target helper drifted';
        END IF;
    ELSE
        EXECUTE $fixed_helper_definition$
CREATE FUNCTION canonical_brain._discord_guild_routeback_target_valid(
    value jsonb
)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
SET search_path = pg_catalog, canonical_brain
AS $function$
    WITH approved_roots(channel_id) AS (
        VALUES
          ('1504852355588423801'::text),
          ('1510888721614901358'::text),
          ('1504852408227069993'::text),
          ('1504852444407140402'::text),
          ('1504852485083496561'::text),
          ('1504852553031221391'::text),
          ('1504852628373373028'::text),
          ('1505499746939174993'::text),
          ('1507239177350283274'::text),
          ('1507239385010016308'::text)
    )
    SELECT COALESCE(
        pg_catalog.jsonb_typeof(value) = 'object'
        AND value->>'guild_id' = '1282725267068157972'
        AND value->>'channel_type' = value->>'target_type'
        AND COALESCE(value->>'channel_id', '') ~ '^[1-9][0-9]{16,19}$'
        AND COALESCE(value->>'thread_id', '') = ''
        AND COALESCE(value->>'chat_id', '') = ''
        AND NOT canonical_brain._contains_forbidden_dm_ref(value)
        AND CASE value->>'target_type'
                WHEN 'guild_channel' THEN
                    (
                        NOT (value ? 'parent_channel_id')
                        OR COALESCE(value->>'parent_channel_id', '') = ''
                    )
                    AND EXISTS (
                        SELECT 1 FROM approved_roots AS root
                         WHERE root.channel_id = value->>'channel_id'
                    )
                WHEN 'guild_thread' THEN
                    COALESCE(value->>'parent_channel_id', '')
                        ~ '^[1-9][0-9]{16,19}$'
                    AND value->>'parent_channel_id'
                        IS DISTINCT FROM value->>'channel_id'
                    AND EXISTS (
                        SELECT 1 FROM approved_roots AS root
                         WHERE root.channel_id = value->>'parent_channel_id'
                    )
                WHEN 'public_guild_channel' THEN
                    value->>'channel_id' = '1526858760100909066'
                    AND (
                        NOT (value ? 'parent_channel_id')
                        OR COALESCE(value->>'parent_channel_id', '') = ''
                    )
                ELSE false
            END,
        false
    )
$function$;
$fixed_helper_definition$;
        REVOKE ALL PRIVILEGES ON FUNCTION
            canonical_brain._discord_guild_routeback_target_valid(jsonb)
            FROM PUBLIC;
        did_apply := true;

        SELECT pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   pg_catalog.pg_get_functiondef(routine.oid), 'UTF8'
               )), 'hex')
          INTO STRICT observed_helper_definition_sha256
          FROM pg_catalog.pg_proc AS routine
         WHERE routine.oid = pg_catalog.to_regprocedure(
             'canonical_brain._discord_guild_routeback_target_valid(jsonb)'
         )
           AND routine.proowner = (
               SELECT oid FROM pg_catalog.pg_roles
                WHERE rolname = 'canonical_brain_migration_owner'
           )
           AND routine.prosrc IS NOT NULL
           AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               routine.prosrc, 'UTF8'
           )), 'hex') =
               'e82ee5b2240d61c1e7c60d76ec87729d9d87e134d4b2083d5cd7b447f5ef093c'
           AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               pg_catalog.pg_get_functiondef(routine.oid), 'UTF8'
           )), 'hex') =
               '2fc8e5334784ae791398033c8b460ccb170324f4c8cc1b3d0387f907e9cf7737'
           AND (
               SELECT pg_catalog.count(*) = 1
                      AND pg_catalog.bool_and(
                          acl.grantee = routine.proowner
                          AND acl.grantor = routine.proowner
                          AND acl.privilege_type = 'EXECUTE'
                          AND acl.is_grantable IS FALSE
                      )
                 FROM pg_catalog.aclexplode(COALESCE(
                     routine.proacl,
                     pg_catalog.acldefault('f', routine.proowner)
                 )) AS acl
           );
    END IF;

    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_proc AS helper
          JOIN pg_catalog.pg_namespace AS helper_namespace
            ON helper_namespace.oid = helper.pronamespace
         WHERE helper_namespace.nspname = 'canonical_brain'
           AND helper.proname = '_discord_guild_routeback_target_valid'
    ) <> 1 THEN
        RAISE EXCEPTION 'schema reconciliation target helper inventory drifted';
    END IF;

    SELECT * INTO STRICT after_observation
      FROM canonical_brain_reconciliation.
           observe_missing_discord_routeback_helper_v1();

    IF after_observation.row_count <> before_observation.row_count
       OR after_observation.canonical14_sha256
          <> before_observation.canonical14_sha256
       OR after_observation.relation_receipts
          <> before_observation.relation_receipts
       OR after_observation.quarantine_anchor_receipts
          <> before_observation.quarantine_anchor_receipts
       OR after_observation.observation_sha256
          <> before_observation.observation_sha256
       OR observed_helper_definition_sha256 <>
          '2fc8e5334784ae791398033c8b460ccb170324f4c8cc1b3d0387f907e9cf7737'
    THEN
        RAISE EXCEPTION 'schema reconciliation fixed apply terminal failed';
    END IF;

    observed_receipt_sha256 := pg_catalog.encode(pg_catalog.sha256(
        pg_catalog.convert_to(
            'canonical-writer-schema-reconciliation-control-apply-v1'
            || E'\n' || did_apply::text
            || E'\n' || bound_plan_sha256
            || E'\n' || bound_authorized_intent_sha256
            || E'\n' || bound_truth_receipt_sha256
            || E'\n' || before_observation.observation_sha256
            || E'\n' || observed_helper_definition_sha256,
            'UTF8'
        )
    ), 'hex');

    RETURN QUERY SELECT did_apply,
                        bound_plan_sha256,
                        bound_authorized_intent_sha256,
                        bound_truth_receipt_sha256,
                        before_observation.observation_sha256,
                        observed_helper_definition_sha256,
                        observed_receipt_sha256;
END
$control_apply$;

REVOKE ALL PRIVILEGES ON FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1() FROM PUBLIC;
REVOKE ALL PRIVILEGES ON FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1()
    TO canonical_brain_schema_reconciler;
GRANT EXECUTE ON FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1()
    TO canonical_brain_schema_reconciler;

RESET ROLE;
SET LOCAL ROLE cloudsqlsuperuser;
REVOKE canonical_brain_migration_owner FROM SESSION_USER;
RESET ROLE;

DO $control_bootstrap_terminal$
DECLARE
    database_oid oid := (
        SELECT oid FROM pg_catalog.pg_database
         WHERE datname = pg_catalog.current_database()
    );
    executor_oid oid := pg_catalog.to_regrole(
        'canonical_brain_schema_reconciler'
    );
    control_namespace_oid oid;
    observer_oid oid;
    apply_oid oid;
    managed_cloudsqladmin_database_exact boolean;
BEGIN
    SELECT oid INTO STRICT control_namespace_oid
      FROM pg_catalog.pg_namespace
     WHERE nspname = 'canonical_brain_reconciliation';
    SELECT oid INTO STRICT observer_oid
      FROM pg_catalog.pg_proc
     WHERE pronamespace = control_namespace_oid
       AND proname = 'observe_missing_discord_routeback_helper_v1'
       AND pronargs = 0;
    SELECT oid INTO STRICT apply_oid
      FROM pg_catalog.pg_proc
     WHERE pronamespace = control_namespace_oid
       AND proname = 'apply_missing_discord_routeback_helper_v1'
       AND pronargs = 0;
    WITH managed_database AS (
        SELECT database.*,
               pg_catalog.pg_get_userbyid(database.datdba) AS owner_name
          FROM pg_catalog.pg_database AS database
         WHERE database.datname = 'cloudsqladmin'
    ), actual_acl AS (
        SELECT CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                    ELSE pg_catalog.pg_get_userbyid(acl.grantee) END
                   AS grantee,
               pg_catalog.pg_get_userbyid(acl.grantor) AS grantor,
               acl.privilege_type,
               acl.is_grantable
          FROM managed_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
              database.datacl,
              pg_catalog.acldefault('d', database.datdba)
          )) AS acl
    ), expected_acl(grantee, grantor, privilege_type, is_grantable) AS (
        VALUES
          ('PUBLIC'::text, 'cloudsqladmin'::text, 'CONNECT'::text, false),
          ('PUBLIC'::text, 'cloudsqladmin'::text, 'TEMPORARY'::text, false),
          ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'CREATE'::text, false),
          ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'CONNECT'::text, false),
          ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'TEMPORARY'::text, false)
    )
    SELECT (SELECT pg_catalog.count(*) = 1 AND COALESCE(
                       pg_catalog.bool_and(
                           datallowconn AND NOT datistemplate
                           AND owner_name = 'cloudsqladmin'
                       ), false
                   ) FROM managed_database)
       AND NOT EXISTS (
            (SELECT * FROM actual_acl EXCEPT SELECT * FROM expected_acl)
            UNION ALL
            (SELECT * FROM expected_acl EXCEPT SELECT * FROM actual_acl)
       )
      INTO STRICT managed_cloudsqladmin_database_exact;

    IF CURRENT_USER <> SESSION_USER
       OR executor_oid IS NULL
       OR database_oid IS NULL
       OR observer_oid IS NULL
       OR apply_oid IS NULL
       OR NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_roles AS role
            WHERE role.oid = executor_oid
              AND NOT role.rolcanlogin
              AND NOT role.rolinherit
              AND NOT role.rolsuper
              AND NOT role.rolcreatedb
              AND NOT role.rolcreaterole
              AND NOT role.rolreplication
              AND NOT role.rolbypassrls
              AND role.rolconnlimit = -1
              AND role.rolvaliduntil IS NULL
              AND role.rolconfig IS NULL
       )
       OR NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_roles AS owner
            WHERE owner.rolname = 'canonical_brain_migration_owner'
              AND NOT owner.rolcanlogin
              AND NOT owner.rolinherit
              AND NOT owner.rolsuper
              AND NOT owner.rolcreatedb
              AND NOT owner.rolcreaterole
              AND NOT owner.rolreplication
              AND NOT owner.rolbypassrls
              AND owner.rolconnlimit = -1
              AND owner.rolvaliduntil IS NULL
              AND owner.rolconfig IS NULL
       )
       OR pg_catalog.pg_get_userbyid((
           SELECT datdba FROM pg_catalog.pg_database
            WHERE oid = database_oid
       )) <> 'cloudsqlsuperuser'
       OR NOT (
           SELECT pg_catalog.count(*) = 1
                  AND pg_catalog.bool_and(
                      member.rolname = SESSION_USER
                      AND grantor.rolname = 'cloudsqladmin'
                      AND membership.admin_option IS TRUE
                      AND membership.inherit_option IS FALSE
                      AND membership.set_option IS FALSE
                  )
             FROM pg_catalog.pg_auth_members AS membership
             JOIN pg_catalog.pg_roles AS member
               ON member.oid = membership.member
             JOIN pg_catalog.pg_roles AS grantor
               ON grantor.oid = membership.grantor
            WHERE membership.roleid = executor_oid
       )
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_auth_members
            WHERE member = executor_oid OR grantor = executor_oid
       )
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.pg_auth_members AS membership
             JOIN pg_catalog.pg_roles AS owner
               ON owner.oid = membership.roleid
            WHERE owner.rolname = 'canonical_brain_migration_owner'
              AND membership.member = (
                  SELECT oid FROM pg_catalog.pg_roles
                   WHERE rolname = SESSION_USER
              )
       )
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_shdepend
            WHERE refclassid = 'pg_catalog.pg_authid'::pg_catalog.regclass
              AND refobjid = executor_oid
              AND deptype = 'o'
       )
       OR NOT (
           SELECT pg_catalog.count(*) = 4
                  AND pg_catalog.bool_and(
                      dependency.objsubid = 0
                      AND (
                          (dependency.dbid = 0
                           AND dependency.classid =
                               'pg_catalog.pg_database'::pg_catalog.regclass
                           AND dependency.objid = database_oid)
                          OR
                          (dependency.dbid = database_oid
                           AND dependency.classid =
                               'pg_catalog.pg_namespace'::pg_catalog.regclass
                           AND dependency.objid = control_namespace_oid)
                          OR
                          (dependency.dbid = database_oid
                           AND dependency.classid =
                               'pg_catalog.pg_proc'::pg_catalog.regclass
                           AND dependency.objid IN (observer_oid, apply_oid))
                      )
                  )
             FROM pg_catalog.pg_shdepend AS dependency
            WHERE dependency.refclassid =
                  'pg_catalog.pg_authid'::pg_catalog.regclass
              AND dependency.refobjid = executor_oid
              AND dependency.deptype = 'a'
       )
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_shdepend AS dependency
            WHERE dependency.refclassid =
                  'pg_catalog.pg_authid'::pg_catalog.regclass
              AND dependency.refobjid = executor_oid
              AND dependency.deptype <> 'a'
       )
       OR pg_catalog.pg_get_userbyid((
           SELECT nspowner FROM pg_catalog.pg_namespace
            WHERE oid = control_namespace_oid
       )) <> 'canonical_brain_migration_owner'
       OR pg_catalog.has_schema_privilege(
           'canonical_brain_schema_reconciler',
           control_namespace_oid, 'CREATE'
       )
       OR NOT pg_catalog.has_schema_privilege(
           'canonical_brain_schema_reconciler',
           control_namespace_oid, 'USAGE'
       )
       OR NOT pg_catalog.has_database_privilege(
           'canonical_brain_schema_reconciler',
           'muncho_canary_brain', 'CONNECT'
       )
       OR pg_catalog.has_database_privilege(
           'canonical_brain_schema_reconciler',
           'muncho_canary_brain', 'CREATE'
       )
       OR pg_catalog.has_database_privilege(
           'canonical_brain_schema_reconciler',
           'muncho_canary_brain', 'TEMPORARY'
       )
       OR NOT (
           SELECT pg_catalog.count(*) = 1
                  AND pg_catalog.bool_and(
                      acl.grantee = executor_oid
                      AND acl.grantor = database.datdba
                      AND acl.privilege_type = 'CONNECT'
                      AND acl.is_grantable IS FALSE
                  )
             FROM pg_catalog.pg_database AS database,
                  LATERAL pg_catalog.aclexplode(COALESCE(
                      database.datacl,
                      pg_catalog.acldefault('d', database.datdba)
                  )) AS acl
            WHERE database.oid = database_oid
              AND acl.grantee = executor_oid
       )
       OR NOT (
           SELECT pg_catalog.count(*) = 3
                  AND COALESCE(pg_catalog.string_agg(
                      database.datname::text,
                      ',' ORDER BY database.datname::text
                  ), '') = 'cloudsqladmin,muncho_canary_brain,postgres'
             FROM pg_catalog.pg_database AS database
            WHERE database.datallowconn AND NOT database.datistemplate
       )
       OR NOT (
           SELECT pg_catalog.count(*) = 4
                  AND COALESCE(pg_catalog.string_agg(
                      database.datname::text,
                      ',' ORDER BY database.datname::text
                  ), '') =
                      'cloudsqladmin,muncho_canary_brain,postgres,template1'
             FROM pg_catalog.pg_database AS database
            WHERE database.datallowconn
       )
       OR NOT managed_cloudsqladmin_database_exact
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.pg_database AS database
            WHERE database.datallowconn
              AND database.datname <> pg_catalog.current_database()
              AND (
                  pg_catalog.pg_get_userbyid(database.datdba) =
                      'canonical_brain_schema_reconciler'
                  OR pg_catalog.has_database_privilege(
                      executor_oid, database.oid, 'CONNECT'
                  )
                  OR pg_catalog.has_database_privilege(
                      executor_oid, database.oid, 'CREATE'
                  )
                  OR pg_catalog.has_database_privilege(
                      executor_oid, database.oid, 'TEMPORARY'
                  )
              )
              AND NOT (
                  database.datname = 'cloudsqladmin'
                  AND managed_cloudsqladmin_database_exact
                  AND pg_catalog.pg_get_userbyid(database.datdba) <>
                      'canonical_brain_schema_reconciler'
                  AND pg_catalog.has_database_privilege(
                      executor_oid, database.oid, 'CONNECT'
                  )
                  AND NOT pg_catalog.has_database_privilege(
                      executor_oid, database.oid, 'CREATE'
                  )
                  AND pg_catalog.has_database_privilege(
                      executor_oid, database.oid, 'TEMPORARY'
                  )
              )
       )
       OR (
           SELECT pg_catalog.count(*) FROM pg_catalog.pg_proc
            WHERE pronamespace = control_namespace_oid
       ) <> 2
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.pg_proc AS routine
             JOIN pg_catalog.pg_language AS language
               ON language.oid = routine.prolang
            WHERE routine.pronamespace = control_namespace_oid
              AND NOT (
                  pg_catalog.pg_get_userbyid(routine.proowner)
                      = 'canonical_brain_migration_owner'
                  AND routine.pronargs = 0
                  AND routine.prokind = 'f'
                  AND routine.prosecdef IS TRUE
                  AND routine.provolatile = 'v'
                  AND routine.proparallel = 'u'
                  AND routine.proleakproof IS FALSE
                  AND routine.proisstrict IS FALSE
                  AND routine.proretset IS TRUE
                  AND language.lanname = 'plpgsql'
                  AND routine.proconfig = ARRAY[
                      'search_path=pg_catalog, pg_temp',
                      'TimeZone=UTC',
                      'DateStyle=ISO, YMD',
                      'IntervalStyle=postgres',
                      'extra_float_digits=3',
                      'bytea_output=hex',
                      'lock_timeout=15s',
                      'statement_timeout=5min'
                  ]::text[]
                  AND (
                      SELECT pg_catalog.count(*) = 2
                             AND pg_catalog.bool_and(
                                 acl.grantor = routine.proowner
                                 AND acl.privilege_type = 'EXECUTE'
                                 AND acl.is_grantable IS FALSE
                                 AND acl.grantee IN (
                                     routine.proowner, executor_oid
                                 )
                             )
                        FROM pg_catalog.aclexplode(COALESCE(
                            routine.proacl,
                            pg_catalog.acldefault('f', routine.proowner)
                        )) AS acl
                  )
              )
       )
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.pg_proc AS helper
             JOIN pg_catalog.pg_namespace AS helper_namespace
               ON helper_namespace.oid = helper.pronamespace
            WHERE helper_namespace.nspname = 'canonical_brain'
              AND helper.proname = '_discord_guild_routeback_target_valid'
       )
    THEN
        RAISE EXCEPTION 'schema reconciliation control bootstrap terminal failed';
    END IF;
END
$control_bootstrap_terminal$;

COMMIT;
