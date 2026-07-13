-- Canonical Writer persistent-foundation legacy reconciliation v1.
--
-- This tracked variant is only for the disposable, isolated PostgreSQL 18
-- v2 canary.  It preserves the complete base reconciliation contract and adds
-- one transaction-local bridge for a durable pre-existing source owner.  The
-- managed temporary administrator never becomes owner of a durable object.
-- Both bridge memberships are revoked and attested before the sole COMMIT.
--
-- The exact durable source owner is supplied through the non-secret session
-- setting muncho.canonical_writer_foundation_legacy_source_owner.  The base
-- artifact's other nine settings remain mandatory and unchanged.
--
-- This artifact is only for a disposable, isolated PostgreSQL 18 copy.  It
-- deliberately refuses the production database name and also requires nine
-- explicit, session-local expectations.  A caller must collect those values
-- from the exact frozen copy before executing this transaction:
--
--   SET muncho.canonical_writer_reconcile_scope = 'isolated_canary_copy';
--   SET muncho.canonical_writer_reconcile_database = '<current database>';
--   SET muncho.canonical_writer_reconcile_server_identity_sha256 = '<sha256>';
--   SET muncho.canonical_writer_reconcile_source_owner = '<19-col owner>';
--   SET muncho.canonical_writer_reconcile_expected_row_count = '<count>';
--   SET muncho.canonical_writer_reconcile_expected_canonical14_sha256 = '<sha256>';
--   SET muncho.canonical_writer_reconcile_expected_extended19_sha256 = '<sha256>';
--   SET muncho.canonical_writer_reconcile_expected_occurred_at_cutoff = '<timestamptz>';
--   SET muncho.canonical_writer_reconcile_approval_receipt_sha256 = '<sha256>';
--
-- Hash contract (with TimeZone pinned to UTC): each row is converted to jsonb,
-- hashed as its UTF-8 jsonb text, then the table receipt hashes newline-joined
-- ``event_id:row_sha256`` records ordered by event_id.  The two table hashes
-- use different domain-separation labels.  canonical14 hashes only the first
-- fourteen fields; extended19 hashes the complete legacy row.
--
-- The original relation is moved, not rewritten, into the quarantine schema.
-- A new exact fourteen-column public relation receives a mechanically exact
-- copy of those fourteen fields.  No row is inserted into writer provenance:
-- accepting legacy meaning is a later owner-reviewed typed reseed, never an
-- automatic consequence of structural reconciliation.

BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET LOCAL TimeZone = 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '5min';

SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

-- Phase A cannot delete the managed temporary administrator.  PostgreSQL 18
-- records the required source/migration ADMIN bridges under the external role
-- creator, so this SQL session cannot retire them and must never claim the
-- final zero-membership contract.  Keep the reviewed reconciliation body
-- sealed for the future Phase-B Cloud-API workflow, but make direct execution
-- fail before its first mutation.
DO $phase_a_requires_admin_delete_integration$
BEGIN
    RAISE EXCEPTION
        'legacy reconciliation requires Cloud-admin deletion integration';
END
$phase_a_requires_admin_delete_integration$;

-- The copied legacy table belongs to a durable source role, not to the fresh
-- API-created administrator.  The approved observation must already contain
-- the exact source-owner ADMIN/SET bridge for this different temporary admin,
-- and prerequisites retain only its PostgreSQL-18 role-creator ADMIN bridge
-- to the migration owner.  Add one SET-only source->migration edge; every
-- bridge is revoked and re-attested before COMMIT.
DO $foundation_legacy_owner_bridge$
DECLARE
    admin_name text := SESSION_USER;
    source_owner text := pg_catalog.current_setting(
        'muncho.canonical_writer_foundation_legacy_source_owner', true
    );
    legacy_relation regclass := COALESCE(
        pg_catalog.to_regclass(
            'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
        ),
        pg_catalog.to_regclass('public.canonical_event_log')
    );
    relation_owner text;
    public_owner oid;
    admin_oid oid;
BEGIN
    IF source_owner IS NULL
       OR source_owner !~ '^[a-z_][a-z0-9_-]{0,63}$'
       OR source_owner ~ '^muncho_canary_admin_[0-9a-f]{16}$'
       OR source_owner IN (
            'postgres', 'cloudsqladmin', 'cloudsqlsuperuser',
            'canonical_brain_migration_owner', 'canonical_brain_writer',
            'canonical_brain_canary_bootstrap',
            'canonical_brain_canary_bootstrap_login',
            'muncho_canary_writer_login'
       ) OR admin_name !~ '^muncho_canary_admin_[0-9a-f]{16}$'
       OR CURRENT_USER <> admin_name
       OR legacy_relation IS NULL THEN
        RAISE EXCEPTION 'foundation legacy source/admin identity is invalid';
    END IF;
    SELECT pg_catalog.pg_get_userbyid(class.relowner)
      INTO STRICT relation_owner
      FROM pg_catalog.pg_class AS class
     WHERE class.oid = legacy_relation;
    IF relation_owner <> source_owner OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles AS role
         WHERE role.rolname = source_owner
           AND NOT role.rolsuper AND NOT role.rolcreatedb
           AND NOT role.rolcreaterole AND NOT role.rolreplication
           AND NOT role.rolbypassrls AND role.rolconnlimit = -1
           AND role.rolvaliduntil IS NULL AND role.rolconfig IS NULL
    ) THEN
        RAISE EXCEPTION 'foundation legacy source owner is not least privilege';
    END IF;
    SELECT oid INTO STRICT admin_oid
      FROM pg_catalog.pg_roles WHERE rolname = admin_name;
    SELECT nspowner INTO STRICT public_owner
      FROM pg_catalog.pg_namespace WHERE nspname = 'public';
    IF NOT pg_catalog.pg_has_role(admin_name, public_owner, 'USAGE')
       AND NOT EXISTS (
            SELECT 1
              FROM pg_catalog.pg_namespace AS namespace
              CROSS JOIN LATERAL pg_catalog.aclexplode(namespace.nspacl) AS acl
             WHERE namespace.nspname = 'public'
               AND acl.privilege_type = 'CREATE'
               AND acl.is_grantable
               AND (
                    acl.grantee = 0
                    OR acl.grantee = admin_oid
                    OR pg_catalog.pg_has_role(admin_name, acl.grantee, 'USAGE')
               )
       ) THEN
        RAISE EXCEPTION
            'foundation administrator lacks provable public CREATE grant authority';
    END IF;
    IF EXISTS (
        (SELECT granted.rolname, member.rolname,
                membership.admin_option, membership.inherit_option,
                membership.set_option
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname IN (
                source_owner, 'canonical_brain_migration_owner'
           ) OR member.rolname IN (
                source_owner, 'canonical_brain_migration_owner'
           ))
        EXCEPT
        (SELECT source_owner, admin_name, true, true, true
         UNION ALL
         SELECT 'canonical_brain_migration_owner', admin_name,
                true, false, false)
    ) THEN
        RAISE EXCEPTION 'foundation legacy bridge prerequisite membership drifted';
    END IF;
    EXECUTE pg_catalog.format(
        'GRANT canonical_brain_migration_owner TO %I '
        'WITH ADMIN FALSE, INHERIT FALSE, SET TRUE',
        source_owner
    );
    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname = source_owner
           AND member.rolname = admin_name
           AND membership.admin_option
           AND membership.inherit_option
           AND membership.set_option
    ) <> 1 OR (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname = 'canonical_brain_migration_owner'
           AND member.rolname = source_owner
           AND NOT membership.admin_option
           AND NOT membership.inherit_option
           AND membership.set_option
    ) <> 1 OR (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname = 'canonical_brain_migration_owner'
           AND member.rolname = admin_name
           AND membership.admin_option
           AND NOT membership.inherit_option
           AND NOT membership.set_option
    ) <> 1 THEN
        RAISE EXCEPTION 'foundation legacy bridge membership was not exact';
    END IF;
END
$foundation_legacy_owner_bridge$;

GRANT CREATE ON SCHEMA public TO canonical_brain_migration_owner;

DO $foundation_public_create_grant_contract$
BEGIN
    IF NOT pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner', 'public', 'CREATE'
    ) THEN
        RAISE EXCEPTION
            'foundation public CREATE grant did not become effective';
    END IF;
END
$foundation_public_create_grant_contract$;

SELECT pg_catalog.set_config(
    'role',
    pg_catalog.current_setting(
        'muncho.canonical_writer_foundation_legacy_source_owner'
    ),
    true
);

DO $foundation_legacy_owner_active$
BEGIN
    IF CURRENT_USER <> pg_catalog.current_setting(
        'muncho.canonical_writer_foundation_legacy_source_owner'
    ) OR CURRENT_USER = SESSION_USER THEN
        RAISE EXCEPTION 'foundation legacy source role did not become active';
    END IF;
END
$foundation_legacy_owner_active$;

DO $reconcile_prerequisites$
DECLARE
    scope_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_scope', true
    );
    database_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_database', true
    );
    server_identity_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_server_identity_sha256', true
    );
    owner_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_source_owner', true
    );
    row_count_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_expected_row_count', true
    );
    canonical_hash_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_expected_canonical14_sha256', true
    );
    extended_hash_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_expected_extended19_sha256', true
    );
    cutoff_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_expected_occurred_at_cutoff', true
    );
    approval_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_approval_receipt_sha256', true
    );
    public_column_count integer;
    archive_column_count integer;
BEGIN
    IF pg_catalog.current_setting('server_version_num')::integer / 10000 <> 18 THEN
        RAISE EXCEPTION
            'legacy reconciliation requires a separately reviewed PostgreSQL 18 copy';
    END IF;
    IF scope_value IS DISTINCT FROM 'isolated_canary_copy' THEN
        RAISE EXCEPTION
            'legacy reconciliation requires explicit isolated_canary_copy scope';
    END IF;
    IF database_value IS DISTINCT FROM 'muncho_canary_brain'
       OR pg_catalog.current_database() <> 'muncho_canary_brain' THEN
        RAISE EXCEPTION
            'legacy reconciliation requires exact muncho_canary_brain database identity';
    END IF;
    IF pg_catalog.current_database() = 'ai_platform_brain' THEN
        RAISE EXCEPTION
            'legacy reconciliation refuses the production database name';
    END IF;
    IF owner_value IS NULL OR owner_value = '' OR owner_value <> CURRENT_USER THEN
        RAISE EXCEPTION
            'legacy source owner must be pinned and must execute the isolated reconciliation';
    END IF;
    IF row_count_value IS NULL
       OR row_count_value !~ '^[1-9][0-9]*$'
       OR row_count_value::numeric > 9223372036854775807::numeric THEN
        RAISE EXCEPTION 'legacy expected row count is missing or invalid';
    END IF;
    IF server_identity_value !~ '^[0-9a-f]{64}$'
       OR canonical_hash_value !~ '^[0-9a-f]{64}$'
       OR extended_hash_value !~ '^[0-9a-f]{64}$'
       OR approval_value !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'legacy reconciliation hash expectations are invalid';
    END IF;
    IF cutoff_value IS NULL OR cutoff_value = '' THEN
        RAISE EXCEPTION 'legacy occurred_at cutoff expectation is missing';
    END IF;
    BEGIN
        PERFORM cutoff_value::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'legacy occurred_at cutoff expectation is invalid';
    END;
    IF pg_catalog.to_regclass('public.canonical_event_log') IS NULL THEN
        RAISE EXCEPTION 'legacy reconciliation requires public.canonical_event_log';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_roles
         WHERE rolname = 'canonical_brain_migration_owner'
           AND NOT rolcanlogin AND NOT rolinherit
           AND NOT rolsuper AND NOT rolcreatedb
           AND NOT rolcreaterole AND NOT rolreplication AND NOT rolbypassrls
    ) THEN
        RAISE EXCEPTION
            'legacy reconciliation requires least-privilege canonical_brain_migration_owner';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_roles
         WHERE rolname = 'canonical_brain_writer'
           AND NOT rolcanlogin AND NOT rolsuper AND NOT rolcreatedb
           AND NOT rolcreaterole AND NOT rolreplication AND NOT rolbypassrls
    ) THEN
        RAISE EXCEPTION
            'legacy reconciliation requires least-privilege canonical_brain_writer';
    END IF;
    IF EXISTS (
        (SELECT granted.rolname, member.rolname,
                membership.admin_option, membership.inherit_option,
                membership.set_option
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname IN (
                owner_value, 'canonical_brain_migration_owner'
           ) OR member.rolname IN (
                owner_value, 'canonical_brain_migration_owner'
           ))
        EXCEPT
        (SELECT owner_value, SESSION_USER, true, true, true
         UNION ALL
         SELECT 'canonical_brain_migration_owner', SESSION_USER,
                true, false, false
         UNION ALL
         SELECT 'canonical_brain_migration_owner', owner_value,
                false, false, true)
    ) THEN
        RAISE EXCEPTION
            'legacy reconciliation bridge membership contract drifted';
    END IF;
    IF NOT pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner', 'public', 'USAGE'
    ) OR NOT pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner', 'public', 'CREATE'
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_migration_owner requires temporary USAGE/CREATE authority';
    END IF;
    IF pg_catalog.to_regprocedure('pg_catalog.sha256(bytea)') IS NULL THEN
        RAISE EXCEPTION 'PostgreSQL core pg_catalog.sha256(bytea) is required';
    END IF;

    SELECT pg_catalog.count(*)::integer
      INTO public_column_count
      FROM pg_catalog.pg_attribute
     WHERE attrelid = 'public.canonical_event_log'::regclass
       AND attnum > 0 AND NOT attisdropped;
    IF pg_catalog.to_regclass(
        'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
    ) IS NOT NULL THEN
        SELECT pg_catalog.count(*)::integer
          INTO archive_column_count
          FROM pg_catalog.pg_attribute
         WHERE attrelid =
               'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'::regclass
           AND attnum > 0 AND NOT attisdropped;
    END IF;
    IF NOT (
        (public_column_count = 19 AND archive_column_count IS NULL)
        OR (public_column_count = 14 AND archive_column_count = 19)
    ) THEN
        RAISE EXCEPTION
            'legacy reconciliation state is neither pristine nor an exact completed rerun';
    END IF;
END
$reconcile_prerequisites$;

-- Lock whichever atomic state was observed, then re-attest everything below.
DO $reconcile_lock$
BEGIN
    IF pg_catalog.to_regclass(
        'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
    ) IS NULL THEN
        LOCK TABLE public.canonical_event_log IN ACCESS EXCLUSIVE MODE;
    ELSE
        LOCK TABLE canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
            IN ACCESS EXCLUSIVE MODE;
        LOCK TABLE canonical_brain_legacy_quarantine.reconciliation_receipts
            IN ACCESS EXCLUSIVE MODE;
    END IF;
END
$reconcile_lock$;

CREATE OR REPLACE FUNCTION pg_temp._cw_reconcile_canonical14(value jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog
AS $function$
    SELECT pg_catalog.jsonb_build_object(
        'event_id', value->'event_id',
        'schema_version', value->'schema_version',
        'event_type', value->'event_type',
        'occurred_at', value->'occurred_at',
        'case_id', value->'case_id',
        'source', value->'source',
        'actor', value->'actor',
        'subject', value->'subject',
        'evidence', value->'evidence',
        'decision', value->'decision',
        'status', value->'status',
        'next_action', value->'next_action',
        'safety', value->'safety',
        'payload', value->'payload'
    )
$function$;

CREATE OR REPLACE FUNCTION pg_temp._cw_reconcile_sha256(value jsonb)
RETURNS text
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog
AS $function$
    SELECT pg_catalog.encode(
        pg_catalog.sha256(pg_catalog.convert_to(value::text, 'UTF8')), 'hex'
    )
$function$;

CREATE TEMPORARY TABLE _cw_reconcile_observed (
    source_row_count bigint NOT NULL,
    canonical14_sha256 text NOT NULL,
    extended19_sha256 text NOT NULL,
    occurred_at_cutoff timestamptz NOT NULL,
    inserted_at_cutoff timestamptz NOT NULL
) ON COMMIT DROP;

-- Pin the complete legacy relation identity before reading a single row.
DO $legacy_relation_contract$
DECLARE
    legacy_relation regclass;
    owner_value text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_source_owner'
    );
    mismatch text;
    relation_record record;
    index_count integer;
    valid_index_count integer;
    distinct_index_key_count integer;
BEGIN
    legacy_relation := COALESCE(
        pg_catalog.to_regclass(
            'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
        ),
        pg_catalog.to_regclass('public.canonical_event_log')
    );
    SELECT class.relkind, class.relpersistence, class.relispartition,
           access_method.amname, class.reltablespace, class.relrowsecurity,
           class.relforcerowsecurity, class.relreplident, class.reloptions,
           pg_catalog.pg_get_userbyid(class.relowner) AS owner_name
      INTO relation_record
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_am AS access_method ON access_method.oid = class.relam
     WHERE class.oid = legacy_relation;
    IF relation_record.relkind <> 'r'
       OR relation_record.relpersistence <> 'p'
       OR relation_record.relispartition
       OR relation_record.amname <> 'heap'
       OR relation_record.reltablespace <> 0
       OR relation_record.relrowsecurity
       OR relation_record.relforcerowsecurity
       OR relation_record.relreplident <> 'd'
       OR relation_record.reloptions IS NOT NULL
       OR relation_record.owner_name <> owner_value THEN
        RAISE EXCEPTION 'legacy relation physical identity or owner drifted';
    END IF;

    WITH expected(name, type_name, ordinal, not_null, default_expression) AS (
        VALUES
          ('event_id','uuid',1,true,NULL::text),
          ('schema_version','text',2,true,NULL::text),
          ('event_type','text',3,true,NULL::text),
          ('occurred_at','timestamp with time zone',4,true,NULL::text),
          ('case_id','text',5,true,NULL::text),
          ('source','jsonb',6,true,'''{}''::jsonb'),
          ('actor','jsonb',7,true,'''{}''::jsonb'),
          ('subject','jsonb',8,true,'''{}''::jsonb'),
          ('evidence','jsonb',9,true,'''[]''::jsonb'),
          ('decision','jsonb',10,true,'''{}''::jsonb'),
          ('status','jsonb',11,true,'''{}''::jsonb'),
          ('next_action','jsonb',12,true,'''{}''::jsonb'),
          ('safety','jsonb',13,true,'''{}''::jsonb'),
          ('payload','jsonb',14,true,'''{}''::jsonb'),
          ('inserted_at','timestamp with time zone',15,true,'now()'),
          ('idempotency_key','text',16,false,NULL::text),
          ('source_spool','text',17,false,NULL::text),
          ('spool_line_number','integer',18,false,NULL::text),
          ('raw_event_sha256','text',19,false,NULL::text)
    ), actual AS (
        SELECT attribute.attname,
               pg_catalog.format_type(attribute.atttypid, attribute.atttypmod),
               attribute.attnum::integer, attribute.attnotnull,
               pg_catalog.pg_get_expr(default_row.adbin, default_row.adrelid),
               attribute.attidentity, attribute.attgenerated,
               attribute.atthasmissing, attribute.attislocal,
               attribute.attinhcount, attribute.attndims,
               attribute.attoptions, attribute.attfdwoptions
          FROM pg_catalog.pg_attribute AS attribute
          LEFT JOIN pg_catalog.pg_attrdef AS default_row
            ON default_row.adrelid = attribute.attrelid
           AND default_row.adnum = attribute.attnum
         WHERE attribute.attrelid = legacy_relation
           AND attribute.attnum > 0 AND NOT attribute.attisdropped
    ), difference AS (
        (SELECT name, type_name, ordinal, not_null, default_expression
           FROM expected
         EXCEPT
         SELECT attname, format_type, attnum, attnotnull, pg_get_expr
           FROM actual)
        UNION ALL
        (SELECT attname, format_type, attnum, attnotnull, pg_get_expr
           FROM actual
         EXCEPT
         SELECT name, type_name, ordinal, not_null, default_expression
           FROM expected)
    )
    SELECT pg_catalog.string_agg(name || ':' || ordinal::text, ',' ORDER BY ordinal)
      INTO mismatch FROM difference;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'legacy exact column/default contract drifted: %', mismatch;
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_attribute AS attribute
         WHERE attribute.attrelid = legacy_relation
           AND attribute.attnum > 0 AND NOT attribute.attisdropped
           AND (attribute.attidentity <> '' OR attribute.attgenerated <> ''
                OR attribute.atthasmissing OR NOT attribute.attislocal
                OR attribute.attinhcount <> 0 OR attribute.attndims <> 0
                OR attribute.attoptions IS NOT NULL
                OR attribute.attfdwoptions IS NOT NULL)
    ) THEN
        RAISE EXCEPTION 'legacy column storage/identity contract drifted';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_constraint AS constraint_row
         WHERE constraint_row.conrelid = legacy_relation
           AND constraint_row.contype <> 'n'
           AND NOT (
                constraint_row.contype = 'p'
                AND pg_catalog.pg_get_constraintdef(constraint_row.oid, true)
                    = 'PRIMARY KEY (event_id)'
           )
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint AS constraint_row
         WHERE constraint_row.conrelid = legacy_relation
           AND constraint_row.contype = 'p'
           AND pg_catalog.pg_get_constraintdef(constraint_row.oid, true)
               = 'PRIMARY KEY (event_id)'
    ) THEN
        RAISE EXCEPTION 'legacy constraint contract drifted';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_trigger AS trigger_row
         WHERE trigger_row.tgrelid = legacy_relation AND NOT trigger_row.tgisinternal
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_rewrite AS rewrite_row
         WHERE rewrite_row.ev_class = legacy_relation
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_policy AS policy_row
         WHERE policy_row.polrelid = legacy_relation
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_inherits AS inheritance
         WHERE inheritance.inhrelid = legacy_relation
            OR inheritance.inhparent = legacy_relation
    ) THEN
        RAISE EXCEPTION 'legacy triggers/rules/policies/inheritance contract drifted';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_class AS class
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(class.relacl, pg_catalog.acldefault('r', class.relowner))
          ) AS acl
         WHERE class.oid = legacy_relation
           AND acl.grantee <> class.relowner
           AND acl.privilege_type IN (
                'SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER'
           )
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_attribute AS attribute
          CROSS JOIN LATERAL pg_catalog.aclexplode(attribute.attacl) AS acl
         WHERE attribute.attrelid = legacy_relation
           AND attribute.attnum > 0 AND NOT attribute.attisdropped
           AND acl.grantee <> (
                SELECT relowner FROM pg_catalog.pg_class WHERE oid = legacy_relation
           )
    ) THEN
        RAISE EXCEPTION 'legacy table or column ACL contract drifted';
    END IF;

    SELECT pg_catalog.count(*)::integer,
           pg_catalog.count(*) FILTER (
               WHERE index.indisvalid AND index.indisready AND index.indislive
                 AND NOT index.indisexclusion AND index.indimmediate
                 AND index.indnkeyatts = 1 AND index.indnatts = 1
                 AND index.indexprs IS NULL
                 AND access_method.amname = 'btree'
                 AND index_class.relpersistence = 'p'
                 AND index_class.reloptions IS NULL
                 AND index_class.reltablespace = 0
                 AND pg_catalog.pg_get_userbyid(index_class.relowner) = owner_value
                 AND (
                    (pg_catalog.pg_get_indexdef(index.indexrelid, 1, true) = 'event_id'
                     AND index.indisprimary AND index.indisunique
                     AND index.indpred IS NULL)
                    OR
                    (pg_catalog.pg_get_indexdef(index.indexrelid, 1, true) = 'case_id'
                     AND NOT index.indisprimary AND NOT index.indisunique
                     AND index.indpred IS NULL)
                    OR
                    (pg_catalog.pg_get_indexdef(index.indexrelid, 1, true) = 'event_type'
                     AND NOT index.indisprimary AND NOT index.indisunique
                     AND index.indpred IS NULL)
                    OR
                    (pg_catalog.pg_get_indexdef(index.indexrelid, 1, true) = 'occurred_at'
                     AND NOT index.indisprimary AND NOT index.indisunique
                     AND index.indpred IS NULL)
                    OR
                    (pg_catalog.pg_get_indexdef(index.indexrelid, 1, true) = 'idempotency_key'
                     AND NOT index.indisprimary AND index.indisunique
                     AND pg_catalog.replace(
                         pg_catalog.replace(
                             pg_catalog.pg_get_expr(index.indpred, index.indrelid, true),
                             '(', ''
                         ), ')', ''
                     ) = 'idempotency_key IS NOT NULL')
                 )
           )::integer,
           pg_catalog.count(DISTINCT pg_catalog.pg_get_indexdef(
               index.indexrelid, 1, true
           ))::integer
      INTO index_count, valid_index_count, distinct_index_key_count
      FROM pg_catalog.pg_index AS index
      JOIN pg_catalog.pg_class AS index_class
        ON index_class.oid = index.indexrelid
      JOIN pg_catalog.pg_am AS access_method
        ON access_method.oid = index_class.relam
     WHERE index.indrelid = legacy_relation;
    IF index_count <> 5 OR valid_index_count <> 5
       OR distinct_index_key_count <> 5 THEN
        RAISE EXCEPTION 'legacy exact five-index contract drifted';
    END IF;
END
$legacy_relation_contract$;

DO $observe_legacy_snapshot$
DECLARE
    legacy_name text;
    observed_count bigint;
    observed_canonical text;
    observed_extended text;
    observed_cutoff timestamptz;
    observed_inserted_cutoff timestamptz;
BEGIN
    legacy_name := CASE WHEN pg_catalog.to_regclass(
        'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
    ) IS NULL THEN 'public.canonical_event_log'
      ELSE 'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
    END;
    EXECUTE pg_catalog.format($sql$
        WITH row_receipts AS (
            SELECT event_id,
                   pg_temp._cw_reconcile_sha256(
                       pg_temp._cw_reconcile_canonical14(pg_catalog.to_jsonb(event_row))
                   ) AS canonical14_row_sha256,
                   pg_temp._cw_reconcile_sha256(
                       pg_catalog.to_jsonb(event_row)
                   ) AS extended19_row_sha256
              FROM %s AS event_row
        )
        SELECT pg_catalog.count(*)::bigint,
               pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   'canonical-writer-legacy-reconcile-v1:canonical14' || E'\n'
                   || COALESCE(pg_catalog.string_agg(
                        event_id::text || ':' || canonical14_row_sha256,
                        E'\n' ORDER BY event_id
                   ), ''), 'UTF8')), 'hex'),
               pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   'canonical-writer-legacy-reconcile-v1:extended19' || E'\n'
                   || COALESCE(pg_catalog.string_agg(
                        event_id::text || ':' || extended19_row_sha256,
                        E'\n' ORDER BY event_id
                   ), ''), 'UTF8')), 'hex')
          FROM row_receipts
    $sql$, legacy_name)
      INTO observed_count, observed_canonical, observed_extended;
    EXECUTE pg_catalog.format(
        'SELECT max(occurred_at), max(inserted_at) FROM %s', legacy_name
    ) INTO observed_cutoff, observed_inserted_cutoff;
    IF observed_count <> pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_expected_row_count'
    )::bigint
       OR observed_canonical <> pg_catalog.current_setting(
            'muncho.canonical_writer_reconcile_expected_canonical14_sha256'
       )
       OR observed_extended <> pg_catalog.current_setting(
            'muncho.canonical_writer_reconcile_expected_extended19_sha256'
       )
       OR observed_cutoff IS DISTINCT FROM pg_catalog.current_setting(
            'muncho.canonical_writer_reconcile_expected_occurred_at_cutoff'
       )::timestamptz THEN
        RAISE EXCEPTION
            'legacy frozen snapshot count/hash/cutoff drifted from explicit expectations';
    END IF;
    INSERT INTO _cw_reconcile_observed VALUES (
        observed_count, observed_canonical, observed_extended,
        observed_cutoff, observed_inserted_cutoff
    );
END
$observe_legacy_snapshot$;

-- The exact SET edge and schema CREATE grant were established and proven
-- before activating the source owner.  Re-prove them here without creating a
-- second, contradictory administrator membership.
DO $temporary_reconcile_owner_authority$
DECLARE
    admin_name text := SESSION_USER;
BEGIN
    IF admin_name = 'canonical_brain_migration_owner' THEN
        RAISE EXCEPTION 'offline migration owner cannot be the login session';
    END IF;
    PERFORM pg_catalog.set_config(
        'muncho.canonical_writer_reconcile_admin', admin_name, true
    );
    IF NOT pg_catalog.pg_has_role(
        CURRENT_USER, 'canonical_brain_migration_owner', 'MEMBER'
    ) OR NOT pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner', 'public', 'CREATE'
    ) THEN
        RAISE EXCEPTION
            'transaction-scoped reconciliation owner authority is not exact';
    END IF;
END
$temporary_reconcile_owner_authority$;

DO $perform_first_reconciliation$
DECLARE
    observed _cw_reconcile_observed%ROWTYPE;
BEGIN
    IF pg_catalog.to_regclass(
        'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
    ) IS NOT NULL THEN
        RETURN;
    END IF;
    SELECT * INTO STRICT observed FROM _cw_reconcile_observed;

    CREATE SCHEMA canonical_brain_legacy_quarantine;
    REVOKE ALL ON SCHEMA canonical_brain_legacy_quarantine FROM PUBLIC;
    ALTER TABLE public.canonical_event_log
        SET SCHEMA canonical_brain_legacy_quarantine;
    ALTER TABLE canonical_brain_legacy_quarantine.canonical_event_log
        RENAME TO canonical_event_log_legacy_v1;

    CREATE TABLE public.canonical_event_log (
        event_id uuid NOT NULL,
        schema_version text NOT NULL,
        event_type text NOT NULL,
        occurred_at timestamptz NOT NULL,
        case_id text NOT NULL,
        source jsonb NOT NULL,
        actor jsonb NOT NULL,
        subject jsonb NOT NULL,
        evidence jsonb NOT NULL,
        decision jsonb NOT NULL,
        status jsonb NOT NULL,
        next_action jsonb NOT NULL,
        safety jsonb NOT NULL,
        payload jsonb NOT NULL,
        PRIMARY KEY (event_id)
    );
    INSERT INTO public.canonical_event_log (
        event_id, schema_version, event_type, occurred_at, case_id,
        source, actor, subject, evidence, decision, status,
        next_action, safety, payload
    )
    SELECT event_id, schema_version, event_type, occurred_at, case_id,
           source, actor, subject, evidence, decision, status,
           next_action, safety, payload
      FROM canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
     ORDER BY event_id;
    REVOKE ALL ON TABLE public.canonical_event_log FROM PUBLIC;
    ALTER TABLE public.canonical_event_log
        OWNER TO canonical_brain_migration_owner;

    CREATE TABLE canonical_brain_legacy_quarantine.reconciliation_receipts (
        reconciliation_id text PRIMARY KEY,
        artifact_version text NOT NULL,
        isolated_scope text NOT NULL,
        source_database text NOT NULL,
        server_identity_sha256 text NOT NULL,
        source_owner text NOT NULL,
        target_owner text NOT NULL,
        source_row_count bigint NOT NULL,
        canonical14_sha256 text NOT NULL,
        extended19_sha256 text NOT NULL,
        occurred_at_cutoff timestamptz NOT NULL,
        inserted_at_cutoff timestamptz NOT NULL,
        approval_receipt_sha256 text NOT NULL,
        postgres_version_num integer NOT NULL,
        reconciled_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp()
    );
    REVOKE ALL ON TABLE
        canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1,
        canonical_brain_legacy_quarantine.reconciliation_receipts
        FROM PUBLIC;
    INSERT INTO canonical_brain_legacy_quarantine.reconciliation_receipts (
        reconciliation_id, artifact_version, isolated_scope,
        source_database, server_identity_sha256, source_owner, target_owner,
        source_row_count,
        canonical14_sha256, extended19_sha256, occurred_at_cutoff,
        inserted_at_cutoff, approval_receipt_sha256, postgres_version_num
    ) VALUES (
        'canonical-writer-legacy-reconcile-v1',
        'canonical-writer-legacy-reconcile-v1',
        'isolated_canary_copy',
        pg_catalog.current_database(),
        pg_catalog.current_setting(
            'muncho.canonical_writer_reconcile_server_identity_sha256'
        ),
        pg_catalog.current_setting(
            'muncho.canonical_writer_reconcile_source_owner'
        ),
        'canonical_brain_migration_owner',
        observed.source_row_count,
        observed.canonical14_sha256,
        observed.extended19_sha256,
        observed.occurred_at_cutoff,
        observed.inserted_at_cutoff,
        pg_catalog.current_setting(
            'muncho.canonical_writer_reconcile_approval_receipt_sha256'
        ),
        pg_catalog.current_setting('server_version_num')::integer
    );
END
$perform_first_reconciliation$;

-- The exact owner role and archive read access below exist only in this
-- transaction, are approval-bound by the required reconciliation receipt, and
-- are removed before COMMIT.
GRANT USAGE ON SCHEMA canonical_brain_legacy_quarantine
    TO canonical_brain_migration_owner;
GRANT SELECT ON TABLE
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1,
    canonical_brain_legacy_quarantine.reconciliation_receipts,
    _cw_reconcile_observed
    TO canonical_brain_migration_owner;

SET LOCAL ROLE canonical_brain_migration_owner;
LOCK TABLE public.canonical_event_log IN ACCESS EXCLUSIVE MODE;

-- The target is exact and every copied legacy row remains byte/JSON-identical.
DO $reconciled_contract$
DECLARE
    mismatch text;
    observed _cw_reconcile_observed%ROWTYPE;
    receipt_record record;
    target_row_count bigint;
    copied_hash text;
    target_relation record;
    target_index_count integer;
BEGIN
    SELECT * INTO STRICT observed FROM _cw_reconcile_observed;
    IF pg_catalog.to_regclass(
        'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
    ) IS NULL OR pg_catalog.to_regclass(
        'canonical_brain_legacy_quarantine.reconciliation_receipts'
    ) IS NULL THEN
        RAISE EXCEPTION 'legacy quarantine objects are missing after reconciliation';
    END IF;

    WITH expected(name, type_name, ordinal) AS (
        VALUES
          ('event_id','uuid',1), ('schema_version','text',2),
          ('event_type','text',3), ('occurred_at','timestamp with time zone',4),
          ('case_id','text',5), ('source','jsonb',6), ('actor','jsonb',7),
          ('subject','jsonb',8), ('evidence','jsonb',9), ('decision','jsonb',10),
          ('status','jsonb',11), ('next_action','jsonb',12),
          ('safety','jsonb',13), ('payload','jsonb',14)
    ), actual AS (
        SELECT attribute.attname,
               pg_catalog.format_type(attribute.atttypid, attribute.atttypmod),
               attribute.attnum::integer
          FROM pg_catalog.pg_attribute AS attribute
         WHERE attribute.attrelid = 'public.canonical_event_log'::regclass
           AND attribute.attnum > 0 AND NOT attribute.attisdropped
    ), difference AS (
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        UNION ALL (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT pg_catalog.string_agg(name || ':' || ordinal::text, ',' ORDER BY ordinal)
      INTO mismatch FROM difference;
    IF mismatch IS NOT NULL OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_attribute AS attribute
         WHERE attribute.attrelid = 'public.canonical_event_log'::regclass
           AND attribute.attnum > 0 AND NOT attribute.attisdropped
           AND (NOT attribute.attnotnull OR attribute.atthasdef
                OR attribute.attidentity <> '' OR attribute.attgenerated <> '')
    ) OR (
        SELECT pg_catalog.pg_get_userbyid(class.relowner)
          FROM pg_catalog.pg_class AS class
         WHERE class.oid = 'public.canonical_event_log'::regclass
    ) <> 'canonical_brain_migration_owner' THEN
        RAISE EXCEPTION 'reconciled public fourteen-column identity drifted: %', mismatch;
    END IF;

    SELECT class.relkind, class.relpersistence, class.relispartition,
           access_method.amname, class.reltablespace, class.relrowsecurity,
           class.relforcerowsecurity, class.relreplident, class.reloptions
      INTO target_relation
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_am AS access_method ON access_method.oid = class.relam
     WHERE class.oid = 'public.canonical_event_log'::regclass;
    IF target_relation.relkind <> 'r'
       OR target_relation.relpersistence <> 'p'
       OR target_relation.relispartition
       OR target_relation.amname <> 'heap'
       OR target_relation.reltablespace <> 0
       OR target_relation.relrowsecurity
       OR target_relation.relforcerowsecurity
       OR target_relation.relreplident <> 'd'
       OR target_relation.reloptions IS NOT NULL THEN
        RAISE EXCEPTION 'reconciled public relation physical identity drifted';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_constraint AS constraint_row
         WHERE constraint_row.conrelid = 'public.canonical_event_log'::regclass
           AND constraint_row.contype <> 'n'
           AND NOT (
                constraint_row.contype = 'p'
                AND pg_catalog.pg_get_constraintdef(constraint_row.oid, true)
                    = 'PRIMARY KEY (event_id)'
           )
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint AS constraint_row
         WHERE constraint_row.conrelid = 'public.canonical_event_log'::regclass
           AND constraint_row.contype = 'p'
           AND pg_catalog.pg_get_constraintdef(constraint_row.oid, true)
               = 'PRIMARY KEY (event_id)'
    ) THEN
        RAISE EXCEPTION 'reconciled public primary-key contract drifted';
    END IF;
    SELECT pg_catalog.count(*)::integer
      INTO target_index_count
      FROM pg_catalog.pg_index AS index
      JOIN pg_catalog.pg_class AS index_class
        ON index_class.oid = index.indexrelid
      JOIN pg_catalog.pg_am AS access_method
        ON access_method.oid = index_class.relam
     WHERE index.indrelid = 'public.canonical_event_log'::regclass
       AND index.indisprimary AND index.indisunique
       AND NOT index.indisexclusion AND index.indimmediate
       AND index.indisvalid AND index.indisready AND index.indislive
       AND index.indnkeyatts = 1 AND index.indnatts = 1
       AND index.indexprs IS NULL AND index.indpred IS NULL
       AND pg_catalog.pg_get_indexdef(index.indexrelid, 1, true) = 'event_id'
       AND access_method.amname = 'btree'
       AND index_class.relpersistence = 'p'
       AND index_class.reloptions IS NULL
       AND index_class.reltablespace = 0
       AND pg_catalog.pg_get_userbyid(index_class.relowner)
           = 'canonical_brain_migration_owner';
    IF target_index_count <> 1 OR (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_index
         WHERE indrelid = 'public.canonical_event_log'::regclass
    ) <> 1 THEN
        RAISE EXCEPTION 'reconciled public exact primary index contract drifted';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_trigger AS trigger_row
         WHERE trigger_row.tgrelid = 'public.canonical_event_log'::regclass
           AND NOT trigger_row.tgisinternal
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_rewrite AS rewrite_row
         WHERE rewrite_row.ev_class = 'public.canonical_event_log'::regclass
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_policy AS policy_row
         WHERE policy_row.polrelid = 'public.canonical_event_log'::regclass
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_inherits AS inheritance
         WHERE inheritance.inhrelid = 'public.canonical_event_log'::regclass
            OR inheritance.inhparent = 'public.canonical_event_log'::regclass
    ) THEN
        RAISE EXCEPTION
            'reconciled public triggers/rules/policies/inheritance contract drifted';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_class AS class
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(class.relacl, pg_catalog.acldefault('r', class.relowner))
          ) AS acl
         WHERE class.oid = 'public.canonical_event_log'::regclass
           AND acl.grantee <> class.relowner
           AND acl.privilege_type IN (
                'SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER'
           )
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_attribute AS attribute
          CROSS JOIN LATERAL pg_catalog.aclexplode(attribute.attacl) AS acl
         WHERE attribute.attrelid = 'public.canonical_event_log'::regclass
           AND attribute.attnum > 0 AND NOT attribute.attisdropped
    ) THEN
        RAISE EXCEPTION 'reconciled public table or column ACL contract drifted';
    END IF;
    IF (
        SELECT pg_catalog.pg_get_userbyid(namespace.nspowner)
          FROM pg_catalog.pg_namespace AS namespace
         WHERE namespace.nspname = 'canonical_brain_legacy_quarantine'
    ) <> pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_source_owner'
    ) OR pg_catalog.has_schema_privilege(
        'canonical_brain_writer', 'canonical_brain_legacy_quarantine', 'USAGE'
    ) THEN
        RAISE EXCEPTION 'legacy quarantine schema ownership/ACL contract drifted';
    END IF;

    IF EXISTS (
        (SELECT event_id, schema_version, event_type, occurred_at, case_id,
                source, actor, subject, evidence, decision, status,
                next_action, safety, payload
           FROM canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
         EXCEPT
         SELECT event_id, schema_version, event_type, occurred_at, case_id,
                source, actor, subject, evidence, decision, status,
                next_action, safety, payload
           FROM public.canonical_event_log)
    ) THEN
        RAISE EXCEPTION 'a legacy canonical14 row was not copied identically';
    END IF;
    SELECT pg_catalog.count(*)::bigint
      INTO target_row_count
      FROM public.canonical_event_log AS target
      JOIN canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 AS legacy
        USING (event_id);
    IF target_row_count <> observed.source_row_count THEN
        RAISE EXCEPTION 'reconciled canonical14 row count drifted';
    END IF;
    WITH row_receipts AS (
        SELECT target.event_id,
               pg_temp._cw_reconcile_sha256(pg_catalog.to_jsonb(target)) AS row_sha
          FROM public.canonical_event_log AS target
          JOIN canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 AS legacy
            USING (event_id)
    )
    SELECT pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               'canonical-writer-legacy-reconcile-v1:canonical14' || E'\n'
               || COALESCE(pg_catalog.string_agg(
                    event_id::text || ':' || row_sha, E'\n' ORDER BY event_id
               ), ''), 'UTF8')), 'hex')
      INTO copied_hash
      FROM row_receipts;
    IF copied_hash <> observed.canonical14_sha256 THEN
        RAISE EXCEPTION 'reconciled canonical14 content receipt drifted';
    END IF;

    SELECT * INTO STRICT receipt_record
      FROM canonical_brain_legacy_quarantine.reconciliation_receipts
     WHERE reconciliation_id = 'canonical-writer-legacy-reconcile-v1';
    IF receipt_record.artifact_version <> 'canonical-writer-legacy-reconcile-v1'
       OR receipt_record.isolated_scope <> 'isolated_canary_copy'
       OR receipt_record.source_database <> pg_catalog.current_database()
       OR receipt_record.server_identity_sha256 <> pg_catalog.current_setting(
            'muncho.canonical_writer_reconcile_server_identity_sha256'
       )
       OR receipt_record.source_owner <> pg_catalog.current_setting(
            'muncho.canonical_writer_reconcile_source_owner'
       )
       OR receipt_record.target_owner <> 'canonical_brain_migration_owner'
       OR receipt_record.source_row_count <> observed.source_row_count
       OR receipt_record.canonical14_sha256 <> observed.canonical14_sha256
       OR receipt_record.extended19_sha256 <> observed.extended19_sha256
       OR receipt_record.occurred_at_cutoff <> observed.occurred_at_cutoff
       OR receipt_record.inserted_at_cutoff <> observed.inserted_at_cutoff
       OR receipt_record.approval_receipt_sha256 <> pg_catalog.current_setting(
            'muncho.canonical_writer_reconcile_approval_receipt_sha256'
       )
       OR receipt_record.postgres_version_num < 180000
       OR receipt_record.postgres_version_num >= 190000 THEN
        RAISE EXCEPTION 'legacy reconciliation receipt drifted';
    END IF;
    IF (
        SELECT pg_catalog.count(*)
          FROM canonical_brain_legacy_quarantine.reconciliation_receipts
    ) <> 1 THEN
        RAISE EXCEPTION 'legacy reconciliation receipt cardinality drifted';
    END IF;
    IF pg_catalog.to_regclass(
        'canonical_brain.writer_event_provenance'
    ) IS NOT NULL THEN
        IF EXISTS (
            SELECT 1
              FROM canonical_brain.writer_event_provenance AS provenance
              JOIN canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
                   AS legacy USING (event_id)
        ) THEN
            RAISE EXCEPTION
                'legacy events must not be auto-promoted into writer provenance';
        END IF;
    END IF;
END
$reconciled_contract$;

RESET ROLE;

SELECT pg_catalog.set_config(
    'role',
    pg_catalog.current_setting(
        'muncho.canonical_writer_foundation_legacy_source_owner'
    ),
    true
);

REVOKE SELECT ON TABLE
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1,
    canonical_brain_legacy_quarantine.reconciliation_receipts,
    _cw_reconcile_observed
    FROM canonical_brain_migration_owner;
REVOKE USAGE ON SCHEMA canonical_brain_legacy_quarantine
    FROM canonical_brain_migration_owner;

RESET ROLE;

REVOKE CREATE ON SCHEMA public FROM canonical_brain_migration_owner;

-- RESET ROLE above restored the managed administrator.  Retire the source
-- bridge before the base artifact's existing zero-membership attestation.
DO $retire_foundation_legacy_owner_bridge$
DECLARE
    admin_name text := SESSION_USER;
    source_owner text := pg_catalog.current_setting(
        'muncho.canonical_writer_foundation_legacy_source_owner'
    );
BEGIN
    IF CURRENT_USER <> admin_name THEN
        RAISE EXCEPTION 'foundation legacy bridge administrator was not restored';
    END IF;
    EXECUTE pg_catalog.format(
        'REVOKE canonical_brain_migration_owner FROM %I', source_owner
    );
    EXECUTE pg_catalog.format(
        'REVOKE canonical_brain_migration_owner FROM %I', admin_name
    );
    EXECUTE pg_catalog.format(
        'REVOKE %I FROM %I', source_owner, admin_name
    );
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname IN (
                source_owner, 'canonical_brain_migration_owner'
           ) OR member.rolname IN (
                source_owner, 'canonical_brain_migration_owner'
           )
    ) THEN
        RAISE EXCEPTION 'foundation legacy bridge membership survived retirement';
    END IF;
END
$retire_foundation_legacy_owner_bridge$;

DO $final_reconcile_membership_contract$
DECLARE
    admin_name text := pg_catalog.current_setting(
        'muncho.canonical_writer_reconcile_admin'
    );
    admin_superuser boolean;
BEGIN
    SELECT rolsuper INTO STRICT admin_superuser
      FROM pg_catalog.pg_roles WHERE rolname = admin_name;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS owner_role
            ON owner_role.rolname = 'canonical_brain_migration_owner'
         WHERE membership.roleid = owner_role.oid
            OR membership.member = owner_role.oid
    ) THEN
        RAISE EXCEPTION
            'temporary reconciliation membership row survived retirement';
    END IF;
    IF NOT admin_superuser AND pg_catalog.pg_has_role(
        admin_name, 'canonical_brain_migration_owner', 'MEMBER'
    ) THEN
        RAISE EXCEPTION
            'temporary reconciliation role membership survived retirement';
    END IF;
    IF pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner', 'public', 'CREATE'
    ) THEN
        RAISE EXCEPTION
            'temporary reconciliation public CREATE survived retirement';
    END IF;
    IF pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner',
        'canonical_brain_legacy_quarantine', 'USAGE'
    ) THEN
        RAISE EXCEPTION
            'temporary reconciliation archive USAGE survived retirement';
    END IF;
    IF pg_catalog.has_table_privilege(
        'canonical_brain_migration_owner',
        'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1',
        'SELECT'
    ) THEN
        RAISE EXCEPTION
            'temporary reconciliation legacy SELECT survived retirement';
    END IF;
    IF pg_catalog.has_table_privilege(
        'canonical_brain_migration_owner',
        'canonical_brain_legacy_quarantine.reconciliation_receipts',
        'SELECT'
    ) THEN
        RAISE EXCEPTION
            'temporary reconciliation receipt SELECT survived retirement';
    END IF;
END
$final_reconcile_membership_contract$;

COMMIT;
