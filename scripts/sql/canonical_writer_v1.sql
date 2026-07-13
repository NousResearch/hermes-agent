-- Privileged Canonical Writer v1 database contract.
--
-- This migration is intentionally fail closed.  It does not create login
-- roles, guess the production Canonical Brain schema, or move existing data.
-- Before applying it, an administrator must create:
--
--   canonical_brain_migration_owner  NOLOGIN, owns this contract
--   canonical_brain_writer           NOLOGIN, granted only to the service login
--   canonical_brain_canary_bootstrap NOLOGIN, inert one-shot bootstrap role
--   canonical_brain_canary_bootstrap_login LOGIN, member only of that role
--
-- The existing public.canonical_event_log and PostgreSQL's core
-- pg_catalog.sha256(bytea) routine are explicit prerequisites.  The runtime
-- writer role receives no table privileges and no
-- general SQL surface: only the eighteen fixed (jsonb, jsonb) routines at the
-- end of this file are executable by it.
--
-- Applying the contract is an owner-approved operation.  The invoking session
-- must pin these settings before BEGIN:
--
--   muncho.canonical_writer_migration_scope
--       = isolated_canary_copy | owner_approved_cutover
--   muncho.canonical_writer_migration_database = current_database()
--   muncho.canonical_writer_migration_approval_receipt_sha256 = 64 lowercase hex
--   muncho.canonical_writer_cloudsqladmin_hba_rejection_sha256
--       = digest emitted and injected only by the root-controlled active-TLS
--         probe-and-apply driver (never an operator-supplied placeholder)
--
-- Managed PostgreSQL administrators are not necessarily superusers.  After
-- preflight proves that the offline owner has zero memberships, this migration
-- grants the invoking SESSION_USER one transaction-scoped SET-only membership,
-- executes owner operations with SET LOCAL ROLE, then revokes and re-attests
-- that membership before COMMIT.  Errors and process loss roll the grant back
-- with the rest of this transaction.
--
-- PostgreSQL 16+ is required because the transaction-scoped owner boundary
-- depends on independent ADMIN, INHERIT, and SET membership options.  This v1
-- artifact pins the current fourteen-column
-- event envelope exactly.
-- It deliberately does not guess compatibility for the legacy Cloud shape
-- known to have carried idempotency_key/source_spool/spool_line_number/
-- raw_event_sha256 columns.  A read-only production schema attestation and a
-- separately reviewed reconciliation migration are mandatory before cutover;
-- applying this artifact directly to a legacy nineteen-column table must fail.

BEGIN;

-- Deployment/preflight takes the same global key in shared mode.  The
-- migration takes it exclusively so no writer process can attest or execute
-- against a partially installed contract.
SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

-- Cloud SQL exposes one provider-owned maintenance database through PUBLIC
-- catalog ACLs while rejecting direct TLS connections to it in pg_hba.  It is
-- the sole cross-database exception, and only with this exact managed catalog
-- fingerprint plus a trusted-preflight rejection receipt.
CREATE OR REPLACE FUNCTION pg_temp._cw_managed_cloudsqladmin_exception(
    database_oid oid,
    hba_rejection_sha256 text
)
RETURNS boolean
LANGUAGE sql
STABLE
SET search_path = pg_catalog
AS $function$
    WITH database_row AS (
        SELECT database.oid, database.datname, database.datallowconn,
               database.datistemplate,
               pg_catalog.pg_get_userbyid(database.datdba) AS owner_name,
               database.datdba, database.datacl
          FROM pg_catalog.pg_database AS database
         WHERE database.oid = database_oid
    ), actual_acl AS (
        SELECT CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                    ELSE pg_catalog.pg_get_userbyid(acl.grantee) END AS grantee,
               pg_catalog.pg_get_userbyid(acl.grantor) AS grantor,
               acl.privilege_type, acl.is_grantable
          FROM database_row
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  database_row.datacl,
                  pg_catalog.acldefault('d', database_row.datdba)
              )
          ) AS acl
    ), expected_acl(grantee, grantor, privilege_type, is_grantable) AS (
        VALUES
          ('PUBLIC','cloudsqladmin','CONNECT',false),
          ('PUBLIC','cloudsqladmin','TEMPORARY',false),
          ('cloudsqladmin','cloudsqladmin','CREATE',false),
          ('cloudsqladmin','cloudsqladmin','CONNECT',false),
          ('cloudsqladmin','cloudsqladmin','TEMPORARY',false)
    )
    SELECT COALESCE(
               hba_rejection_sha256 ~ '^[0-9a-f]{64}$', false
           )
       AND EXISTS (
            SELECT 1 FROM database_row
             WHERE datname = 'cloudsqladmin'
               AND datallowconn AND NOT datistemplate
               AND owner_name = 'cloudsqladmin'
       )
       AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'cloudsqladmin'
               AND rolcanlogin AND rolsuper AND rolcreatedb AND rolcreaterole
               AND rolreplication AND rolbypassrls
       )
       AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'cloudsqlsuperuser'
               AND rolcanlogin AND NOT rolsuper AND rolcreatedb AND rolcreaterole
               AND NOT rolreplication AND NOT rolbypassrls
       )
       AND NOT EXISTS (
            (SELECT * FROM actual_acl EXCEPT SELECT * FROM expected_acl)
            UNION ALL
            (SELECT * FROM expected_acl EXCEPT SELECT * FROM actual_acl)
       )
$function$;

REVOKE ALL ON FUNCTION pg_temp._cw_managed_cloudsqladmin_exception(oid,text)
    FROM PUBLIC;
GRANT EXECUTE ON FUNCTION
    pg_temp._cw_managed_cloudsqladmin_exception(oid,text)
    TO canonical_brain_migration_owner;

DO $prerequisites$
DECLARE
    missing_columns text;
    migration_scope text := pg_catalog.current_setting(
        'muncho.canonical_writer_migration_scope', true
    );
    migration_database text := pg_catalog.current_setting(
        'muncho.canonical_writer_migration_database', true
    );
    migration_approval text := pg_catalog.current_setting(
        'muncho.canonical_writer_migration_approval_receipt_sha256', true
    );
    cloudsqladmin_hba_receipt text := pg_catalog.current_setting(
        'muncho.canonical_writer_cloudsqladmin_hba_rejection_sha256', true
    );
BEGIN
    IF migration_scope NOT IN (
        'isolated_canary_copy', 'owner_approved_cutover'
    ) OR migration_database IS DISTINCT FROM pg_catalog.current_database()
       OR migration_approval !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION
            'canonical writer migration lacks exact scope/database/approval binding';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'canonical_brain_migration_owner'
          AND rolcanlogin IS FALSE
          AND rolinherit IS FALSE
          AND rolsuper IS FALSE
          AND rolcreatedb IS FALSE
          AND rolcreaterole IS FALSE
          AND rolreplication IS FALSE
          AND rolbypassrls IS FALSE
    ) THEN
        RAISE EXCEPTION
            'prerequisite missing: canonical_brain_migration_owner must be a least-privilege NOLOGIN role';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'canonical_brain_writer'
          AND rolcanlogin IS FALSE
          AND rolsuper IS FALSE
          AND rolcreatedb IS FALSE
          AND rolcreaterole IS FALSE
          AND rolreplication IS FALSE
          AND rolbypassrls IS FALSE
    ) THEN
        RAISE EXCEPTION
            'prerequisite missing: canonical_brain_writer must be a least-privilege NOLOGIN role';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'canonical_brain_canary_bootstrap'
          AND rolcanlogin IS FALSE
          AND rolinherit IS FALSE
          AND rolsuper IS FALSE
          AND rolcreatedb IS FALSE
          AND rolcreaterole IS FALSE
          AND rolreplication IS FALSE
          AND rolbypassrls IS FALSE
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'canonical_brain_canary_bootstrap_login'
          AND rolcanlogin IS TRUE
          AND rolinherit IS TRUE
          AND rolsuper IS FALSE
          AND rolcreatedb IS FALSE
          AND rolcreaterole IS FALSE
          AND rolreplication IS FALSE
          AND rolbypassrls IS FALSE
    ) THEN
        RAISE EXCEPTION
            'prerequisite missing: exact least-privilege canary bootstrap role/login';
    END IF;
    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted_role
            ON granted_role.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member_role
            ON member_role.oid = membership.member
         WHERE granted_role.rolname = 'canonical_brain_canary_bootstrap'
           AND member_role.rolname = 'canonical_brain_canary_bootstrap_login'
           AND NOT membership.admin_option
           AND membership.inherit_option
           AND membership.set_option
    ) <> 1 OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted_role
            ON granted_role.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member_role
            ON member_role.oid = membership.member
         WHERE (
                granted_role.rolname = 'canonical_brain_canary_bootstrap'
                OR member_role.rolname = 'canonical_brain_canary_bootstrap_login'
                OR member_role.rolname = 'canonical_brain_canary_bootstrap'
               )
           AND NOT (
                granted_role.rolname = 'canonical_brain_canary_bootstrap'
                AND member_role.rolname = 'canonical_brain_canary_bootstrap_login'
                AND NOT membership.admin_option
                AND membership.inherit_option
                AND membership.set_option
           )
    ) THEN
        RAISE EXCEPTION
            'canary bootstrap role/login membership is not exact';
    END IF;
    IF pg_catalog.pg_has_role(
        'canonical_brain_writer',
        'canonical_brain_migration_owner',
        'MEMBER'
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_writer must not inherit the migration-owner role';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS writer_role
            ON writer_role.rolname = 'canonical_brain_writer'
         WHERE membership.member = writer_role.oid
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_writer must not inherit any other database role';
    END IF;
    IF EXISTS (
        WITH RECURSIVE inheritors(oid, depth, admin_path) AS (
            SELECT membership.member, 1, membership.admin_option
              FROM pg_catalog.pg_auth_members AS membership
              JOIN pg_catalog.pg_roles AS writer_role
                ON writer_role.oid = membership.roleid
             WHERE writer_role.rolname = 'canonical_brain_writer'
            UNION ALL
            SELECT membership.member, inherited.depth + 1,
                   inherited.admin_path OR membership.admin_option
              FROM pg_catalog.pg_auth_members AS membership
              JOIN inheritors AS inherited
                ON inherited.oid = membership.roleid
        )
        SELECT 1
          FROM inheritors
          JOIN pg_catalog.pg_roles AS inheritor
            ON inheritor.oid = inheritors.oid
         WHERE inheritors.depth <> 1
            OR inheritors.admin_path
            OR NOT inheritor.rolcanlogin
            OR inheritor.rolsuper
            OR inheritor.rolcreatedb
            OR inheritor.rolcreaterole
            OR inheritor.rolreplication
            OR inheritor.rolbypassrls
    ) OR (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS writer_role
            ON writer_role.oid = membership.roleid
         WHERE writer_role.rolname = 'canonical_brain_writer'
    ) > 1 THEN
        RAISE EXCEPTION
            'canonical_brain_writer may have at most one direct least-privilege LOGIN member and no nested/admin membership';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS owner_role
            ON owner_role.rolname = 'canonical_brain_migration_owner'
         WHERE membership.roleid = owner_role.oid
            OR membership.member = owner_role.oid
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_migration_owner must have no role memberships or inheriting members';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn
           AND NOT database.datistemplate
           AND database.datname <> pg_catalog.current_database()
           AND pg_catalog.has_database_privilege(
               'canonical_brain_writer', database.datname, 'CONNECT'
           )
           AND NOT pg_temp._cw_managed_cloudsqladmin_exception(
               database.oid, cloudsqladmin_hba_receipt
           )
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_writer can CONNECT to another database; revoke that authority before migration';
    END IF;
    IF pg_catalog.to_regclass('public.canonical_event_log') IS NULL THEN
        RAISE EXCEPTION
            'prerequisite missing: public.canonical_event_log';
    END IF;
    IF (
        SELECT pg_catalog.pg_get_userbyid(class.relowner)
          FROM pg_catalog.pg_class AS class
         WHERE class.oid = 'public.canonical_event_log'::regclass
    ) <> 'canonical_brain_migration_owner' THEN
        RAISE EXCEPTION
            'prerequisite missing: public.canonical_event_log must already be owned by canonical_brain_migration_owner';
    END IF;
    IF NOT pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner', 'public', 'USAGE'
    ) OR pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner', 'public', 'CREATE'
    ) THEN
        RAISE EXCEPTION
            'prerequisite missing: canonical_brain_migration_owner requires USAGE-only effective authority on public schema';
    END IF;
    IF pg_catalog.current_setting('server_version_num')::integer < 160000
       OR pg_catalog.to_regprocedure('pg_catalog.sha256(bytea)') IS NULL THEN
        RAISE EXCEPTION
            'prerequisite missing: PostgreSQL 16+ SET-only role membership and core pg_catalog.sha256(bytea)';
    END IF;

    SELECT pg_catalog.string_agg(required.name, ',' ORDER BY required.name)
      INTO missing_columns
      FROM (
        VALUES
          ('event_id'), ('schema_version'), ('event_type'), ('occurred_at'),
          ('case_id'), ('source'), ('actor'), ('subject'), ('evidence'),
          ('decision'), ('status'), ('next_action'), ('safety'), ('payload')
      ) AS required(name)
     WHERE NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_attribute AS attribute
         WHERE attribute.attrelid = 'public.canonical_event_log'::regclass
           AND attribute.attname = required.name
           AND attribute.attnum > 0
           AND NOT attribute.attisdropped
     );
    IF missing_columns IS NOT NULL THEN
        RAISE EXCEPTION
            'public.canonical_event_log is incompatible; missing columns: %',
            missing_columns;
    END IF;
END
$prerequisites$;

-- Reject a preexisting schema unless it already belongs to the exact offline
-- owner.  Creation happens only after the administrator has acquired the
-- transaction-scoped SET-only membership below.
DO $owner_schema_prerequisite$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_namespace AS namespace
         WHERE namespace.nspname = 'canonical_brain'
           AND pg_catalog.pg_get_userbyid(namespace.nspowner)
               <> 'canonical_brain_migration_owner'
    ) THEN
        RAISE EXCEPTION
            'preexisting canonical_brain schema has an untrusted owner';
    END IF;
END
$owner_schema_prerequisite$;

DO $temporary_owner_membership$
DECLARE
    admin_name text := SESSION_USER;
    membership_valid boolean;
BEGIN
    IF admin_name = 'canonical_brain_migration_owner' THEN
        RAISE EXCEPTION 'offline migration owner cannot be the login session';
    END IF;
    PERFORM pg_catalog.set_config(
        'muncho.canonical_writer_migration_admin', admin_name, true
    );
    EXECUTE pg_catalog.format(
        'GRANT TEMPORARY ON DATABASE %I TO canonical_brain_migration_owner',
        pg_catalog.current_database()
    );
    EXECUTE pg_catalog.format(
        'GRANT canonical_brain_migration_owner TO %I '
        'WITH ADMIN FALSE, INHERIT FALSE, SET TRUE',
        admin_name
    );
    SELECT NOT membership.admin_option
           AND NOT membership.inherit_option
           AND membership.set_option
      INTO membership_valid
      FROM pg_catalog.pg_auth_members AS membership
      JOIN pg_catalog.pg_roles AS owner_role
        ON owner_role.oid = membership.roleid
      JOIN pg_catalog.pg_roles AS admin_role
        ON admin_role.oid = membership.member
     WHERE owner_role.rolname = 'canonical_brain_migration_owner'
       AND admin_role.rolname = admin_name;
    IF membership_valid IS DISTINCT FROM true
       OR NOT pg_catalog.pg_has_role(
            admin_name, 'canonical_brain_migration_owner', 'MEMBER'
       ) THEN
        RAISE EXCEPTION
            'transaction-scoped migration-owner membership was not exact';
    END IF;
END
$temporary_owner_membership$;

-- The managed administrator now has SET authority for the exact NOLOGIN
-- owner, so PostgreSQL permits AUTHORIZATION without granting the owner any
-- database CREATE capability.  This DDL and the membership share one rollback
-- boundary.
DO $owner_schema_create$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_namespace AS namespace
         WHERE namespace.nspname = 'canonical_brain'
    ) THEN
        CREATE SCHEMA canonical_brain
            AUTHORIZATION canonical_brain_migration_owner;
    END IF;
END
$owner_schema_create$;

SET LOCAL ROLE canonical_brain_migration_owner;

LOCK TABLE public.canonical_event_log IN ACCESS EXCLUSIVE MODE;

DO $event_log_contract$
DECLARE
    mismatch text;
    relation_record record;
BEGIN
    SELECT class.relkind, class.relpersistence, class.relispartition,
           class.relam, class.reltablespace, class.relrowsecurity,
           class.relforcerowsecurity, class.relreplident, class.reloptions
      INTO relation_record
      FROM pg_catalog.pg_class AS class
     WHERE class.oid = 'public.canonical_event_log'::regclass;
    IF relation_record.relkind <> 'r'
       OR relation_record.relpersistence <> 'p'
       OR relation_record.relispartition
       OR relation_record.relam <> (
            SELECT access_method.oid
              FROM pg_catalog.pg_am AS access_method
             WHERE access_method.amname = 'heap'
       )
       OR relation_record.reltablespace <> 0
       OR relation_record.relrowsecurity
       OR relation_record.relforcerowsecurity
       OR relation_record.relreplident <> 'd'
       OR relation_record.reloptions IS NOT NULL THEN
        RAISE EXCEPTION
            'public.canonical_event_log must be a permanent ordinary table with RLS disabled';
    END IF;

    WITH expected(column_name, data_type, ordinal) AS (
        VALUES
          ('event_id','uuid',1),
          ('schema_version','text',2),
          ('event_type','text',3),
          ('occurred_at','timestamp with time zone',4),
          ('case_id','text',5),
          ('source','jsonb',6),
          ('actor','jsonb',7),
          ('subject','jsonb',8),
          ('evidence','jsonb',9),
          ('decision','jsonb',10),
          ('status','jsonb',11),
          ('next_action','jsonb',12),
          ('safety','jsonb',13),
          ('payload','jsonb',14)
    ), actual AS (
        SELECT attribute.attname AS column_name,
               pg_catalog.format_type(attribute.atttypid, attribute.atttypmod)
                   AS data_type,
               attribute.attnum::integer AS ordinal,
               attribute.attnotnull,
               attribute.attidentity,
               attribute.attgenerated
          FROM pg_catalog.pg_attribute AS attribute
         WHERE attribute.attrelid = 'public.canonical_event_log'::regclass
           AND attribute.attnum > 0
           AND NOT attribute.attisdropped
    ), difference AS (
        (SELECT expected.column_name, expected.data_type, expected.ordinal FROM expected
         EXCEPT SELECT actual.column_name, actual.data_type, actual.ordinal FROM actual)
        UNION ALL
        (SELECT actual.column_name, actual.data_type, actual.ordinal FROM actual
         EXCEPT SELECT expected.column_name, expected.data_type, expected.ordinal FROM expected)
    )
    SELECT pg_catalog.string_agg(
               difference.ordinal::text || ':' || difference.column_name || ':'
                   || difference.data_type,
               ',' ORDER BY difference.column_name
           )
      INTO mismatch
      FROM difference;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION
            'public.canonical_event_log exact column/type contract mismatch: %',
            mismatch;
    END IF;

    SELECT pg_catalog.string_agg(actual.attname, ',' ORDER BY actual.attnum)
      INTO mismatch
      FROM pg_catalog.pg_attribute AS actual
      JOIN pg_catalog.pg_type AS data_type
        ON data_type.oid = actual.atttypid
     WHERE actual.attrelid = 'public.canonical_event_log'::regclass
       AND actual.attnum > 0
       AND NOT actual.attisdropped
       AND (
            NOT actual.attnotnull
            OR actual.attidentity <> ''
            OR actual.attgenerated <> ''
            OR actual.atthasdef
            OR actual.atthasmissing
            OR NOT actual.attislocal
            OR actual.attinhcount <> 0
            OR actual.attndims <> 0
            OR actual.attcollation <> data_type.typcollation
            OR actual.attstorage <> data_type.typstorage
            OR actual.attstattarget <> -1
            OR actual.attoptions IS NOT NULL
            OR actual.attfdwoptions IS NOT NULL
       );
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION
            'public.canonical_event_log nullability/identity contract mismatch: %',
            mismatch;
    END IF;

    SELECT pg_catalog.string_agg(
               constraint_row.contype::text || ':'
               || pg_catalog.pg_get_constraintdef(constraint_row.oid, true),
               ';' ORDER BY constraint_row.oid
           )
      INTO mismatch
      FROM pg_catalog.pg_constraint AS constraint_row
     WHERE constraint_row.conrelid = 'public.canonical_event_log'::regclass
       -- PostgreSQL 18 materializes column NOT NULL state as ``contype = 'n'``
       -- rows.  Column nullability is attested independently above, so those
       -- version-specific catalog rows are not additional table constraints.
       AND constraint_row.contype <> 'n'
       AND pg_catalog.pg_get_constraintdef(constraint_row.oid, true)
           <> 'PRIMARY KEY (event_id)';
    IF mismatch IS NOT NULL OR NOT EXISTS (
        SELECT 1
         FROM pg_catalog.pg_constraint AS constraint_row
         WHERE constraint_row.conrelid = 'public.canonical_event_log'::regclass
           AND pg_catalog.pg_get_constraintdef(constraint_row.oid, true)
               = 'PRIMARY KEY (event_id)'
    ) THEN
        RAISE EXCEPTION
            'public.canonical_event_log must have only the exact event_id primary key constraint';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_trigger AS trigger
         WHERE trigger.tgrelid = 'public.canonical_event_log'::regclass
           AND NOT trigger.tgisinternal
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_rewrite AS rewrite
         WHERE rewrite.ev_class = 'public.canonical_event_log'::regclass
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_policy AS policy
         WHERE policy.polrelid = 'public.canonical_event_log'::regclass
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_inherits AS inheritance
         WHERE inheritance.inhrelid = 'public.canonical_event_log'::regclass
            OR inheritance.inhparent = 'public.canonical_event_log'::regclass
    ) THEN
        RAISE EXCEPTION
            'public.canonical_event_log forbids user triggers, rules, policies, and inheritance';
    END IF;
END
$event_log_contract$;

REVOKE ALL ON TABLE public.canonical_event_log FROM PUBLIC;
DO $retire_event_log_writers$
DECLARE
    acl_record record;
    grantee_sql text;
BEGIN
    FOR acl_record IN
        SELECT DISTINCT direct_acl.grantee, direct_acl.column_name
          FROM (
                SELECT acl.grantee, class.relowner, NULL::text AS column_name
                  FROM pg_catalog.pg_class AS class
                  CROSS JOIN LATERAL pg_catalog.aclexplode(
                      class.relacl
                  ) AS acl
                 WHERE class.oid = 'public.canonical_event_log'::regclass
                UNION ALL
                SELECT acl.grantee, class.relowner, attribute.attname
                    AS column_name
                  FROM pg_catalog.pg_class AS class
                  JOIN pg_catalog.pg_attribute AS attribute
                    ON attribute.attrelid = class.oid
                   AND attribute.attnum > 0
                   AND NOT attribute.attisdropped
                  CROSS JOIN LATERAL pg_catalog.aclexplode(
                      attribute.attacl
                  ) AS acl
                 WHERE class.oid = 'public.canonical_event_log'::regclass
          ) AS direct_acl
         WHERE direct_acl.grantee <> direct_acl.relowner
    LOOP
        grantee_sql := CASE WHEN acl_record.grantee = 0 THEN 'PUBLIC' ELSE
            pg_catalog.format(
                '%I', pg_catalog.pg_get_userbyid(acl_record.grantee)
            ) END;
        IF acl_record.column_name IS NULL THEN
            EXECUTE pg_catalog.format(
                'REVOKE ALL PRIVILEGES ON TABLE public.canonical_event_log FROM %s CASCADE',
                grantee_sql
            );
        ELSE
            EXECUTE pg_catalog.format(
                'REVOKE ALL PRIVILEGES (%I) ON TABLE public.canonical_event_log FROM %s CASCADE',
                acl_record.column_name,
                grantee_sql
            );
        END IF;
    END LOOP;
END
$retire_event_log_writers$;

DO $event_log_exclusivity$
DECLARE
    owner_oid oid;
BEGIN
    SELECT role.oid INTO owner_oid
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname = 'canonical_brain_migration_owner';
    IF (
        SELECT class.relowner <> owner_oid
            OR class.relrowsecurity
            OR class.relforcerowsecurity
          FROM pg_catalog.pg_class AS class
         WHERE class.oid = 'public.canonical_event_log'::regclass
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_class AS class
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(class.relacl, pg_catalog.acldefault('r', class.relowner))
          ) AS acl
         WHERE class.oid = 'public.canonical_event_log'::regclass
           AND acl.grantee NOT IN (owner_oid)
           AND acl.privilege_type IN (
                'SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER'
           )
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_attribute AS attribute
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              attribute.attacl
          ) AS acl
         WHERE attribute.attrelid = 'public.canonical_event_log'::regclass
           AND attribute.attnum > 0
           AND NOT attribute.attisdropped
           AND acl.grantee NOT IN (owner_oid)
           AND acl.privilege_type IN (
                'SELECT','INSERT','UPDATE','REFERENCES'
           )
    ) OR (
        SELECT pg_catalog.count(*) <> 1
            OR COALESCE(pg_catalog.bool_or(
                NOT index.indisprimary
                OR NOT index.indisunique
                OR index.indisexclusion
                OR NOT index.indimmediate
                OR NOT index.indisvalid
                OR NOT index.indisready
                OR NOT index.indislive
                OR index.indisclustered
                OR index.indisreplident
                OR index.indcheckxmin
                OR index.indnkeyatts <> 1
                OR index.indnatts <> 1
                OR index.indexprs IS NOT NULL
                OR index.indpred IS NOT NULL
                OR ARRAY(
                    SELECT key_part.attnum
                      FROM pg_catalog.unnest(index.indkey)
                           WITH ORDINALITY AS key_part(attnum, ordinal)
                     ORDER BY key_part.ordinal
                ) <> ARRAY[(
                    SELECT attribute.attnum
                      FROM pg_catalog.pg_attribute AS attribute
                     WHERE attribute.attrelid =
                           'public.canonical_event_log'::regclass
                       AND attribute.attname = 'event_id'
                )::smallint]::smallint[]
                OR access_method.amname <> 'btree'
                OR ARRAY(
                    SELECT key_part.opclass_oid
                      FROM pg_catalog.unnest(index.indclass)
                           WITH ORDINALITY AS key_part(opclass_oid, ordinal)
                     ORDER BY key_part.ordinal
                ) <> ARRAY[(
                    SELECT operator_class.oid
                      FROM pg_catalog.pg_opclass AS operator_class
                      JOIN pg_catalog.pg_am AS operator_method
                        ON operator_method.oid = operator_class.opcmethod
                     WHERE operator_method.amname = 'btree'
                       AND operator_class.opcintype = 'uuid'::regtype
                       AND operator_class.opcdefault
                )::oid]::oid[]
                OR ARRAY(
                    SELECT key_part.collation_oid
                      FROM pg_catalog.unnest(index.indcollation)
                           WITH ORDINALITY AS key_part(collation_oid, ordinal)
                     ORDER BY key_part.ordinal
                ) <> ARRAY[0::oid]::oid[]
                OR ARRAY(
                    SELECT key_part.option_value
                      FROM pg_catalog.unnest(index.indoption)
                           WITH ORDINALITY AS key_part(option_value, ordinal)
                     ORDER BY key_part.ordinal
                ) <> ARRAY[0::smallint]::smallint[]
                OR index_class.relpersistence <> 'p'
                OR index_class.reloptions IS NOT NULL
                OR index_class.reltablespace <> 0
                OR pg_catalog.pg_get_userbyid(index_class.relowner)
                   <> 'canonical_brain_migration_owner'
                OR NOT EXISTS (
                    SELECT 1
                      FROM pg_catalog.pg_constraint AS constraint_row
                     WHERE constraint_row.conrelid =
                           'public.canonical_event_log'::regclass
                       AND constraint_row.contype = 'p'
                       AND constraint_row.conindid = index.indexrelid
                )
            ), true)
          FROM pg_catalog.pg_index AS index
          JOIN pg_catalog.pg_class AS index_class
            ON index_class.oid = index.indexrelid
          JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = index_class.relam
         WHERE index.indrelid = 'public.canonical_event_log'::regclass
    ) THEN
        RAISE EXCEPTION
            'public.canonical_event_log owner/RLS/ACL exclusivity attestation failed';
    END IF;
END
$event_log_exclusivity$;

DO $preexisting_tables$
DECLARE
    table_name text;
    actual_owner text;
    unexpected_tables text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'writer_routeback_authorizations',
        'writer_routeback_lifecycle_terminals',
        'writer_routeback_terminals',
        'writer_public_routeback_targets',
        'writer_event_provenance',
        'writer_capability_grants',
        'writer_capability_revocation_scopes',
        'writer_capability_revocations',
        'writer_capability_consumptions',
        'writer_canary_scope_preapprovals',
        'writer_canary_scope_preapproval_retirements',
        'writer_canary_scope_claims'
    ]
    LOOP
        SELECT pg_catalog.pg_get_userbyid(class.relowner)
          INTO actual_owner
          FROM pg_catalog.pg_class AS class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
         WHERE namespace.nspname = 'canonical_brain'
           AND class.relname = table_name
           AND class.relkind IN ('r', 'p');
        IF FOUND AND actual_owner <> 'canonical_brain_migration_owner' THEN
            RAISE EXCEPTION
                'preexisting %.% has untrusted owner %',
                'canonical_brain', table_name, actual_owner;
        END IF;
    END LOOP;
    SELECT pg_catalog.string_agg(class.relname, ',' ORDER BY class.relname)
      INTO unexpected_tables
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
     WHERE namespace.nspname = 'canonical_brain'
       AND class.relkind IN ('r', 'p')
       AND class.relname LIKE 'writer\_%' ESCAPE '\'
       AND class.relname <> ALL (ARRAY[
            'writer_routeback_authorizations',
            'writer_routeback_lifecycle_terminals',
            'writer_routeback_terminals',
            'writer_public_routeback_targets',
            'writer_event_provenance',
            'writer_capability_grants',
            'writer_capability_revocation_scopes',
            'writer_capability_revocations',
            'writer_capability_consumptions',
            'writer_canary_scope_preapprovals',
            'writer_canary_scope_preapproval_retirements',
            'writer_canary_scope_claims'
       ]);
    IF unexpected_tables IS NOT NULL THEN
        RAISE EXCEPTION
            'unexpected preexisting canonical writer tables: %', unexpected_tables;
    END IF;
END
$preexisting_tables$;

-- All writer state is append-only.  Current state is derived from immutable
-- grants/claims plus terminal/revocation/consumption rows.  The canonical event
-- log receives a receipt in the same transaction as every authoritative state
-- transition.
CREATE TABLE IF NOT EXISTS canonical_brain.writer_routeback_authorizations (
    authorization_id text PRIMARY KEY,
    case_id text NOT NULL,
    target_ref jsonb NOT NULL,
    message_summary text NOT NULL,
    source_refs jsonb NOT NULL,
    content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    session_key_sha256 text NOT NULL CHECK (session_key_sha256 ~ '^[0-9a-f]{64}$'),
    capability_epoch_sha256 text NOT NULL
        CHECK (capability_epoch_sha256 ~ '^[0-9a-f]{64}$'),
    runtime_platform text NOT NULL,
    source_thread_id text NOT NULL,
    idempotency_key text NOT NULL
        CHECK (pg_catalog.octet_length(idempotency_key) BETWEEN 1 AND 256),
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    intent_event_id uuid NOT NULL,
    UNIQUE (case_id, idempotency_key)
);

-- A preclaim blocker is terminal for the global case+lifecycle key without
-- fabricating a send authorization.  The claim routine consults this table
-- under the same lifecycle lock before it can create an authorization row.
CREATE TABLE IF NOT EXISTS canonical_brain.writer_routeback_lifecycle_terminals (
    lifecycle_id text PRIMARY KEY,
    case_id text NOT NULL,
    idempotency_key text NOT NULL
        CHECK (pg_catalog.octet_length(idempotency_key) BETWEEN 1 AND 256),
    target_ref jsonb NOT NULL,
    message_summary text NOT NULL,
    source_refs jsonb NOT NULL,
    outcome text NOT NULL CHECK (outcome IN ('sent', 'blocked')),
    receipt jsonb NOT NULL,
    blocker_reason text NOT NULL,
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    session_key_sha256 text NOT NULL CHECK (session_key_sha256 ~ '^[0-9a-f]{64}$'),
    capability_epoch_sha256 text NOT NULL
        CHECK (capability_epoch_sha256 ~ '^[0-9a-f]{64}$'),
    finalized_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    terminal_event_id uuid NOT NULL,
    UNIQUE (case_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS canonical_brain.writer_routeback_terminals (
    authorization_id text PRIMARY KEY
        REFERENCES canonical_brain.writer_routeback_authorizations(authorization_id),
    outcome text NOT NULL CHECK (outcome IN ('sent', 'blocked')),
    receipt jsonb NOT NULL,
    blocker_reason text NOT NULL,
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    finalized_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    terminal_event_id uuid NOT NULL
);

-- Empty by default.  An owner-approved deployment migration may add exact
-- public channel/thread identifiers.  Runtime code cannot populate this ACL.
CREATE TABLE IF NOT EXISTS canonical_brain.writer_public_routeback_targets (
    channel_id text PRIMARY KEY,
    target_type text NOT NULL CHECK (target_type IN ('public_channel', 'public_thread')),
    approved_by text NOT NULL,
    approved_at timestamptz NOT NULL,
    enabled boolean NOT NULL DEFAULT true
);

-- Only _append_event can populate this table.  It starts empty at cutover, so
-- legacy rows written through the shared helper are never silently promoted
-- into trusted runtime scope.
CREATE TABLE IF NOT EXISTS canonical_brain.writer_event_provenance (
    event_id uuid PRIMARY KEY,
    canonical_content_sha256 text NOT NULL
        CHECK (canonical_content_sha256 ~ '^[0-9a-f]{64}$'),
    origin text NOT NULL,
    trusted_runtime jsonb NOT NULL
        CHECK (pg_catalog.jsonb_typeof(trusted_runtime) = 'object'),
    appended_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

CREATE TABLE IF NOT EXISTS canonical_brain.writer_capability_grants (
    approval_id text PRIMARY KEY,
    case_id text NOT NULL,
    plan_id text NOT NULL,
    plan_revision integer NOT NULL CHECK (plan_revision BETWEEN 1 AND 999999999),
    session_key_sha256 text NOT NULL CHECK (session_key_sha256 ~ '^[0-9a-f]{64}$'),
    capability_epoch_sha256 text NOT NULL
        CHECK (capability_epoch_sha256 ~ '^[0-9a-f]{64}$'),
    approved_by_user_id text NOT NULL,
    approval_source_sha256 text NOT NULL UNIQUE
        CHECK (approval_source_sha256 ~ '^[0-9a-f]{64}$'),
    command_hashes jsonb NOT NULL CHECK (pg_catalog.jsonb_typeof(command_hashes) = 'array'),
    expires_at timestamptz NOT NULL,
    max_uses integer NOT NULL CHECK (max_uses BETWEEN 1 AND 1000),
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    granted_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    grant_event_id uuid NOT NULL
);

-- A scope tombstone prevents a revoke/grant race from resurrecting authority.
-- Plan tombstones bind one plan in one routing epoch; session tombstones bind
-- the whole session epoch and store an empty plan_id.  Fresh epochs remain a
-- deliberate, separately authenticated authority boundary.
CREATE TABLE IF NOT EXISTS canonical_brain.writer_capability_revocation_scopes (
    scope_type text NOT NULL CHECK (scope_type IN ('plan', 'session')),
    session_key_sha256 text NOT NULL CHECK (session_key_sha256 ~ '^[0-9a-f]{64}$'),
    capability_epoch_sha256 text NOT NULL
        CHECK (capability_epoch_sha256 ~ '^[0-9a-f]{64}$'),
    plan_id text NOT NULL,
    reason text NOT NULL,
    revoked_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (
        scope_type, session_key_sha256, capability_epoch_sha256, plan_id
    )
);

CREATE TABLE IF NOT EXISTS canonical_brain.writer_capability_revocations (
    approval_id text PRIMARY KEY
        REFERENCES canonical_brain.writer_capability_grants(approval_id),
    reason text NOT NULL,
    revoked_by_session_sha256 text NOT NULL
        CHECK (revoked_by_session_sha256 ~ '^[0-9a-f]{64}$'),
    revoked_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

CREATE TABLE IF NOT EXISTS canonical_brain.writer_capability_consumptions (
    consume_id uuid PRIMARY KEY,
    approval_id text NOT NULL
        REFERENCES canonical_brain.writer_capability_grants(approval_id),
    command_sha256 text NOT NULL CHECK (command_sha256 ~ '^[0-9a-f]{64}$'),
    session_key_sha256 text NOT NULL CHECK (session_key_sha256 ~ '^[0-9a-f]{64}$'),
    capability_epoch_sha256 text NOT NULL
        CHECK (capability_epoch_sha256 ~ '^[0-9a-f]{64}$'),
    idempotency_key text NOT NULL
        CHECK (pg_catalog.octet_length(idempotency_key) BETWEEN 1 AND 256),
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    remaining_uses integer NOT NULL CHECK (remaining_uses >= 0),
    consumed_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    receipt_event_id uuid NOT NULL,
    response jsonb NOT NULL,
    UNIQUE (session_key_sha256, capability_epoch_sha256, idempotency_key)
);

-- A canary scope exists only after an owner-bound, root-configured bootstrap
-- preapproval.  Runtime/model callers cannot create or broaden this row.
CREATE TABLE IF NOT EXISTS canonical_brain.writer_canary_scope_preapprovals (
    grant_id text PRIMARY KEY,
    case_id text NOT NULL UNIQUE,
    release_sha256 text NOT NULL CHECK (release_sha256 ~ '^[0-9a-f]{64}$'),
    fixture_sha256 text NOT NULL CHECK (fixture_sha256 ~ '^[0-9a-f]{64}$'),
    run_id text NOT NULL UNIQUE,
    session_key_sha256 text NOT NULL CHECK (session_key_sha256 ~ '^[0-9a-f]{64}$'),
    expires_at timestamptz NOT NULL,
    approved_by text NOT NULL,
    approval_source_sha256 text NOT NULL UNIQUE
        CHECK (approval_source_sha256 ~ '^[0-9a-f]{64}$'),
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    preapproved_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    receipt_event_id uuid NOT NULL
);

-- A writer shutdown/recovery that wins the preclaim lock records one durable
-- terminal.  The row is append-only; later retries return its first-writer
-- event/provenance instead of authoring a second semantic outcome.
CREATE TABLE IF NOT EXISTS canonical_brain.writer_canary_scope_preapproval_retirements (
    grant_id text PRIMARY KEY
        REFERENCES canonical_brain.writer_canary_scope_preapprovals(grant_id),
    case_id text NOT NULL,
    release_sha256 text NOT NULL CHECK (release_sha256 ~ '^[0-9a-f]{64}$'),
    fixture_sha256 text NOT NULL CHECK (fixture_sha256 ~ '^[0-9a-f]{64}$'),
    run_id text NOT NULL,
    session_key_sha256 text NOT NULL CHECK (session_key_sha256 ~ '^[0-9a-f]{64}$'),
    expires_at timestamptz NOT NULL,
    approved_by text NOT NULL,
    approval_source_sha256 text NOT NULL
        CHECK (approval_source_sha256 ~ '^[0-9a-f]{64}$'),
    provisioning_receipt_sha256 text NOT NULL
        CHECK (provisioning_receipt_sha256 ~ '^[0-9a-f]{64}$'),
    preapproval_event_id uuid NOT NULL,
    bootstrap_consumption_event_id uuid NOT NULL,
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    reason text NOT NULL
        CHECK (reason = 'activation_failed_before_first_claim'),
    retired_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    retirement_event_id uuid NOT NULL
);

-- The claim is append-only and one-shot by grant_id.  Its epoch comes only
-- from the authenticated gateway RuntimeContext, never from request payload.
CREATE TABLE IF NOT EXISTS canonical_brain.writer_canary_scope_claims (
    grant_id text PRIMARY KEY
        REFERENCES canonical_brain.writer_canary_scope_preapprovals(grant_id),
    case_id text NOT NULL UNIQUE,
    release_sha256 text NOT NULL CHECK (release_sha256 ~ '^[0-9a-f]{64}$'),
    fixture_sha256 text NOT NULL CHECK (fixture_sha256 ~ '^[0-9a-f]{64}$'),
    run_id text NOT NULL UNIQUE,
    approval_source_sha256 text NOT NULL
        CHECK (approval_source_sha256 ~ '^[0-9a-f]{64}$'),
    session_key_sha256 text NOT NULL CHECK (session_key_sha256 ~ '^[0-9a-f]{64}$'),
    capability_epoch_sha256 text NOT NULL
        CHECK (capability_epoch_sha256 ~ '^[0-9a-f]{64}$'),
    expires_at timestamptz NOT NULL,
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    claimed_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    claim_event_id uuid NOT NULL
);

CREATE INDEX IF NOT EXISTS writer_routeback_target_idx
    ON canonical_brain.writer_routeback_authorizations
    ((COALESCE(target_ref->>'thread_id', target_ref->>'channel_id')));
CREATE INDEX IF NOT EXISTS writer_event_provenance_session_idx
    ON canonical_brain.writer_event_provenance
    ((trusted_runtime->>'session_key_sha256'));
CREATE INDEX IF NOT EXISTS writer_event_provenance_thread_idx
    ON canonical_brain.writer_event_provenance
    ((trusted_runtime->>'platform'),
     (COALESCE(trusted_runtime->>'thread_id', trusted_runtime->>'chat_id')));
CREATE INDEX IF NOT EXISTS writer_capability_scope_idx
    ON canonical_brain.writer_capability_grants
    (session_key_sha256, capability_epoch_sha256, case_id, plan_id, plan_revision);
CREATE INDEX IF NOT EXISTS writer_capability_command_idx
    ON canonical_brain.writer_capability_grants USING gin (command_hashes);
CREATE INDEX IF NOT EXISTS writer_capability_use_idx
    ON canonical_brain.writer_capability_consumptions
    (approval_id, command_sha256);
CREATE INDEX IF NOT EXISTS writer_canary_preapproval_scope_idx
    ON canonical_brain.writer_canary_scope_preapprovals
    (case_id, session_key_sha256, expires_at);
CREATE INDEX IF NOT EXISTS writer_canary_preapproval_retirement_scope_idx
    ON canonical_brain.writer_canary_scope_preapproval_retirements
    (case_id, session_key_sha256, retired_at);
CREATE INDEX IF NOT EXISTS writer_canary_claim_scope_idx
    ON canonical_brain.writer_canary_scope_claims
    (case_id, session_key_sha256, capability_epoch_sha256);

ALTER TABLE canonical_brain.writer_routeback_authorizations
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_routeback_lifecycle_terminals
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_routeback_terminals
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_public_routeback_targets
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_event_provenance
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_capability_grants
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_capability_revocation_scopes
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_capability_revocations
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_capability_consumptions
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_canary_scope_preapprovals
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_canary_scope_preapproval_retirements
    OWNER TO canonical_brain_migration_owner;
ALTER TABLE canonical_brain.writer_canary_scope_claims
    OWNER TO canonical_brain_migration_owner;

-- PostgreSQL itself parses and deparses the exact CHECK expressions used by
-- this artifact.  Comparing the real tables to this transaction-local template
-- avoids brittle major-version string guesses while still rejecting any
-- logically different or additional CHECK constraint on a rerun.
CREATE TEMPORARY TABLE canonical_writer_check_contract (
    content_sha256 text CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    session_key_sha256 text CHECK (session_key_sha256 ~ '^[0-9a-f]{64}$'),
    idempotency_key text
        CHECK (pg_catalog.octet_length(idempotency_key) BETWEEN 1 AND 256),
    request_sha256 text CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    release_sha256 text CHECK (release_sha256 ~ '^[0-9a-f]{64}$'),
    fixture_sha256 text CHECK (fixture_sha256 ~ '^[0-9a-f]{64}$'),
    outcome text CHECK (outcome IN ('sent', 'blocked')),
    target_type text CHECK (target_type IN ('public_channel', 'public_thread')),
    canonical_content_sha256 text
        CHECK (canonical_content_sha256 ~ '^[0-9a-f]{64}$'),
    trusted_runtime jsonb
        CHECK (pg_catalog.jsonb_typeof(trusted_runtime) = 'object'),
    scope_type text CHECK (scope_type IN ('plan', 'session')),
    plan_revision integer CHECK (plan_revision BETWEEN 1 AND 999999999),
    capability_epoch_sha256 text
        CHECK (capability_epoch_sha256 ~ '^[0-9a-f]{64}$'),
    approval_source_sha256 text
        CHECK (approval_source_sha256 ~ '^[0-9a-f]{64}$'),
    provisioning_receipt_sha256 text
        CHECK (provisioning_receipt_sha256 ~ '^[0-9a-f]{64}$'),
    command_hashes jsonb
        CHECK (pg_catalog.jsonb_typeof(command_hashes) = 'array'),
    max_uses integer CHECK (max_uses BETWEEN 1 AND 1000),
    revoked_by_session_sha256 text
        CHECK (revoked_by_session_sha256 ~ '^[0-9a-f]{64}$'),
    command_sha256 text CHECK (command_sha256 ~ '^[0-9a-f]{64}$'),
    remaining_uses integer CHECK (remaining_uses >= 0),
    reason text CHECK (reason = 'activation_failed_before_first_claim')
) ON COMMIT DROP;

CREATE TEMPORARY TABLE canonical_writer_index_contract (
    target_ref jsonb,
    trusted_runtime jsonb,
    session_key_sha256 text,
    capability_epoch_sha256 text,
    case_id text,
    plan_id text,
    plan_revision integer,
    command_hashes jsonb,
    approval_id text,
    command_sha256 text,
    expires_at timestamptz,
    retired_at timestamptz
) ON COMMIT DROP;
CREATE INDEX contract_routeback_target_idx
    ON canonical_writer_index_contract
    ((COALESCE(target_ref->>'thread_id', target_ref->>'channel_id')));
CREATE INDEX contract_event_provenance_session_idx
    ON canonical_writer_index_contract
    ((trusted_runtime->>'session_key_sha256'));
CREATE INDEX contract_event_provenance_thread_idx
    ON canonical_writer_index_contract
    ((trusted_runtime->>'platform'),
     (COALESCE(trusted_runtime->>'thread_id', trusted_runtime->>'chat_id')));
CREATE INDEX contract_capability_scope_idx
    ON canonical_writer_index_contract
    (session_key_sha256, capability_epoch_sha256, case_id, plan_id, plan_revision);
CREATE INDEX contract_capability_command_idx
    ON canonical_writer_index_contract USING gin (command_hashes);
CREATE INDEX contract_capability_use_idx
    ON canonical_writer_index_contract (approval_id, command_sha256);
CREATE INDEX contract_canary_preapproval_scope_idx
    ON canonical_writer_index_contract (case_id, session_key_sha256, expires_at);
CREATE INDEX contract_canary_preapproval_retirement_scope_idx
    ON canonical_writer_index_contract (case_id, session_key_sha256, retired_at);
CREATE INDEX contract_canary_claim_scope_idx
    ON canonical_writer_index_contract
    (case_id, session_key_sha256, capability_epoch_sha256);

DO $table_contract$
DECLARE
    mismatch text;
BEGIN
    SELECT pg_catalog.string_agg(class.relname, ',' ORDER BY class.relname)
      INTO mismatch
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
     WHERE namespace.nspname = 'canonical_brain'
       AND class.relname IN (
            'writer_routeback_authorizations',
            'writer_routeback_lifecycle_terminals',
            'writer_routeback_terminals',
            'writer_public_routeback_targets',
            'writer_event_provenance',
            'writer_capability_grants',
            'writer_capability_revocation_scopes',
            'writer_capability_revocations',
            'writer_capability_consumptions',
            'writer_canary_scope_preapprovals',
            'writer_canary_scope_preapproval_retirements',
            'writer_canary_scope_claims'
       )
       AND (
            class.relkind <> 'r'
            OR class.relpersistence <> 'p'
            OR class.relispartition
            OR class.relam <> (
                SELECT access_method.oid
                  FROM pg_catalog.pg_am AS access_method
                 WHERE access_method.amname = 'heap'
            )
            OR class.reltablespace <> 0
            OR class.relrowsecurity
            OR class.relforcerowsecurity
            OR class.relreplident <> 'd'
            OR class.reloptions IS NOT NULL
            OR pg_catalog.pg_get_userbyid(class.relowner)
               <> 'canonical_brain_migration_owner'
       );
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'preexisting writer table relation contract mismatch: %', mismatch;
    END IF;

    SELECT pg_catalog.string_agg(surface, ',' ORDER BY surface)
      INTO mismatch
      FROM (
        SELECT class.relname || ':trigger:' || trigger.tgname AS surface
          FROM pg_catalog.pg_trigger AS trigger
          JOIN pg_catalog.pg_class AS class ON class.oid = trigger.tgrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
         WHERE namespace.nspname = 'canonical_brain'
           AND class.relname LIKE 'writer\_%' ESCAPE '\'
           AND NOT trigger.tgisinternal
        UNION ALL
        SELECT class.relname || ':rule:' || rewrite.rulename
          FROM pg_catalog.pg_rewrite AS rewrite
          JOIN pg_catalog.pg_class AS class ON class.oid = rewrite.ev_class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
         WHERE namespace.nspname = 'canonical_brain'
           AND class.relname LIKE 'writer\_%' ESCAPE '\'
        UNION ALL
        SELECT class.relname || ':policy:' || policy.polname
          FROM pg_catalog.pg_policy AS policy
          JOIN pg_catalog.pg_class AS class ON class.oid = policy.polrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
         WHERE namespace.nspname = 'canonical_brain'
           AND class.relname LIKE 'writer\_%' ESCAPE '\'
        UNION ALL
        SELECT child.relname || ':inheritance'
          FROM pg_catalog.pg_inherits AS inheritance
          JOIN pg_catalog.pg_class AS child ON child.oid = inheritance.inhrelid
          JOIN pg_catalog.pg_class AS parent ON parent.oid = inheritance.inhparent
          JOIN pg_catalog.pg_namespace AS child_namespace
            ON child_namespace.oid = child.relnamespace
          JOIN pg_catalog.pg_namespace AS parent_namespace
            ON parent_namespace.oid = parent.relnamespace
         WHERE (child_namespace.nspname = 'canonical_brain'
                AND child.relname LIKE 'writer\_%' ESCAPE '\')
            OR (parent_namespace.nspname = 'canonical_brain'
                AND parent.relname LIKE 'writer\_%' ESCAPE '\')
      ) AS forbidden_surfaces;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'preexisting writer table active surface forbidden: %', mismatch;
    END IF;

    WITH expected(table_name, ordered_columns) AS (
        VALUES
          ('writer_routeback_authorizations', ARRAY[
              'authorization_id','case_id','target_ref','message_summary',
              'source_refs','content_sha256','session_key_sha256',
              'capability_epoch_sha256','runtime_platform',
              'source_thread_id','idempotency_key',
              'request_sha256','created_at','intent_event_id'
          ]),
          ('writer_routeback_lifecycle_terminals', ARRAY[
              'lifecycle_id','case_id','idempotency_key','target_ref',
              'message_summary','source_refs','outcome','receipt',
              'blocker_reason','request_sha256','session_key_sha256',
              'capability_epoch_sha256','finalized_at','terminal_event_id'
          ]),
          ('writer_routeback_terminals', ARRAY[
              'authorization_id','outcome','receipt','blocker_reason',
              'request_sha256','finalized_at','terminal_event_id'
          ]),
          ('writer_public_routeback_targets', ARRAY[
              'channel_id','target_type','approved_by','approved_at','enabled'
          ]),
          ('writer_event_provenance', ARRAY[
              'event_id','canonical_content_sha256','origin','trusted_runtime',
              'appended_at'
          ]),
          ('writer_capability_grants', ARRAY[
              'approval_id','case_id','plan_id','plan_revision',
              'session_key_sha256','capability_epoch_sha256',
              'approved_by_user_id','approval_source_sha256','command_hashes',
              'expires_at','max_uses','request_sha256','granted_at','grant_event_id'
          ]),
          ('writer_capability_revocation_scopes', ARRAY[
              'scope_type','session_key_sha256','capability_epoch_sha256',
              'plan_id','reason','revoked_at'
          ]),
          ('writer_capability_revocations', ARRAY[
              'approval_id','reason','revoked_by_session_sha256','revoked_at'
          ]),
          ('writer_capability_consumptions', ARRAY[
              'consume_id','approval_id','command_sha256','session_key_sha256',
              'capability_epoch_sha256','idempotency_key','request_sha256',
              'remaining_uses','consumed_at','receipt_event_id','response'
          ]),
          ('writer_canary_scope_preapprovals', ARRAY[
              'grant_id','case_id','release_sha256','fixture_sha256','run_id',
              'session_key_sha256','expires_at','approved_by',
              'approval_source_sha256','request_sha256','preapproved_at',
              'receipt_event_id'
          ]),
          ('writer_canary_scope_preapproval_retirements', ARRAY[
              'grant_id','case_id','release_sha256','fixture_sha256','run_id',
              'session_key_sha256','expires_at','approved_by',
              'approval_source_sha256','provisioning_receipt_sha256',
              'preapproval_event_id','bootstrap_consumption_event_id',
              'request_sha256','reason','retired_at','retirement_event_id'
          ]),
          ('writer_canary_scope_claims', ARRAY[
              'grant_id','case_id','release_sha256','fixture_sha256','run_id',
              'approval_source_sha256','session_key_sha256',
              'capability_epoch_sha256','expires_at','request_sha256',
              'claimed_at','claim_event_id'
          ])
    ), actual AS (
        SELECT class.relname AS table_name,
               pg_catalog.array_agg(
                   attribute.attname ORDER BY attribute.attnum
               ) AS ordered_columns
          FROM pg_catalog.pg_class AS class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
          JOIN pg_catalog.pg_attribute AS attribute
            ON attribute.attrelid = class.oid
           AND attribute.attnum > 0
           AND NOT attribute.attisdropped
         WHERE namespace.nspname = 'canonical_brain'
           AND class.relname IN (
                'writer_routeback_authorizations',
                'writer_routeback_lifecycle_terminals',
                'writer_routeback_terminals',
                'writer_public_routeback_targets',
                'writer_event_provenance',
                'writer_capability_grants',
                'writer_capability_revocation_scopes',
                'writer_capability_revocations',
                'writer_capability_consumptions',
                'writer_canary_scope_preapprovals',
                'writer_canary_scope_preapproval_retirements',
                'writer_canary_scope_claims'
           )
         GROUP BY class.relname
    ), difference AS (
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        UNION ALL
        (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT pg_catalog.string_agg(
               table_name || ':' || pg_catalog.array_to_string(ordered_columns, ','),
               ';' ORDER BY table_name
           )
      INTO mismatch
      FROM difference;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'preexisting writer table column contract mismatch: %', mismatch;
    END IF;

    SELECT pg_catalog.string_agg(
               class.relname || '.' || attribute.attname || ':'
               || pg_catalog.format_type(attribute.atttypid, attribute.atttypmod),
               ',' ORDER BY class.relname, attribute.attnum
           )
      INTO mismatch
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
      JOIN pg_catalog.pg_attribute AS attribute
        ON attribute.attrelid = class.oid
       AND attribute.attnum > 0
       AND NOT attribute.attisdropped
      JOIN pg_catalog.pg_type AS data_type
        ON data_type.oid = attribute.atttypid
     WHERE namespace.nspname = 'canonical_brain'
       AND class.relname IN (
            'writer_routeback_authorizations',
            'writer_routeback_lifecycle_terminals',
            'writer_routeback_terminals',
            'writer_public_routeback_targets',
            'writer_event_provenance',
            'writer_capability_grants',
            'writer_capability_revocation_scopes',
            'writer_capability_revocations',
            'writer_capability_consumptions',
            'writer_canary_scope_preapprovals',
            'writer_canary_scope_preapproval_retirements',
            'writer_canary_scope_claims'
       )
       AND pg_catalog.format_type(attribute.atttypid, attribute.atttypmod)
           <> CASE
                WHEN attribute.attname IN (
                    'target_ref','source_refs','receipt','command_hashes','response',
                    'trusted_runtime'
                ) THEN 'jsonb'
                WHEN attribute.attname IN (
                    'intent_event_id','terminal_event_id','grant_event_id',
                    'consume_id','receipt_event_id','claim_event_id','event_id',
                    'preapproval_event_id','bootstrap_consumption_event_id',
                    'retirement_event_id'
                ) THEN 'uuid'
                WHEN attribute.attname IN (
                    'created_at','finalized_at','approved_at','expires_at',
                    'granted_at','revoked_at','consumed_at','appended_at',
                    'preapproved_at','claimed_at','retired_at'
                ) THEN 'timestamp with time zone'
                WHEN attribute.attname IN (
                    'plan_revision','max_uses','remaining_uses'
                ) THEN 'integer'
                WHEN attribute.attname = 'enabled' THEN 'boolean'
                ELSE 'text'
              END;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'preexisting writer table type contract mismatch: %', mismatch;
    END IF;

    SELECT pg_catalog.string_agg(
               class.relname || '.' || attribute.attname,
               ',' ORDER BY class.relname, attribute.attnum
           )
      INTO mismatch
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
      JOIN pg_catalog.pg_attribute AS attribute
        ON attribute.attrelid = class.oid
       AND attribute.attnum > 0
       AND NOT attribute.attisdropped
      JOIN pg_catalog.pg_type AS data_type
        ON data_type.oid = attribute.atttypid
     WHERE namespace.nspname = 'canonical_brain'
       AND class.relname IN (
            'writer_routeback_authorizations',
            'writer_routeback_lifecycle_terminals',
            'writer_routeback_terminals',
            'writer_public_routeback_targets',
            'writer_event_provenance',
            'writer_capability_grants',
            'writer_capability_revocation_scopes',
            'writer_capability_revocations',
            'writer_capability_consumptions',
            'writer_canary_scope_preapprovals',
            'writer_canary_scope_preapproval_retirements',
            'writer_canary_scope_claims'
       )
       AND NOT attribute.attnotnull;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'preexisting writer table nullability mismatch: %', mismatch;
    END IF;

    SELECT pg_catalog.string_agg(
               class.relname || '.' || attribute.attname,
               ',' ORDER BY class.relname, attribute.attnum
           )
      INTO mismatch
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
      JOIN pg_catalog.pg_attribute AS attribute
        ON attribute.attrelid = class.oid
       AND attribute.attnum > 0
       AND NOT attribute.attisdropped
      JOIN pg_catalog.pg_type AS data_type
        ON data_type.oid = attribute.atttypid
     WHERE namespace.nspname = 'canonical_brain'
       AND class.relname IN (
            'writer_routeback_authorizations',
            'writer_routeback_lifecycle_terminals',
            'writer_routeback_terminals',
            'writer_public_routeback_targets',
            'writer_event_provenance',
            'writer_capability_grants',
            'writer_capability_revocation_scopes',
            'writer_capability_revocations',
            'writer_capability_consumptions',
            'writer_canary_scope_preapprovals',
            'writer_canary_scope_preapproval_retirements',
            'writer_canary_scope_claims'
       )
       AND (
            attribute.attidentity <> ''
            OR attribute.attgenerated <> ''
            OR attribute.atthasmissing
            OR NOT attribute.attislocal
            OR attribute.attinhcount <> 0
            OR attribute.attndims <> 0
            OR attribute.attcollation <> data_type.typcollation
            OR attribute.attstorage <> data_type.typstorage
            OR attribute.attstattarget <> -1
            OR attribute.attoptions IS NOT NULL
            OR attribute.attfdwoptions IS NOT NULL
       );
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION
            'preexisting writer table identity/generated/options contract mismatch: %',
            mismatch;
    END IF;

    WITH expected(table_name, column_name, default_expression) AS (
        VALUES
          ('writer_routeback_authorizations','created_at','clock_timestamp()'),
          ('writer_routeback_lifecycle_terminals','finalized_at','clock_timestamp()'),
          ('writer_routeback_terminals','finalized_at','clock_timestamp()'),
          ('writer_public_routeback_targets','enabled','true'),
          ('writer_event_provenance','appended_at','clock_timestamp()'),
          ('writer_capability_grants','granted_at','clock_timestamp()'),
          ('writer_capability_revocation_scopes','revoked_at','clock_timestamp()'),
          ('writer_capability_revocations','revoked_at','clock_timestamp()'),
          ('writer_capability_consumptions','consumed_at','clock_timestamp()'),
          ('writer_canary_scope_preapprovals','preapproved_at','clock_timestamp()'),
          ('writer_canary_scope_preapproval_retirements','retired_at',
              'clock_timestamp()'),
          ('writer_canary_scope_claims','claimed_at','clock_timestamp()')
    ), actual AS (
        SELECT class.relname AS table_name,
               attribute.attname AS column_name,
               pg_catalog.pg_get_expr(
                   default_value.adbin, default_value.adrelid, false
               ) AS default_expression
          FROM pg_catalog.pg_class AS class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
          JOIN pg_catalog.pg_attribute AS attribute
            ON attribute.attrelid = class.oid
           AND attribute.attnum > 0
           AND NOT attribute.attisdropped
          JOIN pg_catalog.pg_attrdef AS default_value
            ON default_value.adrelid = class.oid
           AND default_value.adnum = attribute.attnum
         WHERE namespace.nspname = 'canonical_brain'
           AND class.relname IN (
                'writer_routeback_authorizations',
                'writer_routeback_lifecycle_terminals',
                'writer_routeback_terminals',
                'writer_public_routeback_targets',
                'writer_event_provenance',
                'writer_capability_grants',
                'writer_capability_revocation_scopes',
                'writer_capability_revocations',
                'writer_capability_consumptions',
                'writer_canary_scope_preapprovals',
                'writer_canary_scope_preapproval_retirements',
                'writer_canary_scope_claims'
           )
    ), difference AS (
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        UNION ALL
        (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT pg_catalog.string_agg(
               table_name || '.' || column_name || ':' || default_expression,
               ';' ORDER BY table_name, column_name
           )
      INTO mismatch
      FROM difference;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION
            'preexisting writer table default contract mismatch: %', mismatch;
    END IF;

    WITH expected_columns(table_name, column_name) AS (
        VALUES
          ('writer_routeback_authorizations','content_sha256'),
          ('writer_routeback_authorizations','session_key_sha256'),
          ('writer_routeback_authorizations','capability_epoch_sha256'),
          ('writer_routeback_authorizations','idempotency_key'),
          ('writer_routeback_authorizations','request_sha256'),
          ('writer_routeback_lifecycle_terminals','idempotency_key'),
          ('writer_routeback_lifecycle_terminals','outcome'),
          ('writer_routeback_lifecycle_terminals','request_sha256'),
          ('writer_routeback_lifecycle_terminals','session_key_sha256'),
          ('writer_routeback_lifecycle_terminals','capability_epoch_sha256'),
          ('writer_routeback_terminals','outcome'),
          ('writer_routeback_terminals','request_sha256'),
          ('writer_public_routeback_targets','target_type'),
          ('writer_event_provenance','canonical_content_sha256'),
          ('writer_event_provenance','trusted_runtime'),
          ('writer_capability_grants','plan_revision'),
          ('writer_capability_grants','session_key_sha256'),
          ('writer_capability_grants','capability_epoch_sha256'),
          ('writer_capability_grants','approval_source_sha256'),
          ('writer_capability_grants','command_hashes'),
          ('writer_capability_grants','max_uses'),
          ('writer_capability_grants','request_sha256'),
          ('writer_capability_revocation_scopes','scope_type'),
          ('writer_capability_revocation_scopes','session_key_sha256'),
          ('writer_capability_revocation_scopes','capability_epoch_sha256'),
          ('writer_capability_revocations','revoked_by_session_sha256'),
          ('writer_capability_consumptions','command_sha256'),
          ('writer_capability_consumptions','session_key_sha256'),
          ('writer_capability_consumptions','capability_epoch_sha256'),
          ('writer_capability_consumptions','idempotency_key'),
          ('writer_capability_consumptions','request_sha256'),
          ('writer_capability_consumptions','remaining_uses'),
          ('writer_canary_scope_preapprovals','release_sha256'),
          ('writer_canary_scope_preapprovals','fixture_sha256'),
          ('writer_canary_scope_preapprovals','session_key_sha256'),
          ('writer_canary_scope_preapprovals','approval_source_sha256'),
          ('writer_canary_scope_preapprovals','request_sha256'),
          ('writer_canary_scope_preapproval_retirements','release_sha256'),
          ('writer_canary_scope_preapproval_retirements','fixture_sha256'),
          ('writer_canary_scope_preapproval_retirements','session_key_sha256'),
          ('writer_canary_scope_preapproval_retirements',
              'approval_source_sha256'),
          ('writer_canary_scope_preapproval_retirements',
              'provisioning_receipt_sha256'),
          ('writer_canary_scope_preapproval_retirements','request_sha256'),
          ('writer_canary_scope_preapproval_retirements','reason'),
          ('writer_canary_scope_claims','release_sha256'),
          ('writer_canary_scope_claims','fixture_sha256'),
          ('writer_canary_scope_claims','approval_source_sha256'),
          ('writer_canary_scope_claims','session_key_sha256'),
          ('writer_canary_scope_claims','capability_epoch_sha256'),
          ('writer_canary_scope_claims','request_sha256')
    ), template AS (
        SELECT attribute.attname AS column_name,
               pg_catalog.pg_get_constraintdef(constraint_row.oid, false)
                   AS definition
          FROM pg_catalog.pg_constraint AS constraint_row
          JOIN pg_catalog.pg_attribute AS attribute
            ON attribute.attrelid = constraint_row.conrelid
           AND attribute.attnum = constraint_row.conkey[1]
         WHERE constraint_row.conrelid = 'canonical_writer_check_contract'::regclass
           AND constraint_row.contype = 'c'
           AND pg_catalog.cardinality(constraint_row.conkey) = 1
    ), expected_rows AS (
        SELECT expected_columns.table_name,
               expected_columns.column_name,
               template.definition
          FROM expected_columns
          JOIN template USING (column_name)
    ), expected AS (
        SELECT expected_rows.*, 1::bigint AS occurrences
          FROM expected_rows
    ), actual_rows AS (
        SELECT class.relname AS table_name,
               attribute.attname AS column_name,
               pg_catalog.pg_get_constraintdef(constraint_row.oid, false)
                   AS definition
          FROM pg_catalog.pg_constraint AS constraint_row
          JOIN pg_catalog.pg_class AS class ON class.oid = constraint_row.conrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
          LEFT JOIN pg_catalog.pg_attribute AS attribute
            ON attribute.attrelid = constraint_row.conrelid
           AND attribute.attnum = constraint_row.conkey[1]
         WHERE namespace.nspname = 'canonical_brain'
           AND class.relname IN (
                'writer_routeback_authorizations',
                'writer_routeback_lifecycle_terminals',
                'writer_routeback_terminals',
                'writer_public_routeback_targets',
                'writer_event_provenance',
                'writer_capability_grants',
                'writer_capability_revocation_scopes',
                'writer_capability_revocations',
                'writer_capability_consumptions',
                'writer_canary_scope_preapprovals',
                'writer_canary_scope_preapproval_retirements',
                'writer_canary_scope_claims'
           )
           AND constraint_row.contype = 'c'
    ), actual AS (
        SELECT actual_rows.table_name, actual_rows.column_name,
               actual_rows.definition, pg_catalog.count(*) AS occurrences
          FROM actual_rows
         GROUP BY actual_rows.table_name, actual_rows.column_name,
                  actual_rows.definition
    ), difference AS (
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        UNION ALL
        (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT pg_catalog.string_agg(
               table_name || '.' || COALESCE(column_name, '<multi>') || ':'
               || definition || ':' || occurrences::text,
               ';' ORDER BY table_name, column_name, definition
           )
      INTO mismatch
      FROM difference;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION
            'preexisting writer table CHECK contract mismatch: %', mismatch;
    END IF;

    WITH expected_rows(table_name, constraint_type, columns) AS (
        VALUES
          ('writer_routeback_authorizations','p',ARRAY['authorization_id']),
          ('writer_routeback_lifecycle_terminals','p',ARRAY['lifecycle_id']),
          ('writer_routeback_terminals','p',ARRAY['authorization_id']),
          ('writer_public_routeback_targets','p',ARRAY['channel_id']),
          ('writer_event_provenance','p',ARRAY['event_id']),
          ('writer_capability_grants','p',ARRAY['approval_id']),
          ('writer_capability_revocation_scopes','p',
              ARRAY['scope_type','session_key_sha256','capability_epoch_sha256','plan_id']),
          ('writer_capability_revocations','p',ARRAY['approval_id']),
          ('writer_capability_consumptions','p',ARRAY['consume_id']),
          ('writer_canary_scope_preapprovals','p',ARRAY['grant_id']),
          ('writer_canary_scope_preapproval_retirements','p',ARRAY['grant_id']),
          ('writer_canary_scope_claims','p',ARRAY['grant_id']),
          ('writer_routeback_authorizations','u',
              ARRAY['case_id','idempotency_key']),
          ('writer_routeback_lifecycle_terminals','u',
              ARRAY['case_id','idempotency_key']),
          ('writer_capability_grants','u',ARRAY['approval_source_sha256']),
          ('writer_capability_consumptions','u',
              ARRAY['session_key_sha256','capability_epoch_sha256','idempotency_key']),
          ('writer_canary_scope_preapprovals','u',ARRAY['case_id']),
          ('writer_canary_scope_preapprovals','u',ARRAY['run_id']),
          ('writer_canary_scope_preapprovals','u',ARRAY['approval_source_sha256']),
          ('writer_canary_scope_claims','u',ARRAY['case_id']),
          ('writer_canary_scope_claims','u',ARRAY['run_id']),
          ('writer_routeback_terminals','f',ARRAY['authorization_id']),
          ('writer_capability_revocations','f',ARRAY['approval_id']),
          ('writer_capability_consumptions','f',ARRAY['approval_id']),
          ('writer_canary_scope_preapproval_retirements','f',ARRAY['grant_id']),
          ('writer_canary_scope_claims','f',ARRAY['grant_id'])
    ), expected AS (
        SELECT expected_rows.*, 1::bigint AS occurrences
          FROM expected_rows
    ), actual_rows AS (
        SELECT class.relname AS table_name,
               constraint_row.contype::text AS constraint_type,
               pg_catalog.array_agg(attribute.attname ORDER BY key_position.ordinality)
                   AS columns
          FROM pg_catalog.pg_constraint AS constraint_row
          JOIN pg_catalog.pg_class AS class ON class.oid = constraint_row.conrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
          JOIN LATERAL pg_catalog.unnest(constraint_row.conkey)
               WITH ORDINALITY AS key_position(attnum, ordinality) ON true
          JOIN pg_catalog.pg_attribute AS attribute
            ON attribute.attrelid = class.oid
           AND attribute.attnum = key_position.attnum
         WHERE namespace.nspname = 'canonical_brain'
           AND constraint_row.contype IN ('p','u','f')
           AND class.relname IN (
                'writer_routeback_authorizations',
                'writer_routeback_lifecycle_terminals',
                'writer_routeback_terminals',
                'writer_public_routeback_targets',
                'writer_event_provenance',
                'writer_capability_grants',
                'writer_capability_revocation_scopes',
                'writer_capability_revocations',
                'writer_capability_consumptions',
                'writer_canary_scope_preapprovals',
                'writer_canary_scope_preapproval_retirements',
                'writer_canary_scope_claims'
           )
         GROUP BY class.relname, constraint_row.oid, constraint_row.contype
    ), actual AS (
        SELECT actual_rows.table_name, actual_rows.constraint_type,
               actual_rows.columns, pg_catalog.count(*) AS occurrences
          FROM actual_rows
         GROUP BY actual_rows.table_name, actual_rows.constraint_type,
                  actual_rows.columns
    ), difference AS (
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        UNION ALL
        (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT pg_catalog.string_agg(
               difference.table_name || ':' || difference.constraint_type || ':'
               || pg_catalog.array_to_string(difference.columns, ',') || ':'
               || difference.occurrences::text,
               ';' ORDER BY difference.table_name, difference.constraint_type
           )
      INTO mismatch
      FROM difference;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'preexisting writer table key contract mismatch: %', mismatch;
    END IF;

    SELECT pg_catalog.string_agg(class.relname, ',' ORDER BY class.relname)
      INTO mismatch
      FROM pg_catalog.pg_constraint AS constraint_row
      JOIN pg_catalog.pg_class AS class ON class.oid = constraint_row.conrelid
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
     WHERE namespace.nspname = 'canonical_brain'
       AND constraint_row.contype = 'f'
       AND class.relname IN (
            'writer_routeback_terminals',
            'writer_capability_revocations',
            'writer_capability_consumptions',
            'writer_canary_scope_preapproval_retirements',
            'writer_canary_scope_claims'
       )
       AND (
            constraint_row.condeferrable
            OR constraint_row.condeferred
            OR constraint_row.confupdtype <> 'a'
            OR constraint_row.confdeltype <> 'a'
            OR constraint_row.confmatchtype <> 's'
            OR NOT (
                (class.relname = 'writer_routeback_terminals'
                 AND constraint_row.confrelid =
                     'canonical_brain.writer_routeback_authorizations'::regclass)
                OR (class.relname IN (
                        'writer_capability_revocations',
                        'writer_capability_consumptions'
                    )
                    AND constraint_row.confrelid =
                        'canonical_brain.writer_capability_grants'::regclass)
                OR (class.relname IN (
                        'writer_canary_scope_claims',
                        'writer_canary_scope_preapproval_retirements'
                    )
                    AND constraint_row.confrelid =
                        'canonical_brain.writer_canary_scope_preapprovals'::regclass)
            )
            OR constraint_row.confkey <> ARRAY[
                (
                    SELECT attribute.attnum
                      FROM pg_catalog.pg_attribute AS attribute
                     WHERE attribute.attrelid = constraint_row.confrelid
                       AND attribute.attname = CASE
                           WHEN class.relname = 'writer_routeback_terminals'
                               THEN 'authorization_id'
                           WHEN class.relname IN (
                               'writer_canary_scope_claims',
                               'writer_canary_scope_preapproval_retirements'
                           )
                               THEN 'grant_id'
                           ELSE 'approval_id'
                       END
                )::smallint
            ]::smallint[]
       );
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'preexisting writer foreign-key contract mismatch: %', mismatch;
    END IF;

    -- Primary/UNIQUE constraints own indexes that are excluded from the
    -- explicit non-constraint index catalog below.  Attest their complete
    -- mechanical structure as well: no INCLUDE columns, expressions,
    -- predicates, alternate ordering/collation, or nondefault opclasses.
    SELECT pg_catalog.string_agg(
               class.relname || ':' || constraint_row.conname,
               ',' ORDER BY class.relname, constraint_row.conname
           )
      INTO mismatch
      FROM pg_catalog.pg_constraint AS constraint_row
      JOIN pg_catalog.pg_class AS class ON class.oid = constraint_row.conrelid
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
      JOIN pg_catalog.pg_index AS index
        ON index.indexrelid = constraint_row.conindid
      JOIN pg_catalog.pg_class AS index_class
        ON index_class.oid = index.indexrelid
      JOIN pg_catalog.pg_am AS access_method
        ON access_method.oid = index_class.relam
     WHERE namespace.nspname = 'canonical_brain'
       AND constraint_row.contype IN ('p','u')
       AND class.relname IN (
            'writer_routeback_authorizations',
            'writer_routeback_lifecycle_terminals',
            'writer_routeback_terminals',
            'writer_public_routeback_targets',
            'writer_event_provenance',
            'writer_capability_grants',
            'writer_capability_revocation_scopes',
            'writer_capability_revocations',
            'writer_capability_consumptions',
            'writer_canary_scope_preapprovals',
            'writer_canary_scope_preapproval_retirements',
            'writer_canary_scope_claims'
       )
       AND (
            NOT index.indisunique
            OR index.indisprimary <> (constraint_row.contype = 'p')
            OR index.indisexclusion
            OR NOT index.indimmediate
            OR NOT index.indisvalid
            OR NOT index.indisready
            OR NOT index.indislive
            OR index.indisclustered
            OR index.indisreplident
            OR index.indcheckxmin
            OR index.indnkeyatts <> pg_catalog.cardinality(constraint_row.conkey)
            OR index.indnatts <> pg_catalog.cardinality(constraint_row.conkey)
            OR ARRAY(
                SELECT key_part.attnum
                  FROM pg_catalog.unnest(index.indkey)
                       WITH ORDINALITY AS key_part(attnum, ordinal)
                 ORDER BY key_part.ordinal
            ) <> constraint_row.conkey
            OR index.indexprs IS NOT NULL
            OR index.indpred IS NOT NULL
            OR access_method.amname <> 'btree'
            OR index_class.relpersistence <> 'p'
            OR index_class.reloptions IS NOT NULL
            OR index_class.reltablespace <> 0
            OR pg_catalog.pg_get_userbyid(index_class.relowner)
               <> 'canonical_brain_migration_owner'
            OR EXISTS (
                SELECT 1
                  FROM ROWS FROM (
                      pg_catalog.unnest(index.indkey::smallint[]),
                      pg_catalog.unnest(index.indclass::oid[]),
                      pg_catalog.unnest(index.indcollation::oid[]),
                      pg_catalog.unnest(index.indoption::smallint[])
                  ) WITH ORDINALITY AS key_part(
                      attnum, opclass_oid, collation_oid, option_value, ordinal
                  )
                  JOIN pg_catalog.pg_attribute AS attribute
                    ON attribute.attrelid = class.oid
                   AND attribute.attnum = key_part.attnum
                 WHERE key_part.opclass_oid <> (
                        SELECT operator_class.oid
                          FROM pg_catalog.pg_opclass AS operator_class
                         WHERE operator_class.opcmethod = index_class.relam
                           AND operator_class.opcintype = attribute.atttypid
                           AND operator_class.opcdefault
                         LIMIT 1
                 )
                    OR key_part.collation_oid <> attribute.attcollation
                    OR key_part.option_value <> 0
            )
       );
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION
            'preexisting writer constraint index contract mismatch: %', mismatch;
    END IF;

    SELECT pg_catalog.string_agg(
               class.relname || ':' || constraint_row.conname,
               ',' ORDER BY class.relname, constraint_row.conname
           )
      INTO mismatch
      FROM pg_catalog.pg_constraint AS constraint_row
      JOIN pg_catalog.pg_class AS class ON class.oid = constraint_row.conrelid
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
     WHERE namespace.nspname = 'canonical_brain'
       AND class.relname IN (
            'writer_routeback_authorizations',
            'writer_routeback_lifecycle_terminals',
            'writer_routeback_terminals',
            'writer_public_routeback_targets',
            'writer_event_provenance',
            'writer_capability_grants',
            'writer_capability_revocation_scopes',
            'writer_capability_revocations',
            'writer_capability_consumptions',
            'writer_canary_scope_preapprovals',
            'writer_canary_scope_preapproval_retirements',
            'writer_canary_scope_claims'
       )
       AND (
            -- PostgreSQL 18 adds ``contype = 'n'`` rows for NOT NULL.  Their
            -- exact state is already pinned by the column contract.
            constraint_row.contype NOT IN ('p','u','f','c','n')
            OR NOT constraint_row.convalidated
            OR constraint_row.condeferrable
            OR constraint_row.condeferred
            -- Primary, unique, and foreign-key constraints are inherently
            -- non-inheritable in PostgreSQL; CHECK/NOT-NULL must not opt out
            -- of inheritance.  Pin the semantic value instead of assuming
            -- the catalog flag is always false.
            OR constraint_row.connoinherit <>
               (constraint_row.contype IN ('p','u','f'))
            OR NOT constraint_row.conislocal
            OR constraint_row.coninhcount <> 0
            OR constraint_row.conparentid <> 0
       );
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION
            'preexisting writer constraint state contract mismatch: %', mismatch;
    END IF;

    WITH index_map(actual_name, template_name) AS (
        VALUES
          ('writer_routeback_target_idx','contract_routeback_target_idx'),
          ('writer_event_provenance_session_idx',
              'contract_event_provenance_session_idx'),
          ('writer_event_provenance_thread_idx',
              'contract_event_provenance_thread_idx'),
          ('writer_capability_scope_idx','contract_capability_scope_idx'),
          ('writer_capability_command_idx','contract_capability_command_idx'),
          ('writer_capability_use_idx','contract_capability_use_idx'),
          ('writer_canary_preapproval_scope_idx',
              'contract_canary_preapproval_scope_idx'),
          ('writer_canary_preapproval_retirement_scope_idx',
              'contract_canary_preapproval_retirement_scope_idx'),
          ('writer_canary_claim_scope_idx','contract_canary_claim_scope_idx')
    ), template AS (
        SELECT index_class.relname AS template_name,
               access_method.amname,
               index.indisunique, index.indisprimary, index.indisexclusion,
               index.indimmediate, index.indnkeyatts, index.indnatts,
               COALESCE((
                   SELECT pg_catalog.array_agg(
                              COALESCE(attribute.attname, '')
                              ORDER BY key_position.ordinality
                          )
                     FROM pg_catalog.unnest(index.indkey)
                          WITH ORDINALITY AS key_position(attnum, ordinality)
                     LEFT JOIN pg_catalog.pg_attribute AS attribute
                       ON attribute.attrelid = index.indrelid
                      AND attribute.attnum = key_position.attnum
               ), ARRAY[]::text[]) AS key_columns,
               COALESCE(pg_catalog.pg_get_expr(
                   index.indexprs, index.indrelid, false
               ), '') AS expressions,
               COALESCE(pg_catalog.pg_get_expr(
                   index.indpred, index.indrelid, false
               ), '') AS predicate,
               index.indclass::text AS operator_classes,
               index.indcollation::text AS collations,
               index.indoption::text AS options
          FROM pg_catalog.pg_index AS index
          JOIN pg_catalog.pg_class AS index_class
            ON index_class.oid = index.indexrelid
          JOIN pg_catalog.pg_class AS table_class
            ON table_class.oid = index.indrelid
          JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = index_class.relam
         WHERE table_class.oid = 'canonical_writer_index_contract'::regclass
    ), expected AS (
        SELECT index_map.actual_name AS index_name,
               template.amname, template.indisunique, template.indisprimary,
               template.indisexclusion, template.indimmediate,
               template.indnkeyatts, template.indnatts, template.key_columns,
               template.expressions, template.predicate,
               template.operator_classes, template.collations, template.options
          FROM index_map
          JOIN template USING (template_name)
    ), actual AS (
        SELECT index_class.relname AS index_name,
               access_method.amname,
               index.indisunique, index.indisprimary, index.indisexclusion,
               index.indimmediate, index.indnkeyatts, index.indnatts,
               COALESCE((
                   SELECT pg_catalog.array_agg(
                              COALESCE(attribute.attname, '')
                              ORDER BY key_position.ordinality
                          )
                     FROM pg_catalog.unnest(index.indkey)
                          WITH ORDINALITY AS key_position(attnum, ordinality)
                     LEFT JOIN pg_catalog.pg_attribute AS attribute
                       ON attribute.attrelid = index.indrelid
                      AND attribute.attnum = key_position.attnum
               ), ARRAY[]::text[]) AS key_columns,
               COALESCE(pg_catalog.pg_get_expr(
                   index.indexprs, index.indrelid, false
               ), '') AS expressions,
               COALESCE(pg_catalog.pg_get_expr(
                   index.indpred, index.indrelid, false
               ), '') AS predicate,
               index.indclass::text AS operator_classes,
               index.indcollation::text AS collations,
               index.indoption::text AS options
          FROM pg_catalog.pg_index AS index
          JOIN pg_catalog.pg_class AS index_class
            ON index_class.oid = index.indexrelid
          JOIN pg_catalog.pg_class AS table_class
            ON table_class.oid = index.indrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = table_class.relnamespace
          JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = index_class.relam
         WHERE namespace.nspname = 'canonical_brain'
           AND table_class.relname IN (
                'writer_routeback_authorizations',
                'writer_routeback_lifecycle_terminals',
                'writer_routeback_terminals',
                'writer_public_routeback_targets',
                'writer_event_provenance',
                'writer_capability_grants',
                'writer_capability_revocation_scopes',
                'writer_capability_revocations',
                'writer_capability_consumptions',
                'writer_canary_scope_preapprovals',
                'writer_canary_scope_preapproval_retirements',
                'writer_canary_scope_claims'
           )
           AND NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_constraint AS constraint_row
                 WHERE constraint_row.conindid = index.indexrelid
           )
    ), difference AS (
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        UNION ALL
        (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT pg_catalog.string_agg(index_name, ',' ORDER BY index_name)
      INTO mismatch
      FROM difference;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'preexisting writer table index contract mismatch: %', mismatch;
    END IF;

    SELECT pg_catalog.string_agg(index_class.relname, ',' ORDER BY index_class.relname)
      INTO mismatch
      FROM pg_catalog.pg_index AS index
      JOIN pg_catalog.pg_class AS index_class
        ON index_class.oid = index.indexrelid
      JOIN pg_catalog.pg_class AS table_class
        ON table_class.oid = index.indrelid
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = table_class.relnamespace
     WHERE namespace.nspname = 'canonical_brain'
       AND table_class.relname IN (
            'writer_routeback_authorizations',
            'writer_routeback_lifecycle_terminals',
            'writer_routeback_terminals',
            'writer_public_routeback_targets',
            'writer_event_provenance',
            'writer_capability_grants',
            'writer_capability_revocation_scopes',
            'writer_capability_revocations',
            'writer_capability_consumptions',
            'writer_canary_scope_preapprovals',
            'writer_canary_scope_preapproval_retirements',
            'writer_canary_scope_claims'
       )
       AND (
            NOT index.indisvalid OR NOT index.indisready OR NOT index.indislive
            OR index.indisclustered OR index.indisreplident OR index.indcheckxmin
            OR index_class.relpersistence <> 'p'
            OR index_class.reloptions IS NOT NULL
            OR index_class.reltablespace <> 0
            OR pg_catalog.pg_get_userbyid(index_class.relowner)
               <> 'canonical_brain_migration_owner'
       );
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION
            'preexisting writer table index state/owner contract mismatch: %',
            mismatch;
    END IF;
END
$table_contract$;

CREATE OR REPLACE FUNCTION canonical_brain._ok(result jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog, canonical_brain
AS $function$
    SELECT pg_catalog.jsonb_build_object('ok', true, 'result', result)
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._event_envelope(
    event_row public.canonical_event_log
)
RETURNS jsonb
LANGUAGE sql
STABLE
STRICT
SET search_path = pg_catalog, canonical_brain
AS $function$
    SELECT pg_catalog.jsonb_build_object(
        'event_id', event_row.event_id::text,
        'schema_version', event_row.schema_version,
        'event_type', event_row.event_type,
        'occurred_at', event_row.occurred_at,
        'case_id', event_row.case_id,
        'source', event_row.source,
        'actor', event_row.actor,
        'subject', event_row.subject,
        'evidence', event_row.evidence,
        'decision', event_row.decision,
        'status', event_row.status,
        'next_action', event_row.next_action,
        'safety', event_row.safety,
        'payload', event_row.payload
    )
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._append_event(
    event_type_value text,
    case_id_value text,
    summary_value text,
    source_refs_value jsonb,
    actors_value jsonb,
    payload_value jsonb,
    safety_value jsonb,
    identity_value text,
    origin_value text,
    runtime_value jsonb
)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    event_uuid uuid;
    content_sha text;
    occurred timestamptz := pg_catalog.clock_timestamp();
    canonical_payload jsonb;
    source_doc jsonb;
    actor_doc jsonb;
    subject_doc jsonb;
    evidence_doc jsonb;
    decision_doc jsonb;
    status_doc jsonb;
    next_action_doc jsonb;
    safety_doc jsonb;
    expected_event jsonb;
    readback_event jsonb;
    existing_sha text;
    provenance_sha text;
    provenance_at timestamptz;
    provenance_runtime jsonb;
    inserted_count integer;
BEGIN
    IF event_type_value IS NULL OR event_type_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$' THEN
        RETURN canonical_brain._fail('invalid_event', 'event_type is invalid');
    END IF;
    IF case_id_value IS NULL OR case_id_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$' THEN
        RETURN canonical_brain._fail('invalid_event', 'case_id is invalid');
    END IF;
    IF pg_catalog.length(COALESCE(summary_value, '')) NOT BETWEEN 1 AND 4000
       OR pg_catalog.octet_length(COALESCE(identity_value, '')) NOT BETWEEN 1 AND 256
       OR pg_catalog.jsonb_typeof(source_refs_value) <> 'object'
       OR pg_catalog.jsonb_typeof(actors_value) <> 'object'
       OR pg_catalog.jsonb_typeof(payload_value) <> 'object'
       OR pg_catalog.jsonb_typeof(safety_value) <> 'object'
       OR NOT canonical_brain._runtime_valid(runtime_value) THEN
        RETURN canonical_brain._fail('invalid_event', 'canonical event envelope is invalid');
    END IF;

    content_sha := canonical_brain._sha256_json(
        pg_catalog.jsonb_build_object(
            'event_type', event_type_value,
            'case_id', case_id_value,
            'summary', summary_value,
            'source_refs', source_refs_value,
            'actors', actors_value,
            'payload', payload_value,
            'safety', safety_value,
            'origin', origin_value
        )
    );
    event_uuid := canonical_brain._deterministic_uuid(
        'canonical-writer:' || case_id_value || ':' || event_type_value || ':' || identity_value
    );
    canonical_payload := payload_value || pg_catalog.jsonb_build_object(
        'idempotency_key', identity_value,
        'summary', summary_value,
        'canonical_content_sha256', content_sha
    );
    source_doc := pg_catalog.jsonb_build_object(
        'system', 'hermes_agent',
        'component', 'canonical_writer',
        'source_refs', source_refs_value,
        'observed_session', runtime_value
    );
    actor_doc := COALESCE(actors_value->'actor', pg_catalog.jsonb_build_object(
        'type', 'service', 'id', 'canonical_writer'
    ));
    subject_doc := COALESCE(actors_value->'subject', pg_catalog.jsonb_build_object(
        'type', 'case', 'id', case_id_value
    ));
    evidence_doc := CASE
        WHEN pg_catalog.jsonb_typeof(payload_value->'evidence') = 'array'
            THEN payload_value->'evidence'
        ELSE '[]'::jsonb
    END;
    decision_doc := pg_catalog.jsonb_build_object(
        'kind', 'typed_canonical_writer_operation',
        'decided_by', origin_value,
        'keyword_authority', false,
        'attestation', CASE
            WHEN event_type_value IN (
                'route_back.intent.created', 'route_back.sent',
                'route_back.blocked', 'approval.capability.recorded',
                'approval.capability.revoked',
                'approval.capability.session_revoked',
                'capability.check.recorded', 'lease.shadow.recorded',
                'canary.scope.bootstrap_authorized',
                'canary.scope.bootstrap_consumed',
                'canary.scope.bootstrap_retired',
                'canary.scope.preapproved',
                'canary.scope.preapproval_retired', 'canary.scope.claimed',
                'canary.scope.revoked'
            ) THEN 'privileged_writer_receipt'
            ELSE 'model_authored'
        END
    );
    status_doc := pg_catalog.jsonb_build_object(
        'state', COALESCE(
            payload_value->'plan'->>'state',
            payload_value->'verification'->>'outcome',
            event_type_value
        ),
        'event_type', event_type_value,
        'summary', pg_catalog.left(summary_value, 500)
    );
    next_action_doc := CASE
        WHEN pg_catalog.jsonb_typeof(payload_value->'next_action') = 'object'
            THEN payload_value->'next_action'
        ELSE '{}'::jsonb
    END;
    safety_doc := pg_catalog.jsonb_build_object(
        'secret_value_recorded', false,
        'payment_credential_recorded', false,
        'business_mutation', false
    ) || safety_value;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('canonical-event:' || event_uuid::text, 0)
    );

    SELECT event.payload->>'canonical_content_sha256'
      INTO existing_sha
      FROM public.canonical_event_log AS event
     WHERE event.event_id = event_uuid
     LIMIT 1;
    IF FOUND THEN
        IF existing_sha IS DISTINCT FROM content_sha THEN
            RETURN canonical_brain._fail(
                'idempotency_conflict',
                'event identity is already bound to different canonical content'
            );
        END IF;
        SELECT provenance.canonical_content_sha256, provenance.appended_at,
               provenance.trusted_runtime
          INTO provenance_sha, provenance_at, provenance_runtime
          FROM canonical_brain.writer_event_provenance AS provenance
         WHERE provenance.event_id = event_uuid;
        IF NOT FOUND OR provenance_sha IS DISTINCT FROM content_sha THEN
            RETURN canonical_brain._fail(
                'event_provenance_missing',
                'preexisting event was not inserted by the privileged writer'
            );
        END IF;
        source_doc := pg_catalog.jsonb_build_object(
            'system', 'hermes_agent',
            'component', 'canonical_writer',
            'source_refs', source_refs_value,
            'observed_session', provenance_runtime
        );
        SELECT canonical_brain._event_envelope(event)
          INTO readback_event
          FROM public.canonical_event_log AS event
         WHERE event.event_id = event_uuid;
        expected_event := pg_catalog.jsonb_build_object(
            'event_id', event_uuid::text,
            'schema_version', 'canonical_event.v1',
            'event_type', event_type_value,
            'occurred_at', provenance_at,
            'case_id', case_id_value,
            'source', source_doc,
            'actor', actor_doc,
            'subject', subject_doc,
            'evidence', evidence_doc,
            'decision', decision_doc,
            'status', status_doc,
            'next_action', next_action_doc,
            'safety', safety_doc,
            'payload', canonical_payload
        );
        IF readback_event IS DISTINCT FROM expected_event THEN
            RETURN canonical_brain._fail(
                'canonical_readback_mismatch',
                'provenanced canonical event no longer matches its exact immutable content'
            );
        END IF;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'event_id', event_uuid::text,
            'event_type', event_type_value,
            'case_id', case_id_value,
            'canonical_content_sha256', content_sha,
            'inserted', false,
            'deduped', true
        ));
    END IF;

    INSERT INTO public.canonical_event_log (
        event_id, schema_version, event_type, occurred_at, case_id,
        source, actor, subject, evidence, decision, status,
        next_action, safety, payload
    ) VALUES (
        event_uuid,
        'canonical_event.v1',
        event_type_value,
        occurred,
        case_id_value,
        source_doc,
        actor_doc,
        subject_doc,
        evidence_doc,
        decision_doc,
        status_doc,
        next_action_doc,
        safety_doc,
        canonical_payload
    );
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    IF inserted_count <> 1 THEN
        RAISE EXCEPTION 'canonical event insert did not write exactly one row';
    END IF;
    SELECT canonical_brain._event_envelope(event)
      INTO readback_event
      FROM public.canonical_event_log AS event
     WHERE event.event_id = event_uuid;
    expected_event := pg_catalog.jsonb_build_object(
        'event_id', event_uuid::text,
        'schema_version', 'canonical_event.v1',
        'event_type', event_type_value,
        'occurred_at', occurred,
        'case_id', case_id_value,
        'source', source_doc,
        'actor', actor_doc,
        'subject', subject_doc,
        'evidence', evidence_doc,
        'decision', decision_doc,
        'status', status_doc,
        'next_action', next_action_doc,
        'safety', safety_doc,
        'payload', canonical_payload
    );
    IF readback_event IS DISTINCT FROM expected_event THEN
        RAISE EXCEPTION
            'canonical event readback mismatch before provenance append';
    END IF;
    INSERT INTO canonical_brain.writer_event_provenance (
        event_id, canonical_content_sha256, origin, trusted_runtime, appended_at
    ) VALUES (
        event_uuid, content_sha, origin_value, runtime_value, occurred
    );
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'event_id', event_uuid::text,
        'event_type', event_type_value,
        'case_id', case_id_value,
        'canonical_content_sha256', content_sha,
        'inserted', true,
        'deduped', false
    ));
END
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._plan_head(case_id_value text)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    event_count integer;
    invalid boolean;
    head_count integer;
    head_value jsonb;
BEGIN
    SELECT pg_catalog.count(*)
      INTO event_count
      FROM public.canonical_event_log AS event
      JOIN canonical_brain.writer_event_provenance AS provenance
        ON provenance.event_id = event.event_id
     WHERE event.case_id = case_id_value
       AND event.event_type = 'task.plan.updated';
    IF event_count > 256 THEN
        RETURN canonical_brain._fail(
            'plan_graph_too_large',
            'canonical plan graph exceeds the bounded validation window'
        );
    END IF;
    IF event_count = 0 THEN
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object('head', NULL));
    END IF;

    SELECT pg_catalog.bool_or(
        pg_catalog.jsonb_typeof(event.payload->'plan') <> 'object'
        OR COALESCE(event.payload->'plan'->>'plan_id', '') = ''
        OR COALESCE(event.payload->'plan'->>'revision', '') !~ '^[1-9][0-9]{0,8}$'
    )
      INTO invalid
      FROM public.canonical_event_log AS event
      JOIN canonical_brain.writer_event_provenance AS provenance
        ON provenance.event_id = event.event_id
     WHERE event.case_id = case_id_value
       AND event.event_type = 'task.plan.updated';
    IF invalid THEN
        RETURN canonical_brain._fail('plan_graph_invalid', 'plan row is structurally invalid');
    END IF;

    SELECT EXISTS (
        SELECT 1
          FROM public.canonical_event_log AS event
          JOIN canonical_brain.writer_event_provenance AS provenance
            ON provenance.event_id = event.event_id
         WHERE event.case_id = case_id_value
           AND event.event_type = 'task.plan.updated'
         GROUP BY event.payload->'plan'->>'plan_id',
                  (event.payload->'plan'->>'revision')::integer
        HAVING pg_catalog.count(DISTINCT (event.payload->'plan')::text) > 1
    ) INTO invalid;
    IF invalid THEN
        RETURN canonical_brain._fail(
            'plan_graph_invalid',
            'one plan revision has conflicting canonical content'
        );
    END IF;

    SELECT EXISTS (
        SELECT 1
          FROM public.canonical_event_log AS event
          JOIN canonical_brain.writer_event_provenance AS provenance
            ON provenance.event_id = event.event_id
         WHERE event.case_id = case_id_value
           AND event.event_type = 'task.plan.updated'
           AND COALESCE(event.payload->'plan'->>'supersedes_plan_id', '') <> ''
         GROUP BY event.payload->'plan'->>'plan_id'
        HAVING pg_catalog.count(DISTINCT pg_catalog.jsonb_build_array(
            event.payload->'plan'->>'supersedes_plan_id',
            event.payload->'plan'->>'supersedes_plan_revision'
        )::text) > 1
    ) INTO invalid;
    IF invalid THEN
        RETURN canonical_brain._fail(
            'plan_graph_invalid',
            'plan supersession edge changed across revisions'
        );
    END IF;

    WITH latest AS (
        SELECT DISTINCT ON (event.payload->'plan'->>'plan_id')
               event.event_id,
               event.occurred_at,
               event.payload->'plan' AS plan,
               event.payload->'plan'->>'plan_id' AS plan_id,
               (event.payload->'plan'->>'revision')::integer AS revision
          FROM public.canonical_event_log AS event
          JOIN canonical_brain.writer_event_provenance AS provenance
            ON provenance.event_id = event.event_id
         WHERE event.case_id = case_id_value
           AND event.event_type = 'task.plan.updated'
         ORDER BY event.payload->'plan'->>'plan_id',
                  (event.payload->'plan'->>'revision')::integer DESC,
                  event.occurred_at DESC,
                  event.event_id DESC
    ), invalid_edge AS (
        SELECT child.plan_id
          FROM latest AS child
          LEFT JOIN latest AS predecessor
            ON predecessor.plan_id = child.plan->>'supersedes_plan_id'
         WHERE COALESCE(child.plan->>'supersedes_plan_id', '') <> ''
           AND (
                child.plan->>'supersedes_plan_id' = child.plan_id
                OR predecessor.plan_id IS NULL
                OR COALESCE(child.plan->>'supersedes_plan_revision', '')
                   !~ '^[1-9][0-9]{0,8}$'
                OR (child.plan->>'supersedes_plan_revision')::integer
                   <> predecessor.revision
           )
    )
    SELECT EXISTS (SELECT 1 FROM invalid_edge) INTO invalid;
    IF invalid THEN
        RETURN canonical_brain._fail(
            'plan_graph_invalid',
            'plan supersession predecessor or revision is invalid'
        );
    END IF;

    WITH latest AS (
        SELECT DISTINCT ON (event.payload->'plan'->>'plan_id')
               event.event_id,
               event.occurred_at,
               event.payload->'plan' AS plan,
               event.payload->'plan'->>'plan_id' AS plan_id,
               (event.payload->'plan'->>'revision')::integer AS revision
          FROM public.canonical_event_log AS event
          JOIN canonical_brain.writer_event_provenance AS provenance
            ON provenance.event_id = event.event_id
         WHERE event.case_id = case_id_value
           AND event.event_type = 'task.plan.updated'
         ORDER BY event.payload->'plan'->>'plan_id',
                  (event.payload->'plan'->>'revision')::integer DESC,
                  event.occurred_at DESC,
                  event.event_id DESC
    ), superseded AS (
        SELECT DISTINCT plan->>'supersedes_plan_id' AS plan_id
          FROM latest
         WHERE COALESCE(plan->>'supersedes_plan_id', '') <> ''
    ), heads AS (
        SELECT latest.*
          FROM latest
         WHERE latest.plan_id NOT IN (SELECT superseded.plan_id FROM superseded)
    )
    SELECT pg_catalog.count(*),
           pg_catalog.max(pg_catalog.jsonb_build_object(
               'event_id', heads.event_id::text,
               'occurred_at', heads.occurred_at,
               'plan', heads.plan
           )::text)::jsonb
      INTO head_count, head_value
      FROM heads;
    IF head_count <> 1 THEN
        RETURN canonical_brain._fail(
            'plan_graph_invalid',
            'canonical plan graph must have exactly one head'
        );
    END IF;
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object('head', head_value));
END
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._fail(code text, message text)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog, canonical_brain
AS $function$
    SELECT pg_catalog.jsonb_build_object(
        'ok', false,
        'error', pg_catalog.jsonb_build_object(
            'code', code,
            'message', pg_catalog.left(message, 1000)
        )
    )
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._sha256_text(value text)
RETURNS text
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog, canonical_brain
AS $function$
    SELECT pg_catalog.encode(
        pg_catalog.sha256(pg_catalog.convert_to(value, 'UTF8')),
        'hex'
    )
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._sha256_json(value jsonb)
RETURNS text
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog, canonical_brain
AS $function$
    SELECT canonical_brain._sha256_text(value::text)
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._deterministic_uuid(value text)
RETURNS uuid
LANGUAGE plpgsql
IMMUTABLE
STRICT
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    digest_hex text := canonical_brain._sha256_text(value);
BEGIN
    -- Force RFC 4122 version/variant bits while retaining 122 digest bits.
    RETURN pg_catalog.format(
        '%s-%s-5%s-8%s-%s',
        pg_catalog.substr(digest_hex, 1, 8),
        pg_catalog.substr(digest_hex, 9, 4),
        pg_catalog.substr(digest_hex, 14, 3),
        pg_catalog.substr(digest_hex, 18, 3),
        pg_catalog.substr(digest_hex, 21, 12)
    )::uuid;
END
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._keys_valid(
    value jsonb,
    allowed text[],
    required text[]
)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
SET search_path = pg_catalog, canonical_brain
AS $function$
    SELECT pg_catalog.jsonb_typeof(value) = 'object'
       AND NOT EXISTS (
            SELECT 1 FROM pg_catalog.jsonb_object_keys(value) AS key(name)
             WHERE NOT (key.name = ANY (allowed))
       )
       AND NOT EXISTS (
            SELECT 1 FROM pg_catalog.unnest(required) AS needed(name)
             WHERE NOT (value ? needed.name)
       )
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._runtime_valid(runtime jsonb)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
SET search_path = pg_catalog, canonical_brain
AS $function$
    SELECT canonical_brain._keys_valid(
               runtime,
               ARRAY['request_id','platform','session_key_sha256',
                     'capability_epoch_sha256','user_id','chat_id','thread_id',
                     'message_id','owner_authenticated','service_internal'],
               ARRAY['request_id']
           )
       AND pg_catalog.length(COALESCE(runtime->>'request_id', '')) BETWEEN 1 AND 240
       AND (
            COALESCE(runtime->>'session_key_sha256', '') = ''
            OR runtime->>'session_key_sha256' ~ '^[0-9a-f]{64}$'
       )
       AND (
            COALESCE(runtime->>'capability_epoch_sha256', '') = ''
            OR runtime->>'capability_epoch_sha256' ~ '^[0-9a-f]{64}$'
       )
       AND pg_catalog.length(COALESCE(runtime->>'platform', '')) <= 240
       AND pg_catalog.length(COALESCE(runtime->>'user_id', '')) <= 240
       AND pg_catalog.length(COALESCE(runtime->>'chat_id', '')) <= 240
       AND pg_catalog.length(COALESCE(runtime->>'thread_id', '')) <= 240
       AND pg_catalog.length(COALESCE(runtime->>'message_id', '')) <= 240
       AND (
            NOT (runtime ? 'owner_authenticated')
            OR pg_catalog.jsonb_typeof(runtime->'owner_authenticated') = 'boolean'
       )
       AND (
            NOT (runtime ? 'service_internal')
            OR pg_catalog.jsonb_typeof(runtime->'service_internal') = 'boolean'
       )
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._contains_forbidden_dm_ref(value jsonb)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    item record;
    normalized_key text;
    normalized_value text;
BEGIN
    IF value IS NULL OR value = 'null'::jsonb THEN
        RETURN false;
    END IF;
    IF pg_catalog.jsonb_typeof(value) = 'array' THEN
        FOR item IN SELECT element FROM pg_catalog.jsonb_array_elements(value) AS x(element)
        LOOP
            IF canonical_brain._contains_forbidden_dm_ref(item.element) THEN
                RETURN true;
            END IF;
        END LOOP;
        RETURN false;
    END IF;
    IF pg_catalog.jsonb_typeof(value) <> 'object' THEN
        RETURN false;
    END IF;
    FOR item IN SELECT key, nested FROM pg_catalog.jsonb_each(value) AS x(key, nested)
    LOOP
        normalized_key := pg_catalog.replace(
            pg_catalog.lower(item.key), '-', '_'
        );
        normalized_value := pg_catalog.lower(
            pg_catalog.btrim(item.nested::text, '"')
        );
        IF normalized_key IN (
            'dm_channel_id', 'direct_message_channel_id',
            'recipient_id', 'dm_recipient_id'
        ) AND item.nested NOT IN ('null'::jsonb, 'false'::jsonb, '""'::jsonb) THEN
            RETURN true;
        END IF;
        IF normalized_key IN (
            'channel_type', 'target_type', 'target_kind',
            'delivery_type', 'lane', 'role'
        ) AND normalized_value IN (
            'dm', 'direct_message', 'private_dm', 'user_dm', 'private',
            'group', 'group_dm', 'private_channel', 'private_thread'
        ) THEN
            RETURN true;
        END IF;
        IF canonical_brain._contains_forbidden_dm_ref(item.nested) THEN
            RETURN true;
        END IF;
    END LOOP;
    RETURN false;
END
$function$;

CREATE OR REPLACE FUNCTION canonical_brain._case_scope_authorized(
    case_id_value text,
    runtime_value jsonb,
    allow_new boolean
)
RETURNS boolean
LANGUAGE plpgsql
STABLE
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    case_exists boolean;
    observed_thread text := COALESCE(
        NULLIF(runtime_value->>'thread_id', ''),
        runtime_value->>'chat_id',
        ''
    );
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime_value)
       OR case_id_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$' THEN
        RETURN false;
    END IF;
    -- owner_authenticated is constructed only by the writer service from the
    -- configured Discord owner IDs; it is never accepted from caller payload.
    IF runtime_value->>'platform' = 'discord'
       AND runtime_value->>'owner_authenticated' = 'true' THEN
        RETURN true;
    END IF;
    -- Isolated-canary API cases never inherit ordinary session/thread
    -- provenance.  Their sole authority is the exact one-shot claim and its
    -- current, non-retired session generation.  This branch is deliberately
    -- absent for ordinary cases and for Discord handoff continuation.
    IF runtime_value->>'platform' = 'api_server'
       AND EXISTS (
            SELECT 1
              FROM canonical_brain.writer_canary_scope_preapprovals AS marker
             WHERE marker.case_id = case_id_value
       ) THEN
        RETURN COALESCE(runtime_value->>'session_key_sha256', '')
                   ~ '^[0-9a-f]{64}$'
           AND COALESCE(runtime_value->>'capability_epoch_sha256', '')
                   ~ '^[0-9a-f]{64}$'
           AND EXISTS (
                SELECT 1
                  FROM canonical_brain.writer_canary_scope_preapprovals AS preapproval
                  JOIN canonical_brain.writer_canary_scope_claims AS claim
                    ON claim.grant_id = preapproval.grant_id
                   AND claim.case_id = preapproval.case_id
                   AND claim.release_sha256 = preapproval.release_sha256
                   AND claim.fixture_sha256 = preapproval.fixture_sha256
                   AND claim.run_id = preapproval.run_id
                   AND claim.approval_source_sha256
                       = preapproval.approval_source_sha256
                   AND claim.session_key_sha256
                       = preapproval.session_key_sha256
                   AND claim.expires_at = preapproval.expires_at
                 WHERE preapproval.case_id = case_id_value
                   AND preapproval.expires_at > pg_catalog.statement_timestamp()
                   AND claim.session_key_sha256
                       = runtime_value->>'session_key_sha256'
                   AND claim.capability_epoch_sha256
                       = runtime_value->>'capability_epoch_sha256'
                   AND NOT EXISTS (
                        SELECT 1
                          FROM canonical_brain.
                               writer_canary_scope_preapproval_retirements
                               AS terminal
                         WHERE terminal.grant_id = preapproval.grant_id
                   )
                   AND NOT EXISTS (
                        SELECT 1
                          FROM canonical_brain.writer_capability_revocation_scopes AS scope
                         WHERE scope.scope_type = 'session'
                           AND scope.session_key_sha256
                               = claim.session_key_sha256
                           AND scope.capability_epoch_sha256
                               = claim.capability_epoch_sha256
                   )
           );
    END IF;
    SELECT EXISTS (
        SELECT 1 FROM public.canonical_event_log AS event
         WHERE event.case_id = case_id_value
    ) INTO case_exists;
    IF NOT case_exists THEN
        RETURN allow_new
           AND runtime_value->>'platform' = 'discord'
           AND COALESCE(runtime_value->>'session_key_sha256', '') ~ '^[0-9a-f]{64}$'
           AND observed_thread <> '';
    END IF;
    -- Existing-case authority comes only from immutable trusted runtime data
    -- written by the SECURITY DEFINER append path.  Model-authored source_refs
    -- are deliberately excluded.
    RETURN EXISTS (
        SELECT 1
          FROM public.canonical_event_log AS event
          JOIN canonical_brain.writer_event_provenance AS provenance
            ON provenance.event_id = event.event_id
         WHERE event.case_id = case_id_value
           AND (
                (
                    COALESCE(runtime_value->>'session_key_sha256', '') <> ''
                    AND provenance.trusted_runtime->>'session_key_sha256'
                        = runtime_value->>'session_key_sha256'
                )
                OR (
                    observed_thread <> ''
                    AND provenance.trusted_runtime->>'platform'
                        = runtime_value->>'platform'
                    AND observed_thread IN (
                        COALESCE(provenance.trusted_runtime->>'thread_id', ''),
                        COALESCE(provenance.trusted_runtime->>'chat_id', '')
                    )
                )
           )
    ) OR EXISTS (
        -- A completed public route-back is the exact writer-owned handoff
        -- edge that authorizes the destination thread to continue the case.
        -- Claimed-only and blocked attempts confer no authority.
        SELECT 1
          FROM canonical_brain.writer_routeback_authorizations AS authorization_row
          JOIN canonical_brain.writer_routeback_terminals AS terminal
            ON terminal.authorization_id = authorization_row.authorization_id
           AND terminal.outcome = 'sent'
          JOIN canonical_brain.writer_public_routeback_targets AS allowed
            ON allowed.channel_id = COALESCE(
                authorization_row.target_ref->>'thread_id',
                authorization_row.target_ref->>'channel_id',
                ''
            )
           AND allowed.enabled
           AND allowed.target_type IN ('public_channel', 'public_thread')
         WHERE runtime_value->>'platform' = 'discord'
           AND observed_thread <> ''
           AND authorization_row.case_id = case_id_value
           AND observed_thread = allowed.channel_id
    );
END
$function$;

-- Fixed public routine 1/18.
CREATE OR REPLACE FUNCTION canonical_brain.writer_ping(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
BEGIN
    IF request <> '{}'::jsonb OR NOT canonical_brain._runtime_valid(runtime) THEN
        RETURN canonical_brain._fail('invalid_request', 'ping envelope is invalid');
    END IF;
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'service', 'canonical_writer',
        'protocol', 'v1',
        'database_identity', CURRENT_USER,
        'request_id', runtime->>'request_id'
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'writer ping failed');
END
$function$;

-- Private one-shot bootstrap routine.  It is never granted to the ordinary
-- writer role and is deliberately absent from the public wire catalog.
CREATE OR REPLACE FUNCTION canonical_brain.writer_canary_scope_preapprove(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    expires_value timestamptz;
    request_hash text;
    existing_record canonical_brain.writer_canary_scope_preapprovals%ROWTYPE;
    append_result jsonb;
    consume_result jsonb;
    authorization_record jsonb;
    authorization_count bigint;
    preapproved_value timestamptz;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR runtime->>'service_internal' <> 'true'
       OR runtime->>'platform' <> 'writer_service'
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                  'run_id','session_key_sha256','expires_at','approved_by',
                  'approval_source_sha256','provisioning_receipt_sha256'],
            ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                  'run_id','session_key_sha256','expires_at','approved_by',
                  'approval_source_sha256','provisioning_receipt_sha256']
       )
       OR request->>'grant_id' !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'case_id' !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'run_id' !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'approved_by' !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'release_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'fixture_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'session_key_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'approval_source_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'provisioning_receipt_sha256' !~ '^[0-9a-f]{64}$'
       OR runtime->>'session_key_sha256' IS DISTINCT FROM
            request->>'session_key_sha256' THEN
        RETURN canonical_brain._fail(
            'invalid_request',
            'canary scope preapproval is invalid or not service-internal'
        );
    END IF;
    BEGIN
        expires_value := (request->>'expires_at')::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        RETURN canonical_brain._fail(
            'invalid_request', 'canary scope expiry is invalid'
        );
    END;
    IF expires_value <= pg_catalog.clock_timestamp()
       OR expires_value > pg_catalog.clock_timestamp() + INTERVAL '1 hour' THEN
        RETURN canonical_brain._fail(
            'invalid_expiry', 'canary scope must expire within one hour'
        );
    END IF;
    request_hash := canonical_brain._sha256_json(
        request || pg_catalog.jsonb_build_object(
            'expires_at', pg_catalog.to_char(
                expires_value AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            )
        )
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'canary-preapproval:' || (request->>'grant_id'), 0
        )
    );
    IF SESSION_USER <> 'canonical_brain_canary_bootstrap_login'
       OR CURRENT_USER <> 'canonical_brain_migration_owner'
       OR NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles AS role
             WHERE role.rolname = 'canonical_brain_canary_bootstrap_login'
               AND role.rolcanlogin AND role.rolinherit
               AND NOT role.rolsuper AND NOT role.rolcreatedb
               AND NOT role.rolcreaterole AND NOT role.rolreplication
               AND NOT role.rolbypassrls
       ) OR NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles AS role
             WHERE role.rolname = 'canonical_brain_canary_bootstrap'
               AND NOT role.rolcanlogin AND NOT role.rolinherit
               AND NOT role.rolsuper AND NOT role.rolcreatedb
               AND NOT role.rolcreaterole AND NOT role.rolreplication
               AND NOT role.rolbypassrls
       ) OR (
            SELECT pg_catalog.count(*)
              FROM pg_catalog.pg_auth_members AS membership
              JOIN pg_catalog.pg_roles AS granted_role
                ON granted_role.oid = membership.roleid
              JOIN pg_catalog.pg_roles AS member_role
                ON member_role.oid = membership.member
             WHERE granted_role.rolname = 'canonical_brain_canary_bootstrap'
               AND member_role.rolname =
                   'canonical_brain_canary_bootstrap_login'
               AND NOT membership.admin_option
               AND membership.inherit_option
               AND membership.set_option
       ) <> 1 OR EXISTS (
            SELECT 1
              FROM pg_catalog.pg_auth_members AS membership
              JOIN pg_catalog.pg_roles AS granted_role
                ON granted_role.oid = membership.roleid
              JOIN pg_catalog.pg_roles AS member_role
                ON member_role.oid = membership.member
             WHERE (
                    granted_role.rolname = 'canonical_brain_canary_bootstrap'
                    OR member_role.rolname =
                        'canonical_brain_canary_bootstrap_login'
                    OR member_role.rolname = 'canonical_brain_canary_bootstrap'
                   )
               AND NOT (
                    granted_role.rolname = 'canonical_brain_canary_bootstrap'
                    AND member_role.rolname =
                        'canonical_brain_canary_bootstrap_login'
                    AND NOT membership.admin_option
                    AND membership.inherit_option
                    AND membership.set_option
               )
       ) THEN
        RETURN canonical_brain._fail(
            'bootstrap_authority_mismatch',
            'canary bootstrap role/login authority is not exact'
        );
    END IF;
    IF (
        WITH actual AS (
            SELECT 'schema'::text AS object_kind, namespace.nspname AS identity,
                   acl.privilege_type, acl.is_grantable,
                   pg_catalog.pg_get_userbyid(acl.grantor) AS grantor
              FROM pg_catalog.pg_namespace AS namespace
              CROSS JOIN LATERAL pg_catalog.aclexplode(
                  COALESCE(
                      namespace.nspacl,
                      pg_catalog.acldefault('n', namespace.nspowner)
                  )
              ) AS acl
             WHERE namespace.nspname = 'canonical_brain'
               AND acl.grantee = (
                    SELECT oid FROM pg_catalog.pg_roles
                     WHERE rolname = 'canonical_brain_canary_bootstrap'
               )
            UNION ALL
            SELECT 'function',
                   pg_catalog.format(
                       '%I.%I(%s)', namespace.nspname, routine.proname,
                       pg_catalog.oidvectortypes(routine.proargtypes)
                   ), acl.privilege_type, acl.is_grantable,
                   pg_catalog.pg_get_userbyid(acl.grantor)
              FROM pg_catalog.pg_proc AS routine
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = routine.pronamespace
              CROSS JOIN LATERAL pg_catalog.aclexplode(
                  COALESCE(
                      routine.proacl,
                      pg_catalog.acldefault('f', routine.proowner)
                  )
              ) AS acl
             WHERE namespace.nspname = 'canonical_brain'
               AND acl.grantee = (
                    SELECT oid FROM pg_catalog.pg_roles
                     WHERE rolname = 'canonical_brain_canary_bootstrap'
               )
            UNION ALL
            SELECT CASE WHEN class.relkind = 'S' THEN 'sequence' ELSE 'table' END,
                   namespace.nspname || '.' || class.relname,
                   acl.privilege_type, acl.is_grantable,
                   pg_catalog.pg_get_userbyid(acl.grantor)
              FROM pg_catalog.pg_class AS class
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = class.relnamespace
              CROSS JOIN LATERAL pg_catalog.aclexplode(
                  COALESCE(
                      class.relacl,
                      pg_catalog.acldefault(
                          CASE WHEN class.relkind = 'S'
                               THEN 'S'::"char" ELSE 'r'::"char" END,
                          class.relowner
                      )
                  )
              ) AS acl
             WHERE namespace.nspname = 'canonical_brain'
               AND class.relkind IN ('r','p','S')
               AND acl.grantee = (
                    SELECT oid FROM pg_catalog.pg_roles
                     WHERE rolname = 'canonical_brain_canary_bootstrap'
               )
        ), expected(object_kind, identity, privilege_type, is_grantable, grantor)
        AS (
            VALUES
              ('schema','canonical_brain','USAGE',false,
               'canonical_brain_migration_owner'),
              ('function',
               'canonical_brain.writer_canary_scope_preapprove(jsonb, jsonb)',
               'EXECUTE',false,'canonical_brain_migration_owner')
        )
        SELECT EXISTS (
            (SELECT * FROM actual EXCEPT SELECT * FROM expected)
            UNION ALL
            (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        )
    ) THEN
        RETURN canonical_brain._fail(
            'bootstrap_acl_mismatch',
            'canary bootstrap ACL is not the exact one-shot grant'
        );
    END IF;
    SELECT pg_catalog.count(*),
           pg_catalog.max(
               (event.payload->'canary_scope_bootstrap_authorization')::text
           )::jsonb
      INTO authorization_count, authorization_record
      FROM public.canonical_event_log AS event
      JOIN canonical_brain.writer_event_provenance AS provenance
        ON provenance.event_id = event.event_id
     WHERE event.event_type = 'canary.scope.bootstrap_authorized'
       AND event.case_id = request->>'case_id'
       AND provenance.origin = 'canary_scope_bootstrap_provision'
       AND event.payload->'canary_scope_bootstrap_authorization'->>'grant_id'
           = request->>'grant_id';
    IF authorization_count <> 1
       OR NOT canonical_brain._keys_valid(
            authorization_record,
            ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                  'run_id','session_key_sha256','expires_at','approved_by',
                  'approval_source_sha256','provisioning_receipt_sha256',
                  'bootstrap_login','state'],
            ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                  'run_id','session_key_sha256','expires_at','approved_by',
                  'approval_source_sha256','provisioning_receipt_sha256',
                  'bootstrap_login','state']
       )
       OR authorization_record->>'grant_id' <> request->>'grant_id'
       OR authorization_record->>'case_id' <> request->>'case_id'
       OR authorization_record->>'release_sha256' <> request->>'release_sha256'
       OR authorization_record->>'fixture_sha256' <> request->>'fixture_sha256'
       OR authorization_record->>'run_id' <> request->>'run_id'
       OR authorization_record->>'session_key_sha256'
            <> request->>'session_key_sha256'
       OR authorization_record->>'approved_by' <> request->>'approved_by'
       OR authorization_record->>'approval_source_sha256'
            <> request->>'approval_source_sha256'
       OR authorization_record->>'provisioning_receipt_sha256'
            <> request->>'provisioning_receipt_sha256'
       OR authorization_record->>'bootstrap_login'
            <> 'canonical_brain_canary_bootstrap_login'
       OR authorization_record->>'state' <> 'authorized'
       OR (authorization_record->>'expires_at')::timestamptz
            IS DISTINCT FROM expires_value
       OR EXISTS (
            SELECT 1
              FROM public.canonical_event_log AS event
              JOIN canonical_brain.writer_event_provenance AS provenance
                ON provenance.event_id = event.event_id
             WHERE event.event_type = 'canary.scope.bootstrap_consumed'
               AND event.case_id = request->>'case_id'
               AND provenance.origin = 'canary_scope_bootstrap_consume'
               AND event.payload->'canary_scope_bootstrap_consumption'->>'grant_id'
                   = request->>'grant_id'
       ) OR EXISTS (
            SELECT 1
              FROM public.canonical_event_log AS event
              JOIN canonical_brain.writer_event_provenance AS provenance
                ON provenance.event_id = event.event_id
             WHERE event.event_type = 'canary.scope.bootstrap_retired'
               AND event.case_id = request->>'case_id'
               AND provenance.origin = 'canary_scope_bootstrap_retire'
               AND event.payload->'canary_scope_bootstrap_retirement'->>'grant_id'
                   = request->>'grant_id'
       ) THEN
        RETURN canonical_brain._fail(
            'bootstrap_authorization_missing',
            'exact one-shot canary bootstrap authorization is unavailable'
        );
    END IF;
    SELECT * INTO existing_record
      FROM canonical_brain.writer_canary_scope_preapprovals AS preapproval
     WHERE preapproval.grant_id = request->>'grant_id';
    IF FOUND THEN
        IF existing_record.request_sha256 <> request_hash
           OR existing_record.case_id <> request->>'case_id'
           OR existing_record.session_key_sha256
                <> request->>'session_key_sha256' THEN
            RETURN canonical_brain._fail(
                'idempotency_conflict',
                'canary grant identity is bound to another preapproval'
            );
        END IF;
        IF existing_record.expires_at <= pg_catalog.clock_timestamp() THEN
            RETURN canonical_brain._fail(
                'canary_scope_expired', 'canary scope preapproval has expired'
            );
        END IF;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'grant_id', existing_record.grant_id,
            'case_id', existing_record.case_id,
            'release_sha256', existing_record.release_sha256,
            'fixture_sha256', existing_record.fixture_sha256,
            'run_id', existing_record.run_id,
            'session_key_sha256', existing_record.session_key_sha256,
            'expires_at', existing_record.expires_at,
            'approved_by', existing_record.approved_by,
            'approval_source_sha256', existing_record.approval_source_sha256,
            'preapproved_at', existing_record.preapproved_at,
            'event_id', existing_record.receipt_event_id::text,
            'receipt_event_id', existing_record.receipt_event_id::text,
            'inserted', false,
            'deduped', true
        ));
    END IF;
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_canary_scope_preapprovals AS preapproval
         WHERE preapproval.case_id = request->>'case_id'
            OR preapproval.run_id = request->>'run_id'
            OR preapproval.approval_source_sha256
                = request->>'approval_source_sha256'
    ) THEN
        RETURN canonical_brain._fail(
            'idempotency_conflict',
            'canary case, run, or approval receipt is already preapproved'
        );
    END IF;
    preapproved_value := pg_catalog.clock_timestamp();
    append_result := canonical_brain._append_event(
        'canary.scope.preapproved',
        request->>'case_id',
        'Owner-bound isolated canary execution scope preapproved',
        pg_catalog.jsonb_build_object(
            'manual_ref', 'canary-preapproval:' || (request->>'grant_id')
        ),
        pg_catalog.jsonb_build_object(
            'actor', pg_catalog.jsonb_build_object(
                'type', 'owner_approval', 'id', request->>'approved_by'
            ),
            'subject', pg_catalog.jsonb_build_object(
                'type', 'canary_scope', 'id', request->>'grant_id'
            )
        ),
        pg_catalog.jsonb_build_object(
            'canary_scope_preapproval', pg_catalog.jsonb_build_object(
                'grant_id', request->>'grant_id',
                'case_id', request->>'case_id',
                'release_sha256', request->>'release_sha256',
                'fixture_sha256', request->>'fixture_sha256',
                'run_id', request->>'run_id',
                'session_key_sha256', request->>'session_key_sha256',
                'expires_at', expires_value,
                'approved_by', request->>'approved_by',
                'approval_source_sha256', request->>'approval_source_sha256',
                'state', 'preapproved'
            )
        ),
        pg_catalog.jsonb_build_object(
            'isolated_canary', true,
            'service_internal_preapproval', true
        ),
        'canary-preapprove:' || (request->>'grant_id'),
        'canary_scope_preapprove',
        runtime
    );
    IF NOT (append_result->>'ok')::boolean THEN
        RETURN append_result;
    END IF;
    INSERT INTO canonical_brain.writer_canary_scope_preapprovals (
        grant_id, case_id, release_sha256, fixture_sha256, run_id,
        session_key_sha256, expires_at, approved_by, approval_source_sha256,
        request_sha256, preapproved_at, receipt_event_id
    ) VALUES (
        request->>'grant_id', request->>'case_id', request->>'release_sha256',
        request->>'fixture_sha256', request->>'run_id',
        request->>'session_key_sha256', expires_value, request->>'approved_by',
        request->>'approval_source_sha256', request_hash,
        preapproved_value,
        (append_result->'result'->>'event_id')::uuid
    );
    consume_result := canonical_brain._append_event(
        'canary.scope.bootstrap_consumed',
        request->>'case_id',
        'One-shot isolated canary bootstrap authority consumed',
        pg_catalog.jsonb_build_object(
            'manual_ref', 'canary-bootstrap-consume:' || (request->>'grant_id')
        ),
        pg_catalog.jsonb_build_object(
            'actor', pg_catalog.jsonb_build_object(
                'type', 'writer_bootstrap', 'id', SESSION_USER
            ),
            'subject', pg_catalog.jsonb_build_object(
                'type', 'canary_scope', 'id', request->>'grant_id'
            )
        ),
        pg_catalog.jsonb_build_object(
            'canary_scope_bootstrap_consumption',
            pg_catalog.jsonb_build_object(
                'grant_id', request->>'grant_id',
                'case_id', request->>'case_id',
                'provisioning_receipt_sha256',
                    request->>'provisioning_receipt_sha256',
                'preapproval_event_id', append_result->'result'->>'event_id',
                'state', 'consumed'
            )
        ),
        pg_catalog.jsonb_build_object(
            'isolated_canary', true,
            'one_shot_bootstrap_consumed', true
        ),
        'canary-bootstrap-consume:' || (request->>'grant_id'),
        'canary_scope_bootstrap_consume',
        runtime
    );
    IF NOT (consume_result->>'ok')::boolean THEN
        RAISE EXCEPTION 'canary bootstrap consumption receipt append failed';
    END IF;
    EXECUTE
        'REVOKE EXECUTE ON FUNCTION '
        'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb) '
        'FROM canonical_brain_canary_bootstrap';
    EXECUTE
        'REVOKE USAGE ON SCHEMA canonical_brain '
        'FROM canonical_brain_canary_bootstrap';
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'grant_id', request->>'grant_id',
        'case_id', request->>'case_id',
        'release_sha256', request->>'release_sha256',
        'fixture_sha256', request->>'fixture_sha256',
        'run_id', request->>'run_id',
        'session_key_sha256', request->>'session_key_sha256',
        'expires_at', expires_value,
        'approved_by', request->>'approved_by',
        'approval_source_sha256', request->>'approval_source_sha256',
        'preapproved_at', preapproved_value,
        'event_id', append_result->'result'->>'event_id',
        'receipt_event_id', append_result->'result'->>'event_id',
        'bootstrap_consumption_event_id',
            consume_result->'result'->>'event_id',
        'bootstrap_acl_revoked', true,
        'inserted', true,
        'deduped', false
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'canary scope preapproval failed');
END
$function$;

-- Private writer-UID routine.  It is absent from the 18-operation protocol
-- catalog and may be called only by in-process shutdown, ExecStopPost, or
-- next-start reconciliation through a fixed statement.  Claim and retirement
-- serialize on the same preapproval lock.  When a claim already committed,
-- cleanup retires its exact session epoch instead of fabricating a preclaim
-- terminal.
CREATE OR REPLACE FUNCTION canonical_brain.writer_canary_scope_preapproval_retire(
    request jsonb, runtime jsonb
)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    request_hash text;
    expires_value timestamptz;
    preapproval_record canonical_brain.writer_canary_scope_preapprovals%ROWTYPE;
    retirement_record
        canonical_brain.writer_canary_scope_preapproval_retirements%ROWTYPE;
    observed_claim canonical_brain.writer_canary_scope_claims%ROWTYPE;
    locked_claim canonical_brain.writer_canary_scope_claims%ROWTYPE;
    claim_observed boolean := false;
    consumption_count bigint;
    consumption_event_id text;
    consumption_record jsonb;
    append_result jsonb;
    revoke_response jsonb;
    revoke_result jsonb;
    retired_value timestamptz;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR runtime->>'platform' <> 'writer_service'
       OR runtime->>'service_internal' <> 'true'
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                  'run_id','session_key_sha256','expires_at','approved_by',
                  'approval_source_sha256','provisioning_receipt_sha256'],
            ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                  'run_id','session_key_sha256','expires_at','approved_by',
                  'approval_source_sha256','provisioning_receipt_sha256']
       )
       OR request->>'grant_id' !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'case_id' !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'run_id' !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'approved_by' !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'release_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'fixture_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'session_key_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'approval_source_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'provisioning_receipt_sha256' !~ '^[0-9a-f]{64}$' THEN
        RETURN canonical_brain._fail(
            'invalid_request', 'canary preclaim retirement request is invalid'
        );
    END IF;
    BEGIN
        expires_value := (request->>'expires_at')::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        RETURN canonical_brain._fail(
            'invalid_request', 'canary preclaim retirement expiry is invalid'
        );
    END;
    IF expires_value IS NULL THEN
        RETURN canonical_brain._fail(
            'invalid_request', 'canary preclaim retirement expiry is missing'
        );
    END IF;
    request_hash := canonical_brain._sha256_json(request);

    -- An unlocked append-only claim observation determines lock order.  A
    -- claim that appears after this read triggers SQLSTATE 40001 below so the
    -- retry can acquire capability -> preapproval in canonical order.
    SELECT * INTO observed_claim
      FROM canonical_brain.writer_canary_scope_claims AS claim
     WHERE claim.grant_id = request->>'grant_id';
    claim_observed := FOUND;
    IF claim_observed THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(
                'capability-scope:' || observed_claim.session_key_sha256 || ':'
                    || observed_claim.capability_epoch_sha256,
                0
            )
        );
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'canary-preapproval:' || (request->>'grant_id'), 0
        )
    );
    SELECT * INTO preapproval_record
      FROM canonical_brain.writer_canary_scope_preapprovals AS preapproval
     WHERE preapproval.grant_id = request->>'grant_id'
     FOR UPDATE;
    SELECT * INTO locked_claim
      FROM canonical_brain.writer_canary_scope_claims AS claim
     WHERE claim.grant_id = request->>'grant_id';
    IF NOT claim_observed AND FOUND THEN
        RAISE EXCEPTION USING
            ERRCODE = '40001',
            MESSAGE = 'canary claim committed during preclaim reconciliation';
    END IF;
    IF claim_observed AND (
        NOT FOUND
        OR locked_claim.claim_event_id <> observed_claim.claim_event_id
        OR locked_claim.session_key_sha256 <> observed_claim.session_key_sha256
        OR locked_claim.capability_epoch_sha256
            <> observed_claim.capability_epoch_sha256
    ) THEN
        RAISE EXCEPTION 'canary claim changed during preclaim reconciliation';
    END IF;

    SELECT * INTO retirement_record
      FROM canonical_brain.writer_canary_scope_preapproval_retirements AS terminal
     WHERE terminal.grant_id = request->>'grant_id';
    IF claim_observed AND FOUND THEN
        RAISE EXCEPTION 'canary preclaim terminal truth is contradictory';
    END IF;
    IF preapproval_record.grant_id IS NULL THEN
        IF claim_observed OR retirement_record.grant_id IS NOT NULL THEN
            RAISE EXCEPTION 'canary preclaim truth exists without preapproval';
        END IF;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'outcome', 'not_preapproved',
            'grant_id', request->>'grant_id',
            'case_id', request->>'case_id',
            'release_sha256', request->>'release_sha256',
            'fixture_sha256', request->>'fixture_sha256',
            'run_id', request->>'run_id',
            'session_key_sha256', request->>'session_key_sha256',
            'expires_at', expires_value,
            'approved_by', request->>'approved_by',
            'approval_source_sha256', request->>'approval_source_sha256',
            'provisioning_receipt_sha256',
                request->>'provisioning_receipt_sha256',
            'preapproval_event_id', NULL,
            'bootstrap_consumption_event_id', NULL,
            'claim_event_id', NULL,
            'retirement_event_id', NULL,
            'revocation_event_id', NULL,
            'claimed_at', NULL,
            'retired_at', NULL,
            'reason', 'preapproval_not_committed',
            'scope_retired', false,
            'authority_active', false,
            'inserted', false,
            'deduped', false
        ));
    END IF;
    IF preapproval_record.case_id <> request->>'case_id'
       OR preapproval_record.release_sha256 <> request->>'release_sha256'
       OR preapproval_record.fixture_sha256 <> request->>'fixture_sha256'
       OR preapproval_record.run_id <> request->>'run_id'
       OR preapproval_record.session_key_sha256 <> request->>'session_key_sha256'
       OR preapproval_record.expires_at IS DISTINCT FROM expires_value
       OR preapproval_record.approved_by <> request->>'approved_by'
       OR preapproval_record.approval_source_sha256
            <> request->>'approval_source_sha256' THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'canary preclaim retirement differs from the sealed preapproval'
        );
    END IF;
    SELECT pg_catalog.count(*), pg_catalog.max(event.event_id::text),
           pg_catalog.max(
               (event.payload->'canary_scope_bootstrap_consumption')::text
           )::jsonb
      INTO consumption_count, consumption_event_id, consumption_record
      FROM public.canonical_event_log AS event
      JOIN canonical_brain.writer_event_provenance AS provenance
        ON provenance.event_id = event.event_id
     WHERE event.event_type = 'canary.scope.bootstrap_consumed'
       AND provenance.origin = 'canary_scope_bootstrap_consume'
       AND (
            event.case_id = request->>'case_id'
            OR event.payload->'canary_scope_bootstrap_consumption'->>'grant_id'
                = request->>'grant_id'
       );
    IF consumption_count <> 1
       OR NOT canonical_brain._keys_valid(
            consumption_record,
            ARRAY['grant_id','case_id','provisioning_receipt_sha256',
                  'preapproval_event_id','state'],
            ARRAY['grant_id','case_id','provisioning_receipt_sha256',
                  'preapproval_event_id','state']
       )
       OR consumption_record->>'grant_id' <> request->>'grant_id'
       OR consumption_record->>'case_id' <> request->>'case_id'
       OR consumption_record->>'provisioning_receipt_sha256'
            <> request->>'provisioning_receipt_sha256'
       OR consumption_record->>'preapproval_event_id'
            <> preapproval_record.receipt_event_id::text
       OR consumption_record->>'state' <> 'consumed' THEN
        RAISE EXCEPTION 'canary bootstrap consumption truth is invalid';
    END IF;

    IF claim_observed THEN
        IF locked_claim.case_id <> preapproval_record.case_id
           OR locked_claim.release_sha256 <> preapproval_record.release_sha256
           OR locked_claim.fixture_sha256 <> preapproval_record.fixture_sha256
           OR locked_claim.run_id <> preapproval_record.run_id
           OR locked_claim.approval_source_sha256
                <> preapproval_record.approval_source_sha256
           OR locked_claim.session_key_sha256
                <> preapproval_record.session_key_sha256
           OR locked_claim.expires_at IS DISTINCT FROM preapproval_record.expires_at
        THEN
            RAISE EXCEPTION 'claimed canary scope binding is invalid';
        END IF;
        revoke_response := canonical_brain.writer_capability_revoke_session(
            pg_catalog.jsonb_build_object(
                'session_key_sha256', locked_claim.session_key_sha256,
                'reason', 'preclaim_reconciliation_after_claim'
            ),
            runtime || pg_catalog.jsonb_build_object(
                'session_key_sha256', locked_claim.session_key_sha256,
                'capability_epoch_sha256',
                    locked_claim.capability_epoch_sha256
            )
        );
        IF NOT COALESCE((revoke_response->>'ok')::boolean, false) THEN
            RAISE EXCEPTION 'claimed canary session retirement failed';
        END IF;
        revoke_result := revoke_response->'result';
        IF NOT canonical_brain._keys_valid(
                revoke_result,
                ARRAY['success','session_key_sha256',
                      'capability_epoch_sha256','scope_type','scope_revoked',
                      'authority_active','revocation_event_id','inserted',
                      'deduped','revoked','canary_scopes_revoked'],
                ARRAY['success','session_key_sha256',
                      'capability_epoch_sha256','scope_type','scope_revoked',
                      'authority_active','revocation_event_id','inserted',
                      'deduped','revoked','canary_scopes_revoked']
           )
           OR revoke_result->>'success' <> 'true'
           OR revoke_result->>'session_key_sha256'
                <> locked_claim.session_key_sha256
           OR revoke_result->>'capability_epoch_sha256'
                <> locked_claim.capability_epoch_sha256
           OR revoke_result->>'scope_type' <> 'session'
           OR revoke_result->>'scope_revoked' <> 'true'
           OR revoke_result->>'authority_active' <> 'false'
           OR revoke_result->>'revocation_event_id' !~
                '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
           OR revoke_result->>'inserted' NOT IN ('true','false')
           OR revoke_result->>'deduped' NOT IN ('true','false')
           OR (revoke_result->>'inserted')::boolean
                = (revoke_result->>'deduped')::boolean THEN
            RAISE EXCEPTION 'claimed canary session receipt is invalid';
        END IF;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'outcome', 'claimed',
            'grant_id', preapproval_record.grant_id,
            'case_id', preapproval_record.case_id,
            'release_sha256', preapproval_record.release_sha256,
            'fixture_sha256', preapproval_record.fixture_sha256,
            'run_id', preapproval_record.run_id,
            'session_key_sha256', preapproval_record.session_key_sha256,
            'expires_at', preapproval_record.expires_at,
            'approved_by', preapproval_record.approved_by,
            'approval_source_sha256', preapproval_record.approval_source_sha256,
            'provisioning_receipt_sha256',
                request->>'provisioning_receipt_sha256',
            'preapproval_event_id', preapproval_record.receipt_event_id::text,
            'bootstrap_consumption_event_id', consumption_event_id,
            'claim_event_id', locked_claim.claim_event_id::text,
            'retirement_event_id', NULL,
            'revocation_event_id', revoke_result->>'revocation_event_id',
            'claimed_at', locked_claim.claimed_at,
            'retired_at', NULL,
            'reason', 'claim_already_committed_session_retired',
            'scope_retired', false,
            'authority_active', false,
            'inserted', (revoke_result->>'inserted')::boolean,
            'deduped', (revoke_result->>'deduped')::boolean
        ));
    END IF;

    IF retirement_record.grant_id IS NOT NULL THEN
        IF retirement_record.case_id <> preapproval_record.case_id
           OR retirement_record.release_sha256 <> preapproval_record.release_sha256
           OR retirement_record.fixture_sha256 <> preapproval_record.fixture_sha256
           OR retirement_record.run_id <> preapproval_record.run_id
           OR retirement_record.session_key_sha256
                <> preapproval_record.session_key_sha256
           OR retirement_record.expires_at
                IS DISTINCT FROM preapproval_record.expires_at
           OR retirement_record.approved_by <> preapproval_record.approved_by
           OR retirement_record.approval_source_sha256
                <> preapproval_record.approval_source_sha256
           OR retirement_record.provisioning_receipt_sha256
                <> request->>'provisioning_receipt_sha256'
           OR retirement_record.preapproval_event_id
                <> preapproval_record.receipt_event_id
           OR retirement_record.bootstrap_consumption_event_id::text
                <> consumption_event_id
           OR retirement_record.request_sha256 <> request_hash
           OR retirement_record.reason
                <> 'activation_failed_before_first_claim' THEN
            RAISE EXCEPTION 'existing canary preclaim retirement is invalid';
        END IF;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'outcome', 'retired',
            'grant_id', retirement_record.grant_id,
            'case_id', retirement_record.case_id,
            'release_sha256', retirement_record.release_sha256,
            'fixture_sha256', retirement_record.fixture_sha256,
            'run_id', retirement_record.run_id,
            'session_key_sha256', retirement_record.session_key_sha256,
            'expires_at', retirement_record.expires_at,
            'approved_by', retirement_record.approved_by,
            'approval_source_sha256',
                retirement_record.approval_source_sha256,
            'provisioning_receipt_sha256',
                retirement_record.provisioning_receipt_sha256,
            'preapproval_event_id',
                retirement_record.preapproval_event_id::text,
            'bootstrap_consumption_event_id',
                retirement_record.bootstrap_consumption_event_id::text,
            'claim_event_id', NULL,
            'retirement_event_id',
                retirement_record.retirement_event_id::text,
            'revocation_event_id', NULL,
            'claimed_at', NULL,
            'retired_at', retirement_record.retired_at,
            'reason', retirement_record.reason,
            'scope_retired', true,
            'authority_active', false,
            'inserted', false,
            'deduped', true
        ));
    END IF;

    retired_value := pg_catalog.clock_timestamp();
    append_result := canonical_brain._append_event(
        'canary.scope.preapproval_retired',
        preapproval_record.case_id,
        'Unclaimed isolated canary preapproval durably retired',
        pg_catalog.jsonb_build_object(
            'manual_ref', 'canary-preapproval-retire:'
                || preapproval_record.grant_id
        ),
        pg_catalog.jsonb_build_object(
            'actor', pg_catalog.jsonb_build_object(
                'type', 'writer_reconciliation', 'id', SESSION_USER
            ),
            'subject', pg_catalog.jsonb_build_object(
                'type', 'canary_scope', 'id', preapproval_record.grant_id
            )
        ),
        pg_catalog.jsonb_build_object(
            'canary_scope_preapproval_retirement',
            pg_catalog.jsonb_build_object(
                'grant_id', preapproval_record.grant_id,
                'case_id', preapproval_record.case_id,
                'release_sha256', preapproval_record.release_sha256,
                'fixture_sha256', preapproval_record.fixture_sha256,
                'run_id', preapproval_record.run_id,
                'session_key_sha256', preapproval_record.session_key_sha256,
                'expires_at', preapproval_record.expires_at,
                'approved_by', preapproval_record.approved_by,
                'approval_source_sha256',
                    preapproval_record.approval_source_sha256,
                'provisioning_receipt_sha256',
                    request->>'provisioning_receipt_sha256',
                'preapproval_event_id',
                    preapproval_record.receipt_event_id::text,
                'bootstrap_consumption_event_id', consumption_event_id,
                'reason', 'activation_failed_before_first_claim',
                'state', 'retired'
            )
        ),
        pg_catalog.jsonb_build_object(
            'isolated_canary', true,
            'preclaim_tombstone_recorded', true,
            'authority_active', false
        ),
        'canary-preapproval-retire:' || preapproval_record.grant_id,
        'canary_scope_preapproval_retire',
        runtime
    );
    IF NOT (append_result->>'ok')::boolean THEN
        RAISE EXCEPTION 'canary preclaim retirement receipt append failed';
    END IF;
    INSERT INTO canonical_brain.writer_canary_scope_preapproval_retirements (
        grant_id, case_id, release_sha256, fixture_sha256, run_id,
        session_key_sha256, expires_at, approved_by, approval_source_sha256,
        provisioning_receipt_sha256, preapproval_event_id,
        bootstrap_consumption_event_id, request_sha256, reason, retired_at,
        retirement_event_id
    ) VALUES (
        preapproval_record.grant_id, preapproval_record.case_id,
        preapproval_record.release_sha256, preapproval_record.fixture_sha256,
        preapproval_record.run_id, preapproval_record.session_key_sha256,
        preapproval_record.expires_at, preapproval_record.approved_by,
        preapproval_record.approval_source_sha256,
        request->>'provisioning_receipt_sha256',
        preapproval_record.receipt_event_id, consumption_event_id::uuid,
        request_hash, 'activation_failed_before_first_claim', retired_value,
        (append_result->'result'->>'event_id')::uuid
    );
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'outcome', 'retired',
        'grant_id', preapproval_record.grant_id,
        'case_id', preapproval_record.case_id,
        'release_sha256', preapproval_record.release_sha256,
        'fixture_sha256', preapproval_record.fixture_sha256,
        'run_id', preapproval_record.run_id,
        'session_key_sha256', preapproval_record.session_key_sha256,
        'expires_at', preapproval_record.expires_at,
        'approved_by', preapproval_record.approved_by,
        'approval_source_sha256', preapproval_record.approval_source_sha256,
        'provisioning_receipt_sha256',
            request->>'provisioning_receipt_sha256',
        'preapproval_event_id', preapproval_record.receipt_event_id::text,
        'bootstrap_consumption_event_id', consumption_event_id,
        'claim_event_id', NULL,
        'retirement_event_id', append_result->'result'->>'event_id',
        'revocation_event_id', NULL,
        'claimed_at', NULL,
        'retired_at', retired_value,
        'reason', 'activation_failed_before_first_claim',
        'scope_retired', true,
        'authority_active', false,
        'inserted', true,
        'deduped', false
    ));
END
$function$;

-- Fixed public routine 2/18.  Gateway-callable exact one-shot claim.
CREATE OR REPLACE FUNCTION canonical_brain.writer_canary_scope_claim(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    request_hash text;
    preapproval_record canonical_brain.writer_canary_scope_preapprovals%ROWTYPE;
    existing_claim canonical_brain.writer_canary_scope_claims%ROWTYPE;
    append_result jsonb;
    authority_active boolean;
    claimed_value timestamptz;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR runtime->>'platform' <> 'api_server'
       OR runtime->>'service_internal' = 'true'
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$'
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                  'run_id','approval_source_sha256'],
            ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                  'run_id','approval_source_sha256']
       )
       OR request->>'grant_id' !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'case_id' !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'run_id' !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->>'release_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'fixture_sha256' !~ '^[0-9a-f]{64}$'
       OR request->>'approval_source_sha256' !~ '^[0-9a-f]{64}$' THEN
        RETURN canonical_brain._fail(
            'invalid_request', 'canary scope claim is invalid'
        );
    END IF;
    request_hash := canonical_brain._sha256_json(
        request || pg_catalog.jsonb_build_object(
            'session_key_sha256', session_value,
            'capability_epoch_sha256', epoch_value
        )
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_capability_revocation_scopes AS scope
         WHERE scope.scope_type = 'session'
           AND scope.session_key_sha256 = session_value
           AND scope.capability_epoch_sha256 = epoch_value
    ) THEN
        RETURN canonical_brain._fail(
            'session_epoch_retired',
            'session authority epoch has been durably retired'
        );
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'canary-preapproval:' || (request->>'grant_id'), 0
        )
    );
    SELECT * INTO preapproval_record
     FROM canonical_brain.writer_canary_scope_preapprovals AS preapproval
     WHERE preapproval.grant_id = request->>'grant_id'
     FOR UPDATE;
    IF NOT FOUND THEN
        RETURN canonical_brain._fail(
            'canary_scope_missing', 'canary scope has no owner preapproval'
        );
    END IF;
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_canary_scope_preapproval_retirements
               AS terminal
         WHERE terminal.grant_id = request->>'grant_id'
    ) THEN
        RETURN canonical_brain._fail(
            'canary_scope_preapproval_retired',
            'canary scope preapproval was durably retired before claim'
        );
    END IF;
    IF preapproval_record.case_id <> request->>'case_id'
       OR preapproval_record.release_sha256 <> request->>'release_sha256'
       OR preapproval_record.fixture_sha256 <> request->>'fixture_sha256'
       OR preapproval_record.run_id <> request->>'run_id'
       OR preapproval_record.approval_source_sha256
            <> request->>'approval_source_sha256'
       OR preapproval_record.session_key_sha256 <> session_value THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'canary scope claim differs from its exact preapproval'
        );
    END IF;
    SELECT * INTO existing_claim
      FROM canonical_brain.writer_canary_scope_claims AS claim
     WHERE claim.grant_id = request->>'grant_id';
    IF FOUND THEN
        IF existing_claim.request_sha256 <> request_hash
           OR existing_claim.session_key_sha256 <> session_value
           OR existing_claim.capability_epoch_sha256 <> epoch_value THEN
            RETURN canonical_brain._fail(
                'canary_scope_replayed',
                'one-shot canary scope was already claimed by another session generation'
            );
        END IF;
        authority_active := existing_claim.expires_at
                > pg_catalog.clock_timestamp()
            AND NOT EXISTS (
                SELECT 1
                  FROM canonical_brain.writer_capability_revocation_scopes AS scope
                 WHERE scope.scope_type = 'session'
                   AND scope.session_key_sha256 = session_value
                   AND scope.capability_epoch_sha256 = epoch_value
            );
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'grant_id', existing_claim.grant_id,
            'case_id', existing_claim.case_id,
            'release_sha256', existing_claim.release_sha256,
            'fixture_sha256', existing_claim.fixture_sha256,
            'run_id', existing_claim.run_id,
            'approval_source_sha256', existing_claim.approval_source_sha256,
            'session_key_sha256', existing_claim.session_key_sha256,
            'capability_epoch_sha256', existing_claim.capability_epoch_sha256,
            'expires_at', existing_claim.expires_at,
            'claimed_at', existing_claim.claimed_at,
            'event_id', existing_claim.claim_event_id::text,
            'claim_event_id', existing_claim.claim_event_id::text,
            'authority_active', authority_active,
            'inserted', false,
            'deduped', true
        ));
    END IF;
    IF preapproval_record.expires_at <= pg_catalog.clock_timestamp() THEN
        RETURN canonical_brain._fail(
            'canary_scope_expired', 'canary scope preapproval has expired'
        );
    END IF;
    claimed_value := pg_catalog.clock_timestamp();
    append_result := canonical_brain._append_event(
        'canary.scope.claimed',
        preapproval_record.case_id,
        'Exact isolated canary API session scope claimed',
        pg_catalog.jsonb_build_object(
            'manual_ref', 'canary-claim:' || preapproval_record.grant_id
        ),
        pg_catalog.jsonb_build_object(
            'actor', pg_catalog.jsonb_build_object(
                'type', 'service', 'id', 'api_server'
            ),
            'subject', pg_catalog.jsonb_build_object(
                'type', 'canary_scope', 'id', preapproval_record.grant_id
            )
        ),
        pg_catalog.jsonb_build_object(
            'canary_scope_claim', pg_catalog.jsonb_build_object(
                'grant_id', preapproval_record.grant_id,
                'case_id', preapproval_record.case_id,
                'release_sha256', preapproval_record.release_sha256,
                'fixture_sha256', preapproval_record.fixture_sha256,
                'run_id', preapproval_record.run_id,
                'approval_source_sha256',
                    preapproval_record.approval_source_sha256,
                'session_key_sha256', session_value,
                'capability_epoch_sha256', epoch_value,
                'expires_at', preapproval_record.expires_at,
                'state', 'claimed'
            )
        ),
        pg_catalog.jsonb_build_object(
            'isolated_canary', true,
            'one_shot_scope', true
        ),
        'canary-claim:' || preapproval_record.grant_id,
        'canary_scope_claim',
        runtime
    );
    IF NOT (append_result->>'ok')::boolean THEN
        RETURN append_result;
    END IF;
    INSERT INTO canonical_brain.writer_canary_scope_claims (
        grant_id, case_id, release_sha256, fixture_sha256, run_id,
        approval_source_sha256, session_key_sha256,
        capability_epoch_sha256, expires_at, request_sha256, claimed_at,
        claim_event_id
    ) VALUES (
        preapproval_record.grant_id, preapproval_record.case_id,
        preapproval_record.release_sha256, preapproval_record.fixture_sha256,
        preapproval_record.run_id, preapproval_record.approval_source_sha256,
        session_value, epoch_value, preapproval_record.expires_at, request_hash,
        claimed_value,
        (append_result->'result'->>'event_id')::uuid
    );
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'grant_id', preapproval_record.grant_id,
        'case_id', preapproval_record.case_id,
        'release_sha256', preapproval_record.release_sha256,
        'fixture_sha256', preapproval_record.fixture_sha256,
        'run_id', preapproval_record.run_id,
        'approval_source_sha256', preapproval_record.approval_source_sha256,
        'session_key_sha256', session_value,
        'capability_epoch_sha256', epoch_value,
        'expires_at', preapproval_record.expires_at,
        'claimed_at', claimed_value,
        'event_id', append_result->'result'->>'event_id',
        'claim_event_id', append_result->'result'->>'event_id',
        'authority_active', true,
        'inserted', true,
        'deduped', false
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'canary scope claim failed');
END
$function$;

-- Fixed public routine 3/18.  This is an exact bounded event query, never a
-- semantic classifier.  The model chooses case/thread and requested view.
CREATE OR REPLACE FUNCTION canonical_brain.writer_case_query(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    case_value text := COALESCE(request->>'case_id', '');
    thread_value text := COALESCE(request->>'thread_id', '');
    view_value text := COALESCE(request->>'view', 'summary');
    limit_value integer;
    events_value jsonb;
    support_events_value jsonb := '[]'::jsonb;
    support_incomplete_value jsonb := '[]'::jsonb;
    missing_verification_value jsonb := '[]'::jsonb;
    total_count integer := 0;
    window_count integer := 0;
    total_candidate_count integer := 0;
    window_candidate_count integer := 0;
    plan_count integer := 0;
    plan_lineage_count integer := 0;
    support_candidate_count integer := 0;
    support_returned_count integer := 0;
    quarantined_security_count integer := 0;
    head_result jsonb;
    head_plan jsonb;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['case_id','thread_id','limit','view'],
            ARRAY[]::text[]
       )
       OR ((case_value = '') = (thread_value = ''))
       OR (case_value <> '' AND case_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$')
       OR pg_catalog.length(thread_value) > 240
       OR view_value NOT IN ('summary', 'resume_bundle')
       OR COALESCE(request->>'limit', '80') !~ '^[1-9][0-9]{0,2}$' THEN
        RETURN canonical_brain._fail('invalid_request', 'case query is invalid');
    END IF;
    IF case_value <> ''
       AND NOT canonical_brain._case_scope_authorized(case_value, runtime, false) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'runtime is not authorized for the exact canonical case'
        );
    END IF;
    IF thread_value <> ''
       AND NOT (
            runtime->>'platform' = 'discord'
            AND runtime->>'owner_authenticated' = 'true'
       )
       AND thread_value <> COALESCE(
            NULLIF(runtime->>'thread_id', ''), runtime->>'chat_id', ''
       ) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'thread query must match the exact observed runtime thread'
        );
    END IF;
    limit_value := (COALESCE(request->>'limit', '80'))::integer;
    IF limit_value > 200 THEN
        RETURN canonical_brain._fail('invalid_request', 'case query limit exceeds 200');
    END IF;

    SELECT pg_catalog.count(*), pg_catalog.count(DISTINCT event.case_id)
      INTO total_count, total_candidate_count
      FROM public.canonical_event_log AS event
     WHERE (
          (case_value <> '' AND event.case_id = case_value)
          OR (thread_value <> '' AND EXISTS (
              SELECT 1
                FROM public.canonical_event_log AS scoped_event
                JOIN canonical_brain.writer_event_provenance AS provenance
                  ON provenance.event_id = scoped_event.event_id
               WHERE scoped_event.case_id = event.case_id
                 AND provenance.trusted_runtime->>'platform' = runtime->>'platform'
                 AND thread_value IN (
                     COALESCE(provenance.trusted_runtime->>'thread_id', ''),
                     COALESCE(provenance.trusted_runtime->>'chat_id', '')
                 )
          ))
     )
       AND EXISTS (
            SELECT 1
              FROM canonical_brain.writer_event_provenance AS trusted_event
             WHERE trusted_event.event_id = event.event_id
       )
       AND canonical_brain._case_scope_authorized(
            event.case_id, runtime, false
       );

    WITH candidates AS (
        SELECT canonical_brain._event_envelope(event) AS event_json,
               event.occurred_at,
               event.event_id
          FROM public.canonical_event_log AS event
         WHERE (
              (case_value <> '' AND event.case_id = case_value)
              OR (thread_value <> '' AND EXISTS (
                  SELECT 1
                    FROM public.canonical_event_log AS scoped_event
                    JOIN canonical_brain.writer_event_provenance AS provenance
                      ON provenance.event_id = scoped_event.event_id
                   WHERE scoped_event.case_id = event.case_id
                     AND provenance.trusted_runtime->>'platform' = runtime->>'platform'
                     AND thread_value IN (
                         COALESCE(provenance.trusted_runtime->>'thread_id', ''),
                         COALESCE(provenance.trusted_runtime->>'chat_id', '')
                     )
              ))
         )
           AND EXISTS (
                SELECT 1
                  FROM canonical_brain.writer_event_provenance AS trusted_event
                 WHERE trusted_event.event_id = event.event_id
           )
           AND canonical_brain._case_scope_authorized(
                event.case_id, runtime, false
           )
         ORDER BY event.occurred_at DESC, event.event_id DESC
         LIMIT limit_value
    ), sized AS (
        SELECT candidates.*,
               pg_catalog.sum(
                   pg_catalog.octet_length(candidates.event_json::text) + 1
               ) OVER (
                   ORDER BY candidates.occurred_at DESC, candidates.event_id DESC
               ) AS cumulative_bytes
          FROM candidates
    )
    SELECT COALESCE(
        pg_catalog.jsonb_agg(
            sized.event_json ORDER BY sized.occurred_at, sized.event_id
        ) FILTER (WHERE sized.cumulative_bytes <= 384000),
        '[]'::jsonb
    )
      INTO events_value
      FROM sized;
    window_count := pg_catalog.jsonb_array_length(events_value);
    SELECT pg_catalog.count(DISTINCT item->>'case_id')
      INTO window_candidate_count
      FROM pg_catalog.jsonb_array_elements(events_value) AS event_item(item);

    IF case_value <> '' AND view_value = 'resume_bundle' THEN
        SELECT pg_catalog.count(*) INTO quarantined_security_count
          FROM public.canonical_event_log AS event
         WHERE event.case_id = case_value
           AND NOT EXISTS (
                SELECT 1
                  FROM canonical_brain.writer_event_provenance AS provenance
                 WHERE provenance.event_id = event.event_id
           );
        IF quarantined_security_count > 0 THEN
            support_incomplete_value := support_incomplete_value
                || pg_catalog.jsonb_build_array(
                    'legacy_events_quarantined'
                );
        END IF;
        SELECT pg_catalog.count(*) INTO plan_count
          FROM public.canonical_event_log AS event
          JOIN canonical_brain.writer_event_provenance AS provenance
            ON provenance.event_id = event.event_id
         WHERE event.case_id = case_value
           AND event.event_type = 'task.plan.updated';
        IF plan_count > 256 THEN
            support_incomplete_value := support_incomplete_value
                || pg_catalog.jsonb_build_array('plan_graph_exceeds_256');
        END IF;
        SELECT pg_catalog.count(DISTINCT event.payload->'plan'->>'plan_id')
          INTO plan_lineage_count
          FROM public.canonical_event_log AS event
          JOIN canonical_brain.writer_event_provenance AS provenance
            ON provenance.event_id = event.event_id
         WHERE event.case_id = case_value
           AND event.event_type = 'task.plan.updated';
        IF plan_lineage_count > 16 THEN
            support_incomplete_value := support_incomplete_value
                || pg_catalog.jsonb_build_array('plan_lineage_exceeds_16');
        END IF;
        head_result := canonical_brain._plan_head(case_value);
        IF NOT (head_result->>'ok')::boolean THEN
            support_incomplete_value := support_incomplete_value
                || pg_catalog.jsonb_build_array(
                    COALESCE(head_result->'error'->>'code', 'plan_graph_invalid')
                );
            head_plan := NULL;
        ELSE
            head_plan := head_result->'result'->'head'->'plan';
            IF head_plan = 'null'::jsonb THEN
                head_plan := NULL;
            END IF;
        END IF;

        WITH selected_ids AS (
            (
                SELECT latest.event_id
                  FROM (
                    SELECT DISTINCT ON (event.payload->'plan'->>'plan_id')
                           event.event_id,
                           event.occurred_at,
                           event.payload->'plan'->>'plan_id' AS plan_id,
                           CASE
                               WHEN event.payload->'plan'->>'revision' ~ '^[1-9][0-9]{0,8}$'
                                   THEN (event.payload->'plan'->>'revision')::integer
                               ELSE -1
                           END AS revision
                      FROM public.canonical_event_log AS event
                      JOIN canonical_brain.writer_event_provenance AS provenance
                        ON provenance.event_id = event.event_id
                     WHERE event.case_id = case_value
                       AND event.event_type = 'task.plan.updated'
                     ORDER BY event.payload->'plan'->>'plan_id',
                              CASE
                                  WHEN event.payload->'plan'->>'revision' ~ '^[1-9][0-9]{0,8}$'
                                      THEN (event.payload->'plan'->>'revision')::integer
                                  ELSE -1
                              END DESC,
                              event.occurred_at DESC,
                              event.event_id DESC
                  ) AS latest
                 ORDER BY latest.occurred_at, latest.event_id
                 LIMIT 16
            )
            UNION
            (
                SELECT event.event_id
                  FROM public.canonical_event_log AS event
                  JOIN canonical_brain.writer_event_provenance AS provenance
                    ON provenance.event_id = event.event_id
                 WHERE event.case_id = case_value
                   AND event.event_type IN (
                        'canary.scope.bootstrap_authorized',
                        'canary.scope.bootstrap_consumed',
                        'canary.scope.bootstrap_retired',
                        'canary.scope.preapproved',
                        'canary.scope.preapproval_retired',
                        'canary.scope.claimed',
                        'canary.scope.revoked'
                   )
                 ORDER BY event.occurred_at DESC, event.event_id DESC
                 LIMIT 64
            )
            UNION
            (
                SELECT event.event_id
                  FROM public.canonical_event_log AS event
                  JOIN canonical_brain.writer_event_provenance AS provenance
                    ON provenance.event_id = event.event_id
                 WHERE event.case_id = case_value
                   AND event.event_type = 'task.verification.recorded'
                 ORDER BY event.occurred_at DESC, event.event_id DESC
                 LIMIT 80
            )
            UNION
            (
                SELECT event.event_id
                  FROM public.canonical_event_log AS event
                  JOIN canonical_brain.writer_event_provenance AS provenance
                    ON provenance.event_id = event.event_id
                 WHERE event.case_id = case_value
                   AND event.event_type = 'approval.capability.recorded'
                 ORDER BY event.occurred_at DESC, event.event_id DESC
                 LIMIT 80
            )
            UNION
            (
                SELECT event.event_id
                  FROM public.canonical_event_log AS event
                  JOIN canonical_brain.writer_event_provenance AS provenance
                    ON provenance.event_id = event.event_id
                 WHERE event.case_id = case_value
                   AND event.event_type = 'capability.check.recorded'
                 ORDER BY event.occurred_at DESC, event.event_id DESC
                 LIMIT 64
            )
            UNION
            (
                SELECT event.event_id
                  FROM public.canonical_event_log AS event
                  JOIN canonical_brain.writer_event_provenance AS provenance
                    ON provenance.event_id = event.event_id
                 WHERE event.case_id = case_value
                   AND event.event_type IN ('route_back.sent','route_back.blocked')
                 ORDER BY event.occurred_at DESC, event.event_id DESC
                 LIMIT 80
            )
            UNION
            (
                SELECT event.event_id
                  FROM public.canonical_event_log AS event
                  JOIN canonical_brain.writer_event_provenance AS provenance
                    ON provenance.event_id = event.event_id
                 WHERE event.case_id = case_value
                   AND event.event_type = 'lease.shadow.recorded'
                 ORDER BY event.occurred_at DESC, event.event_id DESC
                 LIMIT 64
            )
            UNION
            (
                SELECT event.event_id
                  FROM public.canonical_event_log AS event
                  JOIN canonical_brain.writer_event_provenance AS provenance
                    ON provenance.event_id = event.event_id
                  JOIN pg_catalog.jsonb_array_elements_text(
                      CASE
                          WHEN pg_catalog.jsonb_typeof(head_plan->'verification_event_ids') = 'array'
                              THEN head_plan->'verification_event_ids'
                          ELSE '[]'::jsonb
                      END
                  ) AS required(event_id)
                    ON event.event_id = CASE
                        WHEN required.event_id ~
                             '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
                            THEN required.event_id::uuid
                        ELSE NULL
                    END
                 WHERE event.case_id = case_value
                   AND event.event_type = 'task.verification.recorded'
            )
        ), bounded_support AS (
            SELECT event.event_id, event.occurred_at
              FROM selected_ids
              JOIN public.canonical_event_log AS event
                ON event.event_id = selected_ids.event_id
             ORDER BY event.occurred_at, event.event_id
             LIMIT 500
        ), support_rows AS (
            SELECT bounded_support.event_id,
                   bounded_support.occurred_at,
                   canonical_brain._event_envelope(event) AS event_json
              FROM bounded_support
              JOIN public.canonical_event_log AS event
                ON event.event_id = bounded_support.event_id
        ), sized_support AS (
            SELECT support_rows.*,
                   pg_catalog.sum(
                       pg_catalog.octet_length(support_rows.event_json::text) + 1
                   ) OVER (
                       ORDER BY support_rows.occurred_at DESC,
                                support_rows.event_id DESC
                   ) AS cumulative_bytes
              FROM support_rows
        )
        SELECT COALESCE(
            pg_catalog.jsonb_agg(
                sized_support.event_json
                ORDER BY sized_support.occurred_at, sized_support.event_id
            ) FILTER (WHERE sized_support.cumulative_bytes <= 600000),
            '[]'::jsonb
        ),
        pg_catalog.count(*),
        pg_catalog.count(*) FILTER (
            WHERE sized_support.cumulative_bytes <= 600000
        )
          INTO support_events_value, support_candidate_count, support_returned_count
          FROM sized_support;
        IF support_returned_count < support_candidate_count THEN
            support_incomplete_value := support_incomplete_value
                || pg_catalog.jsonb_build_array('support_byte_budget_exceeded');
        END IF;

        IF head_plan IS NOT NULL THEN
            IF EXISTS (
                SELECT 1
                  FROM pg_catalog.jsonb_array_elements_text(
                      CASE
                          WHEN pg_catalog.jsonb_typeof(head_plan->'verification_event_ids') = 'array'
                              THEN head_plan->'verification_event_ids'
                          ELSE '[]'::jsonb
                      END
                  ) AS required(event_id)
                 WHERE required.event_id !~
                       '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
            ) THEN
                support_incomplete_value := support_incomplete_value
                    || pg_catalog.jsonb_build_array('referenced_verification_id_invalid');
            END IF;
            SELECT COALESCE(pg_catalog.jsonb_agg(required.event_id), '[]'::jsonb)
              INTO missing_verification_value
              FROM pg_catalog.jsonb_array_elements_text(
                  CASE
                      WHEN pg_catalog.jsonb_typeof(head_plan->'verification_event_ids') = 'array'
                          THEN head_plan->'verification_event_ids'
                      ELSE '[]'::jsonb
                  END
              ) AS required(event_id)
             WHERE required.event_id ~
                       '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
               AND NOT EXISTS (
                    SELECT 1
                      FROM public.canonical_event_log AS event
                      JOIN canonical_brain.writer_event_provenance AS provenance
                        ON provenance.event_id = event.event_id
                     WHERE event.case_id = case_value
                       AND event.event_type = 'task.verification.recorded'
                       AND event.event_id = CASE
                           WHEN required.event_id ~
                                '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
                               THEN required.event_id::uuid
                           ELSE NULL
                       END
                );
            IF pg_catalog.jsonb_array_length(missing_verification_value) > 0 THEN
                support_incomplete_value := support_incomplete_value
                    || pg_catalog.jsonb_build_array('referenced_verification_missing');
            END IF;
        END IF;
    END IF;
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'events', events_value,
        'support_events', support_events_value,
        'view', view_value,
        'bounded', true,
        'event_count', window_count,
        'truncated', total_count > window_count,
        'candidate_cases_truncated', total_candidate_count > window_candidate_count,
        'support_incomplete_reasons', support_incomplete_value,
        'missing_verification_event_ids', missing_verification_value
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'canonical case query failed');
END
$function$;

-- Fixed public routine 4/18.  Only exact sent authorizations are projected
-- back to their source thread, and the result is hard bounded to three cases.
CREATE OR REPLACE FUNCTION canonical_brain.writer_routeback_context(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    thread_value text := COALESCE(request->>'thread_id', '');
    cases_value jsonb;
    total_count integer;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR NOT canonical_brain._keys_valid(request, ARRAY['thread_id'], ARRAY['thread_id'])
       OR pg_catalog.length(thread_value) NOT BETWEEN 1 AND 240
       OR thread_value <> COALESCE(NULLIF(runtime->>'thread_id', ''), runtime->>'chat_id', '') THEN
        RETURN canonical_brain._fail('scope_mismatch', 'route-back context scope is invalid');
    END IF;

    SELECT pg_catalog.count(*)
      INTO total_count
      FROM canonical_brain.writer_routeback_authorizations AS authorization_row
      JOIN canonical_brain.writer_routeback_terminals AS terminal
        ON terminal.authorization_id = authorization_row.authorization_id
     WHERE terminal.outcome = 'sent'
       AND thread_value = COALESCE(
            authorization_row.target_ref->>'thread_id',
            authorization_row.target_ref->>'channel_id',
            ''
       )
       AND COALESCE(
            authorization_row.source_thread_id,
            ''
       ) NOT IN ('', thread_value)
       AND canonical_brain._case_scope_authorized(
            authorization_row.case_id, runtime, false
       );

    SELECT COALESCE(pg_catalog.jsonb_agg(item ORDER BY item->>'case_id'), '[]'::jsonb)
      INTO cases_value
      FROM (
        SELECT pg_catalog.jsonb_build_object(
            'case_id', authorization_row.case_id,
            'source_thread_id', authorization_row.source_thread_id
        ) AS item
          FROM canonical_brain.writer_routeback_authorizations AS authorization_row
          JOIN canonical_brain.writer_routeback_terminals AS terminal
            ON terminal.authorization_id = authorization_row.authorization_id
         WHERE terminal.outcome = 'sent'
           AND thread_value = COALESCE(
                authorization_row.target_ref->>'thread_id',
                authorization_row.target_ref->>'channel_id',
                ''
           )
           AND authorization_row.source_thread_id NOT IN ('', thread_value)
           AND canonical_brain._case_scope_authorized(
                authorization_row.case_id, runtime, false
           )
         ORDER BY terminal.finalized_at DESC, authorization_row.authorization_id DESC
         LIMIT 3
      ) AS bounded;
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'thread_id', thread_value,
        'cases', cases_value,
        'truncated', total_count > 3
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'route-back context read failed');
END
$function$;

-- Fixed public routine 5/18.
CREATE OR REPLACE FUNCTION canonical_brain.writer_plan_active_match(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    head_result jsonb;
    head_plan jsonb;
    matches_value boolean;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR NOT canonical_brain._keys_valid(
            request, ARRAY['case_id','plan_id'], ARRAY['case_id','plan_id']
       )
       OR COALESCE(request->>'case_id', '') !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR pg_catalog.length(COALESCE(request->>'plan_id', '')) NOT BETWEEN 1 AND 240 THEN
        RETURN canonical_brain._fail('invalid_request', 'active plan query is invalid');
    END IF;
    IF NOT canonical_brain._case_scope_authorized(request->>'case_id', runtime, false) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'runtime is not authorized for the exact canonical case'
        );
    END IF;
    head_result := canonical_brain._plan_head(request->>'case_id');
    IF NOT (head_result->>'ok')::boolean THEN
        RETURN head_result;
    END IF;
    head_plan := head_result->'result'->'head'->'plan';
    matches_value := head_plan IS NOT NULL
        AND head_plan <> 'null'::jsonb
        AND head_plan->>'plan_id' = request->>'plan_id'
        AND head_plan->>'state' = 'active';
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'matches', matches_value,
        'active', matches_value,
        'plan_revision', CASE WHEN matches_value
                              THEN (head_plan->>'revision')::integer ELSE NULL END,
        'active_plan_id', CASE WHEN head_plan->>'state' = 'active'
                               THEN head_plan->>'plan_id' ELSE '' END,
        'active_plan_revision', CASE WHEN head_plan->>'state' = 'active'
                                     THEN head_plan->'revision' ELSE NULL END
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'active plan query failed');
END
$function$;

-- Fixed public routine 6/18.  Privileged receipt types and task CAS events are
-- unavailable through this model-authored append entry point.
CREATE OR REPLACE FUNCTION canonical_brain.writer_event_append_model(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    event_type_value text := COALESCE(request->>'event_type', '');
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$' THEN
        RETURN canonical_brain._fail(
            'invalid_runtime',
            'model event append requires exact session and routing epoch'
        );
    END IF;
    IF NOT canonical_brain._keys_valid(
            request,
            ARRAY['event_type','case_id','summary','source_refs','actors','body','safety','idempotency_key'],
            ARRAY['event_type','case_id','summary','source_refs','body','idempotency_key']
       ) THEN
        RETURN canonical_brain._fail('invalid_request', 'model event request is invalid');
    END IF;
    IF event_type_value IN (
        'task.plan.updated', 'task.verification.recorded',
        'route_back.intent.created', 'route_back.sent', 'route_back.blocked',
        'approval.capability.recorded', 'approval.capability.revoked',
        'approval.capability.session_revoked', 'capability.check.recorded',
        'lease.shadow.recorded', 'canary.scope.bootstrap_authorized',
        'canary.scope.bootstrap_consumed', 'canary.scope.bootstrap_retired',
        'canary.scope.preapproved', 'canary.scope.preapproval_retired',
        'canary.scope.claimed', 'canary.scope.revoked'
    ) THEN
        RETURN canonical_brain._fail(
            'privileged_event_forbidden',
            'event type requires its fixed privileged routine'
        );
    END IF;
    -- Session retirement and every initiating mutation serialize on this exact
    -- scope before any case/plan/lifecycle lock.  A mutation either commits
    -- before rotation or observes its durable tombstone and cannot start.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_capability_revocation_scopes AS scope
         WHERE scope.scope_type = 'session'
           AND scope.session_key_sha256 = session_value
           AND scope.capability_epoch_sha256 = epoch_value
    ) THEN
        RETURN canonical_brain._fail(
            'session_epoch_retired',
            'session authority epoch has been durably retired'
        );
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'canonical-case:' || (request->>'case_id'), 0
        )
    );
    IF NOT canonical_brain._case_scope_authorized(request->>'case_id', runtime, true) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'runtime is not authorized to append to the exact canonical case'
        );
    END IF;
    RETURN canonical_brain._append_event(
        event_type_value,
        request->>'case_id',
        request->>'summary',
        request->'source_refs',
        COALESCE(request->'actors', '{}'::jsonb),
        request->'body',
        COALESCE(request->'safety', '{}'::jsonb),
        request->>'idempotency_key',
        'hermes_agent_llm_reasoning',
        runtime
    );
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'model event append failed');
END
$function$;

-- Fixed public routine 7/18.  Plan revision validation and the append share a
-- case-scoped transaction advisory lock, providing the canonical CAS point.
CREATE OR REPLACE FUNCTION canonical_brain.writer_plan_transition(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    case_value text := COALESCE(request->>'case_id', '');
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    plan_value jsonb := request->'body'->'plan';
    plan_id_value text;
    revision_value integer;
    state_value text;
    head_result jsonb;
    previous_plan jsonb;
    previous_plan_id text := '';
    previous_revision integer := 0;
    transition_identity text;
    transition_event_id uuid;
    append_result jsonb;
    revoked_reason text;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$' THEN
        RETURN canonical_brain._fail(
            'invalid_runtime',
            'plan transition requires exact session and routing epoch'
        );
    END IF;
    IF NOT canonical_brain._keys_valid(
            request,
            ARRAY['event_type','case_id','summary','source_refs','actors','body','safety','idempotency_key'],
            ARRAY['event_type','case_id','summary','source_refs','body','idempotency_key']
       )
       OR request->>'event_type' <> 'task.plan.updated'
       OR case_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR pg_catalog.jsonb_typeof(plan_value) <> 'object'
       OR COALESCE(plan_value->>'plan_id', '') !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR COALESCE(plan_value->>'revision', '') !~ '^[1-9][0-9]{0,8}$'
       OR COALESCE(plan_value->>'state', '') NOT IN ('active','completed','blocked','cancelled')
       OR pg_catalog.octet_length(COALESCE(request->>'idempotency_key', ''))
          NOT BETWEEN 1 AND 256 THEN
        RETURN canonical_brain._fail('invalid_request', 'plan transition request is invalid');
    END IF;
    plan_id_value := plan_value->>'plan_id';
    revision_value := (plan_value->>'revision')::integer;
    state_value := plan_value->>'state';
    transition_identity := request->>'idempotency_key';

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_capability_revocation_scopes AS scope
         WHERE scope.scope_type = 'session'
           AND scope.session_key_sha256 = session_value
           AND scope.capability_epoch_sha256 = epoch_value
    ) THEN
        RETURN canonical_brain._fail(
            'session_epoch_retired',
            'session authority epoch has been durably retired'
        );
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('canonical-case:' || case_value, 0)
    );
    IF NOT canonical_brain._case_scope_authorized(case_value, runtime, true) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'runtime is not authorized to transition the exact canonical case'
        );
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('canonical-plan:' || case_value, 0)
    );
    transition_event_id := canonical_brain._deterministic_uuid(
        'canonical-writer:' || case_value || ':task.plan.updated:'
            || transition_identity
    );
    IF EXISTS (
        SELECT 1
          FROM public.canonical_event_log AS event
         WHERE event.event_id = transition_event_id
    ) THEN
        RETURN canonical_brain._append_event(
            'task.plan.updated',
            case_value,
            request->>'summary',
            request->'source_refs',
            COALESCE(request->'actors', '{}'::jsonb),
            request->'body' || pg_catalog.jsonb_build_object(
                'writer_request_idempotency_key', request->>'idempotency_key'
            ),
            COALESCE(request->'safety', '{}'::jsonb),
            transition_identity,
            'canonical_plan_cas',
            runtime
        );
    END IF;
    head_result := canonical_brain._plan_head(case_value);
    IF NOT (head_result->>'ok')::boolean THEN
        RETURN head_result;
    END IF;
    previous_plan := head_result->'result'->'head'->'plan';
    IF previous_plan IS NULL OR previous_plan = 'null'::jsonb THEN
        IF revision_value <> 1
           OR COALESCE(plan_value->>'supersedes_plan_id', '') <> ''
           OR plan_value ? 'supersedes_plan_revision' THEN
            RETURN canonical_brain._fail(
                'plan_cas_conflict',
                'initial plan must be revision 1 without a predecessor'
            );
        END IF;
    ELSE
        previous_plan_id := previous_plan->>'plan_id';
        previous_revision := (previous_plan->>'revision')::integer;
        IF plan_id_value = previous_plan_id THEN
            IF revision_value <> previous_revision + 1 THEN
                RETURN canonical_brain._fail(
                    'plan_cas_conflict',
                    'same plan_id must advance the exact current revision by one'
                );
            END IF;
            IF COALESCE(plan_value->>'supersedes_plan_id', '')
               IS DISTINCT FROM COALESCE(previous_plan->>'supersedes_plan_id', '')
               OR COALESCE(plan_value->>'supersedes_plan_revision', '')
               IS DISTINCT FROM COALESCE(previous_plan->>'supersedes_plan_revision', '') THEN
                RETURN canonical_brain._fail(
                    'plan_cas_conflict',
                    'same plan_id cannot change its supersession edge'
                );
            END IF;
        ELSE
            IF revision_value <> 1
               OR plan_value->>'supersedes_plan_id' <> previous_plan_id
               OR COALESCE(plan_value->>'supersedes_plan_revision', '')
                  <> previous_revision::text THEN
                RETURN canonical_brain._fail(
                    'plan_cas_conflict',
                    'new plan_id must revision-1 supersede the exact current head'
                );
            END IF;
        END IF;
    END IF;

    IF state_value = 'completed' THEN
        IF previous_plan IS NULL OR previous_plan = 'null'::jsonb
           OR previous_plan_id <> plan_id_value
           OR previous_revision <> revision_value - 1
           OR previous_plan->>'state' <> 'active'
           OR plan_value->'success_criteria'
              IS DISTINCT FROM previous_plan->'success_criteria' THEN
            RETURN canonical_brain._fail(
                'completion_plan_mismatch',
                'completed plan requires the exact immediately prior active revision of the same plan_id'
            );
        END IF;
        IF pg_catalog.jsonb_typeof(plan_value->'verification_event_ids') <> 'array'
           OR pg_catalog.jsonb_array_length(plan_value->'verification_event_ids') NOT BETWEEN 1 AND 64
           OR pg_catalog.jsonb_typeof(plan_value->'success_criteria') <> 'array'
           OR pg_catalog.jsonb_array_length(plan_value->'success_criteria') NOT BETWEEN 1 AND 32 THEN
            RETURN canonical_brain._fail(
                'verification_receipt_missing',
                'completed plan requires bounded verification_event_ids and success_criteria arrays'
            );
        END IF;
        IF EXISTS (
            SELECT 1
              FROM pg_catalog.jsonb_array_elements(
                  plan_value->'success_criteria'
              ) AS criterion(value)
             WHERE pg_catalog.jsonb_typeof(criterion.value) <> 'object'
                OR COALESCE(criterion.value->>'id', '') = ''
        ) OR (
            SELECT pg_catalog.count(DISTINCT criterion.value->>'id')
              FROM pg_catalog.jsonb_array_elements(
                  plan_value->'success_criteria'
              ) AS criterion(value)
        ) <> pg_catalog.jsonb_array_length(plan_value->'success_criteria') THEN
            RETURN canonical_brain._fail(
                'verification_criteria_invalid',
                'completed plan success_criteria must have unique non-empty ids'
            );
        END IF;
        IF EXISTS (
            SELECT 1
              FROM pg_catalog.jsonb_array_elements(
                  plan_value->'verification_event_ids'
              ) AS required(value)
             WHERE pg_catalog.jsonb_typeof(required.value) <> 'string'
                OR required.value #>> '{}' !~
                   '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
                OR NOT EXISTS (
                    SELECT 1
                      FROM public.canonical_event_log AS verification_event
                      JOIN canonical_brain.writer_event_provenance AS verification_provenance
                        ON verification_provenance.event_id = verification_event.event_id
                     WHERE verification_event.case_id = case_value
                       AND verification_event.event_id = CASE
                           WHEN required.value #>> '{}' ~
                                '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
                               THEN (required.value #>> '{}')::uuid
                           ELSE NULL
                       END
                       AND verification_event.event_type = 'task.verification.recorded'
                       AND verification_event.payload->'verification'->>'plan_id'
                           = plan_id_value
                       AND verification_event.payload->'verification'->>'plan_revision'
                           = previous_revision::text
                       AND verification_event.payload->'verification'->>'outcome'
                           = 'passed'
                )
        ) THEN
            RETURN canonical_brain._fail(
                'verification_receipt_missing',
                'completed plan requires every referenced provenanced receipt to pass for the exact prior revision'
            );
        END IF;
        IF EXISTS (
            SELECT 1
              FROM pg_catalog.jsonb_array_elements(
                  plan_value->'success_criteria'
              ) AS required_criterion(value)
             WHERE NOT EXISTS (
                SELECT 1
                  FROM pg_catalog.jsonb_array_elements_text(
                      plan_value->'verification_event_ids'
                  ) AS required_event(event_id)
                  JOIN public.canonical_event_log AS verification_event
                    ON verification_event.event_id = required_event.event_id::uuid
                  JOIN canonical_brain.writer_event_provenance AS verification_provenance
                    ON verification_provenance.event_id = verification_event.event_id
                  CROSS JOIN LATERAL pg_catalog.jsonb_array_elements_text(
                      CASE
                          WHEN pg_catalog.jsonb_typeof(
                              verification_event.payload->'verification'->'criterion_ids'
                          ) = 'array'
                              THEN verification_event.payload->'verification'->'criterion_ids'
                          ELSE '[]'::jsonb
                      END
                  ) AS covered(criterion_id)
                 WHERE verification_event.case_id = case_value
                   AND verification_event.event_type = 'task.verification.recorded'
                   AND verification_event.payload->'verification'->>'plan_id'
                       = plan_id_value
                   AND verification_event.payload->'verification'->>'plan_revision'
                       = previous_revision::text
                   AND verification_event.payload->'verification'->>'outcome'
                       = 'passed'
                   AND covered.criterion_id = required_criterion.value->>'id'
             )
        ) THEN
            RETURN canonical_brain._fail(
                'verification_criteria_uncovered',
                'completed plan has success_criteria without exact passing verification coverage'
            );
        END IF;
    END IF;

    append_result := canonical_brain._append_event(
        'task.plan.updated',
        case_value,
        request->>'summary',
        request->'source_refs',
        COALESCE(request->'actors', '{}'::jsonb),
        request->'body' || pg_catalog.jsonb_build_object(
            'writer_request_idempotency_key', request->>'idempotency_key'
        ),
        COALESCE(request->'safety', '{}'::jsonb),
        transition_identity,
        'canonical_plan_cas',
        runtime
    );
    IF NOT (append_result->>'ok')::boolean THEN
        RETURN append_result;
    END IF;

    IF previous_plan_id <> '' AND plan_id_value <> previous_plan_id THEN
        revoked_reason := 'plan_superseded';
        INSERT INTO canonical_brain.writer_capability_revocations (
            approval_id, reason, revoked_by_session_sha256, revoked_at
        )
        SELECT grant_row.approval_id,
               revoked_reason,
               COALESCE(NULLIF(runtime->>'session_key_sha256', ''), pg_catalog.repeat('0', 64)),
               pg_catalog.clock_timestamp()
          FROM canonical_brain.writer_capability_grants AS grant_row
         WHERE grant_row.case_id = case_value
           AND grant_row.plan_id = previous_plan_id
        ON CONFLICT (approval_id) DO NOTHING;
    END IF;
    IF previous_plan_id = plan_id_value
       AND previous_revision > 0
       AND revision_value > previous_revision THEN
        INSERT INTO canonical_brain.writer_capability_revocations (
            approval_id, reason, revoked_by_session_sha256, revoked_at
        )
        SELECT grant_row.approval_id,
               'plan_revision_advanced',
               COALESCE(
                   NULLIF(runtime->>'session_key_sha256', ''),
                   pg_catalog.repeat('0', 64)
               ),
               pg_catalog.clock_timestamp()
          FROM canonical_brain.writer_capability_grants AS grant_row
         WHERE grant_row.case_id = case_value
           AND grant_row.plan_id = plan_id_value
           AND grant_row.plan_revision = previous_revision
        ON CONFLICT (approval_id) DO NOTHING;
    END IF;
    IF state_value IN ('completed','blocked','cancelled') THEN
        revoked_reason := 'plan_' || state_value;
        INSERT INTO canonical_brain.writer_capability_revocations (
            approval_id, reason, revoked_by_session_sha256, revoked_at
        )
        SELECT grant_row.approval_id,
               revoked_reason,
               COALESCE(NULLIF(runtime->>'session_key_sha256', ''), pg_catalog.repeat('0', 64)),
               pg_catalog.clock_timestamp()
          FROM canonical_brain.writer_capability_grants AS grant_row
         WHERE grant_row.case_id = case_value
           AND grant_row.plan_id = plan_id_value
        ON CONFLICT (approval_id) DO NOTHING;
    END IF;
    RETURN append_result;
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'canonical plan transition failed');
END
$function$;

-- Fixed public routine 8/18.  Verification is bound to the exact canonical
-- plan head; the model remains the author of the structured outcome.
CREATE OR REPLACE FUNCTION canonical_brain.writer_verification_append(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    case_value text := COALESCE(request->>'case_id', '');
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    verification_value jsonb := request->'body'->'verification';
    head_result jsonb;
    head_plan jsonb;
    verification_event_id uuid;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$' THEN
        RETURN canonical_brain._fail(
            'invalid_runtime',
            'verification append requires exact session and routing epoch'
        );
    END IF;
    IF NOT canonical_brain._keys_valid(
            request,
            ARRAY['event_type','case_id','summary','source_refs','actors','body','safety','idempotency_key'],
            ARRAY['event_type','case_id','summary','source_refs','body','idempotency_key']
       )
       OR request->>'event_type' <> 'task.verification.recorded'
       OR case_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR pg_catalog.jsonb_typeof(verification_value) <> 'object'
       OR COALESCE(verification_value->>'plan_id', '') = ''
       OR COALESCE(verification_value->>'plan_revision', '') !~ '^[1-9][0-9]{0,8}$'
       OR COALESCE(verification_value->>'outcome', '') NOT IN ('passed','failed','inconclusive') THEN
        RETURN canonical_brain._fail('invalid_request', 'verification request is invalid');
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_capability_revocation_scopes AS scope
         WHERE scope.scope_type = 'session'
           AND scope.session_key_sha256 = session_value
           AND scope.capability_epoch_sha256 = epoch_value
    ) THEN
        RETURN canonical_brain._fail(
            'session_epoch_retired',
            'session authority epoch has been durably retired'
        );
    END IF;
    IF NOT canonical_brain._case_scope_authorized(case_value, runtime, false) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'runtime is not authorized to verify the exact canonical case'
        );
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('canonical-plan:' || case_value, 0)
    );
    verification_event_id := canonical_brain._deterministic_uuid(
        'canonical-writer:' || case_value || ':task.verification.recorded:'
            || (request->>'idempotency_key')
    );
    IF EXISTS (
        SELECT 1
          FROM public.canonical_event_log AS event
         WHERE event.event_id = verification_event_id
    ) THEN
        RETURN canonical_brain._append_event(
            'task.verification.recorded',
            case_value,
            request->>'summary',
            request->'source_refs',
            COALESCE(request->'actors', '{}'::jsonb),
            request->'body',
            COALESCE(request->'safety', '{}'::jsonb),
            request->>'idempotency_key',
            'canonical_verification_receipt',
            runtime
        );
    END IF;
    head_result := canonical_brain._plan_head(case_value);
    IF NOT (head_result->>'ok')::boolean THEN
        RETURN head_result;
    END IF;
    head_plan := head_result->'result'->'head'->'plan';
    IF head_plan IS NULL OR head_plan = 'null'::jsonb
       OR verification_value->>'plan_id' <> head_plan->>'plan_id'
       OR (verification_value->>'plan_revision')::integer
          <> (head_plan->>'revision')::integer THEN
        RETURN canonical_brain._fail(
            'plan_cas_conflict',
            'verification does not match the exact canonical plan head'
        );
    END IF;
    IF head_plan->>'state' <> 'active' THEN
        RETURN canonical_brain._fail(
            'plan_not_active',
            'task verification requires the exact canonical plan head to be active'
        );
    END IF;
    IF pg_catalog.jsonb_typeof(verification_value->'criterion_ids') <> 'array'
       OR pg_catalog.jsonb_array_length(verification_value->'criterion_ids') NOT BETWEEN 1 AND 32
       OR pg_catalog.jsonb_typeof(head_plan->'success_criteria') <> 'array'
       OR EXISTS (
            SELECT 1
              FROM pg_catalog.jsonb_array_elements(
                  verification_value->'criterion_ids'
              ) AS supplied(value)
             WHERE pg_catalog.jsonb_typeof(supplied.value) <> 'string'
                OR pg_catalog.length(supplied.value #>> '{}') = 0
       )
       OR (
            SELECT pg_catalog.count(DISTINCT supplied.value #>> '{}')
              FROM pg_catalog.jsonb_array_elements(
                  verification_value->'criterion_ids'
              ) AS supplied(value)
       ) <> pg_catalog.jsonb_array_length(verification_value->'criterion_ids')
       OR EXISTS (
            SELECT 1
              FROM pg_catalog.jsonb_array_elements_text(
                  verification_value->'criterion_ids'
              ) AS supplied(criterion_id)
             WHERE NOT EXISTS (
                SELECT 1
                  FROM pg_catalog.jsonb_array_elements(
                      head_plan->'success_criteria'
                  ) AS criterion(value)
                 WHERE pg_catalog.jsonb_typeof(criterion.value) = 'object'
                   AND criterion.value->>'id' = supplied.criterion_id
             )
       ) THEN
        RETURN canonical_brain._fail(
            'verification_criteria_invalid',
            'verification criterion_ids must be a unique subset of the active plan success_criteria'
        );
    END IF;
    RETURN canonical_brain._append_event(
        'task.verification.recorded',
        case_value,
        request->>'summary',
        request->'source_refs',
        COALESCE(request->'actors', '{}'::jsonb),
        request->'body',
        COALESCE(request->'safety', '{}'::jsonb),
        request->>'idempotency_key',
        'canonical_verification_receipt',
        runtime
    );
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'verification append failed');
END
$function$;

-- Fixed public routine 9/18.  Claiming is an atomic, durable authorization for
-- one exact public target and one exact rendered-content digest.
CREATE OR REPLACE FUNCTION canonical_brain.writer_routeback_claim(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    case_value text := COALESCE(request->>'case_id', '');
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    runtime_platform_value text := COALESCE(runtime->>'platform', '');
    source_thread_value text := COALESCE(
        NULLIF(runtime->>'thread_id', ''),
        runtime->>'chat_id',
        ''
    );
    target_value jsonb := request->'target_ref';
    target_id text;
    authorization_value text;
    lifecycle_value text;
    request_hash text;
    existing_record canonical_brain.writer_routeback_authorizations%ROWTYPE;
    lifecycle_record canonical_brain.writer_routeback_lifecycle_terminals%ROWTYPE;
    terminal_record canonical_brain.writer_routeback_terminals%ROWTYPE;
    intent_result jsonb;
    intent_uuid uuid;
    claimed_at_value timestamptz := pg_catalog.clock_timestamp();
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$' THEN
        RETURN canonical_brain._fail(
            'invalid_runtime',
            'route-back claim requires exact session and routing epoch'
        );
    END IF;
    IF NOT canonical_brain._keys_valid(
            request,
            ARRAY['case_id','target_ref','message_summary','source_refs','content_sha256','idempotency_key'],
            ARRAY['case_id','target_ref','message_summary','source_refs','content_sha256','idempotency_key']
       )
       OR case_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR pg_catalog.jsonb_typeof(target_value) <> 'object'
       OR pg_catalog.jsonb_typeof(request->'source_refs') <> 'object'
       OR canonical_brain._contains_forbidden_dm_ref(target_value)
       OR COALESCE(request->>'content_sha256', '') !~ '^[0-9a-f]{64}$'
       OR pg_catalog.length(COALESCE(request->>'message_summary', '')) NOT BETWEEN 1 AND 4000
       OR pg_catalog.octet_length(COALESCE(request->>'idempotency_key', ''))
          NOT BETWEEN 1 AND 256 THEN
        RETURN canonical_brain._fail('invalid_request', 'route-back claim is invalid');
    END IF;
    IF source_thread_value = '' THEN
        RETURN canonical_brain._fail(
            'invalid_runtime',
            'route-back claim requires an exact observed source thread'
        );
    END IF;
    target_id := COALESCE(target_value->>'thread_id', target_value->>'channel_id', '');
    IF pg_catalog.length(target_id) NOT BETWEEN 1 AND 240 THEN
        RETURN canonical_brain._fail(
            'invalid_request',
            'route-back requires an exact public channel or thread'
        );
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_capability_revocation_scopes AS scope
         WHERE scope.scope_type = 'session'
           AND scope.session_key_sha256 = session_value
           AND scope.capability_epoch_sha256 = epoch_value
    ) THEN
        RETURN canonical_brain._fail(
            'session_epoch_retired',
            'session authority epoch has been durably retired'
        );
    END IF;
    IF NOT canonical_brain._case_scope_authorized(case_value, runtime, false) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'runtime session is not linked to the canonical case'
        );
    END IF;
    authorization_value := 'routeauth:' || pg_catalog.substr(
        canonical_brain._sha256_text(
            '{"case_id":' || pg_catalog.to_json(case_value)::text
            || ',"idempotency_key":'
            || pg_catalog.to_json(request->>'idempotency_key')::text || '}'
        ),
        1,
        40
    );
    lifecycle_value := 'routeblock:' || pg_catalog.substr(
        canonical_brain._sha256_text(
            '{"case_id":' || pg_catalog.to_json(case_value)::text
            || ',"idempotency_key":'
            || pg_catalog.to_json(request->>'idempotency_key')::text || '}'
        ),
        1,
        40
    );
    request_hash := canonical_brain._sha256_json(pg_catalog.jsonb_build_object(
        'authorization_id', authorization_value,
        'case_id', case_value,
        'target_ref', target_value,
        'message_summary', request->>'message_summary',
        'source_refs', request->'source_refs',
        'content_sha256', request->>'content_sha256'
    ));
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('routeback-lifecycle:' || lifecycle_value, 0)
    );

    SELECT * INTO lifecycle_record
      FROM canonical_brain.writer_routeback_lifecycle_terminals AS lifecycle
     WHERE lifecycle.lifecycle_id = lifecycle_value;
    IF FOUND THEN
        IF lifecycle_record.case_id <> case_value
           OR lifecycle_record.target_ref IS DISTINCT FROM target_value
           OR lifecycle_record.message_summary <> request->>'message_summary'
           OR lifecycle_record.source_refs IS DISTINCT FROM request->'source_refs'
           OR lifecycle_record.idempotency_key <> request->>'idempotency_key' THEN
            RETURN canonical_brain._fail(
                'idempotency_conflict',
                'route-back lifecycle identity is bound to different content'
            );
        END IF;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'preclaim', true,
            'preclaim_block_id', lifecycle_record.lifecycle_id,
            'case_id', lifecycle_record.case_id,
            'target_ref', lifecycle_record.target_ref,
            'state', lifecycle_record.outcome,
            'terminal_event_type', 'route_back.' || lifecycle_record.outcome,
            'terminal_payload', pg_catalog.jsonb_build_object(
                'outcome', lifecycle_record.outcome,
                'receipt', lifecycle_record.receipt,
                'blocker_reason', lifecycle_record.blocker_reason
            ),
            'inserted', false,
            'deduped', true
        ));
    END IF;

    SELECT * INTO existing_record
      FROM canonical_brain.writer_routeback_authorizations AS authorization_row
     WHERE authorization_row.authorization_id = authorization_value;
    IF FOUND THEN
        IF existing_record.request_sha256 <> request_hash THEN
            RETURN canonical_brain._fail(
                'idempotency_conflict',
                'route-back authorization identity is bound to different content'
            );
        END IF;
        SELECT * INTO terminal_record
          FROM canonical_brain.writer_routeback_terminals AS terminal
         WHERE terminal.authorization_id = authorization_value;
        IF terminal_record.authorization_id IS NULL
           AND (
               existing_record.session_key_sha256 IS DISTINCT FROM session_value
               OR existing_record.capability_epoch_sha256 IS DISTINCT FROM epoch_value
               OR existing_record.runtime_platform IS DISTINCT FROM runtime_platform_value
               OR existing_record.source_thread_id IS DISTINCT FROM source_thread_value
           ) THEN
            RETURN canonical_brain._fail(
                'scope_mismatch',
                'pending route-back authorization belongs to another exact runtime scope'
            );
        END IF;
        IF terminal_record.authorization_id IS NULL
           AND NOT EXISTS (
               SELECT 1
                 FROM canonical_brain.writer_public_routeback_targets AS allowed
                WHERE allowed.channel_id = target_id
                  AND allowed.enabled
                  AND allowed.target_type IN ('public_channel', 'public_thread')
           ) THEN
            RETURN canonical_brain._fail(
                'target_not_approved',
                'route-back target is not in the owner-provisioned public target ACL'
            );
        END IF;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'authorization_id', authorization_value,
            'case_id', existing_record.case_id,
            'target_ref', existing_record.target_ref,
            'content_sha256', existing_record.content_sha256,
            'claimed_at', existing_record.created_at,
            'state', CASE WHEN terminal_record.authorization_id IS NULL
                          THEN 'authorized' ELSE terminal_record.outcome END,
            'terminal_event_type', CASE WHEN terminal_record.authorization_id IS NULL
                                        THEN '' ELSE 'route_back.' || terminal_record.outcome END,
            'terminal_payload', CASE WHEN terminal_record.authorization_id IS NULL
                                     THEN '{}'::jsonb ELSE pg_catalog.jsonb_build_object(
                                         'outcome', terminal_record.outcome,
                                         'receipt', terminal_record.receipt,
                                         'blocker_reason', terminal_record.blocker_reason
                                     ) END,
            'inserted', false,
            'deduped', true
        ));
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM canonical_brain.writer_public_routeback_targets AS allowed
         WHERE allowed.channel_id = target_id
           AND allowed.enabled
           AND allowed.target_type IN ('public_channel', 'public_thread')
    ) THEN
        RETURN canonical_brain._fail(
            'target_not_approved',
            'route-back target is not in the owner-provisioned public target ACL'
        );
    END IF;

    intent_result := canonical_brain._append_event(
        'route_back.intent.created',
        case_value,
        request->>'message_summary',
        request->'source_refs',
        pg_catalog.jsonb_build_object(
            'subject', pg_catalog.jsonb_build_object(
                'type', 'route_back', 'id', target_id
            )
        ),
        pg_catalog.jsonb_build_object(
            'authorization_id', authorization_value,
            'route_back', pg_catalog.jsonb_build_object(
                'target_ref', target_value,
                'message_summary', request->>'message_summary',
                'execution_binding', pg_catalog.jsonb_build_object(
                    'target_channel_id', target_id,
                    'content_sha256', request->>'content_sha256'
                )
            )
        ),
        '{}'::jsonb,
        authorization_value,
        'routeback_claim',
        runtime
    );
    IF NOT (intent_result->>'ok')::boolean THEN
        RETURN intent_result;
    END IF;
    intent_uuid := (intent_result->'result'->>'event_id')::uuid;
    INSERT INTO canonical_brain.writer_routeback_authorizations (
        authorization_id, case_id, target_ref, message_summary, source_refs,
        content_sha256, session_key_sha256, capability_epoch_sha256,
        runtime_platform,
        source_thread_id, idempotency_key, request_sha256, created_at,
        intent_event_id
    ) VALUES (
        authorization_value, case_value, target_value,
        request->>'message_summary', request->'source_refs',
        request->>'content_sha256', session_value, epoch_value,
        runtime_platform_value,
        source_thread_value,
        request->>'idempotency_key',
        request_hash, claimed_at_value, intent_uuid
    );
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'authorization_id', authorization_value,
        'case_id', case_value,
        'target_ref', target_value,
        'content_sha256', request->>'content_sha256',
        'state', 'authorized',
        'claimed_at', claimed_at_value,
        'event_id', intent_uuid::text,
        'inserted', true,
        'deduped', false
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'route-back claim failed');
END
$function$;

-- Fixed public routine 10/18.  Epoch-only restart recovery is a separate typed
-- operation.  It never creates a lifecycle and can cross only the capability
-- epoch of the exact same session/platform/source lane.  Signed edge evidence
-- may recover terminal truth after an ACL change; only authenticated no-record
-- recovery can mint fresh dispatch authority and therefore rechecks the ACL.
CREATE OR REPLACE FUNCTION canonical_brain.writer_routeback_recover(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    case_value text := COALESCE(request->>'case_id', '');
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    platform_value text := COALESCE(runtime->>'platform', '');
    source_thread_value text := COALESCE(
        NULLIF(runtime->>'thread_id', ''),
        runtime->>'chat_id',
        ''
    );
    target_value jsonb := request->'target_ref';
    target_id text;
    recovery_value text := COALESCE(request->>'recovery_kind', '');
    authorization_value text;
    request_hash text;
    authorization_record canonical_brain.writer_routeback_authorizations%ROWTYPE;
    terminal_record canonical_brain.writer_routeback_terminals%ROWTYPE;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$'
       OR source_thread_value = '' THEN
        RETURN canonical_brain._fail(
            'invalid_runtime',
            'route-back recovery requires exact current runtime scope'
        );
    END IF;
    IF NOT canonical_brain._keys_valid(
            request,
            ARRAY['case_id','target_ref','message_summary','source_refs',
                  'content_sha256','idempotency_key','recovery_kind'],
            ARRAY['case_id','target_ref','message_summary','source_refs',
                  'content_sha256','idempotency_key','recovery_kind']
       )
       OR recovery_value NOT IN ('edge_evidence', 'edge_no_record')
       OR case_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR pg_catalog.jsonb_typeof(target_value) <> 'object'
       OR canonical_brain._contains_forbidden_dm_ref(target_value)
       OR COALESCE(request->>'content_sha256', '') !~ '^[0-9a-f]{64}$'
       OR pg_catalog.length(COALESCE(request->>'message_summary', ''))
          NOT BETWEEN 1 AND 4000
       OR pg_catalog.octet_length(COALESCE(request->>'idempotency_key', ''))
          NOT BETWEEN 1 AND 256 THEN
        RETURN canonical_brain._fail(
            'invalid_request',
            'route-back recovery request is invalid'
        );
    END IF;
    target_id := COALESCE(
        target_value->>'thread_id',
        target_value->>'channel_id',
        ''
    );
    IF pg_catalog.length(target_id) NOT BETWEEN 1 AND 240 THEN
        RETURN canonical_brain._fail(
            'invalid_request',
            'route-back recovery requires an exact public target'
        );
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_capability_revocation_scopes AS scope
         WHERE scope.scope_type = 'session'
           AND scope.session_key_sha256 = session_value
           AND scope.capability_epoch_sha256 = epoch_value
    ) THEN
        RETURN canonical_brain._fail(
            'session_epoch_retired',
            'current recovery epoch has been durably retired'
        );
    END IF;
    IF NOT canonical_brain._case_scope_authorized(case_value, runtime, false) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'current runtime is not linked to the canonical case'
        );
    END IF;
    authorization_value := 'routeauth:' || pg_catalog.substr(
        canonical_brain._sha256_text(
            '{"case_id":' || pg_catalog.to_json(case_value)::text
            || ',"idempotency_key":'
            || pg_catalog.to_json(request->>'idempotency_key')::text || '}'
        ),
        1,
        40
    );
    request_hash := canonical_brain._sha256_json(pg_catalog.jsonb_build_object(
        'authorization_id', authorization_value,
        'case_id', case_value,
        'target_ref', target_value,
        'message_summary', request->>'message_summary',
        'source_refs', request->'source_refs',
        'content_sha256', request->>'content_sha256'
    ));
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'routeback-authorization:' || authorization_value, 0
        )
    );
    SELECT * INTO authorization_record
      FROM canonical_brain.writer_routeback_authorizations AS authorization_row
     WHERE authorization_row.authorization_id = authorization_value;
    IF NOT FOUND THEN
        RETURN canonical_brain._fail(
            'authorization_missing',
            'route-back recovery requires an existing claim'
        );
    END IF;
    IF authorization_record.request_sha256 <> request_hash THEN
        RETURN canonical_brain._fail(
            'idempotency_conflict',
            'route-back recovery identity is bound to different content'
        );
    END IF;
    IF authorization_record.session_key_sha256 <> session_value
       OR authorization_record.runtime_platform <> platform_value
       OR authorization_record.source_thread_id <> source_thread_value THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'route-back recovery cannot cross session or source lane'
        );
    END IF;
    SELECT * INTO terminal_record
      FROM canonical_brain.writer_routeback_terminals AS terminal
     WHERE terminal.authorization_id = authorization_value;
    IF FOUND THEN
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'authorization_id', authorization_value,
            'state', terminal_record.outcome,
            'terminal_event_type', 'route_back.' || terminal_record.outcome,
            'terminal_payload', pg_catalog.jsonb_build_object(
                'outcome', terminal_record.outcome,
                'receipt', terminal_record.receipt,
                'blocker_reason', terminal_record.blocker_reason
            ),
            'inserted', false,
            'deduped', true
        ));
    END IF;
    IF recovery_value = 'edge_no_record'
       AND NOT EXISTS (
           SELECT 1
             FROM canonical_brain.writer_public_routeback_targets AS allowed
            WHERE allowed.channel_id = target_id
              AND allowed.enabled
              AND allowed.target_type IN ('public_channel', 'public_thread')
       ) THEN
        RETURN canonical_brain._fail(
            'target_not_approved',
            'no-record recovery target is not currently approved'
        );
    END IF;
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'authorization_id', authorization_value,
        'case_id', authorization_record.case_id,
        'target_ref', authorization_record.target_ref,
        'content_sha256', authorization_record.content_sha256,
        'state', 'authorized',
        'recovery_kind', recovery_value,
        'recovered_epoch_sha256', epoch_value,
        'recovered', true,
        'inserted', false,
        'deduped', true
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'route-back recovery failed');
END
$function$;

-- Fixed public routine 11/18.
CREATE OR REPLACE FUNCTION canonical_brain.writer_routeback_finalize_sent(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    authorization_record canonical_brain.writer_routeback_authorizations%ROWTYPE;
    lifecycle_record canonical_brain.writer_routeback_lifecycle_terminals%ROWTYPE;
    terminal_record canonical_brain.writer_routeback_terminals%ROWTYPE;
    receipt_value jsonb := request->'receipt';
    request_hash text;
    target_id text;
    append_result jsonb;
    event_uuid uuid;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR COALESCE(runtime->>'session_key_sha256', '') !~ '^[0-9a-f]{64}$'
       OR COALESCE(runtime->>'capability_epoch_sha256', '')
          !~ '^[0-9a-f]{64}$'
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['authorization_id','outcome','receipt','blocker_reason'],
            ARRAY['authorization_id','outcome','receipt','blocker_reason']
       )
       OR request->>'outcome' <> 'sent'
       OR pg_catalog.jsonb_typeof(receipt_value) <> 'object'
       OR NOT canonical_brain._keys_valid(
            receipt_value,
            ARRAY['platform','adapter_receipt','receipt_readback_verified',
                  'message_id','channel_id','chat_id','channel_type',
                  'target_kind','content_sha256'],
            ARRAY['platform','adapter_receipt','receipt_readback_verified',
                  'message_id','channel_id','content_sha256']
       )
       OR canonical_brain._contains_forbidden_dm_ref(receipt_value)
       OR receipt_value->>'platform' <> 'discord'
       OR pg_catalog.jsonb_typeof(receipt_value->'adapter_receipt') <> 'boolean'
       OR receipt_value->'adapter_receipt' <> 'true'::jsonb
       OR pg_catalog.jsonb_typeof(receipt_value->'receipt_readback_verified') <> 'boolean'
       OR receipt_value->'receipt_readback_verified' <> 'true'::jsonb
       OR COALESCE(receipt_value->>'message_id', '')
          !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR COALESCE(receipt_value->>'channel_id', '')
          !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR COALESCE(receipt_value->>'content_sha256', '') !~ '^[0-9a-f]{64}$'
       OR (
            receipt_value ? 'chat_id'
            AND receipt_value->>'chat_id' <> receipt_value->>'channel_id'
       )
       OR (
            receipt_value ? 'channel_type'
            AND receipt_value->>'channel_type'
                !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       )
       OR (
            receipt_value ? 'target_kind'
            AND receipt_value->>'target_kind'
                !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       )
       OR COALESCE(request->>'blocker_reason', '') <> '' THEN
        RETURN canonical_brain._fail('invalid_receipt', 'sent receipt is invalid');
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'routeback-authorization:' || COALESCE(request->>'authorization_id', ''), 0
        )
    );
    SELECT * INTO authorization_record
      FROM canonical_brain.writer_routeback_authorizations AS authorization_row
     WHERE authorization_row.authorization_id = request->>'authorization_id';
    IF NOT FOUND THEN
        RETURN canonical_brain._fail('authorization_missing', 'route-back authorization not found');
    END IF;
    IF authorization_record.session_key_sha256 <> runtime->>'session_key_sha256'
       OR authorization_record.runtime_platform <> runtime->>'platform'
       OR authorization_record.source_thread_id <> COALESCE(
            NULLIF(runtime->>'thread_id', ''), runtime->>'chat_id', ''
       )
       OR NOT canonical_brain._case_scope_authorized(
            authorization_record.case_id, runtime, false
       ) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'route-back finalization cannot cross session or source lane'
        );
    END IF;
    target_id := COALESCE(
        authorization_record.target_ref->>'thread_id',
        authorization_record.target_ref->>'channel_id',
        ''
    );
    IF receipt_value->>'channel_id' <> target_id
       OR receipt_value->>'content_sha256' <> authorization_record.content_sha256 THEN
        RETURN canonical_brain._fail(
            'invalid_receipt',
            'receipt does not match the claimed target and content digest'
        );
    END IF;
    request_hash := canonical_brain._sha256_json(pg_catalog.jsonb_build_object(
        'authorization_id', request->>'authorization_id',
        'outcome', 'sent',
        'receipt', receipt_value,
        'session', runtime->>'session_key_sha256',
        'capability_epoch_sha256', runtime->>'capability_epoch_sha256'
    ));
    SELECT * INTO terminal_record
      FROM canonical_brain.writer_routeback_terminals AS terminal
     WHERE terminal.authorization_id = request->>'authorization_id';
    IF FOUND THEN
        IF terminal_record.outcome <> 'sent'
           OR terminal_record.request_sha256 <> request_hash THEN
            RETURN canonical_brain._fail(
                'terminal_conflict',
                'route-back authorization is already finalized differently'
            );
        END IF;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'authorization_id', request->>'authorization_id',
            'outcome', 'sent',
            'receipt', receipt_value,
            'blocker_reason', '',
            'event_id', terminal_record.terminal_event_id::text,
            'deduped', true
        ));
    END IF;

    append_result := canonical_brain._append_event(
        'route_back.sent',
        authorization_record.case_id,
        authorization_record.message_summary,
        authorization_record.source_refs,
        pg_catalog.jsonb_build_object(
            'subject', pg_catalog.jsonb_build_object(
                'type', 'route_back', 'id', target_id
            )
        ),
        pg_catalog.jsonb_build_object(
            'authorization_id', authorization_record.authorization_id,
            'receipt', receipt_value,
            'route_back', pg_catalog.jsonb_build_object(
                'target_ref', authorization_record.target_ref,
                'receipt', receipt_value,
                'execution_binding', pg_catalog.jsonb_build_object(
                    'target_channel_id', target_id,
                    'content_sha256', authorization_record.content_sha256
                )
            )
        ),
        pg_catalog.jsonb_build_object('outbound', true),
        authorization_record.authorization_id,
        'routeback_finalize_sent',
        runtime
    );
    IF NOT (append_result->>'ok')::boolean THEN
        RETURN append_result;
    END IF;
    event_uuid := (append_result->'result'->>'event_id')::uuid;
    INSERT INTO canonical_brain.writer_routeback_terminals (
        authorization_id, outcome, receipt, blocker_reason,
        request_sha256, finalized_at, terminal_event_id
    ) VALUES (
        authorization_record.authorization_id, 'sent', receipt_value, '',
        request_hash, pg_catalog.clock_timestamp(), event_uuid
    );
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'authorization_id', authorization_record.authorization_id,
        'outcome', 'sent',
        'receipt', receipt_value,
        'blocker_reason', '',
        'event_id', event_uuid::text,
        'inserted', true,
        'deduped', false
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'route-back sent finalization failed');
END
$function$;

-- Fixed public routine 12/18.
CREATE OR REPLACE FUNCTION canonical_brain.writer_routeback_finalize_blocked(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    authorization_record canonical_brain.writer_routeback_authorizations%ROWTYPE;
    lifecycle_record canonical_brain.writer_routeback_lifecycle_terminals%ROWTYPE;
    terminal_record canonical_brain.writer_routeback_terminals%ROWTYPE;
    blocker_value text := COALESCE(request->>'blocker_reason', '');
    receipt_value jsonb := request->'receipt';
    target_id text;
    request_hash text;
    append_result jsonb;
    event_uuid uuid;
    preclaim_value boolean := request->'preclaim' = 'true'::jsonb;
    preclaim_id text;
    authorization_value text;
    preclaim_request_hash text;
    case_value text := COALESCE(request->>'case_id', '');
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    target_value jsonb := request->'target_ref';
BEGIN
    IF preclaim_value THEN
        IF NOT canonical_brain._runtime_valid(runtime)
           OR session_value !~ '^[0-9a-f]{64}$'
           OR epoch_value !~ '^[0-9a-f]{64}$' THEN
            RETURN canonical_brain._fail(
                'invalid_runtime',
                'preclaim blocked requires exact session and routing epoch'
            );
        END IF;
        IF NOT canonical_brain._keys_valid(
                request,
                ARRAY['preclaim','case_id','target_ref','message_summary',
                      'source_refs','idempotency_key','outcome','receipt',
                      'blocker_reason'],
                ARRAY['preclaim','case_id','target_ref','message_summary',
                      'source_refs','idempotency_key','outcome','receipt',
                      'blocker_reason']
           )
           OR request->'preclaim' <> 'true'::jsonb
           OR request->>'outcome' <> 'blocked'
           OR request->'receipt' <> '{}'::jsonb
           OR case_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
           OR pg_catalog.jsonb_typeof(target_value) <> 'object'
           OR pg_catalog.jsonb_typeof(request->'source_refs') <> 'object'
           OR canonical_brain._contains_forbidden_dm_ref(request)
           OR pg_catalog.length(COALESCE(request->>'message_summary', ''))
              NOT BETWEEN 1 AND 4000
           OR pg_catalog.length(blocker_value) NOT BETWEEN 1 AND 1000
           OR pg_catalog.octet_length(COALESCE(request->>'idempotency_key', ''))
              NOT BETWEEN 1 AND 256 THEN
            RETURN canonical_brain._fail(
                'invalid_request',
                'preclaim blocked finalization is invalid'
            );
        END IF;
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(
                'capability-scope:' || session_value || ':' || epoch_value, 0
            )
        );
        IF EXISTS (
            SELECT 1
              FROM canonical_brain.writer_capability_revocation_scopes AS scope
             WHERE scope.scope_type = 'session'
               AND scope.session_key_sha256 = session_value
               AND scope.capability_epoch_sha256 = epoch_value
        ) THEN
            RETURN canonical_brain._fail(
                'session_epoch_retired',
                'session authority epoch has been durably retired'
            );
        END IF;
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended('canonical-case:' || case_value, 0)
        );
        IF NOT canonical_brain._case_scope_authorized(case_value, runtime, true) THEN
            RETURN canonical_brain._fail(
                'scope_mismatch',
                'runtime is not authorized to block the exact canonical case'
            );
        END IF;
        authorization_value := 'routeauth:' || pg_catalog.substr(
            canonical_brain._sha256_text(
                '{"case_id":' || pg_catalog.to_json(case_value)::text
                || ',"idempotency_key":'
                || pg_catalog.to_json(request->>'idempotency_key')::text || '}'
            ),
            1,
            40
        );
        preclaim_id := 'routeblock:' || pg_catalog.substr(
            canonical_brain._sha256_text(
                '{"case_id":' || pg_catalog.to_json(case_value)::text
                || ',"idempotency_key":'
                || pg_catalog.to_json(request->>'idempotency_key')::text || '}'
            ),
            1,
            40
        );
        preclaim_request_hash := canonical_brain._sha256_json(
            pg_catalog.jsonb_build_object(
                'lifecycle_id', preclaim_id,
                'case_id', case_value,
                'target_ref', target_value,
                'message_summary', request->>'message_summary',
                'source_refs', request->'source_refs',
                'outcome', 'blocked',
                'blocker_reason', blocker_value,
                'partial_receipt', '{}'::jsonb
            )
        );
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(
                'routeback-lifecycle:' || preclaim_id, 0
            )
        );
        SELECT * INTO lifecycle_record
          FROM canonical_brain.writer_routeback_lifecycle_terminals AS lifecycle
         WHERE lifecycle.lifecycle_id = preclaim_id;
        IF FOUND THEN
            IF lifecycle_record.request_sha256 <> preclaim_request_hash THEN
                RETURN canonical_brain._fail(
                    'idempotency_conflict',
                    'route-back lifecycle identity is bound to different content'
                );
            END IF;
            RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
                'success', true,
                'preclaim', true,
                'preclaim_block_id', preclaim_id,
                'outcome', lifecycle_record.outcome,
                'receipt', lifecycle_record.receipt,
                'partial_receipt', lifecycle_record.receipt,
                'blocker_reason', lifecycle_record.blocker_reason,
                'event_id', lifecycle_record.terminal_event_id::text,
                'inserted', false,
                'deduped', true
            ));
        END IF;
        SELECT * INTO authorization_record
          FROM canonical_brain.writer_routeback_authorizations AS authorization_row
         WHERE authorization_row.authorization_id = authorization_value;
        IF FOUND THEN
            IF authorization_record.case_id <> case_value
               OR authorization_record.target_ref IS DISTINCT FROM target_value
               OR authorization_record.message_summary
                  <> request->>'message_summary'
               OR authorization_record.source_refs
                  IS DISTINCT FROM request->'source_refs'
               OR authorization_record.idempotency_key
                  <> request->>'idempotency_key' THEN
                RETURN canonical_brain._fail(
                    'idempotency_conflict',
                    'route-back lifecycle identity is bound to different content'
                );
            END IF;
            SELECT * INTO terminal_record
              FROM canonical_brain.writer_routeback_terminals AS terminal
             WHERE terminal.authorization_id = authorization_value;
            IF NOT FOUND THEN
                RETURN canonical_brain._fail(
                    'routeback_outcome_uncertain',
                    'route-back claim is pending reconciliation'
                );
            END IF;
            IF terminal_record.outcome <> 'blocked' THEN
                RETURN canonical_brain._fail(
                    'terminal_conflict',
                    'route-back lifecycle is already finalized differently'
                );
            END IF;
            RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
                'success', true,
                'preclaim', false,
                'preclaim_block_id', preclaim_id,
                'outcome', 'blocked',
                'receipt', terminal_record.receipt,
                'partial_receipt', terminal_record.receipt,
                'blocker_reason', terminal_record.blocker_reason,
                'event_id', terminal_record.terminal_event_id::text,
                'inserted', false,
                'deduped', true
            ));
        END IF;
        append_result := canonical_brain._append_event(
            'route_back.blocked',
            case_value,
            request->>'message_summary',
            request->'source_refs',
            pg_catalog.jsonb_build_object(
                'subject', pg_catalog.jsonb_build_object(
                    'type', 'route_back_preclaim', 'id', preclaim_id
                )
            ),
            pg_catalog.jsonb_build_object(
                'preclaim', true,
                'preclaim_block_id', preclaim_id,
                'target_ref', target_value,
                'blocker_reason', blocker_value,
                'partial_receipt', '{}'::jsonb,
                'route_back', pg_catalog.jsonb_build_object(
                    'preclaim', true,
                    'target_ref', target_value,
                    'delivery_state', 'not_attempted',
                    'blocker_reason', blocker_value,
                    'partial_receipt', '{}'::jsonb
                )
            ),
            pg_catalog.jsonb_build_object(
                'outbound', false,
                'outbound_delivery_uncertain', false,
                'adapter_acceptance_observed', false
            ),
            preclaim_id,
            'routeback_preclaim_blocked',
            runtime
        );
        IF NOT (append_result->>'ok')::boolean THEN
            RETURN append_result;
        END IF;
        event_uuid := (append_result->'result'->>'event_id')::uuid;
        INSERT INTO canonical_brain.writer_routeback_lifecycle_terminals (
            lifecycle_id, case_id, idempotency_key, target_ref,
            message_summary, source_refs, outcome, receipt, blocker_reason,
            request_sha256, session_key_sha256, capability_epoch_sha256,
            finalized_at, terminal_event_id
        ) VALUES (
            preclaim_id, case_value, request->>'idempotency_key', target_value,
            request->>'message_summary', request->'source_refs', 'blocked',
            '{}'::jsonb, blocker_value, preclaim_request_hash,
            runtime->>'session_key_sha256',
            runtime->>'capability_epoch_sha256',
            pg_catalog.clock_timestamp(), event_uuid
        );
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'preclaim', true,
            'preclaim_block_id', preclaim_id,
            'outcome', 'blocked',
            'receipt', '{}'::jsonb,
            'partial_receipt', '{}'::jsonb,
            'blocker_reason', blocker_value,
            'event_id', append_result->'result'->>'event_id',
            'inserted', append_result->'result'->'inserted',
            'deduped', append_result->'result'->'deduped'
        ));
    END IF;
    IF NOT canonical_brain._runtime_valid(runtime)
       OR COALESCE(runtime->>'session_key_sha256', '') !~ '^[0-9a-f]{64}$'
       OR COALESCE(runtime->>'capability_epoch_sha256', '')
          !~ '^[0-9a-f]{64}$'
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['authorization_id','outcome','receipt','blocker_reason'],
            ARRAY['authorization_id','outcome','receipt','blocker_reason']
       )
       OR request->>'outcome' <> 'blocked'
       OR pg_catalog.jsonb_typeof(receipt_value) <> 'object'
       OR (
            receipt_value <> '{}'::jsonb
            AND (
                NOT canonical_brain._keys_valid(
                    receipt_value,
                    ARRAY['platform','adapter_receipt','receipt_readback_verified',
                          'message_id','channel_id','content_sha256'],
                    ARRAY['platform','adapter_receipt','receipt_readback_verified',
                          'message_id','channel_id','content_sha256']
                )
                OR receipt_value->>'platform' <> 'discord'
                OR receipt_value->'adapter_receipt' IS DISTINCT FROM 'true'::jsonb
                OR receipt_value->'receipt_readback_verified'
                   IS DISTINCT FROM 'true'::jsonb
                OR blocker_value
                   <> 'route_back_sent_receipt_persistence_failed'
                OR COALESCE(receipt_value->>'message_id', '')
                   !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
                OR COALESCE(receipt_value->>'channel_id', '')
                   !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
                OR COALESCE(receipt_value->>'content_sha256', '')
                   !~ '^[0-9a-f]{64}$'
            )
       )
       OR pg_catalog.length(blocker_value) NOT BETWEEN 1 AND 1000
       OR canonical_brain._contains_forbidden_dm_ref(request) THEN
        RETURN canonical_brain._fail('invalid_request', 'blocked finalization is invalid');
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'routeback-authorization:' || COALESCE(request->>'authorization_id', ''), 0
        )
    );
    SELECT * INTO authorization_record
      FROM canonical_brain.writer_routeback_authorizations AS authorization_row
     WHERE authorization_row.authorization_id = request->>'authorization_id';
    IF NOT FOUND THEN
        RETURN canonical_brain._fail('authorization_missing', 'route-back authorization not found');
    END IF;
    IF authorization_record.session_key_sha256 <> runtime->>'session_key_sha256'
       OR authorization_record.runtime_platform <> runtime->>'platform'
       OR authorization_record.source_thread_id <> COALESCE(
            NULLIF(runtime->>'thread_id', ''), runtime->>'chat_id', ''
       )
       OR NOT canonical_brain._case_scope_authorized(
            authorization_record.case_id, runtime, false
       ) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'route-back finalization cannot cross session or source lane'
        );
    END IF;
    target_id := COALESCE(
        authorization_record.target_ref->>'thread_id',
        authorization_record.target_ref->>'channel_id',
        ''
    );
    IF receipt_value <> '{}'::jsonb
       AND (
            receipt_value->>'channel_id' <> target_id
            OR receipt_value->>'content_sha256'
               <> authorization_record.content_sha256
       ) THEN
        RETURN canonical_brain._fail(
            'invalid_receipt',
            'blocked partial receipt does not match the exact accepted send authorization'
        );
    END IF;
    request_hash := canonical_brain._sha256_json(pg_catalog.jsonb_build_object(
        'authorization_id', request->>'authorization_id',
        'outcome', 'blocked',
        'blocker_reason', blocker_value,
        'partial_receipt', receipt_value,
        'session', runtime->>'session_key_sha256',
        'capability_epoch_sha256', runtime->>'capability_epoch_sha256'
    ));
    SELECT * INTO terminal_record
      FROM canonical_brain.writer_routeback_terminals AS terminal
     WHERE terminal.authorization_id = request->>'authorization_id';
    IF FOUND THEN
        IF terminal_record.outcome <> 'blocked'
           OR terminal_record.request_sha256 <> request_hash THEN
            RETURN canonical_brain._fail(
                'terminal_conflict',
                'route-back authorization is already finalized differently'
            );
        END IF;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'authorization_id', request->>'authorization_id',
            'outcome', 'blocked',
            'receipt', terminal_record.receipt,
            'partial_receipt', terminal_record.receipt,
            'blocker_reason', blocker_value,
            'event_id', terminal_record.terminal_event_id::text,
            'deduped', true
        ));
    END IF;

    append_result := canonical_brain._append_event(
        'route_back.blocked',
        authorization_record.case_id,
        authorization_record.message_summary,
        authorization_record.source_refs,
        pg_catalog.jsonb_build_object(
            'subject', pg_catalog.jsonb_build_object(
                'type', 'route_back', 'id', authorization_record.authorization_id
            )
        ),
        pg_catalog.jsonb_build_object(
            'authorization_id', authorization_record.authorization_id,
            'blocker_reason', blocker_value,
            'partial_receipt', receipt_value,
            'route_back', pg_catalog.jsonb_build_object(
                'target_ref', authorization_record.target_ref,
                'receipt', receipt_value,
                'partial_receipt', receipt_value,
                'delivery_state', CASE
                    WHEN receipt_value = '{}'::jsonb THEN 'not_verified'
                    ELSE 'verified_but_sent_terminal_persistence_failed'
                END,
                'blocker_reason', blocker_value,
                'execution_binding', pg_catalog.jsonb_build_object(
                    'target_channel_id', COALESCE(
                        authorization_record.target_ref->>'thread_id',
                        authorization_record.target_ref->>'channel_id'
                    ),
                    'content_sha256', authorization_record.content_sha256
                )
            )
        ),
        pg_catalog.jsonb_build_object(
            'outbound_delivery_uncertain',
                receipt_value = '{}'::jsonb,
            'adapter_acceptance_observed', receipt_value <> '{}'::jsonb,
            'outbound_delivery_verified_but_terminal_blocked',
                receipt_value <> '{}'::jsonb
                AND receipt_value->'receipt_readback_verified' = 'true'::jsonb
        ),
        authorization_record.authorization_id,
        'routeback_finalize_blocked',
        runtime
    );
    IF NOT (append_result->>'ok')::boolean THEN
        RETURN append_result;
    END IF;
    event_uuid := (append_result->'result'->>'event_id')::uuid;
    INSERT INTO canonical_brain.writer_routeback_terminals (
        authorization_id, outcome, receipt, blocker_reason,
        request_sha256, finalized_at, terminal_event_id
    ) VALUES (
        authorization_record.authorization_id, 'blocked', receipt_value,
        blocker_value, request_hash, pg_catalog.clock_timestamp(), event_uuid
    );
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'authorization_id', authorization_record.authorization_id,
        'outcome', 'blocked',
        'receipt', receipt_value,
        'partial_receipt', receipt_value,
        'blocker_reason', blocker_value,
        'event_id', event_uuid::text,
        'inserted', true,
        'deduped', false
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'route-back blocked finalization failed');
END
$function$;

-- Fixed public routine 13/18.
CREATE OR REPLACE FUNCTION canonical_brain.writer_lease_shadow_record(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    case_value text := COALESCE(request->>'case_id', '');
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['intent_event_id','intent_kind','case','case_id',
                  'runtime_lease_enforcement','enforcement_enabled',
                  'send_path_blocking_enabled','audit_runtime_id',
                  'source_platform','session_key_ref'],
            ARRAY['intent_event_id','intent_kind','case','case_id',
                  'runtime_lease_enforcement','enforcement_enabled',
                  'send_path_blocking_enabled','audit_runtime_id',
                  'source_platform','session_key_ref']
       )
       OR case_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR request->'case'->>'case_id' IS DISTINCT FROM case_value
       OR pg_catalog.jsonb_typeof(request->'case') <> 'object'
       OR pg_catalog.jsonb_typeof(request->'runtime_lease_enforcement') <> 'object'
       OR pg_catalog.jsonb_typeof(request->'enforcement_enabled') <> 'boolean'
       OR pg_catalog.jsonb_typeof(request->'send_path_blocking_enabled') <> 'boolean'
       OR pg_catalog.length(COALESCE(request->>'intent_event_id', '')) NOT BETWEEN 1 AND 240
       OR pg_catalog.length(COALESCE(request->>'intent_kind', '')) NOT BETWEEN 1 AND 240 THEN
        RETURN canonical_brain._fail('invalid_request', 'lease shadow receipt is invalid');
    END IF;
    IF NOT canonical_brain._case_scope_authorized(case_value, runtime, false)
       AND runtime->>'service_internal' <> 'true' THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'runtime is not authorized for the lease shadow case'
        );
    END IF;
    RETURN canonical_brain._append_event(
        'lease.shadow.recorded',
        case_value,
        'Deterministic send-path lease shadow receipt',
        pg_catalog.jsonb_build_object(
            'platform', request->>'source_platform',
            'manual_ref', 'lease-shadow:' || (request->>'intent_event_id')
        ),
        pg_catalog.jsonb_build_object(
            'actor', pg_catalog.jsonb_build_object(
                'type', 'service', 'id', request->>'audit_runtime_id'
            )
        ),
        pg_catalog.jsonb_build_object('lease_shadow', request),
        '{}'::jsonb,
        request->>'intent_event_id',
        'lease_shadow_record',
        runtime
    );
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'lease shadow append failed');
END
$function$;

-- Fixed public routine 14/18.  Approval-source uniqueness is the replay
-- boundary.  Grant insertion, receipt append, and exact active-plan check are
-- serialized by source, approval, routing epoch, and case advisory locks.
CREATE OR REPLACE FUNCTION canonical_brain.writer_capability_grant(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    case_value text := COALESCE(request->>'case_id', '');
    plan_value text := COALESCE(request->>'plan_id', '');
    plan_revision_value integer;
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    user_value text := COALESCE(runtime->>'user_id', '');
    expires_value timestamptz;
    max_uses_value integer;
    head_result jsonb;
    head_plan jsonb;
    request_hash text;
    existing_record canonical_brain.writer_capability_grants%ROWTYPE;
    append_result jsonb;
    event_uuid uuid;
    remaining_map jsonb;
    current_state text;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$'
       OR pg_catalog.length(user_value) NOT BETWEEN 1 AND 240
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['approval_id','case_id','plan_id','plan_revision','approval_source_sha256',
                  'command_hashes','expires_at','max_uses'],
            ARRAY['approval_id','case_id','plan_id','plan_revision','approval_source_sha256',
                  'command_hashes','expires_at','max_uses']
       )
       OR case_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR pg_catalog.length(plan_value) NOT BETWEEN 1 AND 240
       OR COALESCE(request->>'plan_revision', '') !~ '^[1-9][0-9]{0,8}$'
       OR pg_catalog.length(COALESCE(request->>'approval_id', '')) NOT BETWEEN 1 AND 240
       OR COALESCE(request->>'approval_source_sha256', '') !~ '^[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(request->'command_hashes') <> 'array'
       OR pg_catalog.jsonb_array_length(request->'command_hashes') NOT BETWEEN 1 AND 64
       OR EXISTS (
            SELECT 1
              FROM pg_catalog.jsonb_array_elements_text(request->'command_hashes') AS command(value)
             WHERE command.value !~ '^[0-9a-f]{64}$'
       )
       OR (
            SELECT pg_catalog.count(DISTINCT command.value)
              FROM pg_catalog.jsonb_array_elements_text(request->'command_hashes') AS command(value)
       ) <> pg_catalog.jsonb_array_length(request->'command_hashes')
       OR COALESCE(request->>'max_uses', '') !~ '^([1-9][0-9]{0,2}|1000)$' THEN
        RETURN canonical_brain._fail('invalid_request', 'capability grant is invalid');
    END IF;
    IF runtime->>'platform' <> 'discord'
       OR runtime->>'owner_authenticated' <> 'true'
       OR NOT canonical_brain._case_scope_authorized(case_value, runtime, false) THEN
        RETURN canonical_brain._fail(
            'owner_required',
            'capability grant requires the authenticated Discord owner and exact case scope'
        );
    END IF;
    max_uses_value := (request->>'max_uses')::integer;
    plan_revision_value := (request->>'plan_revision')::integer;
    IF max_uses_value > 1000 THEN
        RETURN canonical_brain._fail('invalid_request', 'capability max_uses exceeds 1000');
    END IF;
    BEGIN
        expires_value := (request->>'expires_at')::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        RETURN canonical_brain._fail('invalid_request', 'capability expiry is invalid');
    END;
    IF expires_value <= pg_catalog.clock_timestamp() THEN
        RETURN canonical_brain._fail('approval_expired', 'capability expiry must be in the future');
    END IF;
    IF expires_value > pg_catalog.clock_timestamp() + INTERVAL '8 hours' THEN
        RETURN canonical_brain._fail(
            'invalid_request',
            'capability expiry exceeds the bounded eight-hour TTL'
        );
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'approval-source:' || (request->>'approval_source_sha256'), 0
        )
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'approval-id:' || (request->>'approval_id'), 0
        )
    );
    -- Grant, consume, and both explicit revoke operations share this exact
    -- scope lock.  The tombstone check below therefore cannot race a revoke
    -- scan, and a completed revoke cannot be bypassed by a later grant in the
    -- same routing epoch.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_capability_revocation_scopes AS scope
         WHERE scope.session_key_sha256 = session_value
           AND scope.capability_epoch_sha256 = epoch_value
           AND (
                scope.scope_type = 'session'
                OR (scope.scope_type = 'plan' AND scope.plan_id = plan_value)
           )
    ) THEN
        RETURN canonical_brain._fail(
            'capability_scope_revoked',
            'capability scope was durably revoked for this routing epoch'
        );
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('canonical-plan:' || case_value, 0)
    );
    head_result := canonical_brain._plan_head(case_value);
    IF NOT (head_result->>'ok')::boolean THEN
        RETURN head_result;
    END IF;
    head_plan := head_result->'result'->'head'->'plan';
    IF head_plan IS NULL OR head_plan = 'null'::jsonb
       OR head_plan->>'plan_id' <> plan_value
       OR head_plan->>'revision' <> plan_revision_value::text
       OR head_plan->>'state' <> 'active' THEN
        RETURN canonical_brain._fail('plan_not_active', 'exact canonical plan is not active');
    END IF;

    request_hash := canonical_brain._sha256_json(pg_catalog.jsonb_build_object(
        'approval_id', request->>'approval_id',
        'case_id', case_value,
        'plan_id', plan_value,
        'plan_revision', plan_revision_value,
        'session_key_sha256', session_value,
        'capability_epoch_sha256', epoch_value,
        'approved_by_user_id', user_value,
        'approval_source_sha256', request->>'approval_source_sha256',
        'command_hashes', request->'command_hashes',
        'expires_at', pg_catalog.to_char(
            expires_value AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
        ),
        'max_uses', max_uses_value
    ));
    SELECT * INTO existing_record
      FROM canonical_brain.writer_capability_grants AS grant_row
     WHERE grant_row.approval_source_sha256 = request->>'approval_source_sha256';
    IF FOUND THEN
        IF existing_record.request_sha256 <> request_hash THEN
            RETURN canonical_brain._fail(
                'approval_source_replay',
                'approval source is already bound to another durable capability'
            );
        END IF;
        SELECT COALESCE(
            pg_catalog.jsonb_object_agg(
                command.value,
                GREATEST(
                    existing_record.max_uses - (
                        SELECT pg_catalog.count(*)::integer
                          FROM canonical_brain.writer_capability_consumptions AS consumption
                         WHERE consumption.approval_id = existing_record.approval_id
                           AND consumption.command_sha256 = command.value
                    ),
                    0
                )
            ),
            '{}'::jsonb
        ) INTO remaining_map
          FROM pg_catalog.jsonb_array_elements_text(existing_record.command_hashes) AS command(value);
        current_state := CASE
            WHEN EXISTS (
                SELECT 1
                  FROM canonical_brain.writer_capability_revocations AS revocation
                 WHERE revocation.approval_id = existing_record.approval_id
            ) THEN 'revoked'
            WHEN existing_record.expires_at <= pg_catalog.clock_timestamp()
                THEN 'expired'
            ELSE 'granted'
        END;
        RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
            'success', true,
            'approval_id', existing_record.approval_id,
            'case_id', existing_record.case_id,
            'plan_id', existing_record.plan_id,
            'plan_revision', existing_record.plan_revision,
            'session_key_sha256', existing_record.session_key_sha256,
            'capability_epoch_sha256', existing_record.capability_epoch_sha256,
            'approved_by_user_id', existing_record.approved_by_user_id,
            'approval_source_sha256', existing_record.approval_source_sha256,
            'command_hashes', existing_record.command_hashes,
            'expires_at', existing_record.expires_at,
            'remaining_uses', remaining_map,
            'state', current_state,
            'authority_active', current_state = 'granted',
            'event_id', existing_record.grant_event_id::text,
            'deduped', true
        ));
    END IF;
    IF EXISTS (
        SELECT 1 FROM canonical_brain.writer_capability_grants AS grant_row
         WHERE grant_row.approval_id = request->>'approval_id'
    ) THEN
        RETURN canonical_brain._fail('approval_id_conflict', 'approval_id already exists');
    END IF;

    SELECT COALESCE(
        pg_catalog.jsonb_object_agg(command.value, max_uses_value),
        '{}'::jsonb
    ) INTO remaining_map
      FROM pg_catalog.jsonb_array_elements_text(request->'command_hashes') AS command(value);
    append_result := canonical_brain._append_event(
        'approval.capability.recorded',
        case_value,
        'Exact plan capability granted',
        pg_catalog.jsonb_build_object(
            'platform', COALESCE(runtime->>'platform', ''),
            'manual_ref',
            'approval-source:' || (request->>'approval_source_sha256')
        ),
        pg_catalog.jsonb_build_object(
            'actor', pg_catalog.jsonb_build_object(
                'type', 'authenticated_owner', 'id', user_value
            ),
            'subject', pg_catalog.jsonb_build_object(
                'type', 'task_plan', 'id', plan_value
            )
        ),
        pg_catalog.jsonb_build_object(
            'approval_receipt', pg_catalog.jsonb_build_object(
                'approval_id', request->>'approval_id',
                'case_id', case_value,
                'plan_id', plan_value,
                'plan_revision', plan_revision_value,
                'session_key_sha256', session_value,
                'capability_epoch_sha256', epoch_value,
                'approved_by_user_id', user_value,
                'approval_source_sha256', request->>'approval_source_sha256',
                'command_hashes', request->'command_hashes',
                'expires_at', expires_value,
                'remaining_uses', remaining_map,
                'state', 'granted',
                'authority_active', true
            )
        ),
        '{}'::jsonb,
        'grant:' || (request->>'approval_id'),
        'capability_grant',
        runtime
    );
    IF NOT (append_result->>'ok')::boolean THEN
        RETURN append_result;
    END IF;
    event_uuid := (append_result->'result'->>'event_id')::uuid;
    INSERT INTO canonical_brain.writer_capability_grants (
        approval_id, case_id, plan_id, plan_revision, session_key_sha256,
        capability_epoch_sha256,
        approved_by_user_id, approval_source_sha256, command_hashes,
        expires_at, max_uses, request_sha256, granted_at, grant_event_id
    ) VALUES (
        request->>'approval_id', case_value, plan_value, plan_revision_value,
        session_value,
        epoch_value,
        user_value, request->>'approval_source_sha256', request->'command_hashes',
        expires_value, max_uses_value, request_hash,
        pg_catalog.clock_timestamp(), event_uuid
    );
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'approval_id', request->>'approval_id',
        'case_id', case_value,
        'plan_id', plan_value,
        'plan_revision', plan_revision_value,
        'session_key_sha256', session_value,
        'capability_epoch_sha256', epoch_value,
        'approved_by_user_id', user_value,
        'approval_source_sha256', request->>'approval_source_sha256',
        'command_hashes', request->'command_hashes',
        'expires_at', expires_value,
        'remaining_uses', remaining_map,
        'state', 'granted',
        'authority_active', true,
        'event_id', event_uuid::text,
        'inserted', true,
        'deduped', false
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'capability grant failed');
END
$function$;

-- Fixed public routine 15/18.  Idempotency lookup, exact session/command/plan
-- checks, decrement, durable use row, and audit receipt are one transaction.
CREATE OR REPLACE FUNCTION canonical_brain.writer_capability_consume(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    user_value text := COALESCE(runtime->>'user_id', '');
    command_value text := COALESCE(request->>'command_sha256', '');
    idempotency_value text := COALESCE(request->>'idempotency_key', '');
    request_hash text;
    consume_uuid uuid;
    existing_use canonical_brain.writer_capability_consumptions%ROWTYPE;
    grant_count integer;
    grant_record canonical_brain.writer_capability_grants%ROWTYPE;
    used_count integer;
    remaining_value integer;
    head_result jsonb;
    head_plan jsonb;
    append_result jsonb;
    response_value jsonb;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$'
       OR pg_catalog.length(user_value) NOT BETWEEN 1 AND 240
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['command_sha256','idempotency_key'],
            ARRAY['command_sha256','idempotency_key']
       )
       OR command_value !~ '^[0-9a-f]{64}$'
       OR pg_catalog.octet_length(idempotency_value) NOT BETWEEN 1 AND 256 THEN
        RETURN canonical_brain._fail('invalid_request', 'capability consume is invalid');
    END IF;
    request_hash := canonical_brain._sha256_json(pg_catalog.jsonb_build_object(
        'session_key_sha256', session_value,
        'capability_epoch_sha256', epoch_value,
        'user_id', user_value,
        'command_sha256', command_value,
        'idempotency_key', idempotency_value
    ));
    consume_uuid := canonical_brain._deterministic_uuid(
        'capability-consume:' || session_value || ':' || epoch_value || ':'
            || idempotency_value
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-consume:' || session_value || ':' || epoch_value || ':'
                || idempotency_value, 0
        )
    );
    -- Linearize every new consume with grant/revoke for this session epoch.
    -- The idempotency lock is acquired first everywhere a consume identity is
    -- involved, while scope mutators acquire only the scope lock.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );

    SELECT * INTO existing_use
     FROM canonical_brain.writer_capability_consumptions AS consumption
     WHERE consumption.session_key_sha256 = session_value
       AND consumption.capability_epoch_sha256 = epoch_value
       AND consumption.idempotency_key = idempotency_value;
    IF FOUND THEN
        IF existing_use.request_sha256 <> request_hash
           OR existing_use.command_sha256 <> command_value THEN
            RETURN canonical_brain._fail(
                'idempotency_conflict',
                'consume idempotency key is bound to another exact command'
            );
        END IF;
        RETURN canonical_brain._ok(
            existing_use.response || pg_catalog.jsonb_build_object('deduped', true)
        );
    END IF;

    SELECT pg_catalog.count(*)
      INTO grant_count
      FROM canonical_brain.writer_capability_grants AS grant_row
     WHERE grant_row.session_key_sha256 = session_value
       AND grant_row.capability_epoch_sha256 = epoch_value
       AND grant_row.approved_by_user_id = user_value
       AND grant_row.command_hashes @> pg_catalog.jsonb_build_array(command_value)
       AND grant_row.expires_at > pg_catalog.clock_timestamp()
       AND NOT EXISTS (
            SELECT 1
              FROM canonical_brain.writer_capability_revocations AS revocation
             WHERE revocation.approval_id = grant_row.approval_id
       )
       AND NOT EXISTS (
            SELECT 1
              FROM canonical_brain.writer_capability_revocation_scopes AS scope
             WHERE scope.session_key_sha256 = session_value
               AND scope.capability_epoch_sha256 = epoch_value
               AND (
                    scope.scope_type = 'session'
                    OR (
                        scope.scope_type = 'plan'
                        AND scope.plan_id = grant_row.plan_id
                    )
               )
       );
    IF grant_count = 0 THEN
        RETURN canonical_brain._fail(
            'capability_missing',
            'no live durable capability matches the exact command and session'
        );
    END IF;
    IF grant_count <> 1 THEN
        RETURN canonical_brain._fail(
            'capability_ambiguous',
            'multiple live durable capabilities match the exact command and session'
        );
    END IF;
    SELECT * INTO grant_record
      FROM canonical_brain.writer_capability_grants AS grant_row
     WHERE grant_row.session_key_sha256 = session_value
       AND grant_row.capability_epoch_sha256 = epoch_value
       AND grant_row.approved_by_user_id = user_value
       AND grant_row.command_hashes @> pg_catalog.jsonb_build_array(command_value)
       AND grant_row.expires_at > pg_catalog.clock_timestamp()
       AND NOT EXISTS (
            SELECT 1
              FROM canonical_brain.writer_capability_revocations AS revocation
             WHERE revocation.approval_id = grant_row.approval_id
       )
       AND NOT EXISTS (
            SELECT 1
              FROM canonical_brain.writer_capability_revocation_scopes AS scope
             WHERE scope.session_key_sha256 = session_value
               AND scope.capability_epoch_sha256 = epoch_value
               AND (
                    scope.scope_type = 'session'
                    OR (
                        scope.scope_type = 'plan'
                        AND scope.plan_id = grant_row.plan_id
                    )
               )
       )
     LIMIT 1;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('canonical-plan:' || grant_record.case_id, 0)
    );
    -- Recheck every mutable-by-append condition after acquiring the plan lock.
    IF grant_record.expires_at <= pg_catalog.clock_timestamp()
       OR EXISTS (
            SELECT 1
              FROM canonical_brain.writer_capability_revocations AS revocation
             WHERE revocation.approval_id = grant_record.approval_id
       )
       OR EXISTS (
            SELECT 1
              FROM canonical_brain.writer_capability_revocation_scopes AS scope
             WHERE scope.session_key_sha256 = session_value
               AND scope.capability_epoch_sha256 = epoch_value
               AND (
                    scope.scope_type = 'session'
                    OR (
                        scope.scope_type = 'plan'
                        AND scope.plan_id = grant_record.plan_id
                    )
               )
       ) THEN
        RETURN canonical_brain._fail('capability_expired', 'durable capability is no longer live');
    END IF;
    head_result := canonical_brain._plan_head(grant_record.case_id);
    IF NOT (head_result->>'ok')::boolean THEN
        RETURN head_result;
    END IF;
    head_plan := head_result->'result'->'head'->'plan';
    IF head_plan IS NULL OR head_plan = 'null'::jsonb
       OR head_plan->>'plan_id' <> grant_record.plan_id
       OR head_plan->>'revision' <> grant_record.plan_revision::text
       OR head_plan->>'state' <> 'active' THEN
        RETURN canonical_brain._fail('plan_not_active', 'exact canonical plan is not active');
    END IF;

    SELECT pg_catalog.count(*)
      INTO used_count
     FROM canonical_brain.writer_capability_consumptions AS consumption
     WHERE consumption.approval_id = grant_record.approval_id
       AND consumption.capability_epoch_sha256 = epoch_value
       AND consumption.command_sha256 = command_value;
    IF used_count >= grant_record.max_uses THEN
        RETURN canonical_brain._fail(
            'capability_exhausted',
            'exact command use counter is exhausted'
        );
    END IF;
    remaining_value := grant_record.max_uses - used_count - 1;

    append_result := canonical_brain._append_event(
        'capability.check.recorded',
        grant_record.case_id,
        'Exact plan capability authorized command hash',
        pg_catalog.jsonb_build_object(
            'platform', COALESCE(runtime->>'platform', ''),
            'manual_ref', 'capability-consume:' || consume_uuid::text
        ),
        pg_catalog.jsonb_build_object(
            'actor', pg_catalog.jsonb_build_object(
                'type', 'service', 'id', 'canonical_writer'
            ),
            'subject', pg_catalog.jsonb_build_object(
                'type', 'task_plan', 'id', grant_record.plan_id
            )
        ),
        pg_catalog.jsonb_build_object(
            'capability_receipt', pg_catalog.jsonb_build_object(
                'approval_id', grant_record.approval_id,
                'case_id', grant_record.case_id,
                'plan_id', grant_record.plan_id,
                'plan_revision', grant_record.plan_revision,
                'approved_by_user_id', grant_record.approved_by_user_id,
                'session_key_sha256', session_value,
                'capability_epoch_sha256', epoch_value,
                'command_sha256', command_value,
                'remaining_uses_for_command', remaining_value,
                'state', 'authorized'
            )
        ),
        '{}'::jsonb,
        'consume:' || consume_uuid::text,
        'capability_consume',
        runtime
    );
    IF NOT (append_result->>'ok')::boolean THEN
        RETURN append_result;
    END IF;
    response_value := pg_catalog.jsonb_build_object(
        'success', true,
        'authorized', true,
        'approval_id', grant_record.approval_id,
        'case_id', grant_record.case_id,
        'plan_id', grant_record.plan_id,
        'plan_revision', grant_record.plan_revision,
        'approved_by_user_id', grant_record.approved_by_user_id,
        'capability_epoch_sha256', epoch_value,
        'command_sha256', command_value,
        'remaining_uses', remaining_value,
        'event_id', append_result->'result'->>'event_id',
        'inserted', true,
        'deduped', false
    );
    INSERT INTO canonical_brain.writer_capability_consumptions (
        consume_id, approval_id, command_sha256, session_key_sha256,
        capability_epoch_sha256, idempotency_key, request_sha256, remaining_uses,
        consumed_at, receipt_event_id, response
    ) VALUES (
        consume_uuid, grant_record.approval_id, command_value, session_value,
        epoch_value, idempotency_value, request_hash, remaining_value,
        pg_catalog.clock_timestamp(),
        (append_result->'result'->>'event_id')::uuid, response_value
    );
    RETURN canonical_brain._ok(response_value);
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'capability consume failed');
END
$function$;

-- Fixed public routine 16/18.
CREATE OR REPLACE FUNCTION canonical_brain.writer_capability_revoke(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    plan_value text := COALESCE(request->>'plan_id', '');
    reason_value text := COALESCE(request->>'reason', '');
    case_record record;
    append_result jsonb;
    inserted_total integer := 0;
    inserted_now integer;
    revoked_ids text[];
    revocation_set_sha256 text;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$'
       OR NOT canonical_brain._keys_valid(
            request, ARRAY['plan_id','reason'], ARRAY['plan_id','reason']
       )
       OR pg_catalog.length(plan_value) NOT BETWEEN 1 AND 240
       OR pg_catalog.length(reason_value) NOT BETWEEN 1 AND 1000 THEN
        RETURN canonical_brain._fail('invalid_request', 'capability revoke is invalid');
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );
    INSERT INTO canonical_brain.writer_capability_revocation_scopes (
        scope_type, session_key_sha256, capability_epoch_sha256,
        plan_id, reason, revoked_at
    ) VALUES (
        'plan', session_value, epoch_value, plan_value, reason_value,
        pg_catalog.clock_timestamp()
    ) ON CONFLICT (
        scope_type, session_key_sha256, capability_epoch_sha256, plan_id
    ) DO NOTHING;
    FOR case_record IN
        SELECT DISTINCT grant_row.case_id
          FROM canonical_brain.writer_capability_grants AS grant_row
         WHERE grant_row.plan_id = plan_value
           AND grant_row.session_key_sha256 = session_value
           AND grant_row.capability_epoch_sha256 = epoch_value
    LOOP
        WITH inserted AS (
            INSERT INTO canonical_brain.writer_capability_revocations (
                approval_id, reason, revoked_by_session_sha256, revoked_at
            )
            SELECT grant_row.approval_id, reason_value, session_value,
                   pg_catalog.clock_timestamp()
              FROM canonical_brain.writer_capability_grants AS grant_row
             WHERE grant_row.case_id = case_record.case_id
               AND grant_row.plan_id = plan_value
               AND grant_row.session_key_sha256 = session_value
               AND grant_row.capability_epoch_sha256 = epoch_value
            ON CONFLICT (approval_id) DO NOTHING
            RETURNING approval_id
        )
        SELECT COALESCE(
                   pg_catalog.array_agg(approval_id ORDER BY approval_id),
                   ARRAY[]::text[]
               )
          INTO revoked_ids
          FROM inserted;
        inserted_now := pg_catalog.cardinality(revoked_ids);
        inserted_total := inserted_total + inserted_now;
        IF inserted_now > 0 THEN
            revocation_set_sha256 := canonical_brain._sha256_json(
                pg_catalog.to_jsonb(revoked_ids)
            );
            append_result := canonical_brain._append_event(
                'approval.capability.revoked',
                case_record.case_id,
                'Exact plan capability revoked',
                pg_catalog.jsonb_build_object(
                    'platform', COALESCE(runtime->>'platform', ''),
                    'manual_ref', 'capability-revoke:' || plan_value
                ),
                '{}'::jsonb,
                pg_catalog.jsonb_build_object(
                    'plan_id', plan_value,
                    'session_key_sha256', session_value,
                    'capability_epoch_sha256', epoch_value,
                    'reason', reason_value,
                    'revoked', inserted_now,
                    'revocation_set_sha256', revocation_set_sha256
                ),
                '{}'::jsonb,
                'revoke:' || plan_value || ':' || epoch_value || ':'
                    || revocation_set_sha256,
                'capability_revoke',
                runtime
            );
            IF NOT (append_result->>'ok')::boolean THEN
                RAISE EXCEPTION 'capability revocation receipt append failed';
            END IF;
        END IF;
    END LOOP;
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'plan_id', plan_value,
        'capability_epoch_sha256', epoch_value,
        'scope_type', 'plan',
        'scope_revoked', true,
        'revoked', inserted_total
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'capability revoke failed');
END
$function$;

-- Fixed public routine 17/18.  The request's session digest must exactly equal
-- the authenticated runtime digest; callers cannot revoke another session.
CREATE OR REPLACE FUNCTION canonical_brain.writer_capability_revoke_session(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    session_value text := COALESCE(runtime->>'session_key_sha256', '');
    epoch_value text := COALESCE(runtime->>'capability_epoch_sha256', '');
    reason_value text := COALESCE(request->>'reason', '');
    case_record record;
    append_result jsonb;
    inserted_total integer := 0;
    inserted_now integer;
    revoked_ids text[];
    revocation_set_sha256 text;
    scope_reason text;
    scope_receipt_result jsonb;
    canary_claim record;
    canary_scope_count integer := 0;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR session_value !~ '^[0-9a-f]{64}$'
       OR epoch_value !~ '^[0-9a-f]{64}$'
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['session_key_sha256','reason'],
            ARRAY['session_key_sha256','reason']
       )
       OR request->>'session_key_sha256' <> session_value
       OR pg_catalog.length(reason_value) NOT BETWEEN 1 AND 1000 THEN
        RETURN canonical_brain._fail('scope_mismatch', 'session revoke scope is invalid');
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'capability-scope:' || session_value || ':' || epoch_value, 0
        )
    );
    INSERT INTO canonical_brain.writer_capability_revocation_scopes (
        scope_type, session_key_sha256, capability_epoch_sha256,
        plan_id, reason, revoked_at
    ) VALUES (
        'session', session_value, epoch_value, '', reason_value,
        pg_catalog.clock_timestamp()
    ) ON CONFLICT (
        scope_type, session_key_sha256, capability_epoch_sha256, plan_id
    ) DO NOTHING;
    SELECT scope.reason INTO scope_reason
      FROM canonical_brain.writer_capability_revocation_scopes AS scope
     WHERE scope.scope_type = 'session'
       AND scope.session_key_sha256 = session_value
       AND scope.capability_epoch_sha256 = epoch_value
       AND scope.plan_id = '';
    scope_receipt_result := canonical_brain._append_event(
        'approval.capability.session_revoked',
        'case:session-authority:' || session_value,
        'Session capability authority durably retired',
        pg_catalog.jsonb_build_object(
            'manual_ref', 'session-scope-revoke:' || session_value || ':'
                || epoch_value
        ),
        pg_catalog.jsonb_build_object(
            'actor', pg_catalog.jsonb_build_object(
                'type', 'service', 'id', 'canonical_writer'
            ),
            'subject', pg_catalog.jsonb_build_object(
                'type', 'session_authority',
                'id', session_value || ':' || epoch_value
            )
        ),
        pg_catalog.jsonb_build_object(
            'session_scope_revocation', pg_catalog.jsonb_build_object(
                'session_key_sha256', session_value,
                'capability_epoch_sha256', epoch_value,
                'scope_type', 'session',
                'reason', scope_reason,
                'scope_revoked', true,
                'authority_active', false
            )
        ),
        pg_catalog.jsonb_build_object(
            'session_tombstone_recorded', true,
            'authority_active', false
        ),
        'session-scope-revoke:' || session_value || ':' || epoch_value,
        'capability_revoke_session',
        runtime
    );
    IF NOT (scope_receipt_result->>'ok')::boolean THEN
        RAISE EXCEPTION 'session scope revocation receipt append failed';
    END IF;
    FOR case_record IN
        SELECT DISTINCT grant_row.case_id
          FROM canonical_brain.writer_capability_grants AS grant_row
         WHERE grant_row.session_key_sha256 = session_value
           AND grant_row.capability_epoch_sha256 = epoch_value
    LOOP
        WITH inserted AS (
            INSERT INTO canonical_brain.writer_capability_revocations (
                approval_id, reason, revoked_by_session_sha256, revoked_at
            )
            SELECT grant_row.approval_id, reason_value, session_value,
                   pg_catalog.clock_timestamp()
              FROM canonical_brain.writer_capability_grants AS grant_row
             WHERE grant_row.case_id = case_record.case_id
               AND grant_row.session_key_sha256 = session_value
               AND grant_row.capability_epoch_sha256 = epoch_value
            ON CONFLICT (approval_id) DO NOTHING
            RETURNING approval_id
        )
        SELECT COALESCE(
                   pg_catalog.array_agg(approval_id ORDER BY approval_id),
                   ARRAY[]::text[]
               )
          INTO revoked_ids
          FROM inserted;
        inserted_now := pg_catalog.cardinality(revoked_ids);
        inserted_total := inserted_total + inserted_now;
        IF inserted_now > 0 THEN
            revocation_set_sha256 := canonical_brain._sha256_json(
                pg_catalog.to_jsonb(revoked_ids)
            );
            append_result := canonical_brain._append_event(
                'approval.capability.session_revoked',
                case_record.case_id,
                'Session capability grants revoked',
                pg_catalog.jsonb_build_object(
                    'platform', COALESCE(runtime->>'platform', ''),
                    'manual_ref', 'session-revoke:' || session_value
                ),
                '{}'::jsonb,
                pg_catalog.jsonb_build_object(
                    'session_key_sha256', session_value,
                    'capability_epoch_sha256', epoch_value,
                    'reason', reason_value,
                    'revoked', inserted_now,
                    'revocation_set_sha256', revocation_set_sha256
                ),
                '{}'::jsonb,
                'session-revoke:' || session_value || ':' || epoch_value || ':'
                    || revocation_set_sha256,
                'capability_revoke_session',
                runtime
            );
            IF NOT (append_result->>'ok')::boolean THEN
                RAISE EXCEPTION 'session revocation receipt append failed';
            END IF;
        END IF;
    END LOOP;
    SELECT pg_catalog.count(*) INTO canary_scope_count
      FROM canonical_brain.writer_canary_scope_claims AS claim
     WHERE claim.session_key_sha256 = session_value
       AND claim.capability_epoch_sha256 = epoch_value;
    IF canary_scope_count > 64 THEN
        RAISE EXCEPTION 'claimed canary scope retirement exceeds bounded receipt set';
    END IF;
    FOR canary_claim IN
        SELECT claim.grant_id, claim.case_id
          FROM canonical_brain.writer_canary_scope_claims AS claim
         WHERE claim.session_key_sha256 = session_value
           AND claim.capability_epoch_sha256 = epoch_value
         ORDER BY claim.case_id, claim.grant_id
         LIMIT 64
    LOOP
        append_result := canonical_brain._append_event(
            'canary.scope.revoked',
            canary_claim.case_id,
            'Exact isolated canary API session scope revoked',
            pg_catalog.jsonb_build_object(
                'manual_ref', 'canary-scope-revoke:' || canary_claim.grant_id
            ),
            pg_catalog.jsonb_build_object(
                'actor', pg_catalog.jsonb_build_object(
                    'type', 'service', 'id', 'canonical_writer'
                ),
                'subject', pg_catalog.jsonb_build_object(
                    'type', 'canary_scope', 'id', canary_claim.grant_id
                )
            ),
            pg_catalog.jsonb_build_object(
                'canary_scope_revocation', pg_catalog.jsonb_build_object(
                    'grant_id', canary_claim.grant_id,
                    'session_key_sha256', session_value,
                    'capability_epoch_sha256', epoch_value,
                    'reason', scope_reason,
                    'state', 'revoked'
                )
            ),
            pg_catalog.jsonb_build_object(
                'isolated_canary', true,
                'session_tombstone_recorded', true
            ),
            'canary-scope-revoke:' || canary_claim.grant_id || ':'
                || epoch_value,
            'capability_revoke_session',
            runtime
        );
        IF NOT (append_result->>'ok')::boolean THEN
            RAISE EXCEPTION 'canary scope revocation receipt append failed';
        END IF;
    END LOOP;
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'success', true,
        'session_key_sha256', session_value,
        'capability_epoch_sha256', epoch_value,
        'scope_type', 'session',
        'scope_revoked', true,
        'authority_active', false,
        'revocation_event_id',
            scope_receipt_result->'result'->>'event_id',
        'inserted',
            (scope_receipt_result->'result'->>'inserted')::boolean,
        'deduped',
            (scope_receipt_result->'result'->>'deduped')::boolean,
        'revoked', inserted_total,
        'canary_scopes_revoked', canary_scope_count
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'session capability revoke failed');
END
$function$;

-- Fixed public routine 18/18.  Ordinary calls are exact-case scoped.  Only an
-- in-process writer job may set trusted runtime.service_internal=true and read
-- the global append log.  Cursor order is (occurred_at,event_id), never UUID
-- alone, and every non-empty page advances to its last returned event.
CREATE OR REPLACE FUNCTION canonical_brain.writer_projection_read_events(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
DECLARE
    case_value text := COALESCE(request->>'case_id', '');
    after_value text := COALESCE(request->>'after_event_id', '');
    limit_value integer;
    cursor_at timestamptz;
    cursor_id uuid;
    page_value jsonb;
    page_count integer;
    candidate_count integer;
    has_more_value boolean;
    next_value text;
BEGIN
    IF NOT canonical_brain._runtime_valid(runtime)
       OR NOT canonical_brain._keys_valid(
            request,
            ARRAY['case_id','after_event_id','limit'],
            ARRAY['case_id','after_event_id','limit']
       )
       OR (case_value <> '' AND case_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$')
       OR COALESCE(request->>'limit', '') !~ '^[1-9][0-9]{0,2}$'
       OR (after_value <> '' AND after_value !~
           '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$') THEN
        RETURN canonical_brain._fail('invalid_request', 'projection read request is invalid');
    END IF;
    limit_value := (request->>'limit')::integer;
    IF limit_value > 500 THEN
        RETURN canonical_brain._fail('invalid_request', 'projection page exceeds 500');
    END IF;
    IF case_value = '' AND runtime->>'service_internal' <> 'true' THEN
        RETURN canonical_brain._fail(
            'service_internal_required',
            'global projection read requires trusted in-process service authority'
        );
    END IF;
    IF case_value <> ''
       AND runtime->>'service_internal' <> 'true'
       AND NOT canonical_brain._case_scope_authorized(case_value, runtime, false) THEN
        RETURN canonical_brain._fail(
            'scope_mismatch',
            'runtime is not authorized for the exact projection case'
        );
    END IF;

    IF after_value <> '' THEN
        cursor_id := after_value::uuid;
        SELECT event.occurred_at
          INTO cursor_at
          FROM public.canonical_event_log AS event
          JOIN canonical_brain.writer_event_provenance AS provenance
            ON provenance.event_id = event.event_id
         WHERE event.event_id = cursor_id
           AND (case_value = '' OR event.case_id = case_value);
        IF NOT FOUND THEN
            RETURN canonical_brain._fail(
                'cursor_not_found',
                'projection cursor is outside the exact authorized scope'
            );
        END IF;
    END IF;

    WITH page_plus_one AS (
        SELECT event.event_id,
               event.occurred_at,
               canonical_brain._event_envelope(event) AS event_json
          FROM public.canonical_event_log AS event
         WHERE (case_value = '' OR event.case_id = case_value)
           AND EXISTS (
                SELECT 1
                  FROM canonical_brain.writer_event_provenance AS provenance
                 WHERE provenance.event_id = event.event_id
           )
           AND (
                after_value = ''
                OR (event.occurred_at, event.event_id) > (cursor_at, cursor_id)
           )
         ORDER BY event.occurred_at, event.event_id
         LIMIT limit_value + 1
    ), sized AS (
        SELECT page_plus_one.*,
               pg_catalog.sum(
                   pg_catalog.octet_length(page_plus_one.event_json::text) + 1
               ) OVER (
                   ORDER BY page_plus_one.occurred_at, page_plus_one.event_id
               ) AS cumulative_bytes
          FROM page_plus_one
    ), page AS (
        SELECT sized.event_id, sized.occurred_at, sized.event_json
          FROM sized
         -- The PostgreSQL client rejects any field above 1 MiB.  Keep the
         -- aggregate below that bound with room for the response envelope.
         WHERE sized.cumulative_bytes <= 984000
         ORDER BY sized.occurred_at, sized.event_id
         LIMIT limit_value
    )
    SELECT COALESCE(
               pg_catalog.jsonb_agg(
                   page.event_json
                   ORDER BY page.occurred_at, page.event_id
               ),
               '[]'::jsonb
           ),
           (SELECT pg_catalog.count(*) FROM page_plus_one) > pg_catalog.count(*),
           pg_catalog.count(*),
           (SELECT pg_catalog.count(*) FROM page_plus_one)
      INTO page_value, has_more_value, page_count, candidate_count
      FROM page;
    IF page_count = 0 AND candidate_count > 0 THEN
        RETURN canonical_brain._fail(
            'event_exceeds_projection_budget',
            'next canonical event exceeds the bounded projection response budget'
        );
    END IF;
    IF page_count > 0 THEN
        next_value := page_value->(page_count - 1)->>'event_id';
        IF next_value IS NULL OR next_value = after_value THEN
            RETURN canonical_brain._fail(
                'cursor_did_not_advance',
                'projection page failed its strictly advancing cursor invariant'
            );
        END IF;
    ELSE
        next_value := after_value;
    END IF;
    RETURN canonical_brain._ok(pg_catalog.jsonb_build_object(
        'events', page_value,
        'has_more', COALESCE(has_more_value, false),
        'next_after_event_id', next_value,
        'case_id', case_value,
        'bounded', true
    ));
EXCEPTION
WHEN serialization_failure OR deadlock_detected THEN
    RAISE;
WHEN OTHERS THEN
    RETURN canonical_brain._fail('database_failure', 'projection read failed');
END
$function$;

-- SECURITY DEFINER ownership is deliberately separate from the login/service
-- role.  The owner has only the base-table privileges needed by these fixed
-- routines; the service role has no direct table or helper-function access.
ALTER FUNCTION canonical_brain._ok(jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._fail(text,text)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._sha256_text(text)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._sha256_json(jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._deterministic_uuid(text)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._keys_valid(jsonb,text[],text[])
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._runtime_valid(jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._contains_forbidden_dm_ref(jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._case_scope_authorized(text,jsonb,boolean)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._event_envelope(public.canonical_event_log)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._append_event(
    text,text,text,jsonb,jsonb,jsonb,jsonb,text,text,jsonb
) OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain._plan_head(text)
    OWNER TO canonical_brain_migration_owner;

ALTER FUNCTION canonical_brain.writer_ping(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_canary_scope_preapproval_retire(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_canary_scope_claim(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_case_query(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_routeback_context(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_plan_active_match(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_event_append_model(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_plan_transition(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_verification_append(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_routeback_claim(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_routeback_recover(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_routeback_finalize_sent(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_routeback_finalize_blocked(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_lease_shadow_record(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_capability_grant(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_capability_consume(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_capability_revoke(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_capability_revoke_session(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
ALTER FUNCTION canonical_brain.writer_projection_read_events(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;

-- Retire every direct non-owner ACL only on the Canonical Writer's own
-- namespace and objects.  PUBLIC/default privileges outside this namespace are
-- never mutated here; the effective-role attestation below fails closed when a
-- shared database has not been separately hardened for the writer login.
DO $retire_canonical_acl$
DECLARE
    acl_record record;
    grantee_sql text;
BEGIN
    FOR acl_record IN
        SELECT DISTINCT direct_acl.object_kind,
                        direct_acl.object_identity,
                        direct_acl.column_name,
                        direct_acl.grantee
          FROM (
                SELECT 'schema'::text AS object_kind,
                       pg_catalog.format('%I', namespace.nspname)
                           AS object_identity,
                       NULL::text AS column_name,
                       acl.grantee
                  FROM pg_catalog.pg_namespace AS namespace
                  CROSS JOIN LATERAL pg_catalog.aclexplode(
                      COALESCE(
                          namespace.nspacl,
                          pg_catalog.acldefault('n', namespace.nspowner)
                      )
                  ) AS acl
                 WHERE namespace.nspname = 'canonical_brain'
                   AND acl.grantee <> namespace.nspowner
                UNION ALL
                SELECT CASE WHEN class.relkind = 'S'
                            THEN 'sequence' ELSE 'table' END,
                       pg_catalog.format('%I.%I', namespace.nspname, class.relname),
                       NULL::text AS column_name,
                       acl.grantee
                  FROM pg_catalog.pg_class AS class
                  JOIN pg_catalog.pg_namespace AS namespace
                    ON namespace.oid = class.relnamespace
                  CROSS JOIN LATERAL pg_catalog.aclexplode(
                      COALESCE(
                          class.relacl,
                          pg_catalog.acldefault(
                              CASE WHEN class.relkind = 'S'
                                   THEN 'S'::"char" ELSE 'r'::"char" END,
                              class.relowner
                          )
                      )
                  ) AS acl
                 WHERE namespace.nspname = 'canonical_brain'
                   AND class.relkind IN ('r','p','S')
                   AND acl.grantee <> class.relowner
                UNION ALL
                SELECT 'column',
                       pg_catalog.format('%I.%I', namespace.nspname, class.relname),
                       attribute.attname AS column_name,
                       acl.grantee
                  FROM pg_catalog.pg_class AS class
                  JOIN pg_catalog.pg_namespace AS namespace
                    ON namespace.oid = class.relnamespace
                  JOIN pg_catalog.pg_attribute AS attribute
                    ON attribute.attrelid = class.oid
                   AND attribute.attnum > 0
                   AND NOT attribute.attisdropped
                  CROSS JOIN LATERAL pg_catalog.aclexplode(
                      attribute.attacl
                  ) AS acl
                 WHERE namespace.nspname = 'canonical_brain'
                   AND class.relkind IN ('r','p')
                   AND acl.grantee <> class.relowner
                UNION ALL
                SELECT CASE WHEN routine.prokind = 'p'
                            THEN 'procedure' ELSE 'function' END,
                       pg_catalog.format(
                           '%I.%I(%s)', namespace.nspname, routine.proname,
                           pg_catalog.oidvectortypes(routine.proargtypes)
                       ),
                       NULL::text AS column_name,
                       acl.grantee
                  FROM pg_catalog.pg_proc AS routine
                  JOIN pg_catalog.pg_namespace AS namespace
                    ON namespace.oid = routine.pronamespace
                  CROSS JOIN LATERAL pg_catalog.aclexplode(
                      COALESCE(
                          routine.proacl,
                          pg_catalog.acldefault('f', routine.proowner)
                      )
                  ) AS acl
                 WHERE namespace.nspname = 'canonical_brain'
                   AND routine.prokind IN ('f','p')
                   AND acl.grantee <> routine.proowner
          ) AS direct_acl
    LOOP
        grantee_sql := CASE WHEN acl_record.grantee = 0 THEN 'PUBLIC' ELSE
            pg_catalog.format(
                '%I', pg_catalog.pg_get_userbyid(acl_record.grantee)
            ) END;
        IF acl_record.object_kind = 'column' THEN
            EXECUTE pg_catalog.format(
                'REVOKE ALL PRIVILEGES (%I) ON TABLE %s FROM %s CASCADE',
                acl_record.column_name,
                acl_record.object_identity,
                grantee_sql
            );
        ELSE
            EXECUTE pg_catalog.format(
                'REVOKE ALL PRIVILEGES ON %s %s FROM %s CASCADE',
                pg_catalog.upper(acl_record.object_kind),
                acl_record.object_identity,
                grantee_sql
            );
        END IF;
    END LOOP;
END
$retire_canonical_acl$;

REVOKE ALL ON public.canonical_event_log FROM canonical_brain_writer;
GRANT USAGE ON SCHEMA canonical_brain TO canonical_brain_writer;

GRANT EXECUTE ON FUNCTION canonical_brain.writer_ping(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION
    canonical_brain.writer_canary_scope_preapproval_retire(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_canary_scope_claim(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_case_query(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_routeback_context(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_plan_active_match(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_event_append_model(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_plan_transition(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_verification_append(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_routeback_claim(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_routeback_recover(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_routeback_finalize_sent(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_routeback_finalize_blocked(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_lease_shadow_record(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_capability_grant(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_capability_consume(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_capability_revoke(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_capability_revoke_session(jsonb,jsonb)
    TO canonical_brain_writer;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_projection_read_events(jsonb,jsonb)
    TO canonical_brain_writer;

DO $canonical_direct_acl_contract$
DECLARE
    mismatch text;
BEGIN
    WITH actual(
        object_kind, object_identity, column_name, grantor_name,
        grantee_name, privilege_type, is_grantable
    ) AS (
        SELECT 'schema', namespace.nspname::text, ''::text,
               pg_catalog.pg_get_userbyid(acl.grantor),
               CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                    ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
               acl.privilege_type, acl.is_grantable
          FROM pg_catalog.pg_namespace AS namespace
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  namespace.nspacl,
                  pg_catalog.acldefault('n', namespace.nspowner)
              )
          ) AS acl
         WHERE namespace.nspname = 'canonical_brain'
           AND acl.grantee <> namespace.nspowner
        UNION ALL
        SELECT CASE WHEN class.relkind = 'S' THEN 'sequence' ELSE 'table' END,
               pg_catalog.format('%I.%I', namespace.nspname, class.relname),
               ''::text,
               pg_catalog.pg_get_userbyid(acl.grantor),
               CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                    ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
               acl.privilege_type, acl.is_grantable
          FROM pg_catalog.pg_class AS class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  class.relacl,
                  pg_catalog.acldefault(
                      CASE WHEN class.relkind = 'S'
                           THEN 'S'::"char" ELSE 'r'::"char" END,
                      class.relowner
                  )
              )
          ) AS acl
         WHERE (
                namespace.nspname = 'canonical_brain'
                AND class.relkind IN ('r','p','S')
               OR class.oid = 'public.canonical_event_log'::regclass
           )
           AND acl.grantee <> class.relowner
        UNION ALL
        SELECT 'column',
               pg_catalog.format('%I.%I', namespace.nspname, class.relname),
               attribute.attname,
               pg_catalog.pg_get_userbyid(acl.grantor),
               CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                    ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
               acl.privilege_type, acl.is_grantable
          FROM pg_catalog.pg_class AS class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
          JOIN pg_catalog.pg_attribute AS attribute
            ON attribute.attrelid = class.oid
           AND attribute.attnum > 0
           AND NOT attribute.attisdropped
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              attribute.attacl
          ) AS acl
         WHERE (
                namespace.nspname = 'canonical_brain'
                AND class.relkind IN ('r','p')
               OR class.oid = 'public.canonical_event_log'::regclass
           )
           AND acl.grantee <> class.relowner
        UNION ALL
        SELECT CASE WHEN routine.prokind = 'p' THEN 'procedure' ELSE 'function' END,
               pg_catalog.format(
                   '%I.%I(%s)', namespace.nspname, routine.proname,
                   pg_catalog.oidvectortypes(routine.proargtypes)
               ),
               ''::text,
               pg_catalog.pg_get_userbyid(acl.grantor),
               CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                    ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
               acl.privilege_type, acl.is_grantable
          FROM pg_catalog.pg_proc AS routine
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = routine.pronamespace
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  routine.proacl,
                  pg_catalog.acldefault('f', routine.proowner)
              )
          ) AS acl
         WHERE namespace.nspname = 'canonical_brain'
           AND routine.prokind IN ('f','p')
           AND acl.grantee <> routine.proowner
    ), expected(
        object_kind, object_identity, column_name, grantor_name,
        grantee_name, privilege_type, is_grantable
    ) AS (
        SELECT 'schema', 'canonical_brain', '',
               'canonical_brain_migration_owner', 'canonical_brain_writer',
               'USAGE', false
        UNION ALL
        SELECT 'function',
               pg_catalog.format(
                   '%I.%I(%s)', namespace.nspname, routine.proname,
                   pg_catalog.oidvectortypes(routine.proargtypes)
               ),
               '', 'canonical_brain_migration_owner',
               'canonical_brain_writer', 'EXECUTE', false
          FROM pg_catalog.pg_proc AS routine
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = routine.pronamespace
         WHERE namespace.nspname = 'canonical_brain'
           AND routine.proname = ANY (ARRAY[
                'writer_ping','writer_canary_scope_preapproval_retire',
                'writer_canary_scope_claim','writer_case_query',
                'writer_routeback_context',
                'writer_plan_active_match','writer_event_append_model',
                'writer_plan_transition','writer_verification_append',
                'writer_routeback_claim','writer_routeback_recover',
                'writer_routeback_finalize_sent',
                'writer_routeback_finalize_blocked','writer_lease_shadow_record',
                'writer_capability_grant','writer_capability_consume',
                'writer_capability_revoke','writer_capability_revoke_session',
                'writer_projection_read_events'
           ])
           AND pg_catalog.oidvectortypes(routine.proargtypes)
               = 'jsonb, jsonb'
    ), difference AS (
        (SELECT * FROM actual EXCEPT SELECT * FROM expected)
        UNION ALL
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
    )
    SELECT pg_catalog.string_agg(
               object_kind || ':' || object_identity || ':' || column_name || ':'
               || grantor_name || ':' || grantee_name || ':' || privilege_type
               || ':' || is_grantable::text,
               ';' ORDER BY object_kind, object_identity, column_name,
                            grantor_name, grantee_name, privilege_type
           )
      INTO mismatch
      FROM difference;
    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'canonical direct ACL contract mismatch: %', mismatch;
    END IF;
END
$canonical_direct_acl_contract$;

DO $retire_canonical_default_acl$
DECLARE
    acl_record record;
    grantee_sql text;
BEGIN
    FOR acl_record IN
        SELECT DISTINCT defaults.defaclobjtype,
                        acl.grantee
          FROM pg_catalog.pg_default_acl AS defaults
          JOIN pg_catalog.pg_roles AS owner
            ON owner.oid = defaults.defaclrole
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = defaults.defaclnamespace
          CROSS JOIN LATERAL pg_catalog.aclexplode(defaults.defaclacl) AS acl
         WHERE owner.rolname = 'canonical_brain_migration_owner'
           AND namespace.nspname = 'canonical_brain'
           AND acl.grantee <> defaults.defaclrole
    LOOP
        grantee_sql := CASE WHEN acl_record.grantee = 0 THEN 'PUBLIC' ELSE
            pg_catalog.format(
                '%I', pg_catalog.pg_get_userbyid(acl_record.grantee)
            ) END;
        EXECUTE pg_catalog.format(
            'ALTER DEFAULT PRIVILEGES FOR ROLE canonical_brain_migration_owner '
            'IN SCHEMA canonical_brain REVOKE ALL PRIVILEGES ON %s FROM %s CASCADE',
            CASE acl_record.defaclobjtype
                WHEN 'r' THEN 'TABLES'
                WHEN 'S' THEN 'SEQUENCES'
                WHEN 'f' THEN 'FUNCTIONS'
                WHEN 'T' THEN 'TYPES'
                ELSE 'TABLES'
            END,
            grantee_sql
        );
    END LOOP;
END
$retire_canonical_default_acl$;

ALTER DEFAULT PRIVILEGES FOR ROLE canonical_brain_migration_owner
    IN SCHEMA canonical_brain REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE canonical_brain_migration_owner
    IN SCHEMA canonical_brain REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE canonical_brain_migration_owner
    IN SCHEMA canonical_brain REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE canonical_brain_migration_owner
    IN SCHEMA canonical_brain REVOKE ALL ON TYPES FROM PUBLIC;

DO $canonical_default_acl_contract$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_default_acl AS defaults
          JOIN pg_catalog.pg_roles AS owner
            ON owner.oid = defaults.defaclrole
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = defaults.defaclnamespace
          CROSS JOIN LATERAL pg_catalog.aclexplode(defaults.defaclacl) AS acl
         WHERE owner.rolname = 'canonical_brain_migration_owner'
           AND namespace.nspname = 'canonical_brain'
           AND acl.grantee <> defaults.defaclrole
    ) THEN
        RAISE EXCEPTION
            'canonical default ACL contract retains a non-owner grant';
    END IF;
END
$canonical_default_acl_contract$;

DO $effective_writer_acl$
DECLARE
    cloudsqladmin_hba_receipt text := pg_catalog.current_setting(
        'muncho.canonical_writer_cloudsqladmin_hba_rejection_sha256', true
    );
BEGIN
    IF NOT pg_catalog.has_database_privilege(
        'canonical_brain_writer', pg_catalog.current_database(), 'CONNECT'
    ) OR pg_catalog.has_database_privilege(
        'canonical_brain_writer', pg_catalog.current_database(), 'TEMP'
    ) OR pg_catalog.has_database_privilege(
        'canonical_brain_writer', pg_catalog.current_database(), 'CREATE'
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_writer effective current-database privileges are not CONNECT-only';
    END IF;
    IF pg_catalog.has_schema_privilege(
        'canonical_brain_writer', 'public', 'USAGE'
    ) OR pg_catalog.has_schema_privilege(
        'canonical_brain_writer', 'public', 'CREATE'
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_writer retains effective public-schema privileges';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_proc AS routine
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = routine.pronamespace
         WHERE namespace.nspname NOT IN (
                'pg_catalog', 'information_schema', 'canonical_brain'
           )
           AND pg_catalog.has_function_privilege(
               'canonical_brain_writer', routine.oid, 'EXECUTE'
           )
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_writer retains executable non-system routines outside the fixed catalog';
    END IF;
    IF pg_catalog.has_table_privilege(
        'canonical_brain_writer', 'public.canonical_event_log', 'SELECT'
    ) OR pg_catalog.has_table_privilege(
        'canonical_brain_writer', 'public.canonical_event_log', 'INSERT'
    ) OR pg_catalog.has_table_privilege(
        'canonical_brain_writer', 'public.canonical_event_log', 'UPDATE'
    ) OR pg_catalog.has_table_privilege(
        'canonical_brain_writer', 'public.canonical_event_log', 'DELETE'
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_class AS class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
         WHERE namespace.nspname = 'canonical_brain'
           AND class.relkind IN ('r', 'p', 'v', 'm')
           AND (
                pg_catalog.has_table_privilege(
                    'canonical_brain_writer', class.oid, 'SELECT'
                )
                OR pg_catalog.has_table_privilege(
                    'canonical_brain_writer', class.oid, 'INSERT'
                )
                OR pg_catalog.has_table_privilege(
                    'canonical_brain_writer', class.oid, 'UPDATE'
                )
                OR pg_catalog.has_table_privilege(
                    'canonical_brain_writer', class.oid, 'DELETE'
                )
           )
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_writer retains direct table authority';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn
           AND NOT database.datistemplate
           AND database.datname <> pg_catalog.current_database()
           AND pg_catalog.has_database_privilege(
               'canonical_brain_writer', database.datname, 'CONNECT'
           )
           AND NOT pg_temp._cw_managed_cloudsqladmin_exception(
               database.oid, cloudsqladmin_hba_receipt
           )
    ) THEN
        RAISE EXCEPTION
            'canonical_brain_writer retains CONNECT on another database';
    END IF;
END
$effective_writer_acl$;

RESET ROLE;

DO $retire_temporary_owner_database_acl$
BEGIN
    EXECUTE pg_catalog.format(
        'REVOKE TEMPORARY ON DATABASE %I FROM canonical_brain_migration_owner',
        pg_catalog.current_database()
    );
END
$retire_temporary_owner_database_acl$;

DO $retire_temporary_owner_membership$
DECLARE
    admin_name text := pg_catalog.current_setting(
        'muncho.canonical_writer_migration_admin'
    );
BEGIN
    IF CURRENT_USER <> SESSION_USER OR CURRENT_USER <> admin_name THEN
        RAISE EXCEPTION
            'migration administrator identity changed before membership retirement';
    END IF;
    EXECUTE pg_catalog.format(
        'REVOKE canonical_brain_migration_owner FROM %I', admin_name
    );
END
$retire_temporary_owner_membership$;

DO $final_owner_membership_contract$
DECLARE
    admin_name text := pg_catalog.current_setting(
        'muncho.canonical_writer_migration_admin'
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
    ) OR (
        NOT admin_superuser
        AND pg_catalog.pg_has_role(
            admin_name, 'canonical_brain_migration_owner', 'MEMBER'
        )
    ) OR pg_catalog.has_database_privilege(
        'canonical_brain_migration_owner',
        pg_catalog.current_database(), 'TEMP'
    ) THEN
        RAISE EXCEPTION
            'migration-owner membership survived transaction-scoped retirement';
    END IF;
END
$final_owner_membership_contract$;

COMMIT;
