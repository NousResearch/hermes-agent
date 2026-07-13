-- Canonical Writer persistent-foundation pristine prerequisites v1.
--
-- The artifact is permanently bound to the isolated v2 canary database.  It
-- accepts only pre-existing exact offline identities and canonical fourteen-
-- column truth, then seals the database/schema ACLs.  PostgreSQL 18 grants a
-- non-superuser role creator durable ADMIN memberships whose grantor cannot be
-- retired by that creator.  Missing roles and legacy reconciliation therefore
-- fail closed in this Phase A artifact; they require a separate Cloud-admin
-- deletion gate and post-deletion observation before any terminal receipt.

BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET LOCAL TimeZone = 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '5min';

SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

DO $foundation_identity$
BEGIN
    IF pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_setting('server_version_num')::integer / 10000 <> 18
       OR SESSION_USER !~ '^muncho_canary_admin_[0-9a-f]{16}$'
       OR CURRENT_USER <> SESSION_USER
       OR pg_catalog.current_database() = 'ai_platform_brain' THEN
        RAISE EXCEPTION 'persistent writer foundation database/admin identity invalid';
    END IF;
    IF pg_catalog.to_regprocedure('pg_catalog.sha256(bytea)') IS NULL THEN
        RAISE EXCEPTION 'PostgreSQL 18 core sha256(bytea) is required';
    END IF;
END
$foundation_identity$;

DO $require_preexisting_exact_roles$
BEGIN
    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_roles
         WHERE rolname IN (
            'canonical_brain_migration_owner',
            'canonical_brain_writer',
            'canonical_brain_canary_bootstrap',
            'canonical_brain_canary_bootstrap_login',
            'muncho_canary_writer_login'
         )
    ) <> 5 THEN
        RAISE EXCEPTION
            'pre-existing exact foundation roles are required';
    END IF;
END
$require_preexisting_exact_roles$;

DO $exact_role_contract$
DECLARE
    mismatch text;
BEGIN
    WITH expected(name, can_login, inherits) AS (
        VALUES
          ('canonical_brain_migration_owner', false, false),
          ('canonical_brain_writer', false, true),
          ('canonical_brain_canary_bootstrap', false, false),
          ('canonical_brain_canary_bootstrap_login', true, true),
          ('muncho_canary_writer_login', false, true)
    ), actual AS (
        SELECT role.rolname, role.rolcanlogin, role.rolinherit
          FROM pg_catalog.pg_roles AS role
         WHERE role.rolname IN (SELECT name FROM expected)
           AND NOT role.rolsuper
           AND NOT role.rolcreatedb
           AND NOT role.rolcreaterole
           AND NOT role.rolreplication
           AND NOT role.rolbypassrls
           AND role.rolconnlimit = -1
           AND role.rolvaliduntil IS NULL
           AND role.rolconfig IS NULL
    ), difference AS (
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        UNION ALL
        (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT pg_catalog.string_agg(name, ',' ORDER BY name)
      INTO mismatch FROM difference;
    IF mismatch IS NOT NULL OR (
        SELECT pg_catalog.count(*) FROM pg_catalog.pg_roles
         WHERE rolname IN (
            'canonical_brain_migration_owner',
            'canonical_brain_writer',
            'canonical_brain_canary_bootstrap',
            'canonical_brain_canary_bootstrap_login',
            'muncho_canary_writer_login'
         )
    ) <> 5 THEN
        RAISE EXCEPTION 'persistent writer prerequisite role contract drifted: %', mismatch;
    END IF;
END
$exact_role_contract$;

DO $pregrant_membership_contract$
BEGIN
    IF EXISTS (
        SELECT granted.rolname, member.rolname,
               membership.admin_option, membership.inherit_option,
               membership.set_option
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname IN (
                'canonical_brain_migration_owner',
                'canonical_brain_writer',
                'canonical_brain_canary_bootstrap',
                'canonical_brain_canary_bootstrap_login',
                'muncho_canary_writer_login'
               )
            OR member.rolname IN (
                'canonical_brain_migration_owner',
                'canonical_brain_writer',
                'canonical_brain_canary_bootstrap',
                'canonical_brain_canary_bootstrap_login',
                'muncho_canary_writer_login'
               )
        EXCEPT
        SELECT 'canonical_brain_canary_bootstrap',
               'canonical_brain_canary_bootstrap_login',
               false, true, true
    ) THEN
        RAISE EXCEPTION 'unexpected target-role membership exists before foundation';
    END IF;
    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname = 'canonical_brain_canary_bootstrap'
           AND member.rolname = 'canonical_brain_canary_bootstrap_login'
           AND NOT membership.admin_option
           AND membership.inherit_option
           AND membership.set_option
    ) <> 1 THEN
        RAISE EXCEPTION
            'pre-existing bootstrap membership contract is absent';
    END IF;
END
$pregrant_membership_contract$;

-- Reject legacy or absent truth before the first ACL mutation.  The complete
-- relation contract is repeated below after the ACLs are sealed.
DO $preexisting_canonical_truth_prerequisite$
DECLARE
    identity text[];
    owner_name text;
BEGIN
    SELECT pg_catalog.array_agg(
               attribute.attname || ':'
               || pg_catalog.format_type(attribute.atttypid, attribute.atttypmod)
               || ':' || attribute.attnotnull::text
               ORDER BY attribute.attnum
           ), pg_catalog.pg_get_userbyid(class.relowner)
      INTO identity, owner_name
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_attribute AS attribute
        ON attribute.attrelid = class.oid
       AND attribute.attnum > 0 AND NOT attribute.attisdropped
     WHERE class.oid = pg_catalog.to_regclass('public.canonical_event_log')
     GROUP BY class.relowner;
    IF identity IS DISTINCT FROM ARRAY[
        'event_id:uuid:true','schema_version:text:true','event_type:text:true',
        'occurred_at:timestamp with time zone:true','case_id:text:true',
        'source:jsonb:true','actor:jsonb:true','subject:jsonb:true',
        'evidence:jsonb:true','decision:jsonb:true','status:jsonb:true',
        'next_action:jsonb:true','safety:jsonb:true','payload:jsonb:true'
    ]::text[] OR owner_name IS DISTINCT FROM 'canonical_brain_migration_owner'
       OR pg_catalog.to_regclass(
            'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
       ) IS NOT NULL THEN
        RAISE EXCEPTION
            'pre-existing exact canonical14 truth is required';
    END IF;
END
$preexisting_canonical_truth_prerequisite$;

-- A managed/non-superuser reconciliation administrator cannot mutate ACLs on
-- databases it does not own.  Cross-database isolation is therefore a sealed
-- precondition, not an authority the foundation pretends to have.  The API
-- bootstrap must establish it before this transaction; this artifact proves
-- that PUBLIC, the administrator, and every foundation identity have neither
-- CONNECT nor TEMPORARY on the stock databases.
DO $cross_database_isolation_prerequisite$
DECLARE
    observed_database_count integer;
BEGIN
    SELECT pg_catalog.count(*)::integer
      INTO observed_database_count
      FROM pg_catalog.pg_database
     WHERE datname IN ('postgres', 'template1');
    IF observed_database_count <> 2 OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  database.datacl,
                  pg_catalog.acldefault('d', database.datdba)
              )
          ) AS acl
          LEFT JOIN pg_catalog.pg_roles AS grantee
            ON grantee.oid = acl.grantee
         WHERE database.datname IN ('postgres', 'template1')
           AND acl.privilege_type IN ('CONNECT', 'TEMPORARY')
           AND (
                acl.grantee = 0
                OR grantee.rolname = SESSION_USER
                OR grantee.rolname IN (
                    'canonical_brain_migration_owner',
                    'canonical_brain_writer',
                    'canonical_brain_canary_bootstrap',
                    'canonical_brain_canary_bootstrap_login',
                    'muncho_canary_writer_login'
                )
           )
    ) THEN
        RAISE EXCEPTION
            'cross-database CONNECT/TEMPORARY isolation prerequisite is absent';
    END IF;
END
$cross_database_isolation_prerequisite$;

REVOKE ALL PRIVILEGES ON DATABASE muncho_canary_brain FROM PUBLIC;
REVOKE ALL PRIVILEGES ON DATABASE muncho_canary_brain FROM
    canonical_brain_migration_owner,
    canonical_brain_writer,
    canonical_brain_canary_bootstrap,
    canonical_brain_canary_bootstrap_login,
    muncho_canary_writer_login;
GRANT CONNECT ON DATABASE muncho_canary_brain TO
    canonical_brain_writer,
    canonical_brain_canary_bootstrap;

REVOKE ALL PRIVILEGES ON SCHEMA public FROM PUBLIC;
REVOKE ALL PRIVILEGES ON SCHEMA public FROM
    canonical_brain_migration_owner,
    canonical_brain_writer,
    canonical_brain_canary_bootstrap,
    canonical_brain_canary_bootstrap_login,
    muncho_canary_writer_login;
GRANT USAGE ON SCHEMA public TO canonical_brain_migration_owner;

DO $event_log_contract$
DECLARE
    identity text[];
    owner_name text;
    primary_count integer;
BEGIN
    SELECT pg_catalog.array_agg(
               attribute.attname || ':'
               || pg_catalog.format_type(attribute.atttypid, attribute.atttypmod)
               || ':' || attribute.attnotnull::text
               ORDER BY attribute.attnum
           ), pg_catalog.pg_get_userbyid(class.relowner)
      INTO identity, owner_name
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_attribute AS attribute
        ON attribute.attrelid = class.oid
       AND attribute.attnum > 0 AND NOT attribute.attisdropped
     WHERE class.oid = 'public.canonical_event_log'::regclass
     GROUP BY class.relowner;
    IF identity = ARRAY[
        'event_id:uuid:true','schema_version:text:true','event_type:text:true',
        'occurred_at:timestamp with time zone:true','case_id:text:true',
        'source:jsonb:true','actor:jsonb:true','subject:jsonb:true',
        'evidence:jsonb:true','decision:jsonb:true','status:jsonb:true',
        'next_action:jsonb:true','safety:jsonb:true','payload:jsonb:true'
    ]::text[] THEN
        IF owner_name <> 'canonical_brain_migration_owner' OR EXISTS (
            SELECT 1 FROM pg_catalog.pg_attribute AS attribute
            LEFT JOIN pg_catalog.pg_attrdef AS default_row
              ON default_row.adrelid = attribute.attrelid
             AND default_row.adnum = attribute.attnum
             WHERE attribute.attrelid = 'public.canonical_event_log'::regclass
               AND attribute.attnum > 0 AND NOT attribute.attisdropped
               AND (default_row.oid IS NOT NULL OR attribute.attidentity <> ''
                    OR attribute.attgenerated <> '' OR attribute.atthasmissing
                    OR NOT attribute.attislocal OR attribute.attinhcount <> 0
                    OR attribute.attndims <> 0 OR attribute.attoptions IS NOT NULL
                    OR attribute.attfdwoptions IS NOT NULL)
        ) THEN
            RAISE EXCEPTION 'canonical fourteen-column event log identity drifted';
        END IF;
        SELECT pg_catalog.count(*)::integer INTO primary_count
          FROM pg_catalog.pg_constraint
         WHERE conrelid = 'public.canonical_event_log'::regclass
           AND contype = 'p'
           AND pg_catalog.pg_get_constraintdef(oid, true) = 'PRIMARY KEY (event_id)';
        IF primary_count <> 1 OR (
            SELECT pg_catalog.count(*) FROM pg_catalog.pg_constraint
             WHERE conrelid = 'public.canonical_event_log'::regclass
               AND contype <> 'n'
        ) <> 1 THEN
            RAISE EXCEPTION 'canonical event log primary-key contract drifted';
        END IF;
    ELSIF identity = ARRAY[
        'event_id:uuid:true','schema_version:text:true','event_type:text:true',
        'occurred_at:timestamp with time zone:true','case_id:text:true',
        'source:jsonb:true','actor:jsonb:true','subject:jsonb:true',
        'evidence:jsonb:true','decision:jsonb:true','status:jsonb:true',
        'next_action:jsonb:true','safety:jsonb:true','payload:jsonb:true',
        'inserted_at:timestamp with time zone:true','idempotency_key:text:false',
        'source_spool:text:false','spool_line_number:integer:false',
        'raw_event_sha256:text:false'
    ]::text[] THEN
        IF owner_name IS NULL
           OR owner_name IN (
                'postgres', 'cloudsqladmin', 'cloudsqlsuperuser',
                'canonical_brain_migration_owner', 'canonical_brain_writer',
                'canonical_brain_canary_bootstrap',
                'canonical_brain_canary_bootstrap_login',
                'muncho_canary_writer_login'
           ) OR pg_catalog.to_regclass(
            'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
        ) IS NOT NULL THEN
            RAISE EXCEPTION 'legacy event log owner is not a durable isolated source role';
        END IF;
    ELSE
        RAISE EXCEPTION 'event log is neither exact canonical14 nor legacy19';
    END IF;
END
$event_log_contract$;

DO $final_prerequisite_contract$
DECLARE
    admin_name text := SESSION_USER;
    legacy_shape boolean := (
        SELECT pg_catalog.count(*) = 19
          FROM pg_catalog.pg_attribute
         WHERE attrelid = 'public.canonical_event_log'::regclass
           AND attnum > 0 AND NOT attisdropped
    );
BEGIN
    IF pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner', 'public', 'CREATE'
    ) OR NOT pg_catalog.has_schema_privilege(
        'canonical_brain_migration_owner', 'public', 'USAGE'
    ) THEN
        RAISE EXCEPTION 'temporary migration-owner authority survived prerequisites';
    END IF;
    IF legacy_shape THEN
        IF (
            SELECT pg_catalog.count(*)
              FROM pg_catalog.pg_auth_members AS membership
              JOIN pg_catalog.pg_roles AS granted
                ON granted.oid = membership.roleid
              JOIN pg_catalog.pg_roles AS member
                ON member.oid = membership.member
             WHERE granted.rolname = 'canonical_brain_migration_owner'
               AND member.rolname = admin_name
               AND membership.admin_option
               AND NOT membership.inherit_option
               AND NOT membership.set_option
        ) <> 1 OR EXISTS (
            SELECT 1
              FROM pg_catalog.pg_auth_members AS membership
              JOIN pg_catalog.pg_roles AS role
                ON role.oid = membership.roleid OR role.oid = membership.member
             WHERE role.rolname = 'canonical_brain_migration_owner'
               AND NOT (
                    membership.roleid = (
                        SELECT oid FROM pg_catalog.pg_roles
                         WHERE rolname = 'canonical_brain_migration_owner'
                    )
                    AND membership.member = (
                        SELECT oid FROM pg_catalog.pg_roles
                         WHERE rolname = admin_name
                    )
                    AND membership.admin_option
                    AND NOT membership.inherit_option
                    AND NOT membership.set_option
               )
        ) THEN
            RAISE EXCEPTION
                'legacy migration-owner administrator bridge is not exact';
        END IF;
    ELSIF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS role
            ON role.oid = membership.roleid OR role.oid = membership.member
         WHERE role.rolname = 'canonical_brain_migration_owner'
    ) THEN
        RAISE EXCEPTION
            'temporary migration-owner membership survived pristine prerequisites';
    END IF;
    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname = 'canonical_brain_canary_bootstrap'
           AND member.rolname = 'canonical_brain_canary_bootstrap_login'
           AND NOT membership.admin_option
           AND membership.inherit_option
           AND membership.set_option
    ) <> 1 OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE (granted.rolname LIKE 'canonical_brain_%'
                OR member.rolname LIKE 'canonical_brain_%'
                OR granted.rolname = 'muncho_canary_writer_login'
                OR member.rolname = 'muncho_canary_writer_login')
           AND NOT (
                granted.rolname = 'canonical_brain_canary_bootstrap'
                AND member.rolname = 'canonical_brain_canary_bootstrap_login'
                AND NOT membership.admin_option
                AND membership.inherit_option
                AND membership.set_option
           )
           AND NOT (
                legacy_shape
                AND granted.rolname = 'canonical_brain_migration_owner'
                AND member.rolname = admin_name
                AND membership.admin_option
                AND NOT membership.inherit_option
                AND NOT membership.set_option
           )
    ) THEN
        RAISE EXCEPTION 'prerequisite target membership contract drifted';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn AND NOT database.datistemplate
           AND database.datname NOT IN (
                pg_catalog.current_database(), 'cloudsqladmin'
           )
           AND pg_catalog.has_database_privilege(
                'canonical_brain_writer', database.datname, 'CONNECT'
           )
    ) THEN
        RAISE EXCEPTION 'writer role retains cross-database CONNECT authority';
    END IF;
END
$final_prerequisite_contract$;

COMMIT;
