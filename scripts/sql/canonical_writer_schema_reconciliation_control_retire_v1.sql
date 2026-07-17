-- P0 sealed retirement of the one-purpose schema-reconciliation control.
--
-- Run only after the fixed helper is exact, all normal executor logins are
-- deleted, and writer/gateway services are stopped.  No CASCADE or DROP OWNED
-- is permitted: any unexpected dependency aborts the whole transaction.

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

SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

DO $control_retire_authority_preflight$
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
       OR NOT pg_catalog.pg_has_role(
           SESSION_USER, 'cloudsqlsuperuser', 'SET'
       )
    THEN
        RAISE EXCEPTION 'schema reconciliation control retire authority failed';
    END IF;
END
$control_retire_authority_preflight$;

SET LOCAL ROLE cloudsqlsuperuser;
GRANT canonical_brain_migration_owner TO SESSION_USER
    WITH ADMIN FALSE, INHERIT FALSE, SET TRUE;
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

DO $control_retire_foundation_preflight$
DECLARE
    database_oid oid := (
        SELECT oid FROM pg_catalog.pg_database
         WHERE datname = pg_catalog.current_database()
    );
    executor_oid oid := pg_catalog.to_regrole(
        'canonical_brain_schema_reconciler'
    );
    control_namespace_oid oid := (
        SELECT oid FROM pg_catalog.pg_namespace
         WHERE nspname = 'canonical_brain_reconciliation'
    );
    observer_oid oid := pg_catalog.to_regprocedure(
        'canonical_brain_reconciliation.'
        'observe_missing_discord_routeback_helper_v1()'
    );
    apply_oid oid := pg_catalog.to_regprocedure(
        'canonical_brain_reconciliation.'
        'apply_missing_discord_routeback_helper_v1()'
    );
    helper_oid oid := pg_catalog.to_regprocedure(
        'canonical_brain._discord_guild_routeback_target_valid(jsonb)'
    );
    managed_cloudsqladmin_database_exact boolean;
BEGIN
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

    IF CURRENT_USER <> 'canonical_brain_migration_owner'
       OR executor_oid IS NULL
       OR control_namespace_oid IS NULL
       OR observer_oid IS NULL
       OR apply_oid IS NULL
       OR helper_oid IS NULL
       OR (
           SELECT pg_catalog.count(*)
             FROM pg_catalog.pg_proc AS helper
             JOIN pg_catalog.pg_namespace AS helper_namespace
               ON helper_namespace.oid = helper.pronamespace
            WHERE helper_namespace.nspname = 'canonical_brain'
              AND helper.proname = '_discord_guild_routeback_target_valid'
       ) <> 1
       OR NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_roles AS role
            WHERE role.oid = executor_oid
              AND NOT role.rolcanlogin AND NOT role.rolinherit
              AND NOT role.rolsuper AND NOT role.rolcreatedb
              AND NOT role.rolcreaterole AND NOT role.rolreplication
              AND NOT role.rolbypassrls AND role.rolconnlimit = -1
              AND role.rolvaliduntil IS NULL AND role.rolconfig IS NULL
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
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_auth_members
            WHERE roleid = executor_oid OR member = executor_oid
               OR grantor = executor_oid
       )
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_shdepend
            WHERE refclassid = 'pg_catalog.pg_authid'::pg_catalog.regclass
              AND refobjid = executor_oid AND deptype = 'o'
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
       OR (
           SELECT pg_catalog.count(*) FROM pg_catalog.pg_proc
            WHERE pronamespace = control_namespace_oid
       ) <> 2
       OR (
           SELECT pg_catalog.count(*) FROM pg_catalog.pg_class
            WHERE relnamespace = control_namespace_oid
              AND relkind IN ('r', 'p', 'v', 'm', 'f', 'S', 'c')
       ) <> 0
       OR NOT pg_catalog.has_database_privilege(
           'canonical_brain_schema_reconciler', database_oid, 'CONNECT'
       )
       OR pg_catalog.has_database_privilege(
           'canonical_brain_schema_reconciler', database_oid, 'CREATE'
       )
       OR pg_catalog.has_database_privilege(
           'canonical_brain_schema_reconciler', database_oid, 'TEMPORARY'
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
       OR NOT pg_catalog.has_schema_privilege(
           'canonical_brain_schema_reconciler', control_namespace_oid, 'USAGE'
       )
       OR pg_catalog.has_schema_privilege(
           'canonical_brain_schema_reconciler', control_namespace_oid, 'CREATE'
       )
       OR NOT pg_catalog.has_function_privilege(
           'canonical_brain_schema_reconciler', observer_oid, 'EXECUTE'
       )
       OR NOT pg_catalog.has_function_privilege(
           'canonical_brain_schema_reconciler', apply_oid, 'EXECUTE'
       )
       OR NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_proc AS routine
            WHERE routine.oid = observer_oid
              AND routine.proowner = (
                  SELECT oid FROM pg_catalog.pg_roles
                   WHERE rolname = 'canonical_brain_migration_owner'
              )
              AND routine.pronargs = 0 AND routine.prokind = 'f'
              AND routine.prosecdef IS TRUE AND routine.provolatile = 'v'
              AND routine.proparallel = 'u'
              AND routine.proleakproof IS FALSE
              AND routine.proisstrict IS FALSE AND routine.proretset IS TRUE
              AND routine.proconfig = ARRAY[
                  'search_path=pg_catalog, pg_temp', 'TimeZone=UTC',
                  'DateStyle=ISO, YMD', 'IntervalStyle=postgres',
                  'extra_float_digits=3', 'bytea_output=hex',
                  'lock_timeout=15s', 'statement_timeout=5min'
              ]::text[]
              AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                  routine.prosrc, 'UTF8'
              )), 'hex') =
                  '47b63aa737d29e1d5b3a54fc824606d91c322a7869118b6f331040e0a3ef96fe'
              AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                  pg_catalog.pg_get_functiondef(routine.oid), 'UTF8'
              )), 'hex') =
                  '7813ead62d79011f2f2c4e1895405bb35a8edc959e244a14fc22d1ab1be56974'
       )
       OR NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_proc AS routine
            WHERE routine.oid = apply_oid
              AND routine.proowner = (
                  SELECT oid FROM pg_catalog.pg_roles
                   WHERE rolname = 'canonical_brain_migration_owner'
              )
              AND routine.pronargs = 0 AND routine.prokind = 'f'
              AND routine.prosecdef IS TRUE AND routine.provolatile = 'v'
              AND routine.proparallel = 'u'
              AND routine.proleakproof IS FALSE
              AND routine.proisstrict IS FALSE AND routine.proretset IS TRUE
              AND routine.proconfig = ARRAY[
                  'search_path=pg_catalog, pg_temp', 'TimeZone=UTC',
                  'DateStyle=ISO, YMD', 'IntervalStyle=postgres',
                  'extra_float_digits=3', 'bytea_output=hex',
                  'lock_timeout=15s', 'statement_timeout=5min'
              ]::text[]
              AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                  routine.prosrc, 'UTF8'
              )), 'hex') =
                  '2a28d4700d550bcc8ddc56ea870fc5f669f55a47f9abc7e1993b99b178db1719'
              AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                  pg_catalog.pg_get_functiondef(routine.oid), 'UTF8'
              )), 'hex') =
                  '63d6388e50086bf2203bafb7d74291cbec32d04c1f2e05af4f007df4c1e9c8d6'
       )
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.pg_proc AS routine
            WHERE routine.oid IN (observer_oid, apply_oid)
              AND NOT (
                  SELECT pg_catalog.count(*) = 2
                         AND pg_catalog.bool_and(
                             acl.grantor = routine.proowner
                             AND acl.grantee IN (
                                 routine.proowner, executor_oid
                             )
                             AND acl.privilege_type = 'EXECUTE'
                             AND acl.is_grantable IS FALSE
                         )
                    FROM pg_catalog.aclexplode(COALESCE(
                        routine.proacl,
                        pg_catalog.acldefault('f', routine.proowner)
                    )) AS acl
              )
       )
       OR NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_proc AS helper
            WHERE helper.oid = helper_oid
              AND helper.proowner = (
                  SELECT oid FROM pg_catalog.pg_roles
                   WHERE rolname = 'canonical_brain_migration_owner'
              )
              AND helper.prosecdef IS FALSE AND helper.provolatile = 'i'
              AND helper.proparallel = 'u' AND helper.proleakproof IS FALSE
              AND helper.proisstrict IS FALSE AND helper.proretset IS FALSE
              AND helper.proconfig = ARRAY[
                  'search_path=pg_catalog, canonical_brain'
              ]::text[]
              AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                  helper.prosrc, 'UTF8'
              )), 'hex') =
                  'e82ee5b2240d61c1e7c60d76ec87729d9d87e134d4b2083d5cd7b447f5ef093c'
              AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                  pg_catalog.pg_get_functiondef(helper.oid), 'UTF8'
              )), 'hex') =
                  '2fc8e5334784ae791398033c8b460ccb170324f4c8cc1b3d0387f907e9cf7737'
              AND (
                  SELECT pg_catalog.count(*) = 1
                         AND pg_catalog.bool_and(
                             acl.grantor = helper.proowner
                             AND acl.grantee = helper.proowner
                             AND acl.privilege_type = 'EXECUTE'
                             AND acl.is_grantable IS FALSE
                         )
                    FROM pg_catalog.aclexplode(COALESCE(
                        helper.proacl,
                        pg_catalog.acldefault('f', helper.proowner)
                    )) AS acl
              )
       )
    THEN
        RAISE EXCEPTION 'schema reconciliation control retire preflight failed';
    END IF;
END
$control_retire_foundation_preflight$;

REVOKE EXECUTE ON FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1()
    FROM canonical_brain_schema_reconciler;
REVOKE EXECUTE ON FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1()
    FROM canonical_brain_schema_reconciler;
REVOKE USAGE ON SCHEMA canonical_brain_reconciliation
    FROM canonical_brain_schema_reconciler;

DROP FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1();
DROP FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1();
DROP SCHEMA canonical_brain_reconciliation;

RESET ROLE;
SET LOCAL ROLE cloudsqlsuperuser;
REVOKE canonical_brain_migration_owner FROM SESSION_USER;
REVOKE CONNECT ON DATABASE muncho_canary_brain
    FROM canonical_brain_schema_reconciler;
DROP ROLE canonical_brain_schema_reconciler;
RESET ROLE;

DO $control_retire_terminal$
DECLARE
    managed_cloudsqladmin_database_exact boolean;
BEGIN
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
       OR pg_catalog.to_regrole('canonical_brain_schema_reconciler') IS NOT NULL
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
       OR NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_database AS database
            WHERE database.datname = pg_catalog.current_database()
              AND pg_catalog.pg_get_userbyid(database.datdba) =
                  'cloudsqlsuperuser'
       )
       OR EXISTS (
           SELECT 1 FROM pg_catalog.pg_namespace
            WHERE nspname = 'canonical_brain_reconciliation'
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
       OR NOT EXISTS (
           SELECT 1
             FROM pg_catalog.pg_proc AS helper
             JOIN pg_catalog.pg_namespace AS namespace
               ON namespace.oid = helper.pronamespace
            WHERE namespace.nspname = 'canonical_brain'
              AND helper.proname = '_discord_guild_routeback_target_valid'
              AND pg_catalog.oidvectortypes(helper.proargtypes) = 'jsonb'
              AND pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                  pg_catalog.pg_get_functiondef(helper.oid), 'UTF8'
              )), 'hex') =
                  '2fc8e5334784ae791398033c8b460ccb170324f4c8cc1b3d0387f907e9cf7737'
       )
       OR (
           SELECT pg_catalog.count(*)
             FROM pg_catalog.pg_proc AS helper
             JOIN pg_catalog.pg_namespace AS helper_namespace
               ON helper_namespace.oid = helper.pronamespace
            WHERE helper_namespace.nspname = 'canonical_brain'
              AND helper.proname = '_discord_guild_routeback_target_valid'
       ) <> 1
    THEN
        RAISE EXCEPTION 'schema reconciliation control retire terminal failed';
    END IF;
END
$control_retire_terminal$;

COMMIT;
