-- Finalize the sole persistent writer membership after the base migration.

BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '2min';

SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

DO $membership_prerequisites$
BEGIN
    IF pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_setting('server_version_num')::integer / 10000 <> 18
       OR SESSION_USER !~ '^muncho_canary_admin_[0-9a-f]{16}$'
       OR CURRENT_USER <> SESSION_USER
       OR pg_catalog.to_regprocedure(
            'canonical_brain.writer_ping(jsonb,jsonb)'
          ) IS NULL
       OR pg_catalog.to_regprocedure(
            'canonical_brain.writer_event_append_model(jsonb,jsonb)'
          ) IS NULL
       OR pg_catalog.to_regprocedure(
            'canonical_brain.writer_routeback_finalize_sent(jsonb,jsonb)'
          ) IS NULL
       OR pg_catalog.to_regprocedure(
            'canonical_brain.writer_routeback_finalize_blocked(jsonb,jsonb)'
          ) IS NULL THEN
        RAISE EXCEPTION 'writer base migration is absent or database identity drifted';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
         WHERE rolname = 'muncho_canary_writer_login'
           AND rolcanlogin AND rolinherit
           AND NOT rolsuper AND NOT rolcreatedb AND NOT rolcreaterole
           AND NOT rolreplication AND NOT rolbypassrls
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE member.rolname = 'muncho_canary_writer_login'
           AND NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_roles AS granted
                 WHERE granted.oid = membership.roleid
                   AND granted.rolname = 'canonical_brain_writer'
           )
    ) THEN
        RAISE EXCEPTION 'writer login authority is not exact before membership';
    END IF;
END
$membership_prerequisites$;

GRANT canonical_brain_writer TO muncho_canary_writer_login
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;

DO $final_membership_contract$
BEGIN
    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE granted.rolname = 'canonical_brain_writer'
           AND member.rolname = 'muncho_canary_writer_login'
           AND NOT membership.admin_option
           AND membership.inherit_option
           AND membership.set_option
    ) <> 1 OR (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE member.rolname = 'muncho_canary_writer_login'
    ) <> 1 OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS writer ON writer.oid = membership.roleid
         WHERE writer.rolname = 'canonical_brain_writer'
           AND NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_roles AS member
                 WHERE member.oid = membership.member
                   AND member.rolname = 'muncho_canary_writer_login'
           )
    ) THEN
        RAISE EXCEPTION 'writer membership is not the sole exact membership';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_class AS class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = class.relnamespace
          CROSS JOIN LATERAL pg_catalog.aclexplode(class.relacl) AS acl
         WHERE namespace.nspname IN ('public', 'canonical_brain')
           AND pg_catalog.pg_get_userbyid(acl.grantee) IN (
                'canonical_brain_writer', 'muncho_canary_writer_login'
           )
           AND acl.privilege_type IN (
                'SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER'
           )
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_attribute AS attribute
          CROSS JOIN LATERAL pg_catalog.aclexplode(attribute.attacl) AS acl
         WHERE pg_catalog.pg_get_userbyid(acl.grantee) IN (
                'canonical_brain_writer', 'muncho_canary_writer_login'
           )
    ) THEN
        RAISE EXCEPTION 'writer received forbidden direct table/column authority';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn AND NOT database.datistemplate
           AND database.datname NOT IN (
                pg_catalog.current_database(), 'cloudsqladmin'
           )
           AND pg_catalog.has_database_privilege(
                'muncho_canary_writer_login', database.datname, 'CONNECT'
           )
    ) OR pg_catalog.pg_has_role(
        'muncho_canary_writer_login',
        'canonical_brain_migration_owner',
        'MEMBER'
    ) THEN
        RAISE EXCEPTION 'writer login has cross-database or owner authority';
    END IF;
END
$final_membership_contract$;

COMMIT;
