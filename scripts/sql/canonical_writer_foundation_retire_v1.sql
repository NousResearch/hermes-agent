-- Explicit persistent-writer authority retirement.  Durable data, the fixed
-- routines, and offline roles are retained; only login authority and its sole
-- writer membership are retired.  The exact credential inode is removed by
-- the root boundary only after this transaction and an authentication-denial
-- proof both succeed.

BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '2min';

SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

DO $retirement_prerequisites$
BEGIN
    IF pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_setting('server_version_num')::integer / 10000 <> 18
       OR SESSION_USER !~ '^muncho_canary_admin_[0-9a-f]{16}$'
       OR CURRENT_USER <> SESSION_USER
       OR NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'canonical_brain_writer'
               AND NOT rolcanlogin AND NOT rolsuper AND NOT rolcreatedb
               AND NOT rolcreaterole AND NOT rolreplication AND NOT rolbypassrls
       ) OR NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'muncho_canary_writer_login'
               AND rolinherit AND NOT rolsuper AND NOT rolcreatedb
               AND NOT rolcreaterole AND NOT rolreplication AND NOT rolbypassrls
       ) THEN
        RAISE EXCEPTION 'persistent writer retirement identity drifted';
    END IF;
END
$retirement_prerequisites$;

REVOKE canonical_brain_writer FROM muncho_canary_writer_login;
ALTER ROLE muncho_canary_writer_login NOLOGIN PASSWORD NULL;

DO $retirement_contract$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE member.rolname = 'muncho_canary_writer_login'
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
         WHERE rolname = 'muncho_canary_writer_login'
           AND NOT rolcanlogin
           AND rolinherit AND NOT rolsuper AND NOT rolcreatedb
           AND NOT rolcreaterole AND NOT rolreplication AND NOT rolbypassrls
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS owner_role ON owner_role.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member_role ON member_role.oid = membership.member
         WHERE owner_role.rolname = 'canonical_brain_migration_owner'
            OR member_role.rolname = 'canonical_brain_migration_owner'
    ) THEN
        RAISE EXCEPTION 'persistent writer authority retirement was not exact';
    END IF;
END
$retirement_contract$;

COMMIT;
