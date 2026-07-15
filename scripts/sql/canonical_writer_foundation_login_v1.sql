-- Secret-free preflight for the isolated-canary persistent writer login.
-- Password installation is intentionally not expressible through this generic
-- SQL artifact.  It follows through the dedicated PostgreSQL CopyData boundary,
-- where the password is sent only as CopyData to a private temporary relation;
-- generic SQL, argv, results, logs, and evidence remain secret-free.

BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '2min';

SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

DO $writer_login_password_copy_preflight$
BEGIN
    IF pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_setting('server_version_num')::integer / 10000 <> 18
       OR SESSION_USER !~ '^muncho_canary_admin_[0-9a-f]{16}$'
       OR CURRENT_USER <> SESSION_USER THEN
        RAISE EXCEPTION 'writer login password-copy identity invalid';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles AS role
         WHERE role.rolname = 'muncho_canary_writer_login'
           AND role.rolinherit AND NOT role.rolsuper AND NOT role.rolcreatedb
           AND NOT role.rolcreaterole AND NOT role.rolreplication
           AND NOT role.rolbypassrls AND role.rolconnlimit = -1
           AND role.rolvaliduntil IS NULL AND role.rolconfig IS NULL
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
         WHERE member.rolname = 'muncho_canary_writer_login'
    ) THEN
        RAISE EXCEPTION 'writer login prerequisite authority drifted';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
         WHERE rolname = 'muncho_canary_writer_login'
           AND rolcanlogin
    ) THEN
        RAISE EXCEPTION 'writer login changed before password-copy boundary';
    END IF;
END
$writer_login_password_copy_preflight$;

COMMIT;
