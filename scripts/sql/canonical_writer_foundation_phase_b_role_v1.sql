-- Canonical Writer isolated-canary Phase-B role + CONNECT foundation v1.
--
-- This sealed artifact has one fixed PostgreSQL 18 target.  It creates only
-- the inert canonical_brain_canary_bootstrap role and its sole target-database
-- CONNECT ACL.  It never creates a login, installs a password, issues a role-
-- membership grant, or changes a schema, table, routine, or legacy relation.
--
-- PostgreSQL 18 automatically grants a newly-created role back to a
-- non-superuser CREATEROLE session with ADMIN TRUE, INHERIT FALSE, SET FALSE.
-- That bootstrap-superuser-owned membership cannot be revoked by its member.
-- It is therefore attested here as temporary authority, never as terminal
-- truth.  Cloud SQL must delete the temporary SESSION_USER, after which a
-- separate writer-authenticated observation must prove that the automatic
-- membership disappeared while the provider-owned CONNECT ACL survived.
--
-- The caller must install these four secret-free, digest-only bindings on the
-- already verified session before executing the sealed bytes:
--
-- muncho.canonical_writer_phase_b_release_revision
-- muncho.canonical_writer_phase_b_role_artifact_sha256
-- muncho.canonical_writer_phase_b_initial_observation_sha256
-- muncho.canonical_writer_phase_b_approved_plan_sha256

BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET LOCAL TimeZone = 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '2min';

SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

DO $phase_b_role_preflight$
DECLARE
    bootstrap_oid oid;
    database_owner_oid oid;
    writer_oid oid;
    release_revision text := pg_catalog.current_setting(
        'muncho.canonical_writer_phase_b_release_revision', true
    );
    artifact_sha256 text := pg_catalog.current_setting(
        'muncho.canonical_writer_phase_b_role_artifact_sha256', true
    );
    observation_sha256 text := pg_catalog.current_setting(
        'muncho.canonical_writer_phase_b_initial_observation_sha256', true
    );
    approved_plan_sha256 text := pg_catalog.current_setting(
        'muncho.canonical_writer_phase_b_approved_plan_sha256', true
    );
BEGIN
    IF pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_database() = 'ai_platform_brain'
       OR pg_catalog.current_setting('server_version_num')::integer < 180000
       OR pg_catalog.current_setting('server_version_num')::integer >= 190000
       OR SESSION_USER !~ '^muncho_canary_admin_[0-9a-f]{16}$'
       OR CURRENT_USER <> SESSION_USER
       OR release_revision !~ '^[0-9a-f]{40}$'
       OR artifact_sha256 !~ '^[0-9a-f]{64}$'
       OR observation_sha256 !~ '^[0-9a-f]{64}$'
       OR approved_plan_sha256 !~ '^[0-9a-f]{64}$'
       OR pg_catalog.current_setting('createrole_self_grant') <> ''
       OR pg_catalog.to_regprocedure('pg_catalog.sha256(bytea)') IS NULL THEN
        RAISE EXCEPTION 'phase-b role identity or sealed binding is invalid';
    END IF;

    SELECT database.datdba
      INTO database_owner_oid
      FROM pg_catalog.pg_database AS database
     WHERE database.datname = pg_catalog.current_database();
    IF database_owner_oid IS DISTINCT FROM (
        SELECT role.oid FROM pg_catalog.pg_roles AS role
         WHERE role.rolname = 'cloudsqlsuperuser'
    ) THEN
        RAISE EXCEPTION 'phase-b role database owner is not cloudsqlsuperuser';
    END IF;

    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_roles AS role
         WHERE role.rolname = SESSION_USER
           AND role.rolcanlogin
           AND role.rolinherit
           AND NOT role.rolsuper
           AND role.rolcreatedb
           AND role.rolcreaterole
           AND NOT role.rolreplication
           AND NOT role.rolbypassrls
           AND role.rolconnlimit = -1
           AND role.rolvaliduntil IS NULL
           AND role.rolconfig IS NULL
    ) <> 1 OR (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_roles AS role
         WHERE role.rolname ~ '^muncho_canary_admin_[0-9a-f]{16}$'
    ) <> 1 THEN
        RAISE EXCEPTION 'phase-b temporary admin role identity is not exact';
    END IF;

    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted
            ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member
            ON member.oid = membership.member
          JOIN pg_catalog.pg_roles AS grantor
            ON grantor.oid = membership.grantor
         WHERE granted.rolname = 'cloudsqlsuperuser'
           AND member.rolname = SESSION_USER
           AND grantor.rolname = 'cloudsqladmin'
           AND NOT membership.admin_option
           AND membership.inherit_option
           AND membership.set_option
    ) <> 1 OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted
            ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member
            ON member.oid = membership.member
          JOIN pg_catalog.pg_roles AS grantor
            ON grantor.oid = membership.grantor
         WHERE (
                member.rolname = SESSION_USER
                AND NOT (
                    granted.rolname = 'cloudsqlsuperuser'
                    AND grantor.rolname = 'cloudsqladmin'
                    AND NOT membership.admin_option
                    AND membership.inherit_option
                    AND membership.set_option
                )
                AND granted.rolname <> 'canonical_brain_canary_bootstrap'
           )
            OR granted.rolname = SESSION_USER
            OR grantor.rolname = SESSION_USER
    ) THEN
        RAISE EXCEPTION 'phase-b temporary admin provider authority is not exact';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  database.datacl,
                  pg_catalog.acldefault('d', database.datdba)
              )
          ) AS acl
         WHERE database.datname = pg_catalog.current_database()
           AND acl.grantee = 0
    ) THEN
        RAISE EXCEPTION 'phase-b target database PUBLIC denial is not exact';
    END IF;

    SELECT role.oid INTO writer_oid
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname = 'canonical_brain_writer'
       AND NOT role.rolcanlogin
       AND role.rolinherit
       AND NOT role.rolsuper
       AND NOT role.rolcreatedb
       AND NOT role.rolcreaterole
       AND NOT role.rolreplication
       AND NOT role.rolbypassrls
       AND role.rolconnlimit = -1
       AND role.rolvaliduntil IS NULL
       AND role.rolconfig IS NULL;
    IF writer_oid IS NULL OR (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  database.datacl,
                  pg_catalog.acldefault('d', database.datdba)
              )
          ) AS acl
         WHERE database.datname = pg_catalog.current_database()
           AND acl.grantee = writer_oid
           AND acl.grantor = database_owner_oid
           AND acl.privilege_type = 'CONNECT'
           AND NOT acl.is_grantable
    ) <> 1 OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  database.datacl,
                  pg_catalog.acldefault('d', database.datdba)
              )
          ) AS acl
         WHERE database.datname = pg_catalog.current_database()
           AND acl.grantee = writer_oid
           AND NOT (
                acl.grantor = database_owner_oid
                AND acl.privilege_type = 'CONNECT'
                AND NOT acl.is_grantable
           )
    ) THEN
        RAISE EXCEPTION 'phase-b existing writer CONNECT authority is not exact';
    END IF;

    SELECT role.oid INTO bootstrap_oid
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname = 'canonical_brain_canary_bootstrap';
    IF bootstrap_oid IS NULL THEN
        PERFORM pg_catalog.set_config(
            'muncho.canonical_writer_phase_b_role_outcome', 'created', true
        );
        RETURN;
    END IF;

    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_roles AS role
         WHERE role.oid = bootstrap_oid
           AND role.rolname = 'canonical_brain_canary_bootstrap'
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
    ) <> 1 THEN
        RAISE EXCEPTION 'phase-b existing bootstrap role is partial or drifted';
    END IF;

    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  database.datacl,
                  pg_catalog.acldefault('d', database.datdba)
              )
          ) AS acl
         WHERE database.datname = pg_catalog.current_database()
           AND acl.grantee = bootstrap_oid
           AND acl.grantor = database_owner_oid
           AND acl.privilege_type = 'CONNECT'
           AND NOT acl.is_grantable
    ) <> 1 OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  database.datacl,
                  pg_catalog.acldefault('d', database.datdba)
              )
          ) AS acl
         WHERE acl.grantee = bootstrap_oid
           AND NOT (
                database.datname = pg_catalog.current_database()
                AND acl.grantor = database_owner_oid
                AND acl.privilege_type = 'CONNECT'
                AND NOT acl.is_grantable
           )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn
           AND database.datname NOT IN (
                pg_catalog.current_database(), 'cloudsqladmin'
           )
           AND (
                pg_catalog.has_database_privilege(
                    bootstrap_oid, database.oid, 'CONNECT'
                ) OR pg_catalog.has_database_privilege(
                    bootstrap_oid, database.oid, 'TEMPORARY'
                )
           )
    ) THEN
        RAISE EXCEPTION 'phase-b existing bootstrap database ACL is drifted';
    END IF;
    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_shdepend AS dependency
         WHERE dependency.refclassid = 'pg_catalog.pg_authid'::regclass
           AND dependency.refobjid = bootstrap_oid
    ) <> 1 OR NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_shdepend AS dependency
         WHERE dependency.refclassid = 'pg_catalog.pg_authid'::regclass
           AND dependency.refobjid = bootstrap_oid
           AND dependency.dbid = 0
           AND dependency.classid = 'pg_catalog.pg_database'::regclass
           AND dependency.objid = (
                SELECT database.oid FROM pg_catalog.pg_database AS database
                 WHERE database.datname = pg_catalog.current_database()
           )
           AND dependency.objsubid = 0
           AND dependency.deptype = 'a'
    ) THEN
        RAISE EXCEPTION 'phase-b existing bootstrap shared dependency is drifted';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_namespace AS namespace
         WHERE namespace.nspowner = bootstrap_oid
            OR (
                namespace.nspacl IS NOT NULL AND EXISTS (
                    SELECT 1 FROM pg_catalog.aclexplode(namespace.nspacl) AS acl
                     WHERE acl.grantee = bootstrap_oid
                )
            )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_class AS class
         WHERE class.relowner = bootstrap_oid
            OR (
                class.relacl IS NOT NULL AND EXISTS (
                    SELECT 1 FROM pg_catalog.aclexplode(class.relacl) AS acl
                     WHERE acl.grantee = bootstrap_oid
                )
            )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_attribute AS attribute
         WHERE attribute.attacl IS NOT NULL
           AND EXISTS (
                SELECT 1 FROM pg_catalog.aclexplode(attribute.attacl) AS acl
                 WHERE acl.grantee = bootstrap_oid
           )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_proc AS procedure
         WHERE procedure.proowner = bootstrap_oid
            OR (
                procedure.proacl IS NOT NULL AND EXISTS (
                    SELECT 1 FROM pg_catalog.aclexplode(procedure.proacl) AS acl
                     WHERE acl.grantee = bootstrap_oid
                )
            )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_default_acl AS defaults
         WHERE defaults.defaclrole = bootstrap_oid
            OR EXISTS (
                SELECT 1 FROM pg_catalog.aclexplode(defaults.defaclacl) AS acl
                 WHERE acl.grantee = bootstrap_oid
            )
    ) THEN
        RAISE EXCEPTION 'phase-b existing bootstrap object authority is drifted';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_auth_members AS membership
         WHERE membership.roleid = bootstrap_oid
            OR membership.member = bootstrap_oid
    ) THEN
        PERFORM pg_catalog.set_config(
            'muncho.canonical_writer_phase_b_role_outcome',
            'adopted_zero_membership',
            true
        );
    ELSIF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS member
            ON member.oid = membership.member
          JOIN pg_catalog.pg_roles AS grantor
            ON grantor.oid = membership.grantor
         WHERE membership.roleid = bootstrap_oid
           AND member.rolname = SESSION_USER
           AND grantor.rolname = 'cloudsqladmin'
           AND membership.admin_option
           AND NOT membership.inherit_option
           AND NOT membership.set_option
    ) = 1 AND (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
         WHERE membership.roleid = bootstrap_oid
            OR membership.member = bootstrap_oid
    ) = 1 THEN
        PERFORM pg_catalog.set_config(
            'muncho.canonical_writer_phase_b_role_outcome',
            'adopted_same_admin_predelete',
            true
        );
    ELSE
        RAISE EXCEPTION 'phase-b existing bootstrap membership is partial or foreign';
    END IF;
END
$phase_b_role_preflight$;

DO $phase_b_role_create_if_missing$
BEGIN
    IF pg_catalog.current_setting(
        'muncho.canonical_writer_phase_b_role_outcome'
    ) = 'created' THEN
        EXECUTE $create_role$
            CREATE ROLE canonical_brain_canary_bootstrap
                NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
                NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1
        $create_role$;
    END IF;
END
$phase_b_role_create_if_missing$;

-- Object privileges are granted only while acting as the exact database
-- owner.  The resulting ACL is owned by cloudsqlsuperuser, not by the
-- temporary administrator that Cloud SQL will delete.
SET LOCAL ROLE cloudsqlsuperuser;
DO $phase_b_role_connect_if_created$
BEGIN
    IF pg_catalog.current_setting(
        'muncho.canonical_writer_phase_b_role_outcome'
    ) = 'created' THEN
        EXECUTE $grant_connect$
            GRANT CONNECT ON DATABASE muncho_canary_brain
                TO canonical_brain_canary_bootstrap
        $grant_connect$;
    END IF;
END
$phase_b_role_connect_if_created$;
RESET ROLE;

DO $phase_b_role_postcondition$
DECLARE
    bootstrap_oid oid;
    database_owner_oid oid;
    outcome text := pg_catalog.current_setting(
        'muncho.canonical_writer_phase_b_role_outcome'
    );
BEGIN
    IF CURRENT_USER <> SESSION_USER
       OR outcome NOT IN (
            'created',
            'adopted_same_admin_predelete',
            'adopted_zero_membership'
       ) THEN
        RAISE EXCEPTION 'phase-b role execution identity drifted';
    END IF;
    SELECT role.oid INTO STRICT bootstrap_oid
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname = 'canonical_brain_canary_bootstrap'
       AND NOT role.rolcanlogin
       AND NOT role.rolinherit
       AND NOT role.rolsuper
       AND NOT role.rolcreatedb
       AND NOT role.rolcreaterole
       AND NOT role.rolreplication
       AND NOT role.rolbypassrls
       AND role.rolconnlimit = -1
       AND role.rolvaliduntil IS NULL
       AND role.rolconfig IS NULL;
    SELECT database.datdba INTO STRICT database_owner_oid
      FROM pg_catalog.pg_database AS database
     WHERE database.datname = pg_catalog.current_database();

    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  database.datacl,
                  pg_catalog.acldefault('d', database.datdba)
              )
          ) AS acl
         WHERE database.datname = pg_catalog.current_database()
           AND acl.grantee = bootstrap_oid
           AND acl.grantor = database_owner_oid
           AND acl.privilege_type = 'CONNECT'
           AND NOT acl.is_grantable
    ) <> 1 OR EXISTS (
        SELECT 1
          FROM pg_catalog.pg_database AS database
          CROSS JOIN LATERAL pg_catalog.aclexplode(
              COALESCE(
                  database.datacl,
                  pg_catalog.acldefault('d', database.datdba)
              )
          ) AS acl
         WHERE acl.grantee = bootstrap_oid
           AND NOT (
                database.datname = pg_catalog.current_database()
                AND acl.grantor = database_owner_oid
                AND acl.privilege_type = 'CONNECT'
                AND NOT acl.is_grantable
           )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_database AS database
         WHERE database.datallowconn
           AND database.datname NOT IN (
                pg_catalog.current_database(), 'cloudsqladmin'
           )
           AND (
                pg_catalog.has_database_privilege(
                    bootstrap_oid, database.oid, 'CONNECT'
                ) OR pg_catalog.has_database_privilege(
                    bootstrap_oid, database.oid, 'TEMPORARY'
                )
           )
    ) THEN
        RAISE EXCEPTION 'phase-b bootstrap CONNECT postcondition failed';
    END IF;
    IF (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_shdepend AS dependency
         WHERE dependency.refclassid = 'pg_catalog.pg_authid'::regclass
           AND dependency.refobjid = bootstrap_oid
    ) <> 1 OR NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_shdepend AS dependency
         WHERE dependency.refclassid = 'pg_catalog.pg_authid'::regclass
           AND dependency.refobjid = bootstrap_oid
           AND dependency.dbid = 0
           AND dependency.classid = 'pg_catalog.pg_database'::regclass
           AND dependency.objid = (
                SELECT database.oid FROM pg_catalog.pg_database AS database
                 WHERE database.datname = pg_catalog.current_database()
           )
           AND dependency.objsubid = 0
           AND dependency.deptype = 'a'
    ) THEN
        RAISE EXCEPTION 'phase-b bootstrap shared dependency postcondition failed';
    END IF;

    IF outcome IN ('created', 'adopted_same_admin_predelete') AND (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS granted
            ON granted.oid = membership.roleid
          JOIN pg_catalog.pg_roles AS member
            ON member.oid = membership.member
          JOIN pg_catalog.pg_roles AS grantor
            ON grantor.oid = membership.grantor
         WHERE granted.oid = bootstrap_oid
           AND member.rolname = SESSION_USER
           AND grantor.rolname = 'cloudsqladmin'
           AND membership.admin_option
           AND NOT membership.inherit_option
           AND NOT membership.set_option
    ) <> 1 THEN
        RAISE EXCEPTION 'phase-b automatic creator membership is not exact';
    END IF;
    IF outcome = 'adopted_zero_membership' AND EXISTS (
        SELECT 1 FROM pg_catalog.pg_auth_members AS membership
         WHERE membership.roleid = bootstrap_oid
            OR membership.member = bootstrap_oid
    ) THEN
        RAISE EXCEPTION 'phase-b adopted role acquired a membership';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS grantor
            ON grantor.oid = membership.grantor
         WHERE (membership.roleid = bootstrap_oid
                OR membership.member = bootstrap_oid)
           AND grantor.rolname = SESSION_USER
    ) OR (
        SELECT pg_catalog.count(*)
          FROM pg_catalog.pg_auth_members AS membership
         WHERE membership.roleid = bootstrap_oid
            OR membership.member = bootstrap_oid
    ) <> (CASE
        WHEN outcome IN ('created', 'adopted_same_admin_predelete') THEN 1
        ELSE 0
    END) THEN
        RAISE EXCEPTION 'phase-b bootstrap membership postcondition failed';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_namespace AS namespace
         WHERE namespace.nspowner = bootstrap_oid
            OR (
                namespace.nspacl IS NOT NULL AND EXISTS (
                    SELECT 1 FROM pg_catalog.aclexplode(namespace.nspacl) AS acl
                     WHERE acl.grantee = bootstrap_oid
                )
            )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_class AS class
         WHERE class.relowner = bootstrap_oid
            OR (
                class.relacl IS NOT NULL AND EXISTS (
                    SELECT 1 FROM pg_catalog.aclexplode(class.relacl) AS acl
                     WHERE acl.grantee = bootstrap_oid
                )
            )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_attribute AS attribute
         WHERE attribute.attacl IS NOT NULL
           AND EXISTS (
                SELECT 1 FROM pg_catalog.aclexplode(attribute.attacl) AS acl
                 WHERE acl.grantee = bootstrap_oid
           )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_proc AS procedure
         WHERE procedure.proowner = bootstrap_oid
            OR (
                procedure.proacl IS NOT NULL AND EXISTS (
                    SELECT 1 FROM pg_catalog.aclexplode(procedure.proacl) AS acl
                     WHERE acl.grantee = bootstrap_oid
                )
            )
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_default_acl AS defaults
         WHERE defaults.defaclrole = bootstrap_oid
            OR EXISTS (
                SELECT 1 FROM pg_catalog.aclexplode(defaults.defaclacl) AS acl
                 WHERE acl.grantee = bootstrap_oid
            )
    ) THEN
        RAISE EXCEPTION 'phase-b bootstrap received object authority';
    END IF;
END
$phase_b_role_postcondition$;

WITH receipt AS (
    SELECT pg_catalog.jsonb_build_object(
        'schema',
            'muncho-canonical-writer-foundation-phase-b-role-preterminal.v1',
        'phase', 'phase_b_role_and_connect',
        'preterminal', true,
        'database', pg_catalog.current_database(),
        'postgres_version_num',
            pg_catalog.current_setting('server_version_num')::integer,
        'session_user', SESSION_USER,
        'role', 'canonical_brain_canary_bootstrap',
        'role_outcome', pg_catalog.current_setting(
            'muncho.canonical_writer_phase_b_role_outcome'
        ),
        'role_contract', pg_catalog.jsonb_build_object(
            'can_login', false,
            'inherits', false,
            'superuser', false,
            'create_database', false,
            'create_role', false,
            'replication', false,
            'bypass_row_security', false,
            'connection_limit', -1,
            'validity_is_unbounded', true,
            'configuration_is_empty', true
        ),
        'connect_contract', pg_catalog.jsonb_build_object(
            'database', 'muncho_canary_brain',
            'privilege', 'CONNECT',
            'grantor', 'cloudsqlsuperuser',
            'grantable', false,
            'managed_cloudsqladmin_hba_boundary_separate', true
        ),
        'temporary_auto_membership', CASE
            WHEN pg_catalog.current_setting(
                'muncho.canonical_writer_phase_b_role_outcome'
            ) IN (
                'created', 'adopted_same_admin_predelete'
            ) THEN pg_catalog.jsonb_build_object(
                'granted_role', 'canonical_brain_canary_bootstrap',
                'member_role', SESSION_USER,
                'grantor', 'cloudsqladmin',
                'admin_option', true,
                'inherit_option', false,
                'set_option', false
            )
            ELSE NULL
        END,
        'temporary_admin_delete_required', true,
        'release_revision', pg_catalog.current_setting(
            'muncho.canonical_writer_phase_b_release_revision'
        ),
        'artifact_sha256', pg_catalog.current_setting(
            'muncho.canonical_writer_phase_b_role_artifact_sha256'
        ),
        'initial_observation_sha256', pg_catalog.current_setting(
            'muncho.canonical_writer_phase_b_initial_observation_sha256'
        ),
        'approved_plan_sha256', pg_catalog.current_setting(
            'muncho.canonical_writer_phase_b_approved_plan_sha256'
        ),
        'secret_material_recorded', false
    ) AS value
)
SELECT (
    receipt.value || pg_catalog.jsonb_build_object(
        'receipt_sha256', pg_catalog.encode(
            pg_catalog.sha256(
                pg_catalog.convert_to(receipt.value::text, 'UTF8')
            ),
            'hex'
        )
    )
)::text AS phase_b_role_receipt
FROM receipt;

COMMIT;
