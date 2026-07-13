-- Owner-approved one-shot isolated-canary bootstrap authority.
--
-- This artifact is intentionally separate from canonical_writer_v1.sql.
-- The base migration always leaves the bootstrap role inert; rerunning it
-- revokes any unconsumed bootstrap ACL and never recreates authority.
-- Before BEGIN the root-controlled provisioner must set every binding below:
--
-- muncho.canonical_canary_bootstrap_database
-- muncho.canonical_canary_bootstrap_grant_id
-- muncho.canonical_canary_bootstrap_case_id
-- muncho.canonical_canary_bootstrap_release_sha256
-- muncho.canonical_canary_bootstrap_fixture_sha256
-- muncho.canonical_canary_bootstrap_run_id
-- muncho.canonical_canary_bootstrap_session_key_sha256
-- muncho.canonical_canary_bootstrap_expires_at
-- muncho.canonical_canary_bootstrap_approved_by
-- muncho.canonical_canary_bootstrap_approval_source_sha256
-- muncho.canonical_canary_bootstrap_provisioning_receipt_sha256

BEGIN;

SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

DO $bootstrap_prerequisites$
DECLARE
    grant_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_grant_id', true
    );
    case_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_case_id', true
    );
    release_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_release_sha256', true
    );
    fixture_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_fixture_sha256', true
    );
    run_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_run_id', true
    );
    session_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_session_key_sha256', true
    );
    expiry_value timestamptz;
    approved_by_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_approved_by', true
    );
    approval_source_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_approval_source_sha256', true
    );
    provisioning_receipt_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_provisioning_receipt_sha256', true
    );
BEGIN
    IF pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_database', true
       ) IS DISTINCT FROM pg_catalog.current_database()
       OR grant_id_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR case_id_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR run_id_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR approved_by_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR release_value !~ '^[0-9a-f]{64}$'
       OR fixture_value !~ '^[0-9a-f]{64}$'
       OR session_value !~ '^[0-9a-f]{64}$'
       OR approval_source_value !~ '^[0-9a-f]{64}$'
       OR provisioning_receipt_value !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'canary bootstrap provisioning binding is invalid';
    END IF;
    BEGIN
        expiry_value := pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_expires_at', true
        )::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'canary bootstrap expiry is invalid';
    END;
    IF expiry_value <= pg_catalog.clock_timestamp()
       OR expiry_value > pg_catalog.clock_timestamp() + INTERVAL '1 hour' THEN
        RAISE EXCEPTION 'canary bootstrap expiry must be within one hour';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
         WHERE rolname = 'canonical_brain_canary_bootstrap'
           AND NOT rolcanlogin AND NOT rolinherit AND NOT rolsuper
           AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication
           AND NOT rolbypassrls
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
         WHERE rolname = 'canonical_brain_canary_bootstrap_login'
           AND rolcanlogin AND rolinherit AND NOT rolsuper
           AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication
           AND NOT rolbypassrls
    ) THEN
        RAISE EXCEPTION 'canary bootstrap role/login identity is invalid';
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
           AND NOT membership.set_option
    ) <> 1 THEN
        RAISE EXCEPTION 'canary bootstrap login membership is invalid';
    END IF;
    IF pg_catalog.has_schema_privilege(
        'canonical_brain_canary_bootstrap', 'canonical_brain', 'USAGE'
    ) OR pg_catalog.has_function_privilege(
        'canonical_brain_canary_bootstrap',
        'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)',
        'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'stale canary bootstrap ACL already exists';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM canonical_brain.writer_canary_scope_preapprovals AS preapproval
         WHERE preapproval.grant_id = grant_id_value
            OR preapproval.case_id = case_id_value
            OR preapproval.run_id = run_id_value
            OR preapproval.approval_source_sha256 = approval_source_value
    ) OR EXISTS (
        SELECT 1
          FROM public.canonical_event_log AS event
          JOIN canonical_brain.writer_event_provenance AS provenance
            ON provenance.event_id = event.event_id
         WHERE event.event_type IN (
                'canary.scope.bootstrap_authorized',
                'canary.scope.bootstrap_consumed',
                'canary.scope.bootstrap_retired'
         )
           AND (
                event.case_id = case_id_value
                OR event.payload->'canary_scope_bootstrap_authorization'->>'grant_id'
                    = grant_id_value
                OR event.payload->'canary_scope_bootstrap_consumption'->>'grant_id'
                    = grant_id_value
                OR event.payload->'canary_scope_bootstrap_retirement'->>'grant_id'
                    = grant_id_value
                OR event.payload->'canary_scope_bootstrap_authorization'
                       ->>'provisioning_receipt_sha256'
                    = provisioning_receipt_value
                OR event.payload->'canary_scope_bootstrap_retirement'
                       ->>'provisioning_receipt_sha256'
                    = provisioning_receipt_value
           )
    ) THEN
        RAISE EXCEPTION 'canary bootstrap authorization is not fresh and one-shot';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS owner_role
            ON owner_role.rolname = 'canonical_brain_migration_owner'
         WHERE membership.roleid = owner_role.oid
            OR membership.member = owner_role.oid
    ) THEN
        RAISE EXCEPTION 'migration owner must be offline before provisioning';
    END IF;
END
$bootstrap_prerequisites$;

DO $bootstrap_temporary_owner$
DECLARE
    admin_name text := SESSION_USER;
BEGIN
    IF admin_name IN (
        'canonical_brain_migration_owner',
        'canonical_brain_writer',
        'canonical_brain_canary_bootstrap',
        'canonical_brain_canary_bootstrap_login'
    ) THEN
        RAISE EXCEPTION 'bootstrap provisioner login is forbidden';
    END IF;
    PERFORM pg_catalog.set_config(
        'muncho.canonical_canary_bootstrap_admin', admin_name, true
    );
    EXECUTE pg_catalog.format(
        'GRANT canonical_brain_migration_owner TO %I '
        'WITH ADMIN FALSE, INHERIT FALSE, SET TRUE',
        admin_name
    );
END
$bootstrap_temporary_owner$;

SET LOCAL ROLE canonical_brain_migration_owner;

DO $bootstrap_authorize$
DECLARE
    grant_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_grant_id'
    );
    case_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_case_id'
    );
    expiry_value timestamptz := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_expires_at'
    )::timestamptz;
    append_result jsonb;
BEGIN
    append_result := canonical_brain._append_event(
        'canary.scope.bootstrap_authorized',
        case_id_value,
        'Owner-approved one-shot isolated canary bootstrap authority',
        pg_catalog.jsonb_build_object(
            'manual_ref', 'canary-bootstrap-authorize:' || grant_id_value
        ),
        pg_catalog.jsonb_build_object(
            'actor', pg_catalog.jsonb_build_object(
                'type', 'owner_approval',
                'id', pg_catalog.current_setting(
                    'muncho.canonical_canary_bootstrap_approved_by'
                )
            ),
            'subject', pg_catalog.jsonb_build_object(
                'type', 'canary_scope', 'id', grant_id_value
            )
        ),
        pg_catalog.jsonb_build_object(
            'canary_scope_bootstrap_authorization',
            pg_catalog.jsonb_build_object(
                'grant_id', grant_id_value,
                'case_id', case_id_value,
                'release_sha256', pg_catalog.current_setting(
                    'muncho.canonical_canary_bootstrap_release_sha256'
                ),
                'fixture_sha256', pg_catalog.current_setting(
                    'muncho.canonical_canary_bootstrap_fixture_sha256'
                ),
                'run_id', pg_catalog.current_setting(
                    'muncho.canonical_canary_bootstrap_run_id'
                ),
                'session_key_sha256', pg_catalog.current_setting(
                    'muncho.canonical_canary_bootstrap_session_key_sha256'
                ),
                'expires_at', expiry_value,
                'approved_by', pg_catalog.current_setting(
                    'muncho.canonical_canary_bootstrap_approved_by'
                ),
                'approval_source_sha256', pg_catalog.current_setting(
                    'muncho.canonical_canary_bootstrap_approval_source_sha256'
                ),
                'provisioning_receipt_sha256', pg_catalog.current_setting(
                    'muncho.canonical_canary_bootstrap_provisioning_receipt_sha256'
                ),
                'bootstrap_login',
                    'canonical_brain_canary_bootstrap_login',
                'state', 'authorized'
            )
        ),
        pg_catalog.jsonb_build_object(
            'isolated_canary', true,
            'owner_approved_one_shot_bootstrap', true
        ),
        'canary-bootstrap-authorize:' || grant_id_value,
        'canary_scope_bootstrap_provision',
        pg_catalog.jsonb_build_object(
            'request_id', 'canary-bootstrap-provision:' || grant_id_value,
            'platform', 'writer_provisioner',
            'session_key_sha256', pg_catalog.current_setting(
                'muncho.canonical_canary_bootstrap_session_key_sha256'
            ),
            'service_internal', true
        )
    );
    IF NOT (append_result->>'ok')::boolean THEN
        RAISE EXCEPTION 'canonical canary bootstrap authorization append failed';
    END IF;
END
$bootstrap_authorize$;

GRANT USAGE ON SCHEMA canonical_brain
    TO canonical_brain_canary_bootstrap;
GRANT EXECUTE ON FUNCTION
    canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)
    TO canonical_brain_canary_bootstrap;

RESET ROLE;

DO $bootstrap_retire_owner$
DECLARE
    admin_name text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_admin'
    );
BEGIN
    EXECUTE pg_catalog.format(
        'REVOKE canonical_brain_migration_owner FROM %I', admin_name
    );
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS owner_role
            ON owner_role.rolname = 'canonical_brain_migration_owner'
         WHERE membership.roleid = owner_role.oid
            OR membership.member = owner_role.oid
    ) THEN
        RAISE EXCEPTION 'temporary migration-owner membership survived';
    END IF;
END
$bootstrap_retire_owner$;

COMMIT;
