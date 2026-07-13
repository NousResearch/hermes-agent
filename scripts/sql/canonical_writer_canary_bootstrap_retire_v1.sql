-- Idempotent owner-operated reconciliation for isolated-canary bootstrap ACL.
--
-- The root runtime supplies the exact eleven provisioning bindings plus three
-- mechanical retirement-only receipt digests.  No reason or semantic choice
-- is caller-controlled: the only retirement reason is fixed below.
--
-- Original eleven GUCs:
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
--
-- Retirement-only mechanical bindings:
-- muncho.canonical_canary_bootstrap_plan_sha256
-- muncho.canonical_canary_bootstrap_owner_approval_sha256
-- muncho.canonical_canary_bootstrap_executor_session_identity_sha256

BEGIN;

SELECT pg_catalog.pg_advisory_xact_lock(4841739663211427921);

DO $bootstrap_retire_prerequisites$
DECLARE
    admin_name text := SESSION_USER;
    grant_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_grant_id', true
    );
    case_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_case_id', true
    );
    run_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_run_id', true
    );
    approved_by_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_approved_by', true
    );
    expiry_value timestamptz;
BEGIN
    IF pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_database', true
       ) IS DISTINCT FROM pg_catalog.current_database()
       OR grant_id_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR case_id_value !~ '^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR run_id_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR approved_by_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$'
       OR pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_release_sha256', true
          ) !~ '^[0-9a-f]{64}$'
       OR pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_fixture_sha256', true
          ) !~ '^[0-9a-f]{64}$'
       OR pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_session_key_sha256', true
          ) !~ '^[0-9a-f]{64}$'
       OR pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_approval_source_sha256', true
          ) !~ '^[0-9a-f]{64}$'
       OR pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_provisioning_receipt_sha256', true
          ) !~ '^[0-9a-f]{64}$'
       OR pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_plan_sha256', true
          ) !~ '^[0-9a-f]{64}$'
       OR pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_owner_approval_sha256', true
          ) !~ '^[0-9a-f]{64}$'
       OR pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_executor_session_identity_sha256',
            true
          ) !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'canary bootstrap retirement binding is invalid';
    END IF;
    BEGIN
        expiry_value := pg_catalog.current_setting(
            'muncho.canonical_canary_bootstrap_expires_at', true
        )::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'canary bootstrap retirement expiry is invalid';
    END;
    IF expiry_value IS NULL THEN
        RAISE EXCEPTION 'canary bootstrap retirement expiry is missing';
    END IF;
    IF admin_name IN (
        'canonical_brain_migration_owner',
        'canonical_brain_writer',
        'canonical_brain_canary_bootstrap',
        'canonical_brain_canary_bootstrap_login'
    ) THEN
        RAISE EXCEPTION 'bootstrap retirement provisioner login is forbidden';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
         WHERE rolname = 'canonical_brain_migration_owner'
           AND NOT rolcanlogin AND NOT rolinherit AND NOT rolsuper
           AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication
           AND NOT rolbypassrls
    ) OR NOT EXISTS (
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
        RAISE EXCEPTION 'canary bootstrap retirement role identity is invalid';
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
                AND NOT membership.set_option
           )
    ) THEN
        RAISE EXCEPTION 'canary bootstrap retirement membership is invalid';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS owner_role
            ON owner_role.rolname = 'canonical_brain_migration_owner'
         WHERE membership.roleid = owner_role.oid
            OR membership.member = owner_role.oid
    ) THEN
        RAISE EXCEPTION 'migration owner must be offline before retirement';
    END IF;
    PERFORM pg_catalog.set_config(
        'muncho.canonical_canary_bootstrap_retire_admin', admin_name, true
    );
    EXECUTE pg_catalog.format(
        'GRANT canonical_brain_migration_owner TO %I '
        'WITH ADMIN FALSE, INHERIT FALSE, SET TRUE',
        admin_name
    );
END
$bootstrap_retire_prerequisites$;

SET LOCAL ROLE canonical_brain_migration_owner;

DO $bootstrap_retire_reconcile$
DECLARE
    grant_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_grant_id'
    );
    case_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_case_id'
    );
    release_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_release_sha256'
    );
    fixture_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_fixture_sha256'
    );
    run_id_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_run_id'
    );
    session_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_session_key_sha256'
    );
    expiry_value timestamptz := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_expires_at'
    )::timestamptz;
    approved_by_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_approved_by'
    );
    approval_source_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_approval_source_sha256'
    );
    provisioning_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_provisioning_receipt_sha256'
    );
    plan_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_plan_sha256'
    );
    owner_approval_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_owner_approval_sha256'
    );
    executor_value text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_executor_session_identity_sha256'
    );
    authorization_count bigint;
    authorization_event_id text;
    authorization_record jsonb;
    consumption_count bigint;
    consumption_event_id text;
    consumption_record jsonb;
    retirement_count bigint;
    retirement_event_id text;
    retirement_record jsonb;
    preapproval_count bigint;
    preapproval_record canonical_brain.writer_canary_scope_preapprovals%ROWTYPE;
    append_result jsonb;
    retirement_payload jsonb;
BEGIN
    SELECT pg_catalog.count(*), pg_catalog.max(event.event_id::text),
           pg_catalog.max(
               (event.payload->'canary_scope_bootstrap_authorization')::text
           )::jsonb
      INTO authorization_count, authorization_event_id, authorization_record
      FROM public.canonical_event_log AS event
      JOIN canonical_brain.writer_event_provenance AS provenance
        ON provenance.event_id = event.event_id
     WHERE event.event_type = 'canary.scope.bootstrap_authorized'
       AND provenance.origin = 'canary_scope_bootstrap_provision'
       AND (
            event.case_id = case_id_value
            OR event.payload->'canary_scope_bootstrap_authorization'->>'grant_id'
                = grant_id_value
       );

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
            event.case_id = case_id_value
            OR event.payload->'canary_scope_bootstrap_consumption'->>'grant_id'
                = grant_id_value
       );
    SELECT pg_catalog.count(*), pg_catalog.max(event.event_id::text),
           pg_catalog.max(
               (event.payload->'canary_scope_bootstrap_retirement')::text
           )::jsonb
      INTO retirement_count, retirement_event_id, retirement_record
      FROM public.canonical_event_log AS event
      JOIN canonical_brain.writer_event_provenance AS provenance
        ON provenance.event_id = event.event_id
     WHERE event.event_type = 'canary.scope.bootstrap_retired'
       AND provenance.origin = 'canary_scope_bootstrap_retire'
       AND (
            event.case_id = case_id_value
            OR event.payload->'canary_scope_bootstrap_retirement'->>'grant_id'
                = grant_id_value
       );
    SELECT pg_catalog.count(*)
      INTO preapproval_count
      FROM canonical_brain.writer_canary_scope_preapprovals AS preapproval
     WHERE preapproval.grant_id = grant_id_value
        OR preapproval.case_id = case_id_value;
    IF preapproval_count = 1 THEN
        SELECT * INTO STRICT preapproval_record
          FROM canonical_brain.writer_canary_scope_preapprovals AS preapproval
         WHERE preapproval.grant_id = grant_id_value
            OR preapproval.case_id = case_id_value;
    END IF;
    IF authorization_count > 1 OR consumption_count > 1 OR retirement_count > 1
       OR (consumption_count = 1 AND retirement_count = 1) THEN
        RAISE EXCEPTION 'canary bootstrap terminal truth is contradictory';
    END IF;

    EXECUTE
        'REVOKE EXECUTE ON FUNCTION '
        'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb) '
        'FROM canonical_brain_canary_bootstrap';
    EXECUTE
        'REVOKE USAGE ON SCHEMA canonical_brain '
        'FROM canonical_brain_canary_bootstrap';
    IF pg_catalog.has_schema_privilege(
        'canonical_brain_canary_bootstrap', 'canonical_brain', 'USAGE'
    ) OR pg_catalog.has_function_privilege(
        'canonical_brain_canary_bootstrap',
        'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)',
        'EXECUTE'
    ) OR pg_catalog.has_schema_privilege(
        'canonical_brain_canary_bootstrap_login', 'canonical_brain', 'USAGE'
    ) OR pg_catalog.has_function_privilege(
        'canonical_brain_canary_bootstrap_login',
        'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)',
        'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'canary bootstrap ACL retirement failed';
    END IF;

    IF authorization_count = 0 THEN
        IF consumption_count <> 0 OR retirement_count <> 0
           OR preapproval_count <> 0 THEN
            RAISE EXCEPTION
                'unauthorized canary bootstrap has contradictory durable truth';
        END IF;
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_outcome',
            'not_authorized', true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_reason',
            'provisioning_not_committed', true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_authorization_event_id',
            '', true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_consumption_event_id',
            '', true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_retirement_event_id',
            '', true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_inserted', 'false', true
        );
        RETURN;
    END IF;

    IF NOT canonical_brain._keys_valid(
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
       OR authorization_record->>'grant_id' <> grant_id_value
       OR authorization_record->>'case_id' <> case_id_value
       OR authorization_record->>'release_sha256' <> release_value
       OR authorization_record->>'fixture_sha256' <> fixture_value
       OR authorization_record->>'run_id' <> run_id_value
       OR authorization_record->>'session_key_sha256' <> session_value
       OR authorization_record->>'approved_by' <> approved_by_value
       OR authorization_record->>'approval_source_sha256'
            <> approval_source_value
       OR authorization_record->>'provisioning_receipt_sha256'
            <> provisioning_value
       OR authorization_record->>'bootstrap_login'
            <> 'canonical_brain_canary_bootstrap_login'
       OR authorization_record->>'state' <> 'authorized'
       OR (authorization_record->>'expires_at')::timestamptz
            IS DISTINCT FROM expiry_value THEN
        RAISE EXCEPTION
            'exact canary bootstrap authorization is unavailable for retirement';
    END IF;

    IF consumption_count = 1 THEN
        IF preapproval_count <> 1
           OR NOT canonical_brain._keys_valid(
                consumption_record,
                ARRAY['grant_id','case_id','provisioning_receipt_sha256',
                      'preapproval_event_id','state'],
                ARRAY['grant_id','case_id','provisioning_receipt_sha256',
                      'preapproval_event_id','state']
           )
           OR consumption_record->>'grant_id' <> grant_id_value
           OR consumption_record->>'case_id' <> case_id_value
           OR consumption_record->>'provisioning_receipt_sha256'
                <> provisioning_value
           OR consumption_record->>'state' <> 'consumed'
           OR preapproval_record.grant_id <> grant_id_value
           OR preapproval_record.case_id <> case_id_value
           OR preapproval_record.release_sha256 <> release_value
           OR preapproval_record.fixture_sha256 <> fixture_value
           OR preapproval_record.run_id <> run_id_value
           OR preapproval_record.session_key_sha256 <> session_value
           OR preapproval_record.expires_at IS DISTINCT FROM expiry_value
           OR preapproval_record.approved_by <> approved_by_value
           OR preapproval_record.approval_source_sha256 <> approval_source_value
           OR preapproval_record.receipt_event_id::text
                <> consumption_record->>'preapproval_event_id' THEN
            RAISE EXCEPTION
                'consumed canary bootstrap terminal binding is invalid';
        END IF;
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_outcome',
            'consumed', true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_reason',
            'bootstrap_consumed', true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_authorization_event_id',
            authorization_event_id, true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_consumption_event_id',
            consumption_event_id, true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_retirement_event_id',
            '', true
        );
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_inserted', 'false', true
        );
        RETURN;
    END IF;

    IF preapproval_count <> 0 THEN
        RAISE EXCEPTION
            'unconsumed canary bootstrap unexpectedly has a preapproval row';
    END IF;
    retirement_payload := pg_catalog.jsonb_build_object(
        'grant_id', grant_id_value,
        'case_id', case_id_value,
        'release_sha256', release_value,
        'fixture_sha256', fixture_value,
        'run_id', run_id_value,
        'session_key_sha256', session_value,
        'expires_at', expiry_value,
        'approved_by', approved_by_value,
        'approval_source_sha256', approval_source_value,
        'provisioning_receipt_sha256', provisioning_value,
        'authorization_event_id', authorization_event_id,
        'plan_sha256', plan_value,
        'owner_approval_sha256', owner_approval_value,
        'executor_session_identity_sha256', executor_value,
        'reason', 'activation_failed_before_consumption',
        'acl_revoked', true
    );
    IF retirement_count = 1 THEN
        IF NOT canonical_brain._keys_valid(
                retirement_record,
                ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                      'run_id','session_key_sha256','expires_at','approved_by',
                      'approval_source_sha256','provisioning_receipt_sha256',
                      'authorization_event_id','plan_sha256',
                      'owner_approval_sha256',
                      'executor_session_identity_sha256','reason','acl_revoked'],
                ARRAY['grant_id','case_id','release_sha256','fixture_sha256',
                      'run_id','session_key_sha256','expires_at','approved_by',
                      'approval_source_sha256','provisioning_receipt_sha256',
                      'authorization_event_id','plan_sha256',
                      'owner_approval_sha256',
                      'executor_session_identity_sha256','reason','acl_revoked']
           )
           OR retirement_record->>'executor_session_identity_sha256'
                !~ '^[0-9a-f]{64}$'
           OR retirement_record - 'executor_session_identity_sha256'
                IS DISTINCT FROM
                retirement_payload - 'executor_session_identity_sha256' THEN
            RAISE EXCEPTION
                'existing canary bootstrap retirement binding is invalid';
        END IF;
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_inserted', 'false', true
        );
    ELSE
        append_result := canonical_brain._append_event(
            'canary.scope.bootstrap_retired',
            case_id_value,
            'Unconsumed isolated canary bootstrap authority retired',
            pg_catalog.jsonb_build_object(
                'manual_ref', 'canary-bootstrap-retire:' || grant_id_value
            ),
            pg_catalog.jsonb_build_object(
                'actor', pg_catalog.jsonb_build_object(
                    'type', 'owner_approved_reconciliation',
                    'id', approved_by_value
                ),
                'subject', pg_catalog.jsonb_build_object(
                    'type', 'canary_scope', 'id', grant_id_value
                )
            ),
            pg_catalog.jsonb_build_object(
                'canary_scope_bootstrap_retirement', retirement_payload
            ),
            pg_catalog.jsonb_build_object(
                'isolated_canary', true,
                'compensating_cleanup', true,
                'activation_failed_before_consumption', true
            ),
            'canary-bootstrap-retire:' || grant_id_value,
            'canary_scope_bootstrap_retire',
            pg_catalog.jsonb_build_object(
                'request_id', 'canary-bootstrap-retire:' || grant_id_value,
                'platform', 'writer_reconciliation',
                'session_key_sha256', session_value,
                'service_internal', true
            )
        );
        IF NOT (append_result->>'ok')::boolean THEN
            RAISE EXCEPTION 'canary bootstrap retirement append failed';
        END IF;
        retirement_event_id := append_result->'result'->>'event_id';
        PERFORM pg_catalog.set_config(
            'muncho.canonical_canary_bootstrap_retire_inserted', 'true', true
        );
    END IF;
    PERFORM pg_catalog.set_config(
        'muncho.canonical_canary_bootstrap_retire_outcome', 'retired', true
    );
    PERFORM pg_catalog.set_config(
        'muncho.canonical_canary_bootstrap_retire_reason',
        'activation_failed_before_consumption', true
    );
    PERFORM pg_catalog.set_config(
        'muncho.canonical_canary_bootstrap_retire_authorization_event_id',
        authorization_event_id, true
    );
    PERFORM pg_catalog.set_config(
        'muncho.canonical_canary_bootstrap_retire_consumption_event_id', '', true
    );
    PERFORM pg_catalog.set_config(
        'muncho.canonical_canary_bootstrap_retire_retirement_event_id',
        retirement_event_id, true
    );
END
$bootstrap_retire_reconcile$;

RESET ROLE;

DO $bootstrap_retire_owner$
DECLARE
    admin_name text := pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_retire_admin'
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
        RAISE EXCEPTION 'temporary migration-owner membership survived retirement';
    END IF;
END
$bootstrap_retire_owner$;

SELECT
    pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_retire_outcome'
    ) AS outcome,
    pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_grant_id'
    ) AS grant_id,
    pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_case_id'
    ) AS case_id,
    NULLIF(pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_retire_authorization_event_id'
    ), '') AS authorization_event_id,
    NULLIF(pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_retire_consumption_event_id'
    ), '') AS consumption_event_id,
    NULLIF(pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_retire_retirement_event_id'
    ), '') AS retirement_event_id,
    pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_retire_inserted'
    )::boolean AS retired_inserted,
    pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_plan_sha256'
    ) AS plan_sha256,
    pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_owner_approval_sha256'
    ) AS owner_approval_sha256,
    pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_executor_session_identity_sha256'
    ) AS executor_session_identity_sha256,
    pg_catalog.current_setting(
        'muncho.canonical_canary_bootstrap_retire_reason'
    ) AS reason,
    NOT pg_catalog.has_schema_privilege(
        'canonical_brain_canary_bootstrap', 'canonical_brain', 'USAGE'
    ) AND NOT pg_catalog.has_function_privilege(
        'canonical_brain_canary_bootstrap',
        'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)',
        'EXECUTE'
    ) AND NOT pg_catalog.has_schema_privilege(
        'canonical_brain_canary_bootstrap_login', 'canonical_brain', 'USAGE'
    ) AND NOT pg_catalog.has_function_privilege(
        'canonical_brain_canary_bootstrap_login',
        'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)',
        'EXECUTE'
    ) AS bootstrap_acl_revoked,
    NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_auth_members AS membership
          JOIN pg_catalog.pg_roles AS owner_role
            ON owner_role.rolname = 'canonical_brain_migration_owner'
         WHERE membership.roleid = owner_role.oid
            OR membership.member = owner_role.oid
    ) AS migration_owner_membership_absent;

COMMIT;
