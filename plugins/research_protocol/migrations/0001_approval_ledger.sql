-- Research Protocol PR2: one-shot approval ledger and fixed security-definer APIs.
-- Apply once as a trusted migration owner. Existing objects make this migration
-- fail rather than silently accepting an unknown or older schema.

BEGIN;

CREATE SCHEMA research_protocol;
REVOKE ALL ON SCHEMA research_protocol FROM PUBLIC;

CREATE TABLE research_protocol.approvals (
    approval_id text PRIMARY KEY,
    scope_sha256 character(64) NOT NULL,
    plan_sha256 character(64) NOT NULL,
    scope_json jsonb NOT NULL,
    verdict text NOT NULL,
    surface text NOT NULL,
    created_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    last_consumed_at timestamptz,
    max_executions integer NOT NULL,
    consumed_count integer NOT NULL DEFAULT 0,
    CONSTRAINT approvals_id_length
        CHECK (char_length(approval_id) BETWEEN 22 AND 256),
    CONSTRAINT approvals_scope_sha256_hex
        CHECK (scope_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT approvals_plan_sha256_hex
        CHECK (plan_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT approvals_scope_json_object
        CHECK (jsonb_typeof(scope_json) = 'object'),
    CONSTRAINT approvals_scope_required_keys
        CHECK (
            scope_json ?& ARRAY[
                'plan_sha256',
                'max_executions',
                'expires_at',
                'run_id',
                'budgets'
            ]
            AND jsonb_typeof(scope_json -> 'budgets') = 'object'
            AND (scope_json -> 'budgets') ? 'max_executions'
            AND scope_json ->> 'plan_sha256' IS NOT NULL
            AND scope_json ->> 'max_executions' IS NOT NULL
            AND scope_json ->> 'expires_at' IS NOT NULL
            AND scope_json ->> 'run_id' IS NOT NULL
            AND scope_json -> 'budgets' ->> 'max_executions' IS NOT NULL
        ),
    CONSTRAINT approvals_scope_json_size
        CHECK (octet_length(scope_json::text) <= 1000000),
    CONSTRAINT approvals_scope_plan_bound
        CHECK (scope_json ->> 'plan_sha256' = plan_sha256::text),
    CONSTRAINT approvals_scope_execution_bound
        CHECK (scope_json ->> 'max_executions' = max_executions::text),
    CONSTRAINT approvals_scope_budget_bound
        CHECK (
            (scope_json -> 'budgets' ->> 'max_executions')::integer
            >= max_executions
        ),
    CONSTRAINT approvals_scope_expiry_bound
        CHECK ((scope_json ->> 'expires_at')::timestamptz = expires_at),
    CONSTRAINT approvals_scope_run_id_bound
        CHECK (
            char_length(scope_json ->> 'run_id') BETWEEN 1 AND 128
            AND scope_json ->> 'run_id' ~ '^[A-Za-z0-9][A-Za-z0-9._-]*$'
        ),
    CONSTRAINT approvals_verdict_closed
        CHECK (verdict IN ('approved', 'denied')),
    CONSTRAINT approvals_surface_length
        CHECK (char_length(surface) BETWEEN 1 AND 128),
    CONSTRAINT approvals_max_executions_positive
        CHECK (max_executions > 0),
    CONSTRAINT approvals_consumed_count_nonnegative
        CHECK (consumed_count >= 0),
    CONSTRAINT approvals_consumed_within_cap
        CHECK (consumed_count <= max_executions),
    CONSTRAINT approvals_denied_unconsumed
        CHECK (verdict = 'approved' OR consumed_count = 0),
    CONSTRAINT approvals_expiry_after_creation
        CHECK (verdict = 'denied' OR expires_at > created_at),
    CONSTRAINT approvals_consumed_timestamp_consistent
        CHECK (
            (consumed_count = 0 AND last_consumed_at IS NULL)
            OR (consumed_count > 0 AND last_consumed_at IS NOT NULL)
        )
);

CREATE INDEX approvals_active_expiry_idx
    ON research_protocol.approvals (expires_at)
    WHERE verdict = 'approved' AND consumed_count < max_executions;

REVOKE ALL ON research_protocol.approvals FROM PUBLIC;

CREATE FUNCTION research_protocol.store_approval(
    p_approval_id text,
    p_scope_sha256 text,
    p_plan_sha256 text,
    p_scope_json text,
    p_verdict text,
    p_surface text,
    p_created_at timestamptz,
    p_expires_at timestamptz,
    p_max_executions integer
)
RETURNS text
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, research_protocol
AS $store$
DECLARE
    v_scope_json jsonb;
BEGIN
    v_scope_json := p_scope_json::jsonb;

    IF p_approval_id IS NULL
       OR p_scope_sha256 IS NULL
       OR p_plan_sha256 IS NULL
       OR p_scope_json IS NULL
       OR p_verdict IS NULL
       OR p_surface IS NULL
       OR p_created_at IS NULL
       OR p_expires_at IS NULL
       OR p_max_executions IS NULL
       OR NOT (
           v_scope_json ?& ARRAY[
               'plan_sha256',
               'max_executions',
               'expires_at',
               'run_id',
               'budgets'
           ]
           AND jsonb_typeof(v_scope_json -> 'budgets') = 'object'
           AND (v_scope_json -> 'budgets') ? 'max_executions'
           AND v_scope_json ->> 'plan_sha256' IS NOT NULL
           AND v_scope_json ->> 'max_executions' IS NOT NULL
           AND v_scope_json ->> 'expires_at' IS NOT NULL
           AND v_scope_json ->> 'run_id' IS NOT NULL
           AND v_scope_json -> 'budgets' ->> 'max_executions' IS NOT NULL
       )
       OR char_length(p_approval_id) NOT BETWEEN 22 AND 256
       OR p_scope_sha256 !~ '^[0-9a-f]{64}$'
       OR p_plan_sha256 !~ '^[0-9a-f]{64}$'
       OR jsonb_typeof(v_scope_json) <> 'object'
       OR octet_length(p_scope_json) > 1000000
       OR pg_catalog.encode(
           pg_catalog.sha256(
               pg_catalog.convert_to(p_scope_json, 'UTF8')
           ),
           'hex'
       ) <> p_scope_sha256
       OR v_scope_json ->> 'plan_sha256' <> p_plan_sha256
       OR v_scope_json ->> 'max_executions' <> p_max_executions::text
       OR (v_scope_json -> 'budgets' ->> 'max_executions')::integer
          < p_max_executions
       OR (v_scope_json ->> 'expires_at')::timestamptz <> p_expires_at
       OR char_length(v_scope_json ->> 'run_id') NOT BETWEEN 1 AND 128
       OR v_scope_json ->> 'run_id' !~ '^[A-Za-z0-9][A-Za-z0-9._-]*$'
       OR p_verdict NOT IN ('approved', 'denied')
       OR char_length(p_surface) NOT BETWEEN 1 AND 128
       OR p_max_executions <= 0
       OR p_created_at > clock_timestamp()
       OR (
           p_verdict = 'approved'
           AND (
               p_expires_at <= clock_timestamp()
               OR p_expires_at <= p_created_at
           )
       )
    THEN
        RAISE EXCEPTION 'invalid approval record' USING ERRCODE = '22023';
    END IF;

    INSERT INTO research_protocol.approvals (
        approval_id,
        scope_sha256,
        plan_sha256,
        scope_json,
        verdict,
        surface,
        created_at,
        expires_at,
        max_executions,
        consumed_count
    ) VALUES (
        p_approval_id,
        p_scope_sha256::character(64),
        p_plan_sha256::character(64),
        v_scope_json,
        p_verdict,
        p_surface,
        p_created_at,
        p_expires_at,
        p_max_executions,
        0
    );

    RETURN p_approval_id;
END
$store$;

REVOKE ALL ON FUNCTION research_protocol.store_approval(
    text, text, text, text, text, text, timestamptz, timestamptz, integer
) FROM PUBLIC;

CREATE FUNCTION research_protocol.claim_approval(
    p_approval_id text,
    p_scope_sha256 text,
    p_plan_sha256 text
)
RETURNS TABLE (
    approval_id text,
    scope_sha256 text,
    plan_sha256 text,
    consumed_count integer,
    max_executions integer
)
LANGUAGE sql
SECURITY DEFINER
SET search_path = pg_catalog, research_protocol
AS $claim$
    UPDATE research_protocol.approvals AS approval
    SET consumed_count = approval.consumed_count + 1,
        last_consumed_at = clock_timestamp()
    WHERE approval.approval_id = p_approval_id
      AND approval.scope_sha256::text = p_scope_sha256
      AND approval.plan_sha256::text = p_plan_sha256
      AND p_scope_sha256 ~ '^[0-9a-f]{64}$'
      AND p_plan_sha256 ~ '^[0-9a-f]{64}$'
      AND approval.verdict = 'approved'
      AND approval.expires_at > clock_timestamp()
      AND approval.consumed_count < approval.max_executions
    RETURNING approval.approval_id,
              approval.scope_sha256::text,
              approval.plan_sha256::text,
              approval.consumed_count,
              approval.max_executions
$claim$;

REVOKE ALL ON FUNCTION research_protocol.claim_approval(text, text, text)
    FROM PUBLIC;

COMMIT;
