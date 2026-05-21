-- migrations/005_skill_recommendations.sql
-- Plan 004-D: LLM-driven skill recommendations
--
-- Prerequisite: migrations/004_skill_drift_alerts.sql must be applied first.
-- Apply: psql "$NEON_DSN_HERMES_APP" -f migrations/005_skill_recommendations.sql
--
-- Design decisions:
-- - skill_recommendations stores one recommended-but-not-yet-created skill per row.
--   Each row represents a cluster of similar tasks that Hermes detected has no
--   matching existing skill.
-- - cluster_examples JSONB: array of {turn_id, summary, timestamp} for the
--   task instances that formed the cluster. Used in the UI to show Blake what
--   tasks were observed before suggesting a skill.
-- - generated_skill_content TEXT: the LLM-drafted SKILL.md content (null until
--   Blake clicks "Generate skill"). Stored here so re-generation is avoidable.
-- - status: 'pending' (suggestion visible in dashboard) → 'generated' (draft
--   produced) → 'dismissed' (Blake rejected).
-- - No auto-push: generated_skill_content is the end state here. Blake
--   commits manually (per D-004-7 decision: editor-only, no auto-commit).
-- - budget_gate: week-level LLM cost tracking is done in-code (BudgetGate).
--   This table does not enforce the $20/mo cap — that's enforced in recommender.py.

CREATE TABLE IF NOT EXISTS skill_recommendations (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id              UUID NOT NULL,
    -- Suggested skill name derived from LLM cluster analysis.
    suggested_skill_name   TEXT NOT NULL,
    -- One-paragraph task summary from the LLM.
    task_summary           TEXT NOT NULL,
    -- JSON array of {turn_id, summary, timestamp} task instances in the cluster.
    cluster_examples       JSONB NOT NULL DEFAULT '[]',
    -- Number of cluster instances observed in the detection window.
    cluster_size           INT NOT NULL DEFAULT 0,
    -- LLM-drafted SKILL.md content (null until "Generate skill" is clicked).
    generated_skill_content TEXT,
    -- "pending" | "generated" | "dismissed"
    status                 TEXT NOT NULL DEFAULT 'pending'
                               CHECK (status IN ('pending', 'generated', 'dismissed')),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    generated_at           TIMESTAMPTZ,
    dismissed_at           TIMESTAMPTZ,
    -- Approximate LLM cost (USD) for analysis + generation of this recommendation.
    llm_cost_usd           NUMERIC(8,6),
    -- Monthly budget tracking: which month this was billed to.
    billed_month           CHAR(7)  -- YYYY-MM format
);

CREATE INDEX IF NOT EXISTS skill_recommendations_tenant_status_idx
    ON skill_recommendations (tenant_id, status, created_at DESC);

COMMENT ON TABLE skill_recommendations IS
    'LLM-detected task clusters with no matching existing skill. '
    'One row per recommended-but-not-created skill. '
    'generated_skill_content is set when Blake clicks "Generate skill" in the dashboard.';

-- Monthly LLM cost tracking for budget gate enforcement.
-- Separate from skill_recommendations to allow fast aggregation without
-- scanning the full recommendations table.
CREATE TABLE IF NOT EXISTS recommender_budget (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID NOT NULL,
    month       CHAR(7) NOT NULL,  -- YYYY-MM
    cost_usd    NUMERIC(8,6) NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, month)
);

COMMENT ON TABLE recommender_budget IS
    'Monthly LLM cost accumulator for Phase D recommender budget gate ($20/mo cap).';

-- RLS
ALTER TABLE skill_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender_budget    ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_skill_recommendations ON skill_recommendations
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

CREATE POLICY tenant_isolation_recommender_budget ON recommender_budget
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

-- Grants
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hermes_app') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON skill_recommendations, recommender_budget TO hermes_app;
    END IF;
END
$$;
