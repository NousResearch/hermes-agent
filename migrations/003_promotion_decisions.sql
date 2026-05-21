-- migrations/003_promotion_decisions.sql
-- Plan 004-B: Auto-suggest + Blake-approve promotion decisions
--
-- Prerequisite: migrations/002_skill_feedback.sql must be applied first.
-- Apply: psql "$NEON_DSN_HERMES_APP" -f migrations/003_promotion_decisions.sql
--
-- Design decisions:
-- - promotion_decisions stores every suggestion (pending, approved, dismissed,
--   auto-demoted). This creates an audit trail for retrospective scoring:
--   "was the system's suggestion correct?" (Plan 014.1).
-- - Bidirectional: action IN ('promote', 'demote') covers both promotion and
--   demotion candidates surfaced by promotion_proposer.py.
-- - Scope columns: from_scope → to_scope record the proposed scope transition
--   so the UI can show "personal → team" or "team → global" clearly.
-- - Decision timestamp is nullable — NULL means pending (not yet acted on).
-- - No FK to skill_scores: skill_scores is a rolling summary that changes over
--   time; we snapshot the scores at suggestion time into score_snapshot JSONB.
-- - RLS: tenant-scoped like all Plan 004 tables.

CREATE TABLE IF NOT EXISTS promotion_decisions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL,
    skill_name      TEXT NOT NULL,
    -- "promote" = personal → team, "demote" = team → personal (or team → global demote)
    action          TEXT NOT NULL CHECK (action IN ('promote', 'demote')),
    from_scope      TEXT NOT NULL,  -- e.g. "personal", "team"
    to_scope        TEXT NOT NULL,  -- e.g. "team", "global"
    -- Snapshot of key metrics at suggestion time for retrospective analysis.
    score_snapshot  JSONB NOT NULL DEFAULT '{}',
    -- "pending" | "approved" | "dismissed"
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'dismissed')),
    suggested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at      TIMESTAMPTZ,
    -- Who approved/dismissed (Slack user_id or "system" for auto-resolved).
    decided_by      TEXT,
    -- Slack DM message_ts of the suggestion DM (for update/delete on decision).
    slack_dm_ts     TEXT,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS promotion_decisions_tenant_status_idx
    ON promotion_decisions (tenant_id, status, suggested_at DESC);

CREATE INDEX IF NOT EXISTS promotion_decisions_tenant_skill_idx
    ON promotion_decisions (tenant_id, skill_name, suggested_at DESC);

COMMENT ON TABLE promotion_decisions IS
    'History of skill promotion/demotion suggestions and Blake approval decisions. '
    'Pending rows are surfaced in the /skills dashboard and Slack DM. '
    'score_snapshot preserves metrics at suggestion time for retrospective analysis.';

-- RLS
ALTER TABLE promotion_decisions ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_promotion_decisions ON promotion_decisions
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

-- Grants
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hermes_app') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON promotion_decisions TO hermes_app;
    END IF;
END
$$;
