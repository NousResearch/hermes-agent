-- migrations/002_skill_feedback.sql
-- Plan 004-A: Self-Improvement — Skill Feedback Telemetry
--
-- Applied against: Neon PostgreSQL (hermes_app role, hermes-saas project)
-- Prerequisite: migrations/001_tenants_and_users.sql must be applied first.
-- !! DO NOT APPLY AUTOMATICALLY — requires human review !!
-- Apply: psql "$NEON_DSN_HERMES_APP" -f migrations/002_skill_feedback.sql
--
-- Design decisions:
-- - skill_feedback stores one row per (skill_name, slack_ts, tenant_id, reaction) pair.
--   Removing a reaction deletes the row (UNIQUE enforces at-most-one reaction per
--   user per message, matching Slack's own "one emoji type per user per message" rule).
-- - slack_ts is the Slack message timestamp (reaction event's item.ts field).
--   This is the correlation key: Slack reaction events reference message by (channel, ts).
-- - skill_scores is a materialized summary updated by the scorer job (Plan 004-A
--   skill_scorer.py). It is NOT a PostgreSQL MATERIALIZED VIEW because we need
--   RLS-aware row-by-row updates without full refresh overhead at this scale.
-- - RLS on both tables: tenant_id GUC-scoped per 001_tenants_and_users.sql pattern.
-- - skill_scores: thumbs_rate is NULL until total_reactions >= 3 (insufficient signal rule).

-- ---------------------------------------------------------------------------
-- 1. skill_feedback
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS skill_feedback (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id    UUID NOT NULL,
    -- Name of the Hermes skill that produced the message (e.g. "blake-ops:daily").
    skill_name   TEXT NOT NULL,
    -- Slack message timestamp (item.ts from reaction_added event).
    -- Used to correlate reaction events back to the Hermes output message.
    slack_ts     TEXT NOT NULL,
    -- Slack channel_id of the message (item.channel from reaction event).
    channel_id   TEXT NOT NULL,
    -- "thumbs_up" | "thumbs_down"
    reaction     TEXT NOT NULL CHECK (reaction IN ('thumbs_up', 'thumbs_down')),
    -- Slack user who reacted (external_id, e.g. U-prefixed).
    reactor_id   TEXT NOT NULL,
    reacted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- One reaction per (tenant, skill, message, reactor). Slack enforces this
    -- at the platform level; we enforce it in DB to prevent duplicates from
    -- race conditions between reaction_added events.
    UNIQUE (tenant_id, skill_name, slack_ts, reactor_id)
);

CREATE INDEX IF NOT EXISTS skill_feedback_tenant_skill_idx
    ON skill_feedback (tenant_id, skill_name, reacted_at DESC);

CREATE INDEX IF NOT EXISTS skill_feedback_slack_ts_idx
    ON skill_feedback (tenant_id, slack_ts, channel_id);

COMMENT ON TABLE skill_feedback IS
    'One row per user reaction (thumbs_up/thumbs_down) on a Hermes skill output.';
COMMENT ON COLUMN skill_feedback.slack_ts IS
    'Slack message timestamp (item.ts). Correlation key: reaction events reference '
    'message by (channel, ts). Stored as TEXT to preserve Slack precision.';

-- ---------------------------------------------------------------------------
-- 2. skill_output_map
-- ---------------------------------------------------------------------------
-- Maps (slack_ts, channel_id) → skill_name at message-send time.
-- This is the correlation table: when a reaction_added event arrives, we look up
-- which skill produced the message with that ts.
-- Populated by feedback_capture.register_output() at Hermes send time.

CREATE TABLE IF NOT EXISTS skill_output_map (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id    UUID NOT NULL,
    -- Slack ts of the bot message (populated at send time).
    slack_ts     TEXT NOT NULL,
    channel_id   TEXT NOT NULL,
    skill_name   TEXT NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- One mapping per (tenant, channel, ts). If Slack delivers the same ts
    -- twice (reconnect dedup), this prevents duplicate rows.
    UNIQUE (tenant_id, channel_id, slack_ts)
);

CREATE INDEX IF NOT EXISTS skill_output_map_ts_idx
    ON skill_output_map (tenant_id, channel_id, slack_ts);

COMMENT ON TABLE skill_output_map IS
    'Maps (slack_ts, channel_id) → skill_name at Hermes output time. '
    'Required for reaction correlation: reaction events carry (channel, ts) '
    'but not skill_name.';

-- ---------------------------------------------------------------------------
-- 3. skill_scores
-- ---------------------------------------------------------------------------
-- Per-skill aggregated telemetry, maintained by skill_scorer.py.
-- Separate from skill_feedback to allow fast reads without aggregation at query time.

CREATE TABLE IF NOT EXISTS skill_scores (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id        UUID NOT NULL,
    skill_name       TEXT NOT NULL,
    -- Rolling window counts.
    usage_7d         INT NOT NULL DEFAULT 0,
    usage_30d        INT NOT NULL DEFAULT 0,
    usage_all        INT NOT NULL DEFAULT 0,
    thumbs_up_7d     INT NOT NULL DEFAULT 0,
    thumbs_down_7d   INT NOT NULL DEFAULT 0,
    thumbs_up_30d    INT NOT NULL DEFAULT 0,
    thumbs_down_30d  INT NOT NULL DEFAULT 0,
    -- thumbs_rate_30d is NULL when (thumbs_up_30d + thumbs_down_30d) < 3
    -- (insufficient signal). Application enforces this; DB stores raw value.
    thumbs_rate_30d  NUMERIC(5,4),  -- 0.0000 to 1.0000
    thumbs_rate_7d   NUMERIC(5,4),
    last_used_at     TIMESTAMPTZ,
    last_scored_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, skill_name)
);

CREATE INDEX IF NOT EXISTS skill_scores_tenant_skill_idx
    ON skill_scores (tenant_id, skill_name);

CREATE INDEX IF NOT EXISTS skill_scores_thumbs_rate_idx
    ON skill_scores (tenant_id, thumbs_rate_30d DESC NULLS LAST);

COMMENT ON TABLE skill_scores IS
    'Per-skill telemetry summary, maintained by skill_scorer.py. '
    'thumbs_rate_30d is NULL when total reactions < 3 (insufficient signal rule).';

-- ---------------------------------------------------------------------------
-- 4. Row Level Security
-- ---------------------------------------------------------------------------

ALTER TABLE skill_feedback     ENABLE ROW LEVEL SECURITY;
ALTER TABLE skill_output_map   ENABLE ROW LEVEL SECURITY;
ALTER TABLE skill_scores       ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_skill_feedback ON skill_feedback
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

CREATE POLICY tenant_isolation_skill_output_map ON skill_output_map
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

CREATE POLICY tenant_isolation_skill_scores ON skill_scores
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

-- ---------------------------------------------------------------------------
-- 5. Grants (hermes_app role)
-- ---------------------------------------------------------------------------
-- hermes_app role needs SELECT/INSERT/UPDATE/DELETE on all new tables.
-- Run after applying migration if hermes_app role exists.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hermes_app') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON skill_feedback, skill_output_map, skill_scores
            TO hermes_app;
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- Apply instructions
-- ---------------------------------------------------------------------------
-- 1. Ensure migrations/001_tenants_and_users.sql is already applied.
-- 2. Export the hermes_app DSN:
--      export NEON_DSN_HERMES_APP="postgres://hermes_app:pass@host/dbname?sslmode=require"
-- 3. Apply:
--      psql "$NEON_DSN_HERMES_APP" -f migrations/002_skill_feedback.sql
-- 4. Verify:
--      psql "$NEON_DSN_HERMES_APP" -c "\dt skill_*"
--      psql "$NEON_DSN_HERMES_APP" -c "\dp skill_feedback"  -- shows RLS policy
