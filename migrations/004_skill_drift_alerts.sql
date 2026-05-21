-- migrations/004_skill_drift_alerts.sql
-- Plan 004-C: Drift / regression alerts
--
-- Prerequisite: migrations/003_promotion_decisions.sql must be applied first.
-- Apply: psql "$NEON_DSN_HERMES_APP" -f migrations/004_skill_drift_alerts.sql
--
-- Design decisions:
-- - skill_drift_alerts stores one alert per (tenant, skill) — a skill can only
--   be in ONE alert state at a time (alerted, resolved, dismissed, iterate).
--   New alert for the same skill requires the prior to be resolved/dismissed first.
-- - Deduplicate: if the skill already has status='alerted', the daily run skips
--   it. This prevents re-alerting every day for the same regression.
-- - Auto-resolution: when thumbs_rate climbs back to >= RESOLUTION_THRESHOLD (0.65),
--   the drift_detector marks the alert resolved. The skill can then be re-alerted
--   if it degrades again.
-- - baseline_rate: captured when the alert fires (the "before" value). Lets
--   Blake see "was at 85%, dropped to 45%".
-- - Manual actions: 'dismiss' (acknowledged but not fixing now) and 'iterate'
--   (skill author should improve it) are tracked in status.
-- - No FK to skill_scores: skill_scores is a rolling summary; drift_alerts captures
--   a point-in-time snapshot of the regression context.

CREATE TABLE IF NOT EXISTS skill_drift_alerts (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id         UUID NOT NULL,
    skill_name        TEXT NOT NULL,
    -- "alerted" | "resolved" | "dismissed" | "iterate"
    status            TEXT NOT NULL DEFAULT 'alerted'
                          CHECK (status IN ('alerted', 'resolved', 'dismissed', 'iterate')),
    -- Rate when the skill was in its good state (before regression).
    baseline_rate     NUMERIC(5,4),
    -- Rate that triggered the alert.
    alert_rate        NUMERIC(5,4),
    -- Rate at resolution time (if resolved).
    resolved_rate     NUMERIC(5,4),
    alerted_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at       TIMESTAMPTZ,
    -- Who dismissed / marked iterate (Slack user_id or "system").
    actioned_by       TEXT,
    -- Slack DM ts for the alert notification (to avoid re-sending).
    slack_dm_ts       TEXT,
    notes             TEXT,
    -- Only one active alert per (tenant, skill).
    UNIQUE (tenant_id, skill_name)
);

-- Allow deleting resolved/dismissed alerts to make room for new ones.
-- (UNIQUE constraint on (tenant_id, skill_name) enforces one-alert-per-skill.)

CREATE INDEX IF NOT EXISTS skill_drift_alerts_tenant_status_idx
    ON skill_drift_alerts (tenant_id, status, alerted_at DESC);

COMMENT ON TABLE skill_drift_alerts IS
    'One active drift alert per (tenant, skill). Auto-resolved when thumbs_rate '
    'recovers. Deduplicated: existing alerted rows are not re-alerted daily.';

-- RLS
ALTER TABLE skill_drift_alerts ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_skill_drift_alerts ON skill_drift_alerts
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

-- Grants
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hermes_app') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON skill_drift_alerts TO hermes_app;
    END IF;
END
$$;
