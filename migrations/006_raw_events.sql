-- =====================================================================
-- Migration 006: raw_events — compliance-grade audit log
-- =====================================================================
-- Plan 007 (Sessions to Neon) Phase A.
--
-- Captures every Slack inbound, Hermes outbound, and tool call as a
-- durable row, separate from the agent-facing `messages` table. This is
-- the compliance / debug / replay surface.
--
-- Why separate from `messages`:
--   - `messages` is what the agent reads back (cleaned-up conversation view).
--   - `raw_events` is what compliance + ops + debugging read (full payload,
--     full provenance, never edited, indexed by event_kind for filtering).
--
-- Idempotency:
--   The (tenant_id, conversation_id, event_kind, platform_message_id)
--   uniqueness constraint guarantees Slack redeliveries do not double-write.
--   Inserts use ON CONFLICT DO NOTHING.
--
-- RLS:
--   Mirrors the pattern used by `tenants`, `messages`, `skill_feedback`:
--   policy USING (tenant_id = current_setting('app.tenant_id')::uuid).
--
-- Grants:
--   `hermes_app` gets INSERT + SELECT + (sequence usage where applicable).
-- =====================================================================

-- 1) Table -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_events (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id            uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    conversation_id      uuid REFERENCES conversations(id) ON DELETE SET NULL,
    event_kind           text NOT NULL,
    platform_message_id  text,
    raw_payload          jsonb NOT NULL DEFAULT '{}'::jsonb,
    recorded_at          timestamp with time zone NOT NULL DEFAULT now(),

    CONSTRAINT raw_events_event_kind_check
        CHECK (event_kind = ANY (ARRAY[
            'slack_inbound',
            'slack_outbound',
            'tool_call_request',
            'tool_call_response'
        ]))
);

-- 2) Indexes ---------------------------------------------------------
-- Per-conversation timeline scan (debug + replay use case)
CREATE INDEX IF NOT EXISTS raw_events_conversation_recorded_idx
    ON raw_events (conversation_id, recorded_at DESC);

-- Per-tenant time scan (analytics + audit use case)
CREATE INDEX IF NOT EXISTS raw_events_tenant_recorded_idx
    ON raw_events (tenant_id, recorded_at DESC);

-- Idempotency key — Slack redeliveries land on the same (event_kind, msg_id)
-- so we use a partial unique index that ignores NULL platform_message_id
-- (tool_call_request / tool_call_response may not have one).
CREATE UNIQUE INDEX IF NOT EXISTS raw_events_dedup_key
    ON raw_events (tenant_id, conversation_id, event_kind, platform_message_id)
    WHERE platform_message_id IS NOT NULL;

-- 3) Row-Level Security ---------------------------------------------
ALTER TABLE raw_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_raw_events ON raw_events;
CREATE POLICY tenant_isolation_raw_events
    ON raw_events
    USING (tenant_id = (current_setting('app.tenant_id', false))::uuid)
    WITH CHECK (tenant_id = (current_setting('app.tenant_id', false))::uuid);

-- 4) Grants ----------------------------------------------------------
-- hermes_app gets read + write; no UPDATE (raw_events is append-only by design)
GRANT SELECT, INSERT ON raw_events TO hermes_app;

-- 5) Comments (docs in schema for future spelunkers) ----------------
COMMENT ON TABLE raw_events IS
    'Plan 007: compliance-grade audit log. Every Slack inbound, Hermes outbound, '
    'and tool call lands here. Separate from messages (the agent view). '
    'Append-only — no UPDATE permission. RLS-tenant-scoped.';

COMMENT ON COLUMN raw_events.event_kind IS
    'One of: slack_inbound, slack_outbound, tool_call_request, tool_call_response. '
    'Enforced by check constraint; new kinds require migration.';

COMMENT ON COLUMN raw_events.platform_message_id IS
    'Slack ts for inbound/outbound; deterministic hash (session_id+turn+tool_name+args) '
    'for tool calls. Used as idempotency key.';

COMMENT ON COLUMN raw_events.raw_payload IS
    'Full event payload, post-redaction. NEVER write unredacted secrets here.';
