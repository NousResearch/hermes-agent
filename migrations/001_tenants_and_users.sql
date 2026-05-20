-- migrations/001_tenants_and_users.sql
-- Phase 0: Identity & Tenant Model — Multi-User SaaS Hermes
--
-- Applied against: Neon PostgreSQL (WAL-native, serverless)
-- !! DO NOT APPLY AUTOMATICALLY — requires human review + Neon project provisioning !!
-- See plans/001-saas-multi-user/phases/phase-0-identity.md Step 7 for apply commands.
--
-- Schema design decisions:
-- - All tables use UUID primary keys (gen_random_uuid()) to avoid sequential-ID
--   enumeration attacks and to support multi-region fan-out without coordination.
-- - tenant_id is denormalised onto messages to allow efficient RLS + WAL fan-out
--   queries without joining conversations on every read.
-- - tool_calls + metadata are JSONB so agent turn schemas can evolve without
--   migrations for every new tool.
-- - RLS policies use current_setting('app.tenant_id') which the application MUST
--   set at connection time: SET LOCAL app.tenant_id = '<uuid>'.
-- - WAL logical replication is Neon-native; no extra configuration required.
--   Consumers subscribe to the publication and filter by tenant_id.

-- ---------------------------------------------------------------------------
-- 1. tenants
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tenants (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- e.g. "slack", "discord", "telegram"
    platform    TEXT NOT NULL,
    -- Platform-native workspace identifier (e.g. Slack team_id = T0123ABCDE).
    external_id TEXT NOT NULL,
    -- Human-readable URL slug for display and API paths.
    slug        TEXT UNIQUE NOT NULL,
    -- Billing tier gate.  'free' → single user; 'team' → unlimited users.
    tier        TEXT NOT NULL DEFAULT 'free'
                    CHECK (tier IN ('free', 'team', 'enterprise')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (platform, external_id)
);

COMMENT ON TABLE tenants IS
    'One row per platform workspace/organisation. The root isolation boundary.';
COMMENT ON COLUMN tenants.external_id IS
    'Platform-native workspace ID, e.g. Slack team_id (T-prefixed).';

-- ---------------------------------------------------------------------------
-- 2. users
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id    UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    -- Platform-native user identifier, e.g. Slack user_id (U-prefixed).
    external_id  TEXT NOT NULL,
    platform     TEXT NOT NULL,
    display_name TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, platform, external_id)
);

CREATE INDEX IF NOT EXISTS users_tenant_id_idx ON users (tenant_id);

COMMENT ON TABLE users IS
    'One row per platform user within a tenant. Scoped by tenant_id.';
COMMENT ON COLUMN users.external_id IS
    'Platform-native user ID, e.g. Slack user_id (U-prefixed).';

-- ---------------------------------------------------------------------------
-- 3. conversations
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS conversations (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id        UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    initiating_user  UUID REFERENCES users(id) ON DELETE SET NULL,
    -- Channel/chat identifier on the platform (e.g. Slack channel_id C-prefixed).
    channel_id       TEXT NOT NULL,
    -- Thread anchor.  NULL for top-level channel conversations.
    thread_id        TEXT,
    platform         TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_tenant_channel_thread_idx
    ON conversations (tenant_id, channel_id, thread_id);

COMMENT ON TABLE conversations IS
    'One row per Hermes conversation (channel + optional thread).';
COMMENT ON COLUMN conversations.thread_id IS
    'Null for top-level messages; Slack thread_ts or equivalent for threads.';

-- ---------------------------------------------------------------------------
-- 4. messages
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS messages (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    -- Denormalised from conversations for efficient RLS + WAL fan-out.
    tenant_id        UUID NOT NULL,
    -- NULL for agent turns; set for user turns.
    user_id          UUID REFERENCES users(id) ON DELETE SET NULL,
    -- "user" | "assistant" | "tool"
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content          TEXT,
    -- Structured tool-call payloads (OpenAI-compatible format).
    tool_calls       JSONB,
    -- Arbitrary per-message metadata (model, tokens, latency, etc.).
    metadata         JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_created_idx
    ON messages (conversation_id, created_at);

COMMENT ON TABLE messages IS
    'One row per agent/user message turn. tenant_id denormalised for RLS.';
COMMENT ON COLUMN messages.tenant_id IS
    'Denormalised from conversations.tenant_id for RLS performance. '
    'Application MUST keep this in sync.';
COMMENT ON COLUMN messages.tool_calls IS
    'Array of tool-call objects (OpenAI-format JSON). NULL when role != tool.';

-- ---------------------------------------------------------------------------
-- 5. Row Level Security
-- ---------------------------------------------------------------------------
-- Strategy: application sets SET LOCAL app.tenant_id = '<uuid>' at the start
-- of each database transaction.  All DML is automatically scoped to that tenant.
-- The application service role is NOT a superuser — it uses a dedicated role
-- that only sees rows matching its tenant_id.
--
-- WARNING: current_setting with raise_exception=false returns '' (empty string)
-- when the GUC is not set.  Casting '' to UUID raises.  To avoid silent full
-- scans when app.tenant_id is unset, we use raise_exception=true so unscoped
-- queries fail loudly rather than returning all rows.
-- ---------------------------------------------------------------------------

ALTER TABLE messages      ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE users         ENABLE ROW LEVEL SECURITY;

-- messages: tenant isolation (named per phase-0 AC requirement)
CREATE POLICY tenant_isolation_messages ON messages
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

-- conversations: same isolation model
CREATE POLICY tenant_isolation_conversations ON conversations
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

-- users: scoped by tenant_id column
CREATE POLICY tenant_isolation_users ON users
    USING (tenant_id = current_setting('app.tenant_id', false)::uuid);

-- ---------------------------------------------------------------------------
-- 6. Auto-update updated_at trigger (optional but recommended)
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- Apply instructions (Phase-0 Step 7 — GATED, human action required)
-- ---------------------------------------------------------------------------
-- 1. Provision a Neon project (hermes-saas) in your preferred region.
-- 2. Export the connection string:
--      export NEON_DSN="postgres://user:pass@host/dbname?sslmode=require"
-- 3. Apply this migration:
--      psql "$NEON_DSN" -f migrations/001_tenants_and_users.sql
-- 4. Verify:
--      psql "$NEON_DSN" -c "\dt"       -- lists all four tables
--      psql "$NEON_DSN" -c "\dp messages" -- shows RLS policies
-- 5. Create an application role with RLS enforcement:
--      CREATE ROLE hermes_app LOGIN PASSWORD '...';
--      GRANT CONNECT ON DATABASE <db> TO hermes_app;
--      GRANT USAGE ON SCHEMA public TO hermes_app;
--      GRANT SELECT, INSERT, UPDATE, DELETE
--          ON tenants, users, conversations, messages TO hermes_app;
-- 6. Set NEON_DSN in .env and guard behind HERMES_MODE=saas.
