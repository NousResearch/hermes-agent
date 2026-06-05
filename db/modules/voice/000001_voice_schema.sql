-- Voice module schema in the shared Agent Core DB.
-- Vapi is the first provider adapter. The local schema remains the canonical
-- call/assistant ledger for Zeus-style agents; provider state is mirrored here.
CREATE SCHEMA IF NOT EXISTS voice;

INSERT INTO agent_core.modules(module, description, owner, schema_name)
VALUES ('voice', 'Agent-native voice/telephony core: assistants, phone numbers, calls, and provider webhook events.', 'agent-runtime', 'voice')
ON CONFLICT (module) DO UPDATE SET updated_at = now();

INSERT INTO agent_core.module_databases(module, database_name, connection_role, migration_role, metadata)
VALUES ('voice', current_database(), 'voice_runtime', 'agent_admin', '{"option":"same-agent-db-schema","primary_adapter":"vapi"}'::jsonb)
ON CONFLICT (module) DO UPDATE SET database_name = EXCLUDED.database_name, connection_role = EXCLUDED.connection_role, migration_role = EXCLUDED.migration_role, metadata = EXCLUDED.metadata;

CREATE TABLE IF NOT EXISTS voice.providers (
  provider text PRIMARY KEY,
  base_url text NOT NULL,
  status text NOT NULL DEFAULT 'active',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO voice.providers(provider, base_url, status, metadata)
VALUES ('vapi', 'https://api.vapi.ai', 'active', '{"auth":"bearer","secret_env":"VAPI_API_KEY"}'::jsonb)
ON CONFLICT (provider) DO UPDATE SET base_url = EXCLUDED.base_url, status = EXCLUDED.status, metadata = EXCLUDED.metadata, updated_at = now();

CREATE TABLE IF NOT EXISTS voice.assistants (
  assistant_id text PRIMARY KEY,
  provider text NOT NULL DEFAULT 'vapi' REFERENCES voice.providers(provider),
  provider_assistant_id text,
  name text NOT NULL,
  business_id text,
  client_id text,
  status text NOT NULL DEFAULT 'active',
  config jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(provider, provider_assistant_id)
);

CREATE TABLE IF NOT EXISTS voice.phone_numbers (
  phone_number_id text PRIMARY KEY,
  provider text NOT NULL DEFAULT 'vapi' REFERENCES voice.providers(provider),
  provider_phone_number_id text,
  number text,
  name text,
  assistant_id text REFERENCES voice.assistants(assistant_id) ON DELETE SET NULL,
  inbound_enabled boolean NOT NULL DEFAULT true,
  outbound_enabled boolean NOT NULL DEFAULT true,
  status text NOT NULL DEFAULT 'active',
  config jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(provider, provider_phone_number_id)
);

CREATE TABLE IF NOT EXISTS voice.calls (
  call_id text PRIMARY KEY,
  provider text NOT NULL DEFAULT 'vapi' REFERENCES voice.providers(provider),
  provider_call_id text,
  direction text NOT NULL CHECK (direction IN ('inbound','outbound','web','unknown')) DEFAULT 'unknown',
  status text NOT NULL DEFAULT 'created',
  assistant_id text REFERENCES voice.assistants(assistant_id) ON DELETE SET NULL,
  phone_number_id text REFERENCES voice.phone_numbers(phone_number_id) ON DELETE SET NULL,
  from_number text,
  to_number text,
  customer_id text,
  contact_id text,
  organization_id text,
  opportunity_id text,
  started_at timestamptz,
  ended_at timestamptz,
  summary text,
  recording_url text,
  transcript_url text,
  artifact jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(provider, provider_call_id)
);

CREATE TABLE IF NOT EXISTS voice.call_events (
  event_id bigserial PRIMARY KEY,
  call_id text REFERENCES voice.calls(call_id) ON DELETE SET NULL,
  provider text NOT NULL DEFAULT 'vapi',
  provider_call_id text,
  event_type text NOT NULL,
  verified boolean NOT NULL DEFAULT false,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  received_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_voice_calls_provider_call ON voice.calls(provider, provider_call_id);
CREATE INDEX IF NOT EXISTS idx_voice_calls_status_created ON voice.calls(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_voice_call_events_call_time ON voice.call_events(call_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_voice_phone_numbers_number ON voice.phone_numbers(number);
