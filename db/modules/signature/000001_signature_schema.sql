-- Signature Core: agent-native document requests, signature capture, approval hashes, and audit trails.
CREATE SCHEMA IF NOT EXISTS signature;

INSERT INTO agent_core.modules(module, description, owner, schema_name)
VALUES ('signature', 'Agent-native e-signature core: document requests, signers, signature fields, audit events, approval hashes, and completed documents.', 'agent-runtime', 'signature')
ON CONFLICT (module) DO UPDATE SET updated_at = now();

INSERT INTO agent_core.module_databases(module, database_name, connection_role, migration_role, metadata)
VALUES ('signature', current_database(), 'signature_runtime', 'agent_admin', '{"option":"same-agent-db-schema","scope":"document-signature-core"}'::jsonb)
ON CONFLICT (module) DO UPDATE SET database_name = EXCLUDED.database_name, connection_role = EXCLUDED.connection_role, migration_role = EXCLUDED.migration_role, metadata = EXCLUDED.metadata;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'signature_runtime') THEN
    CREATE ROLE signature_runtime LOGIN;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS signature.templates (
  template_id text PRIMARY KEY,
  name text NOT NULL,
  document_url text,
  fields jsonb NOT NULL DEFAULT '[]'::jsonb,
  submitters jsonb NOT NULL DEFAULT '[]'::jsonb,
  preferences jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS signature.document_requests (
  request_id text PRIMARY KEY,
  template_id text REFERENCES signature.templates(template_id) ON DELETE SET NULL,
  source_type text,
  source_id text,
  title text NOT NULL,
  status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','sent','viewed','partially_signed','completed','declined','expired','cancelled')),
  document_url text,
  completed_document_url text,
  audit_url text,
  document_hash_sha256 text,
  approval_hash text,
  public_url text,
  fields_snapshot jsonb NOT NULL DEFAULT '[]'::jsonb,
  submitters_snapshot jsonb NOT NULL DEFAULT '[]'::jsonb,
  preferences jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  expires_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS signature.submitters (
  submitter_id text PRIMARY KEY,
  request_id text NOT NULL REFERENCES signature.document_requests(request_id) ON DELETE CASCADE,
  role text NOT NULL DEFAULT 'signer' CHECK (role IN ('signer','approver','viewer','owner','agent')),
  signing_order integer NOT NULL DEFAULT 1,
  name text,
  email text,
  phone text,
  slug text NOT NULL UNIQUE,
  token_hash_sha256 text NOT NULL UNIQUE,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','sent','viewed','signed','approved','declined','expired','cancelled')),
  values jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  sent_at timestamptz,
  opened_at timestamptz,
  signed_at timestamptz,
  declined_at timestamptz,
  ip_address text,
  user_agent text,
  timezone text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS signature.attachments (
  attachment_id text PRIMARY KEY,
  request_id text REFERENCES signature.document_requests(request_id) ON DELETE CASCADE,
  submitter_id text REFERENCES signature.submitters(submitter_id) ON DELETE SET NULL,
  kind text NOT NULL DEFAULT 'signature' CHECK (kind IN ('signature','initials','stamp','file','image','completed_pdf','audit_pdf')),
  filename text,
  mime_type text,
  storage_path text,
  public_url text,
  byte_size bigint,
  sha256 text,
  width integer,
  height integer,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS signature.events (
  signature_event_id bigserial PRIMARY KEY,
  request_id text NOT NULL REFERENCES signature.document_requests(request_id) ON DELETE CASCADE,
  submitter_id text REFERENCES signature.submitters(submitter_id) ON DELETE SET NULL,
  event_type text NOT NULL CHECK (event_type IN ('created','sent','viewed','started','field_updated','signed','approved','declined','completed','downloaded','agent_note','hash_created')),
  actor_type text NOT NULL DEFAULT 'customer' CHECK (actor_type IN ('customer','owner','agent','system','adapter')),
  actor_ref text,
  ip_address text,
  user_agent text,
  event_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  previous_event_hash text,
  event_hash text NOT NULL,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS signature.approvals (
  approval_id text PRIMARY KEY,
  request_id text NOT NULL REFERENCES signature.document_requests(request_id) ON DELETE CASCADE,
  submitter_id text REFERENCES signature.submitters(submitter_id) ON DELETE SET NULL,
  source_type text,
  source_id text,
  approval_context jsonb NOT NULL DEFAULT '{}'::jsonb,
  signature_text text,
  signature_image_sha256 text,
  document_hash_sha256 text,
  approval_hash text NOT NULL UNIQUE,
  ip_address text,
  user_agent text,
  signed_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_signature_requests_source ON signature.document_requests(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_signature_requests_status ON signature.document_requests(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_signature_submitters_request ON signature.submitters(request_id, signing_order);
CREATE INDEX IF NOT EXISTS idx_signature_submitters_email ON signature.submitters(email);
CREATE INDEX IF NOT EXISTS idx_signature_events_request ON signature.events(request_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_signature_approvals_source ON signature.approvals(source_type, source_id);

GRANT USAGE ON SCHEMA signature TO signature_runtime, agent_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA signature TO signature_runtime;
GRANT SELECT ON ALL TABLES IN SCHEMA signature TO agent_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA signature TO signature_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA signature GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO signature_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA signature GRANT SELECT ON TABLES TO agent_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA signature GRANT USAGE, SELECT ON SEQUENCES TO signature_runtime;
