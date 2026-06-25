CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email text UNIQUE NOT NULL,
  display_name text,
  status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS groups (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug text UNIQUE NOT NULL,
  name text NOT NULL,
  type text NOT NULL CHECK (type IN ('company', 'department', 'project', 'role')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_groups (
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  group_id uuid NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, group_id)
);

CREATE TABLE IF NOT EXISTS rag_workspaces (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug text UNIQUE NOT NULL,
  name text NOT NULL,
  visibility_boundary text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title text NOT NULL,
  source_uri text,
  owner_user_id uuid REFERENCES users(id),
  department_slug text,
  classification text NOT NULL CHECK (classification IN ('public', 'internal', 'confidential', 'restricted')),
  checksum text,
  version integer NOT NULL DEFAULT 1,
  status text NOT NULL DEFAULT 'pending_classification',
  queued_at timestamptz,
  indexed_at timestamptz,
  ingest_error text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_acl (
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  principal_type text NOT NULL CHECK (principal_type IN ('user', 'group', 'role')),
  principal_id text NOT NULL,
  permission text NOT NULL CHECK (permission IN ('read', 'write', 'admin')),
  PRIMARY KEY (document_id, principal_type, principal_id, permission)
);

CREATE TABLE IF NOT EXISTS document_workspace_membership (
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  workspace_id uuid NOT NULL REFERENCES rag_workspaces(id) ON DELETE CASCADE,
  PRIMARY KEY (document_id, workspace_id)
);

CREATE TABLE IF NOT EXISTS document_ingest_payloads (
  document_id uuid PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  workspace_slug text NOT NULL,
  title text NOT NULL,
  source_text text NOT NULL,
  attempts integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_sources (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  source_type text NOT NULL CHECK (source_type IN ('notion', 'drive_public')),
  target text NOT NULL DEFAULT 'public',
  workspace_slug text NOT NULL,
  classification text NOT NULL CHECK (classification IN ('public', 'internal', 'confidential', 'restricted')),
  config jsonb NOT NULL DEFAULT '{}'::jsonb,
  interval_minutes integer NOT NULL DEFAULT 1440,
  enabled boolean NOT NULL DEFAULT true,
  next_scan_at timestamptz,
  last_scan_at timestamptz,
  last_status text,
  last_error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_scan_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_id uuid NOT NULL REFERENCES document_sources(id) ON DELETE CASCADE,
  trigger text NOT NULL CHECK (trigger IN ('manual', 'scheduled', 'startup')),
  status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'complete', 'failed')),
  queued_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz,
  items_found integer NOT NULL DEFAULT 0,
  documents_queued integer NOT NULL DEFAULT 0,
  error text
);

CREATE TABLE IF NOT EXISTS source_items (
  source_id uuid NOT NULL REFERENCES document_sources(id) ON DELETE CASCADE,
  external_id text NOT NULL,
  checksum text NOT NULL,
  document_id uuid REFERENCES documents(id) ON DELETE SET NULL,
  title text NOT NULL,
  source_uri text,
  last_seen_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (source_id, external_id)
);

CREATE TABLE IF NOT EXISTS document_dedupe_keys (
  workspace_slug text NOT NULL,
  checksum text NOT NULL,
  document_id uuid REFERENCES documents(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (workspace_slug, checksum)
);

CREATE TABLE IF NOT EXISTS audit_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_user_id uuid,
  actor_agent_id text,
  action text NOT NULL,
  resource_type text,
  resource_id text,
  decision text NOT NULL CHECK (decision IN ('allow', 'deny', 'info')),
  reason text,
  request_id text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_email text,
  actor_role text NOT NULL DEFAULT 'member',
  actor_groups text[] NOT NULL DEFAULT '{}'::text[],
  query_text text NOT NULL,
  mode text NOT NULL DEFAULT 'mix',
  allowed_workspaces text[] NOT NULL DEFAULT '{}'::text[],
  status text NOT NULL CHECK (status IN ('ok', 'error')),
  latency_ms integer NOT NULL DEFAULT 0,
  error text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_workspace_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  query_event_id uuid NOT NULL REFERENCES query_events(id) ON DELETE CASCADE,
  workspace_slug text NOT NULL,
  latency_ms integer NOT NULL DEFAULT 0,
  status text NOT NULL CHECK (status IN ('ok', 'error')),
  reference_count integer NOT NULL DEFAULT 0,
  error text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_document_hits (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  query_event_id uuid NOT NULL REFERENCES query_events(id) ON DELETE CASCADE,
  workspace_slug text NOT NULL,
  reference_id text,
  title text,
  source_uri text,
  rank integer NOT NULL DEFAULT 0,
  document_id uuid REFERENCES documents(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_events_created_at ON query_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_events_actor ON query_events (actor_email, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_workspace_events_workspace ON query_workspace_events (workspace_slug, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_document_hits_lookup ON query_document_hits (workspace_slug, title, created_at DESC);

INSERT INTO users (email, display_name, status)
VALUES ('admin@example.com', 'Initial Admin', 'active')
ON CONFLICT (email) DO NOTHING;

INSERT INTO groups (slug, name, type) VALUES
  ('company_all', 'Company All', 'company'),
  ('role_admin', 'Admin', 'role')
ON CONFLICT (slug) DO NOTHING;

INSERT INTO rag_workspaces (slug, name, visibility_boundary) VALUES
  ('company_public', 'Company Public', 'company'),
  ('department_c_level', 'C Level', 'admin')
ON CONFLICT (slug) DO NOTHING;

INSERT INTO user_groups (user_id, group_id)
SELECT u.id, g.id
FROM users u
CROSS JOIN groups g
WHERE u.email = 'admin@example.com'
  AND g.slug IN ('company_all', 'role_admin')
ON CONFLICT DO NOTHING;
