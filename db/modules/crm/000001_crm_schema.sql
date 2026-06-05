-- CRM module schema in the shared Agent Core DB.
CREATE SCHEMA IF NOT EXISTS crm;

INSERT INTO agent_core.modules(module, description, owner, schema_name)
VALUES ('crm', 'Agent-native CRM: organizations, contacts, opportunities, and interaction history.', 'agent-runtime', 'crm')
ON CONFLICT (module) DO UPDATE SET updated_at = now();

INSERT INTO agent_core.module_databases(module, database_name, connection_role, migration_role, metadata)
VALUES ('crm', current_database(), 'crm_runtime', 'agent_admin', '{"option":"same-agent-db-schema"}'::jsonb)
ON CONFLICT (module) DO UPDATE SET database_name = EXCLUDED.database_name, connection_role = EXCLUDED.connection_role, migration_role = EXCLUDED.migration_role;

CREATE TABLE IF NOT EXISTS crm.organizations (
  organization_id text PRIMARY KEY,
  name text NOT NULL,
  domain text,
  phone text,
  email text,
  website text,
  status text NOT NULL DEFAULT 'active',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS crm.contacts (
  contact_id text PRIMARY KEY,
  organization_id text REFERENCES crm.organizations(organization_id) ON DELETE SET NULL,
  full_name text NOT NULL,
  email text,
  phone text,
  title text,
  status text NOT NULL DEFAULT 'active',
  source text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS crm.opportunities (
  opportunity_id text PRIMARY KEY,
  organization_id text REFERENCES crm.organizations(organization_id) ON DELETE SET NULL,
  contact_id text REFERENCES crm.contacts(contact_id) ON DELETE SET NULL,
  title text NOT NULL,
  stage text NOT NULL DEFAULT 'lead',
  value_amount numeric,
  currency text DEFAULT 'USD',
  expected_close_date date,
  status text NOT NULL DEFAULT 'open',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS crm.interactions (
  interaction_id bigserial PRIMARY KEY,
  organization_id text REFERENCES crm.organizations(organization_id) ON DELETE SET NULL,
  contact_id text REFERENCES crm.contacts(contact_id) ON DELETE SET NULL,
  opportunity_id text REFERENCES crm.opportunities(opportunity_id) ON DELETE SET NULL,
  channel text NOT NULL,
  direction text NOT NULL DEFAULT 'note',
  summary text NOT NULL,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  actor text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_crm_organizations_name ON crm.organizations USING gin (to_tsvector('simple', name));
CREATE INDEX IF NOT EXISTS idx_crm_contacts_name ON crm.contacts USING gin (to_tsvector('simple', full_name));
CREATE INDEX IF NOT EXISTS idx_crm_contacts_email ON crm.contacts(email);
CREATE INDEX IF NOT EXISTS idx_crm_interactions_contact_time ON crm.interactions(contact_id, occurred_at DESC);
