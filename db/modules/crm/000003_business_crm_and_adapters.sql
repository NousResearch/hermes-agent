-- Extend CRM module for agent-operated relationship, pipeline, quote, invoice, and adapter state.
CREATE SCHEMA IF NOT EXISTS crm;

CREATE TABLE IF NOT EXISTS crm.relationships (
  relationship_id text PRIMARY KEY,
  source_type text NOT NULL CHECK (source_type IN ('organization','contact','opportunity','product','quote','invoice','external')),
  source_id text NOT NULL,
  target_type text NOT NULL CHECK (target_type IN ('organization','contact','opportunity','product','quote','invoice','external')),
  target_id text NOT NULL,
  relationship_type text NOT NULL,
  strength numeric,
  status text NOT NULL DEFAULT 'active',
  notes text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source_type, source_id, target_type, target_id, relationship_type)
);

CREATE TABLE IF NOT EXISTS crm.products (
  product_id text PRIMARY KEY,
  sku text,
  name text NOT NULL,
  description text,
  unit_price numeric,
  currency text NOT NULL DEFAULT 'USD',
  status text NOT NULL DEFAULT 'active',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS crm.quotes (
  quote_id text PRIMARY KEY,
  organization_id text REFERENCES crm.organizations(organization_id) ON DELETE SET NULL,
  contact_id text REFERENCES crm.contacts(contact_id) ON DELETE SET NULL,
  opportunity_id text REFERENCES crm.opportunities(opportunity_id) ON DELETE SET NULL,
  title text NOT NULL,
  status text NOT NULL DEFAULT 'draft',
  valid_until date,
  currency text NOT NULL DEFAULT 'USD',
  subtotal numeric NOT NULL DEFAULT 0,
  tax_amount numeric NOT NULL DEFAULT 0,
  total numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS crm.quote_items (
  quote_item_id bigserial PRIMARY KEY,
  quote_id text NOT NULL REFERENCES crm.quotes(quote_id) ON DELETE CASCADE,
  product_id text REFERENCES crm.products(product_id) ON DELETE SET NULL,
  description text NOT NULL,
  quantity numeric NOT NULL DEFAULT 1,
  unit_price numeric NOT NULL DEFAULT 0,
  tax_rate numeric NOT NULL DEFAULT 0,
  line_total numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS crm.invoices (
  invoice_id text PRIMARY KEY,
  quote_id text REFERENCES crm.quotes(quote_id) ON DELETE SET NULL,
  organization_id text REFERENCES crm.organizations(organization_id) ON DELETE SET NULL,
  contact_id text REFERENCES crm.contacts(contact_id) ON DELETE SET NULL,
  title text NOT NULL,
  status text NOT NULL DEFAULT 'draft',
  issue_date date,
  due_date date,
  currency text NOT NULL DEFAULT 'USD',
  subtotal numeric NOT NULL DEFAULT 0,
  tax_amount numeric NOT NULL DEFAULT 0,
  total numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS crm.follow_ups (
  follow_up_id bigserial PRIMARY KEY,
  organization_id text REFERENCES crm.organizations(organization_id) ON DELETE SET NULL,
  contact_id text REFERENCES crm.contacts(contact_id) ON DELETE SET NULL,
  opportunity_id text REFERENCES crm.opportunities(opportunity_id) ON DELETE SET NULL,
  due_at timestamptz NOT NULL,
  summary text NOT NULL,
  status text NOT NULL DEFAULT 'open',
  priority text NOT NULL DEFAULT 'normal',
  assignee text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS crm.external_links (
  external_link_id bigserial PRIMARY KEY,
  local_type text NOT NULL,
  local_id text NOT NULL,
  provider text NOT NULL,
  external_type text NOT NULL,
  external_id text NOT NULL,
  external_url text,
  sync_status text NOT NULL DEFAULT 'linked',
  last_synced_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (local_type, local_id, provider, external_type),
  UNIQUE (provider, external_type, external_id)
);

CREATE INDEX IF NOT EXISTS idx_crm_relationships_source ON crm.relationships(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_crm_relationships_target ON crm.relationships(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_crm_products_name ON crm.products USING gin (to_tsvector('simple', name));
CREATE INDEX IF NOT EXISTS idx_crm_products_sku ON crm.products(sku);
CREATE INDEX IF NOT EXISTS idx_crm_quotes_org_status ON crm.quotes(organization_id, status);
CREATE INDEX IF NOT EXISTS idx_crm_invoices_org_status ON crm.invoices(organization_id, status);
CREATE INDEX IF NOT EXISTS idx_crm_followups_due ON crm.follow_ups(status, due_at);
CREATE INDEX IF NOT EXISTS idx_crm_external_links_local ON crm.external_links(local_type, local_id);
CREATE INDEX IF NOT EXISTS idx_crm_external_links_provider ON crm.external_links(provider, external_type, external_id);

GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA crm TO crm_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA crm TO crm_runtime;
