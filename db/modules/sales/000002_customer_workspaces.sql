-- Customer-facing commercial workspace links for quotes, catalogs, and invoices.
-- These rows back URLs such as https://zeus.kidu.app/w/<token>, where clients
-- can review/comment/approve/sign/pay without needing a back-office login.
CREATE SCHEMA IF NOT EXISTS sales;

CREATE TABLE IF NOT EXISTS sales.customer_workspaces (
  workspace_id text PRIMARY KEY,
  document_type text NOT NULL CHECK (document_type IN ('quote', 'catalog', 'invoice')),
  document_id text NOT NULL,
  customer_email text,
  customer_name text,
  public_token text NOT NULL UNIQUE,
  public_url text NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'viewed', 'commented', 'approved', 'rejected', 'signed', 'paid', 'expired', 'cancelled')),
  expires_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (document_type, document_id, customer_email)
);

CREATE TABLE IF NOT EXISTS sales.customer_workspace_events (
  workspace_event_id bigserial PRIMARY KEY,
  workspace_id text NOT NULL REFERENCES sales.customer_workspaces(workspace_id) ON DELETE CASCADE,
  event_type text NOT NULL CHECK (event_type IN ('created', 'sent', 'opened', 'commented', 'approved', 'rejected', 'signed', 'payment_started', 'paid', 'expired', 'cancelled')),
  actor_type text NOT NULL DEFAULT 'customer' CHECK (actor_type IN ('agent', 'owner', 'customer', 'system', 'adapter')),
  actor_ref text,
  comment text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  occurred_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sales_customer_workspaces_token ON sales.customer_workspaces(public_token);
CREATE INDEX IF NOT EXISTS idx_sales_customer_workspaces_document ON sales.customer_workspaces(document_type, document_id);
CREATE INDEX IF NOT EXISTS idx_sales_customer_workspaces_status ON sales.customer_workspaces(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sales_customer_workspace_events_workspace ON sales.customer_workspace_events(workspace_id, occurred_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA sales TO sales_runtime;
GRANT SELECT ON ALL TABLES IN SCHEMA sales TO agent_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA sales TO sales_runtime;
