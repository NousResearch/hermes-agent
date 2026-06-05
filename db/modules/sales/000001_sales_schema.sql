-- Sales Core: operational catalog, inventory, quotes, orders, invoices, and payment requests.
CREATE SCHEMA IF NOT EXISTS sales;

INSERT INTO agent_core.modules(module, description, owner, schema_name)
VALUES ('sales', 'Agent-native commercial/sales core: products, inventory, quotes, orders, invoices, and payment requests.', 'agent-runtime', 'sales')
ON CONFLICT (module) DO UPDATE SET updated_at = now();

INSERT INTO agent_core.module_databases(module, database_name, connection_role, migration_role, metadata)
VALUES ('sales', current_database(), 'sales_runtime', 'agent_admin', '{"option":"same-agent-db-schema","scope":"commercial-sales-core"}'::jsonb)
ON CONFLICT (module) DO UPDATE SET database_name = EXCLUDED.database_name, connection_role = EXCLUDED.connection_role, migration_role = EXCLUDED.migration_role, metadata = EXCLUDED.metadata;

CREATE TABLE IF NOT EXISTS sales.products (
  product_id text PRIMARY KEY,
  sku text,
  name text NOT NULL,
  description text,
  unit_price numeric,
  currency text NOT NULL DEFAULT 'USD',
  status text NOT NULL DEFAULT 'active',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sku)
);

CREATE TABLE IF NOT EXISTS sales.inventory_balances (
  product_id text PRIMARY KEY REFERENCES sales.products(product_id) ON DELETE CASCADE,
  quantity_on_hand numeric NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sales.inventory_movements (
  inventory_movement_id bigserial PRIMARY KEY,
  product_id text NOT NULL REFERENCES sales.products(product_id) ON DELETE CASCADE,
  quantity_delta numeric NOT NULL,
  reason text NOT NULL DEFAULT 'adjustment',
  reference_type text,
  reference_id text,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sales.quotes (
  quote_id text PRIMARY KEY,
  organization_id text,
  contact_id text,
  opportunity_id text,
  title text NOT NULL,
  status text NOT NULL DEFAULT 'draft',
  valid_until date,
  currency text NOT NULL DEFAULT 'USD',
  subtotal numeric NOT NULL DEFAULT 0,
  discount_amount numeric NOT NULL DEFAULT 0,
  tax_amount numeric NOT NULL DEFAULT 0,
  total numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sales.quote_items (
  quote_item_id bigserial PRIMARY KEY,
  quote_id text NOT NULL REFERENCES sales.quotes(quote_id) ON DELETE CASCADE,
  product_id text REFERENCES sales.products(product_id) ON DELETE SET NULL,
  description text NOT NULL,
  quantity numeric NOT NULL DEFAULT 1,
  unit_price numeric NOT NULL DEFAULT 0,
  discount_rate numeric NOT NULL DEFAULT 0,
  discount_amount numeric NOT NULL DEFAULT 0,
  tax_rate numeric NOT NULL DEFAULT 0,
  line_subtotal numeric NOT NULL DEFAULT 0,
  line_discount numeric NOT NULL DEFAULT 0,
  line_tax numeric NOT NULL DEFAULT 0,
  line_total numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sales.orders (
  order_id text PRIMARY KEY,
  quote_id text REFERENCES sales.quotes(quote_id) ON DELETE SET NULL,
  organization_id text,
  contact_id text,
  opportunity_id text,
  title text NOT NULL,
  status text NOT NULL DEFAULT 'confirmed',
  currency text NOT NULL DEFAULT 'USD',
  subtotal numeric NOT NULL DEFAULT 0,
  discount_amount numeric NOT NULL DEFAULT 0,
  tax_amount numeric NOT NULL DEFAULT 0,
  total numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sales.order_items (
  order_item_id bigserial PRIMARY KEY,
  order_id text NOT NULL REFERENCES sales.orders(order_id) ON DELETE CASCADE,
  product_id text REFERENCES sales.products(product_id) ON DELETE SET NULL,
  description text NOT NULL,
  quantity numeric NOT NULL DEFAULT 1,
  unit_price numeric NOT NULL DEFAULT 0,
  discount_amount numeric NOT NULL DEFAULT 0,
  tax_rate numeric NOT NULL DEFAULT 0,
  line_subtotal numeric NOT NULL DEFAULT 0,
  line_discount numeric NOT NULL DEFAULT 0,
  line_tax numeric NOT NULL DEFAULT 0,
  line_total numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sales.invoices (
  invoice_id text PRIMARY KEY,
  order_id text REFERENCES sales.orders(order_id) ON DELETE SET NULL,
  organization_id text,
  contact_id text,
  title text NOT NULL,
  status text NOT NULL DEFAULT 'draft',
  issue_date date,
  due_date date,
  currency text NOT NULL DEFAULT 'USD',
  subtotal numeric NOT NULL DEFAULT 0,
  discount_amount numeric NOT NULL DEFAULT 0,
  tax_amount numeric NOT NULL DEFAULT 0,
  total numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sales.invoice_items (
  invoice_item_id bigserial PRIMARY KEY,
  invoice_id text NOT NULL REFERENCES sales.invoices(invoice_id) ON DELETE CASCADE,
  product_id text REFERENCES sales.products(product_id) ON DELETE SET NULL,
  description text NOT NULL,
  quantity numeric NOT NULL DEFAULT 1,
  unit_price numeric NOT NULL DEFAULT 0,
  discount_amount numeric NOT NULL DEFAULT 0,
  tax_rate numeric NOT NULL DEFAULT 0,
  line_subtotal numeric NOT NULL DEFAULT 0,
  line_discount numeric NOT NULL DEFAULT 0,
  line_tax numeric NOT NULL DEFAULT 0,
  line_total numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sales.payment_requests (
  payment_request_id text PRIMARY KEY,
  invoice_id text REFERENCES sales.invoices(invoice_id) ON DELETE SET NULL,
  organization_id text,
  contact_id text,
  amount numeric NOT NULL DEFAULT 0,
  currency text NOT NULL DEFAULT 'USD',
  status text NOT NULL DEFAULT 'unavailable',
  adapter text,
  payment_url text,
  adapter_response jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sales_products_name ON sales.products USING gin (to_tsvector('simple', name));
CREATE INDEX IF NOT EXISTS idx_sales_products_sku ON sales.products(sku);
CREATE INDEX IF NOT EXISTS idx_sales_inventory_movements_product ON sales.inventory_movements(product_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_sales_quotes_org_status ON sales.quotes(organization_id, status);
CREATE INDEX IF NOT EXISTS idx_sales_orders_org_status ON sales.orders(organization_id, status);
CREATE INDEX IF NOT EXISTS idx_sales_invoices_org_status ON sales.invoices(organization_id, status);
CREATE INDEX IF NOT EXISTS idx_sales_payment_requests_status ON sales.payment_requests(status, updated_at DESC);

GRANT USAGE ON SCHEMA sales TO sales_runtime, agent_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA sales TO sales_runtime;
GRANT SELECT ON ALL TABLES IN SCHEMA sales TO agent_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA sales TO sales_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA sales GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO sales_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA sales GRANT SELECT ON TABLES TO agent_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA sales GRANT USAGE, SELECT ON SEQUENCES TO sales_runtime;
