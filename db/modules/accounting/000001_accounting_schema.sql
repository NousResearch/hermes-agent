-- Accounting Lite Core: receipts, payment records, bank/cash accounts, and basic double-entry bookkeeping.
CREATE SCHEMA IF NOT EXISTS accounting;

INSERT INTO agent_core.modules(module, description, owner, schema_name)
VALUES ('accounting', 'Agent-native accounting-lite core: receipts, bank/cash accounts, payment evidence, income/expense ledger, and accountant exports.', 'agent-runtime', 'accounting')
ON CONFLICT (module) DO UPDATE SET updated_at = now();

INSERT INTO agent_core.module_databases(module, database_name, connection_role, migration_role, metadata)
VALUES ('accounting', current_database(), 'accounting_runtime', 'agent_admin', '{"option":"same-agent-db-schema","scope":"accounting-lite-core"}'::jsonb)
ON CONFLICT (module) DO UPDATE SET database_name = EXCLUDED.database_name, connection_role = EXCLUDED.connection_role, migration_role = EXCLUDED.migration_role, metadata = EXCLUDED.metadata;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'accounting_runtime') THEN
    CREATE ROLE accounting_runtime LOGIN;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS accounting.accounts (
  account_id text PRIMARY KEY,
  business_id text,
  name text NOT NULL,
  account_type text NOT NULL CHECK (account_type IN ('asset','liability','equity','income','expense')),
  subtype text,
  currency text NOT NULL DEFAULT 'USD',
  institution text,
  account_number_last4 text,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','inactive','archived')),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS accounting.receipts (
  receipt_id text PRIMARY KEY,
  receipt_number text UNIQUE,
  business_id text,
  payer_organization_id text,
  payer_contact_id text,
  payee_organization_id text,
  payee_contact_id text,
  direction text NOT NULL CHECK (direction IN ('incoming','outgoing')),
  status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','sent','viewed','commented','approved','rejected','signed','paid','cancelled')),
  issue_date date,
  payment_date date,
  concept text NOT NULL,
  payment_method text,
  payment_reference text,
  amount numeric NOT NULL DEFAULT 0,
  currency text NOT NULL DEFAULT 'USD',
  source_account_id text REFERENCES accounting.accounts(account_id) ON DELETE SET NULL,
  destination_account_id text REFERENCES accounting.accounts(account_id) ON DELETE SET NULL,
  public_token text UNIQUE,
  public_url text,
  pdf_url text,
  signed_pdf_url text,
  approval_hash text,
  rejection_reason text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS accounting.receipt_events (
  receipt_event_id bigserial PRIMARY KEY,
  receipt_id text NOT NULL REFERENCES accounting.receipts(receipt_id) ON DELETE CASCADE,
  event_type text NOT NULL CHECK (event_type IN ('created','sent','opened','commented','approved','rejected','signed','pdf_generated','agent_note','exported')),
  actor_type text NOT NULL DEFAULT 'agent' CHECK (actor_type IN ('customer','counterparty','owner','agent','system','adapter')),
  actor_ref text,
  comment text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  occurred_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS accounting.journal_entries (
  journal_entry_id text PRIMARY KEY,
  business_id text,
  entry_date date NOT NULL DEFAULT CURRENT_DATE,
  description text NOT NULL,
  source_type text,
  source_id text,
  status text NOT NULL DEFAULT 'posted' CHECK (status IN ('draft','posted','void')),
  currency text NOT NULL DEFAULT 'USD',
  total_debit numeric NOT NULL DEFAULT 0,
  total_credit numeric NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CHECK (total_debit = total_credit)
);

CREATE TABLE IF NOT EXISTS accounting.journal_lines (
  journal_line_id bigserial PRIMARY KEY,
  journal_entry_id text NOT NULL REFERENCES accounting.journal_entries(journal_entry_id) ON DELETE CASCADE,
  account_id text NOT NULL REFERENCES accounting.accounts(account_id) ON DELETE RESTRICT,
  line_no integer NOT NULL DEFAULT 1,
  description text,
  debit numeric NOT NULL DEFAULT 0,
  credit numeric NOT NULL DEFAULT 0,
  contact_id text,
  organization_id text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK (debit >= 0 AND credit >= 0 AND NOT (debit > 0 AND credit > 0))
);

CREATE TABLE IF NOT EXISTS accounting.export_runs (
  export_id text PRIMARY KEY,
  business_id text,
  start_date date,
  end_date date,
  format text NOT NULL DEFAULT 'csv' CHECK (format IN ('csv','xlsx','json','pdf')),
  status text NOT NULL DEFAULT 'created' CHECK (status IN ('created','sent','failed','cancelled')),
  file_path text,
  public_url text,
  row_count integer NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  sent_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_accounting_accounts_business ON accounting.accounts(business_id, account_type, status);
CREATE INDEX IF NOT EXISTS idx_accounting_receipts_business_status ON accounting.receipts(business_id, status, issue_date DESC);
CREATE INDEX IF NOT EXISTS idx_accounting_receipts_counterparty ON accounting.receipts(payee_contact_id, payer_contact_id, status);
CREATE INDEX IF NOT EXISTS idx_accounting_receipt_events_receipt ON accounting.receipt_events(receipt_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_accounting_journal_entries_business_date ON accounting.journal_entries(business_id, entry_date DESC);
CREATE INDEX IF NOT EXISTS idx_accounting_journal_entries_source ON accounting.journal_entries(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_accounting_journal_lines_account ON accounting.journal_lines(account_id, created_at DESC);

GRANT USAGE ON SCHEMA accounting TO accounting_runtime, agent_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA accounting TO accounting_runtime;
GRANT SELECT ON ALL TABLES IN SCHEMA accounting TO agent_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA accounting TO accounting_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA accounting GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO accounting_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA accounting GRANT SELECT ON TABLES TO agent_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA accounting GRANT USAGE, SELECT ON SEQUENCES TO accounting_runtime;
