-- Customer intent escalation queue for constrained customer-service channels.
-- Sophie/customer-service records intents here; Zeus/supervisors process them asynchronously.
CREATE SCHEMA IF NOT EXISTS crm;

CREATE TABLE IF NOT EXISTS crm.customer_intents (
  intent_id text PRIMARY KEY,
  organization_id text REFERENCES crm.organizations(organization_id) ON DELETE SET NULL,
  contact_id text REFERENCES crm.contacts(contact_id) ON DELETE SET NULL,
  opportunity_id text REFERENCES crm.opportunities(opportunity_id) ON DELETE SET NULL,
  interaction_id bigint REFERENCES crm.interactions(interaction_id) ON DELETE SET NULL,
  channel text NOT NULL DEFAULT 'whatsapp',
  conversation_ref text,
  source_ref text,
  intent_type text NOT NULL DEFAULT 'other',
  customer_request_raw text NOT NULL,
  summary text NOT NULL,
  required_action text,
  priority text NOT NULL DEFAULT 'normal',
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','processing','completed','blocked','cancelled')),
  assigned_to text NOT NULL DEFAULT 'zeus',
  due_at timestamptz,
  claimed_at timestamptz,
  claimed_by text,
  processed_at timestamptz,
  result_summary text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_crm_customer_intents_status_created ON crm.customer_intents(status, created_at);
CREATE INDEX IF NOT EXISTS idx_crm_customer_intents_contact_created ON crm.customer_intents(contact_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_crm_customer_intents_conversation ON crm.customer_intents(channel, conversation_ref);
CREATE INDEX IF NOT EXISTS idx_crm_customer_intents_source ON crm.customer_intents(channel, source_ref);

GRANT SELECT, INSERT, UPDATE ON crm.customer_intents TO crm_runtime;
GRANT SELECT ON crm.customer_intents TO agent_runtime;
