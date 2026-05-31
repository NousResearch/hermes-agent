-- Agent Core base schema. Runs in the primary agent database (for Zeus: zeus_agent).
CREATE SCHEMA IF NOT EXISTS agent_core;

CREATE TABLE IF NOT EXISTS agent_core.schema_migrations (
  module text NOT NULL,
  version text NOT NULL,
  checksum text NOT NULL,
  applied_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (module, version)
);

CREATE TABLE IF NOT EXISTS agent_core.modules (
  module text PRIMARY KEY,
  description text NOT NULL,
  owner text NOT NULL DEFAULT 'agent-runtime',
  status text NOT NULL DEFAULT 'active',
  database_name text NOT NULL DEFAULT current_database(),
  schema_name text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS agent_core.module_databases (
  module text PRIMARY KEY REFERENCES agent_core.modules(module) ON DELETE CASCADE,
  database_name text NOT NULL,
  connection_role text NOT NULL DEFAULT 'app',
  migration_role text NOT NULL DEFAULT 'admin',
  created_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO agent_core.modules(module, description, owner, schema_name)
VALUES ('agent_core', 'Shared SQL substrate and module migration ledger for this agent instance.', 'agent-runtime', 'agent_core')
ON CONFLICT (module) DO UPDATE SET updated_at = now();
