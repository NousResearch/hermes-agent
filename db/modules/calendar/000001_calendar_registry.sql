-- Calendar module registry in the primary Agent Core DB.
-- The first scheduler backend uses Option A: same Postgres server, separate database.
CREATE SCHEMA IF NOT EXISTS calendar;

INSERT INTO agent_core.modules(module, description, owner, schema_name, metadata)
VALUES (
  'calendar',
  'Agent-first calendar capability. Canonical tools provide interoperability; Nettu is the first backend adapter.',
  'agent-runtime',
  'calendar',
  '{"backend":"nettu","database_mode":"same_postgres_separate_database"}'::jsonb
)
ON CONFLICT (module) DO UPDATE SET updated_at = now(), metadata = EXCLUDED.metadata;

INSERT INTO agent_core.module_databases(module, database_name, connection_role, migration_role, metadata)
VALUES (
  'calendar',
  'nettu_calendar',
  'agent_runtime',
  'agent_admin',
  '{"adapter":"nettu","option":"A","description":"Same Agent Core Postgres server, separate Nettu database."}'::jsonb
)
ON CONFLICT (module) DO UPDATE SET database_name = EXCLUDED.database_name, metadata = EXCLUDED.metadata;

CREATE TABLE IF NOT EXISTS calendar.scheduler_instances (
  instance_id text PRIMARY KEY,
  backend text NOT NULL,
  base_url text NOT NULL,
  database_name text NOT NULL,
  status text NOT NULL DEFAULT 'active',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO calendar.scheduler_instances(instance_id, backend, base_url, database_name, metadata)
VALUES (
  'local-nettu',
  'nettu',
  'http://127.0.0.1:5055/api/v1',
  'nettu_calendar',
  '{"scope":"zeus-prototype","interoperability":"via_hermes_tools"}'::jsonb
)
ON CONFLICT (instance_id) DO UPDATE SET updated_at = now(), base_url = EXCLUDED.base_url, database_name = EXCLUDED.database_name;
