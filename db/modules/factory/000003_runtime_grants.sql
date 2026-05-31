-- Least-privilege runtime grants for Factory tools.
GRANT USAGE ON SCHEMA factory TO factory_runtime, agent_runtime;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA factory TO factory_runtime;
GRANT SELECT ON ALL TABLES IN SCHEMA factory TO agent_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA factory TO factory_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA factory GRANT SELECT, INSERT, UPDATE ON TABLES TO factory_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA factory GRANT SELECT ON TABLES TO agent_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA factory GRANT USAGE, SELECT ON SEQUENCES TO factory_runtime;

UPDATE agent_core.module_databases
SET connection_role = 'factory_runtime', migration_role = 'agent_admin'
WHERE module = 'factory';
