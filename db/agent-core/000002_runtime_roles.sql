-- Runtime roles are part of the Agent Core DB contract.
-- This migration creates role shells without secrets; scripts/agent_core_roles.py
-- sets LOGIN/PASSWORD from runtime secrets injected by Infisical or .env.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_runtime') THEN
    CREATE ROLE agent_runtime NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'factory_runtime') THEN
    CREATE ROLE factory_runtime NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'calendar_runtime') THEN
    CREATE ROLE calendar_runtime NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'crm_runtime') THEN
    CREATE ROLE crm_runtime NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sales_runtime') THEN
    CREATE ROLE sales_runtime NOLOGIN;
  END IF;
END $$;

GRANT CONNECT ON DATABASE zeus_agent TO agent_runtime, factory_runtime, calendar_runtime, crm_runtime, sales_runtime;
GRANT USAGE ON SCHEMA agent_core TO agent_runtime, factory_runtime, calendar_runtime, crm_runtime, sales_runtime;
GRANT SELECT ON ALL TABLES IN SCHEMA agent_core TO agent_runtime, factory_runtime, calendar_runtime, crm_runtime, sales_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA agent_core GRANT SELECT ON TABLES TO agent_runtime, factory_runtime, calendar_runtime, crm_runtime, sales_runtime;
