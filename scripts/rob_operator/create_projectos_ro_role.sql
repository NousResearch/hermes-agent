-- Rob read-only operator P0/P1 — ONE-TIME, HUMAN-RUN role creation.
--
-- This script is NOT executed automatically by any Rob/Hermès code. It
-- exists as a reviewed, documented command for Nicolas to run manually
-- (or explicitly authorize running) against the production `projectos`
-- database on NiPoGi, exactly once, before Rob's db_select/schema_inspect
-- tools become usable against production (they refuse to run against the
-- app's own superuser credential — see db_readonly_profile.py).
--
-- Confirmed this engagement: the only existing Postgres role in
-- production, `projectos`, has rolsuper=t, rolcreatedb=t, rolcreaterole=t
-- — a full superuser. This script creates a separate, minimal-privilege
-- role instead of ever reusing that credential for inspection.
--
-- Run as: docker exec -i project-os-db psql -U projectos -d projectos < create_projectos_ro_role.sql
-- (requires the existing superuser to grant privileges to the new role —
-- this is the one and only place project needs the superuser identity in
-- this whole effort, and only for this one-time bootstrap.)

-- Replace CHANGE_ME_STRONG_PASSWORD before running. Never commit the
-- real password anywhere; generate it fresh (e.g. `openssl rand -hex 24`)
-- at run time and store it only in NiPoGi's own `.env` /
-- ROB_DB_PROFILE_PROJECTOS_DSN, never in this repo.
CREATE ROLE projectos_ro WITH
  LOGIN
  PASSWORD 'CHANGE_ME_STRONG_PASSWORD'
  NOSUPERUSER
  NOCREATEDB
  NOCREATEROLE
  NOREPLICATION
  CONNECTION LIMIT 5;

GRANT CONNECT ON DATABASE projectos TO projectos_ro;
GRANT USAGE ON SCHEMA public TO projectos_ro;
GRANT USAGE ON SCHEMA drizzle TO projectos_ro;

GRANT SELECT ON ALL TABLES IN SCHEMA public TO projectos_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA drizzle TO projectos_ro;

-- Future tables (new migrations) automatically inherit SELECT for this
-- role without needing this script re-run — the whole point of "default
-- privileges" rather than a one-off GRANT list that goes stale.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO projectos_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA drizzle GRANT SELECT ON TABLES TO projectos_ro;

-- Belt-and-suspenders beyond the GRANT model itself: force every session
-- authenticated as this role into a read-only transaction mode and a
-- bounded statement timeout, so even a bug in Rob's own tool code (or a
-- future consumer of this role that forgets to call
-- db_readonly_profile.build_session_init_statements()) still can't write
-- or run away with a long query.
ALTER ROLE projectos_ro SET default_transaction_read_only = on;
ALTER ROLE projectos_ro SET statement_timeout = '5s';

-- Verification queries (run these after, not part of the mutation itself):
--   select rolname, rolsuper, rolcreatedb, rolcreaterole from pg_roles where rolname = 'projectos_ro';
--   -- expect: f | f | f
--   select has_table_privilege('projectos_ro', 'product_changes', 'SELECT');  -- expect: t
--   select has_table_privilege('projectos_ro', 'product_changes', 'INSERT'); -- expect: f
