# ADR-003: Shared Agent SQL Database for Modular Capabilities

## Status

Accepted and implemented for the Zeus prototype runtime.

## Context

The first Calendar Capability prototype used a service-local Postgres container for Nettu Scheduler. That works for a narrow backend smoke test, but it is not the clean long-term architecture for a personalized business agent.

Zeus is the prototype of a single-tenant business agent. Future capabilities such as calendar, CRM, tasks, documents, reservations, payments, and analytics should not each install an isolated database by default. Each agent instance should have a shared SQL substrate that functional modules can assume exists.

## Decision

Create an independent Agent Core SQL database per agent instance. Functional modules use that shared database and run their own versioned migrations into it.

The database is part of the agent runtime substrate, not owned by any one module.

## Architecture

Layering:

1. Agent Runtime
   - Hermes/Zeus process, tools, gateway, profiles, cron/jobs.
2. Agent Core DB
   - Shared PostgreSQL instance/database for the agent instance.
   - Stores module state and operational data.
3. Module Schemas / Migrations
   - `factory` schema/tables for progress/gates/events.
   - `calendar` schema/tables or scheduler-owned tables for scheduling.
   - `crm` schema/tables for contacts, companies, deals, activities.
   - Future modules add their own schemas/tables through migrations.
4. External Adapters
   - Nettu Scheduler, Google/Outlook, CRMs, payments, etc. can use or sync with the Agent Core DB depending on the module design.

## Recommended Local Prototype

Use local PostgreSQL as the Agent Core DB server for Zeus:

- Container: `agent-postgres`
- Docker network: `agent-core`
- Host bind: `127.0.0.1`
- Port: `55430`
- Primary agent database: `zeus_agent`
- Calendar backend database: `nettu_calendar`
- Admin user: `agent_admin` locally; production should use explicit app/migration roles.
- Secrets: injected through Infisical/runtime env, never committed.

Existing SQLite factory DB remains migration source/history until migrated:

- Current: `~/.hermes/factory/factory.db`
- Target: `factory.*` schema/tables in Agent Core DB.

## Module Migration Contract

Each module owns:

- A stable module name.
- A migrations directory.
- A migration ledger entry in a shared table such as `agent_core.schema_migrations`.
- Idempotent or ledger-controlled migrations.
- No hardcoded tenant-specific facts.

Suggested migration layout in the Hermes repo:

```text
db/
  agent-core/
    000001_init.sql
  modules/
    factory/
      000001_factory_schema.sql
    calendar/
      000001_calendar_schema.sql
    crm/
      000001_crm_schema.sql
```

Suggested ledger:

```sql
CREATE SCHEMA IF NOT EXISTS agent_core;
CREATE TABLE IF NOT EXISTS agent_core.schema_migrations (
  module text NOT NULL,
  version text NOT NULL,
  checksum text NOT NULL,
  applied_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (module, version)
);
```

## Calendar Implication

The current Nettu-specific Postgres container should be treated as prototype scaffolding. The canonical next step is to point Nettu at the shared Agent Core DB, preferably under either:

- a dedicated database inside the same Postgres instance, or
- a dedicated schema if Nettu can support it cleanly.

Because Nettu migrations are not fully idempotent and SQLx validates against a real DB at compile time, the shared DB migration runner must prepare the calendar schema before building/running Nettu.

## CRM Implication

CRM should not create a new Postgres container. It should add CRM migrations to the shared Agent Core DB and expose canonical CRM tools over those tables/adapters.

## Consequences

Positive:

- Cleaner modular architecture.
- One durable SQL substrate per agent instance.
- Easier backup, observability, migrations, and future deployment.
- Functional modules become reusable across future agents.

Tradeoffs:

- Need a real migration runner.
- Need connection config and secrets as first-class agent runtime configuration.
- Need module boundaries to avoid schema coupling.
- Need a migration path from current SQLite factory DB.

## Next Implementation Steps

Implemented now:

1. Agent Core DB compose/service for local Zeus prototype.
2. Migration runner script in Hermes repo.
3. `agent_core.schema_migrations` ledger.
4. Factory, calendar registry, and Nettu external migration application into the shared Postgres server.
5. Calendar/Nettu points to the shared Postgres server using Option A: separate database `nettu_calendar` on the same server.

Still pending:

1. Migrate current Factory SQLite data rows to Postgres schema `factory`.
2. Add production app/migration roles instead of using the local admin role for all prototype operations.
3. Add CRM module migrations against the same DB.
