# Agent Core DB

The Agent Core DB is the shared PostgreSQL substrate for one agent instance. Zeus uses it as part of the core runtime, not as a service-specific database.

## Local Zeus Runtime

Compose file:

```text
runtime/agent-core-db/docker-compose.agent-core.yml
```

Local env file, ignored by git:

```text
runtime/agent-core-db/.env
```

Example:

```text
runtime/agent-core-db/.env.example
```

Local defaults:

- Postgres container: `agent-postgres`
- Docker network: `agent-core`
- Host bind: `127.0.0.1:55430`
- Primary agent database: `zeus_agent`
- Calendar backend database: `nettu_calendar`

## Commands

Start the DB server:

```bash
python scripts/agent_core_db.py up
```

Apply Hermes-owned module migrations:

```bash
python scripts/agent_core_db.py migrate
```

Import the legacy local Factory SQLite progress rows into Postgres:

```bash
python scripts/migrate_factory_sqlite_to_agent_core.py
```

Show status:

```bash
python scripts/agent_core_db.py status
```

Apply an external module's SQL directory into a database on the same server:

```bash
python scripts/agent_core_db.py apply-external \
  --module nettu_calendar \
  --database nettu_calendar \
  --path /home/jean/services/nettu-scheduler/scheduler/crates/infra/migrations
```

## Module Boundary

Modules own their migrations and data model. Interoperability between modules belongs in Hermes tools and agent workflows, not cross-module database coupling.

Examples:

- Factory uses schema/tables in `zeus_agent` under `factory`.
- Calendar/Nettu uses separate database `nettu_calendar` on the same Postgres server.
- CRM should add a CRM schema/module on the same Agent Core DB instead of creating a separate Postgres server.

## Secrets

Real passwords belong in Infisical/runtime env, not source control. The `.env.example` files are templates only.
