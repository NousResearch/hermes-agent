# ADR-001 — Agent-native CRM Core over replaceable adapters

## Status
Accepted for v0 implementation.

## Context

Zeus is the prototype for personalized business agents. Each future client agent is independent/single-tenant with its own tools, server, context, and shared Agent Core SQL database. Jean wants CRM capability similar to Agenda Core/Nettu: a canonical agent tool layer first, backend adapters second.

Directly binding Zeus to Twenty, Odoo, Dolibarr, or another app would make the agent inherit a vendor's object model and workflow semantics. That is not canonical enough for SitioUno agents.

## Decision

Create `CRM Core` as an agent-native module in the shared Agent Core Postgres DB.

- Canonical Hermes tools operate local CRM objects.
- Twenty is the first external adapter, not the source of the tool contract.
- External system identifiers live in `crm.external_links`.
- Vendor APIs are accessed only through adapter helpers/tools.
- Direct SQL writes to vendor databases are forbidden.

## Consequences

Positive:
- Agents get stable tools independent of backend changes.
- Twenty can be replaced or supplemented with Odoo/Lago/Medusa later.
- Local CRM remains available even when external adapter credentials are missing.
- The module follows Jean's shared per-agent DB architecture.

Trade-offs:
- v0 duplicates some data between CRM Core and Twenty.
- Sync conflict resolution is intentionally minimal in v0.
- Quote/invoice support is operational metadata, not legal accounting.

## Implementation Notes

- Use schema `crm` in database `zeus_agent`.
- Use `crm_runtime` runtime role.
- Secrets/config come from Infisical/runtime env: `TWENTY_BASE_URL`, `TWENTY_API_KEY`.
- Tool names stay generic: `crm_contact_upsert`, `crm_quote_create`, etc.
