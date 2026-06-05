# PRD-001 — Agent CRM Core + Twenty Adapter

## Goal

Give Zeus and future single-tenant SitioUno agents a canonical CRM capability operated mainly through chat, voice, WhatsApp, Telegram, and websocket/tool calls — not through a human-first UI.

The capability must mirror the Calendar/Nettu pattern: Hermes exposes stable, generic CRM tools; a replaceable backend adapter handles vendor-specific APIs. Twenty is the first CRM adapter because it is open source, Postgres-backed, self-hostable, and schema/API friendly.

## Users

- Owner/user: Jean or future business owner served by an independent agent instance.
- Agent: Zeus or inherited client agent operating CRM workflows autonomously.
- Optional human operator: can inspect or enrich records through Twenty UI, but UI is not required for normal usage.

## Scope v0

- Shared Agent Core Postgres schema under `crm`.
- Canonical tools for:
  - organizations/companies
  - contacts/people
  - relationships
  - opportunities
  - products/services
  - quotes with line items
  - invoices
  - interactions and follow-ups
  - customer timeline/search
- Twenty adapter:
  - stores API config via runtime env/Infisical (`TWENTY_BASE_URL`, `TWENTY_API_KEY`)
  - syncs organizations, contacts, and opportunities
  - stores external links in `crm.external_links`
  - provides raw escape hatch for advanced Twenty REST endpoints
- Skill/runbook for getting maximum value from Twenty as CRM adapter.

## Non-goals v0

- Full accounting/tax/e-invoicing compliance.
- Payments collection.
- Odoo/Lago adapters.
- Direct writes into Twenty Postgres.
- UI automation.

## Acceptance Criteria

1. Agent can create/update an organization and contact in Postgres through tools.
2. Agent can model an opportunity tied to a customer/contact.
3. Agent can record interactions and follow-up tasks.
4. Agent can create products, quotes, quote items, and invoices.
5. Agent can fetch a customer timeline with interactions, opportunities, quotes, invoices, relationships, and follow-ups.
6. Agent can sync organization/contact/opportunity to Twenty when env is configured.
7. If Twenty is not configured, CRM Core remains usable and reports adapter status without failing core tools.
8. New tools are listed in the `crm` toolset.
9. Migrations run against the shared Agent Core DB; no separate Postgres service is introduced for CRM.
10. Tests validate key SQL/tool and Twenty request behavior without requiring real Twenty credentials.

## Quality Gates

- DB migration applies cleanly to `zeus_agent`.
- Python import/lint passes for `tools/crm_tool.py`.
- Unit tests pass.
- Live smoke test creates a synthetic CRM graph in local Agent Core DB.
- Twenty adapter remains optional; no secret is hardcoded.
