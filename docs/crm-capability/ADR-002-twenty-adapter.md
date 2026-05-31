# ADR-002 — Twenty as first CRM adapter

## Status
Accepted for v0 adapter.

## Context

The CRM research compared Twenty, Odoo, ERPNext/Frappe, Dolibarr, Ever Gauzy, Lago, Medusa, Saleor, EspoCRM, SuiteCRM, and invoicing tools. Twenty is the best CRM-first agent-friendly candidate because it is Postgres-backed, self-hostable, open source, and exposes REST/GraphQL APIs generated from each workspace schema plus a Metadata API.

Odoo is stronger for mature quote-to-invoice/accounting, but its RPC/ORM model is too vendor-specific to become the agent's canonical interface.

## Decision

Use the SitioUno fork of Twenty as the first CRM adapter:

- Fork: `https://github.com/SiteOneTech/twenty`
- Upstream: `https://github.com/twentyhq/twenty`
- Hermes env:
  - `TWENTY_BASE_URL`
  - `TWENTY_API_KEY`
- Adapter writes external IDs into `crm.external_links`.

## Adapter Responsibilities v0

- Sync organization -> Twenty `companies`.
- Sync contact -> Twenty `people`.
- Sync opportunity -> Twenty `opportunities`.
- Preserve local source of truth in CRM Core.
- Return clear adapter status when env is missing.
- Expose raw request escape hatch for advanced Twenty API experiments.

## Non-goals v0

- Installing/running Twenty locally in this branch.
- Deep schema management through Twenty Metadata API.
- Conflict-free two-way sync.
- Products/quotes/invoices as native Twenty custom objects. Those are local CRM Core first; Twenty custom object mapping can be added in v1.

## Future Work

- Create Twenty custom objects: Product, Quote, QuoteItem, Invoice, FollowUp.
- Add metadata API migrations for Twenty workspace schema.
- Add webhook ingestion from Twenty to CRM Core.
- Add Odoo adapter for formal sales orders/invoices.
- Add Lago adapter for SaaS billing.
