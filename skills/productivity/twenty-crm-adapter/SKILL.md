---
name: twenty-crm-adapter
description: "Use Twenty CRM as the first external adapter for Zeus/SitioUno Agent CRM Core."
version: 1.0.0
category: productivity
metadata:
  hermes:
    tags: [twenty, crm, adapter, api, self-hosted]
    created_by: agent
---

# Twenty CRM Adapter

Use this skill when configuring, syncing, or extending the Twenty CRM adapter for Zeus/SitioUno CRM Core.

## Canonical Role

Twenty is an adapter, not the canonical tool contract. Agents should operate `crm_*` tools first, then sync selected records to Twenty for UI/team workflows.

Fork:
- SitioUno fork: `https://github.com/SiteOneTech/twenty`
- Upstream: `https://github.com/twentyhq/twenty`

## Required Runtime Configuration

Secrets/config belong in Infisical/runtime env, not hardcoded files:

- `TWENTY_BASE_URL`: base URL of the self-hosted Twenty instance, e.g. `https://crm.example.com`
- `TWENTY_API_KEY`: API key created in Twenty Settings → API & Webhooks

`crm_status` reports whether the Twenty adapter is configured.

## Twenty API Facts

Twenty exposes schema-per-tenant APIs:

- Core API: `/rest/` and `/graphql/` for objects such as companies, people, opportunities, and custom objects.
- Metadata API: `/rest/metadata/` and `/metadata/` to manage objects, fields, and relations.
- Custom objects automatically get REST/GraphQL endpoints.
- API auth uses a bearer token in the HTTP Authorization header.

## Current Adapter Coverage

v0 syncs:

- CRM organization -> Twenty `companies`
- CRM contact -> Twenty `people`
- CRM opportunity -> Twenty `opportunities`

The adapter stores mapping rows in `crm.external_links` with provider `twenty`.

## Usage Pattern

1. Create/update local CRM Core record.
2. Sync to Twenty if needed:
   - use `sync_twenty=true` during upsert, or
   - call `crm_twenty_sync(local_type, local_id)` later.
3. Inspect `crm.external_links` via `crm_customer_timeline` or DB/debug query when needed.
4. Use `crm_twenty_raw_request` only for advanced Twenty endpoints after preferring canonical tools.

## Recommended Twenty Workspace Extensions

For future v1, create custom objects in Twenty through the Metadata API:

- Product
- Quote
- QuoteItem
- Invoice
- FollowUp

Keep object/field names aligned with CRM Core IDs so sync stays deterministic:

- `localId`
- `businessId`
- `sourceChannel`
- `externalRef`
- `syncStatus`

## Installation Pattern

Follow Twenty's self-host Docker Compose flow from the SitioUno fork. Keep environment values in Infisical. Do not commit API keys or encryption keys.

Typical deployment inputs:

- `SERVER_URL`
- `PG_DATABASE_PASSWORD`
- `ENCRYPTION_KEY`
- object storage configuration if production-like

## Pitfalls

- Do not make Twenty the only source of CRM truth for Zeus; local CRM Core remains canonical.
- Do not rely on UI automation; use REST/GraphQL/Metadata APIs.
- Do not sync everything by default; sync when UI/team integration needs it.
- Do not store Twenty secrets in repo files.
- Do not assume quote/invoice custom objects exist until the Twenty workspace schema is explicitly created.
