# Twenty Adapter Runbook

## Fork

- SitioUno fork: https://github.com/SiteOneTech/twenty
- Upstream: https://github.com/twentyhq/twenty

The fork exists so SitioUno can pin, patch, and install Twenty from its own organization, mirroring the Nettu/Agenda Core pattern.

## Self-host installation outline

Use the fork's Docker Compose package as the base. Do not commit secrets.

Required secrets/config from Infisical/runtime:

- `TWENTY_BASE_URL` — public/internal base URL consumed by Hermes CRM adapter.
- `TWENTY_API_KEY` — API key from Twenty Settings -> API & Webhooks.
- `ENCRYPTION_KEY` — Twenty server encryption key.
- `PG_DATABASE_PASSWORD` — Twenty Postgres password.
- `SERVER_URL` — Twenty server URL.

## Hermes adapter configuration

CRM Core remains usable with no Twenty env. To enable sync:

1. Declare `TWENTY_BASE_URL` and `TWENTY_API_KEY` in Infisical for this agent runtime.
2. Restart/remount Hermes runtime so env reaches `~/.hermes/runtime-secrets.env` or process env.
3. Run `crm_status` and verify `adapters.twenty.configured=true` in a fresh Hermes session.
4. Use `crm_twenty_sync` or `sync_twenty=true` on organization/contact/opportunity upserts.

## API endpoints used by v0

- `POST /rest/companies`
- `PATCH /rest/companies/{id}`
- `POST /rest/people`
- `PATCH /rest/people/{id}`
- `POST /rest/opportunities`
- `PATCH /rest/opportunities/{id}`

External IDs are stored in `crm.external_links`.

## v1 workspace schema extensions

Twenty's Metadata API should create custom objects for:

- Product
- Quote
- QuoteItem
- Invoice
- FollowUp

Fields should include:

- `localId`
- `businessId`
- `sourceChannel`
- `externalRef`
- `syncStatus`
- `lastSyncedAt`

## Verification

- Local core smoke must pass without Twenty env.
- Adapter smoke with fake/missing env must return configured=false, not crash.
- Adapter smoke with real env should create records and populate `crm.external_links`.
