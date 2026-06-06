# Notion API Knownness Packet

Generated from official Notion docs and specs on 2026-05-18.

## Identity

Provider: Notion

API family: Notion REST API, Notion CLI (`ntn`), hosted Notion MCP, integration webhooks, Workers/Developer Platform, official JS/TS SDK.

Protocol shape: REST JSON API plus multipart file upload endpoints, webhook event delivery, OAuth 2.0 public-connection auth, hosted MCP, and TypeScript Workers runtime.

Primary version/date: REST `Notion-Version: 2026-03-11` for new code. SDK v5+ supports `2025-09-03` and `2026-03-11`; README default was `2025-09-03` in the 2026-05-18 fetch.

Environments: production public API at `https://api.notion.com`; no sandbox found in public docs. Webhooks require public HTTPS receiver. Hosted MCP endpoint at `https://mcp.notion.com/mcp`.

Access boundary: public docs/specs. No credentials used. No live workspace read/write probes run.

Why Stockitup/Hermes needs it: future agents can correctly operate Notion pages/data sources/markdown/webhooks without stale database semantics, malformed curl snippets, or credential leakage.

## Source Ranking

Best machine-readable spec:

- `https://developers.notion.com/openapi.json`
- Retrieved to `/tmp/notion-api-official-md/openapi.json`.
- OpenAPI 3.1.0; server `https://api.notion.com`; security schemes `bearerAuth` and `basicAuth`.
- Public documented paths observed: users, pages, page markdown, blocks, data sources, databases, search, comments, file uploads, custom emojis, views, meeting notes query, OAuth token/revoke/introspect.

Lower-authority machine-readable lead:

- `https://developers.notion.com/openapi-undocumented.json`
- Treat as unstable/private; do not build production integrations against it without explicit validation.

Official docs/API reference:

- `https://developers.notion.com/llms.txt`
- `https://developers.notion.com/llms-full.txt`
- `.md` docs under `developers.notion.com/reference`, `guides`, `cli`, and `page/changelog`.

Official SDKs/repos:

- JS/TS SDK: `https://github.com/makenotion/notion-sdk-js`, npm package `@notionhq/client`.
- CLI: `ntn`, docs at `https://developers.notion.com/cli/...`; source build path mentioned as `https://github.com/makenotion/cli.git`.
- Workers SDK / Agent SDK: product/dev docs mention `@notionhq/workers` and `@notionhq/agents-client`; treat as beta/alpha and re-check docs before production.

Changelog/release notes:

- `https://developers.notion.com/page/changelog.md`
- `https://developers.notion.com/reference/changes-by-version.md`

Existing Stockitup evidence:

- No actual Notion API integration found under `/home/snaz/stockitup` during the read-only scout.
- Only incidental/product-positioning Notion mentions were found.

## Protocol Shape

Base URL:

```text
https://api.notion.com
```

API style:

- REST resources under `/v1`.
- JSON bodies for normal endpoints.
- Multipart/form-data for `POST /v1/file_uploads/{file_upload_id}/send`.
- Webhooks are configured in the Developer Portal and delivered as POSTs to external HTTPS URLs.
- Hosted MCP is Streamable HTTP/SSE with OAuth-user auth.

Versioning:

- Required `Notion-Version` request header.
- Latest fetched docs recommend/support `2026-03-11` for current REST behavior.
- Breaking versions listed include `2026-03-11`, `2025-09-03`, and older versions.
- Additive changes can happen without version bumps; clients must tolerate unknown fields.

## Auth and Trust

Auth mechanisms:

- Bearer token for REST API internal connections, PATs, and OAuth access tokens.
- OAuth 2.0 for public connections.
- Basic auth with client id/secret for OAuth token endpoint.
- HMAC-SHA256 webhook signature via `X-Notion-Signature` and subscription verification token.
- Hosted MCP OAuth-user auth; no bearer-token headless auth.

OAuth details:

- Auth URL uses `client_id`, `redirect_uri`, `response_type=code`, `owner=user`, optional `state`.
- Token exchange at `POST /v1/oauth/token`.
- Refresh returns new access and refresh tokens.
- Store `bot_id` as authorization key per docs.

Secrets handling:

- Hermes env: `NOTION_API_KEY`.
- Notion CLI env: `NOTION_API_TOKEN`.
- Tokens are opaque; new prefix `ntn_`, old `secret_` may work.
- Webhook verification tokens are secrets.

## Resource Model

Core objects:

- Database: container object with data sources and container-level metadata/permissions.
- Data source: table/schema/row parent. Query/update schema here.
- Page: metadata/properties and parent; body content lives as blocks or markdown endpoint output.
- Block: tree node with type-specific object and optional children.
- View: data-source/database view config and cached query surface.
- Comment: page/block comment and discussion reply object.
- File Upload: upload lifecycle object, then reusable attachment ID.
- User/bot: current user/bot and workspace user data subject to capabilities.

Important lifecycle/status facts:

- Use `in_trash`, not `archived`, in `2026-03-11` payloads.
- File Upload status enum: `pending`, `uploaded`, `expired`, `failed`.
- Data-source query/view query pagination can cap around 10,000 results.
- File download URLs expire after about one hour.

## Request Mechanics

Pagination: cursor-based `has_more`, `next_cursor`, `start_cursor`, max `page_size` generally 100.

Rate limits: average 3 requests/second per connection. Respect integer-second `Retry-After` on 429.

Payload limits: 500KB overall, 1000 block elements, many arrays max 100, rich text 2000 chars.

Retries: backoff 502/503/504; narrow data-source queries on 503; avoid unsafe create retries without dedupe.

Idempotency: no idempotency-key documented. Store IDs, use external keys, and read-after-write.

Search: eventually consistent and not exhaustive. Use known IDs/data-source query for inventory.

File upload: create upload object, send multipart bytes, attach within one hour; no public delete/revoke API found.

## Errors and Edge Cases

Important status/code patterns:

- `400 missing_version`, `validation_error`, `invalid_json`, `invalid_grant`.
- `401 unauthorized` for invalid token.
- `403 restricted_resource` for capability/permission issues.
- `404 object_not_found` for absent or not-shared objects.
- `409 conflict_error` for collision/storage conflict.
- `429 rate_limited` with `Retry-After`.
- `502`, `503`, `504` for transient server/upstream/timeouts.

Open conflict list:

- Some docs still say views unsupported; newer changelog/OpenAPI/views docs show views API.
- File upload guide says `archived` where object enum says `expired`.
- Multipart part count has 10,000 vs 1,000 doc/schema mismatch.
- MCP tool catalog differs between current tool page and changelog.

## Webhooks / Events

Subscription management: Developer Portal, public HTTPS URL, verification token flow.

Signature: `X-Notion-Signature`, HMAC-SHA256 over raw body with verification token.

Delivery: most within 1 minute, target within 5 minutes; not ordered; retries up to 8 attempts with final retry about 24 hours later.

Event families: page, database, data source, comment; docs/changelog/index also mention file-upload/view/page-transcript events that need fresh validation before hardcoding.

Receiver strategy: validate signature, enqueue/dedupe by event id, fetch latest state from REST API, handle out-of-order and aggregated events.

## SDKs and Examples

Official JS/TS SDK:

- Package: `@notionhq/client`.
- Runtime: Node >=18.
- Current README supports `2025-09-03` and `2026-03-11`, defaulting to `2025-09-03`.
- Set `notionVersion: "2026-03-11"` explicitly for new code.
- Provides pagination helpers and typed error/code helpers.

CLI:

- `ntn api` for REST calls, `ntn files`, `ntn pages`, `ntn datasources`, `ntn workers`.
- `NOTION_API_TOKEN` overrides keychain auth.
- CLI can inspect endpoint docs/specs.

MCP:

- Hosted Notion MCP supports interactive AI clients with OAuth-user access.
- It does not currently support file uploads or bearer-token headless auth.

## Stockitup Integration Map

Existing locations checked: `/home/snaz/stockitup` broad Notion/API token/package searches.

Existing code/specs found: none for real Notion API integration.

Target owner if needed later: decide by product boundary. JAD server integration likely belongs under `packages/jad/server/integrations/notion/` for tenant/workspace use; central/control-plane use would belong in Buttler/future `butt`; C/Ring projection requires Stockitup Ring/API skills first.

Deployment/ops concerns: server-side token storage only, no frontend tokens, explicit tenant/workspace/content sharing model, webhook receiver public HTTPS/HMAC verification, rate-limit queue.

## Monitoring Candidate

Should monitor: yes, if Stockitup/Hermes begins depending on Notion API behavior.

Monitor sources:

- `https://developers.notion.com/openapi.json`
- `https://developers.notion.com/llms.txt`
- `https://developers.notion.com/page/changelog.md`
- `https://developers.notion.com/reference/changes-by-version.md`
- `https://developers.notion.com/reference/request-limits.md`
- `https://developers.notion.com/reference/webhooks.md`
- `https://raw.githubusercontent.com/makenotion/notion-sdk-js/main/README.md`

Expected noise: moderate. Changelog and OpenAPI may change often; monitor should digest diffs and only alert on API version, path/schema, auth, webhook, rate-limit, or SDK default changes.

## Final Confidence

Confidence: high for documented REST API core, auth, data-source split, pages/blocks/markdown/files, and request mechanics because official OpenAPI plus official markdown docs were fetched.

Confidence: medium for webhooks/MCP/Workers details because docs show active drift and beta/alpha surfaces.

Invalidators:

- New Notion API version after `2026-03-11`.
- SDK default version changes.
- Webhook/MCP/Worker catalog changes.
- Production account behavior differing from public docs due plan/admin policy.

Next useful step:

- If Notion becomes operationally important, create a docs/spec monitor and run a credentialed read-only smoke test with an explicitly approved Notion workspace/token.
