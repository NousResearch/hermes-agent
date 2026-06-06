# Notion Deep Edge Cases, Drift, and Codegen Notes — 2026-05-18

Generated/synthesized from official Notion sources and the public OpenAPI on `2026-05-18T04:46:46Z`.

Primary source anchors:

- `https://developers.notion.com/openapi.json` — SHA-256 `c781691e6316b679648c83ff8f18a9dd70943fa24dbe35be69d19ff1cb274174` at this fetch.
- `https://developers.notion.com/llms.txt` — official docs link index.
- `https://developers.notion.com/page/changelog.md`
- `https://developers.notion.com/reference/changes-by-version.md`
- `https://raw.githubusercontent.com/makenotion/notion-sdk-js/main/README.md`

## Problem this reference solves

The first Notion pass taught the big model: `2026-03-11`, data sources, markdown, files, webhooks, MCP, and SDK defaults. This second pass records the edge cases that break integrations and typed clients: stale operation IDs, official-doc contradictions, view/query semantics, OAuth lifecycle traps, property-value incompleteness, webhook version drift, and package/tool-surface drift.

## Claim matrix

### Official spec says one thing; operation IDs sometimes say another

- Source: `openapi.json` paths under `/v1/data_sources`.
- Direct fact: `POST /v1/data_sources/{data_source_id}/query` still has operationId `post-database-query`; `POST /v1/data_sources` still has operationId `create-a-database`.
- Future behavior: generated clients should key canonical routing on HTTP method + path + current docs, then optionally override stale operation IDs.

### Views are first-class current API, despite stale older prose

- Sources: `openapi.json` `/v1/views*`, `https://developers.notion.com/guides/data-apis/working-with-views.md`, `https://developers.notion.com/reference/view.md`.
- Direct fact: views expose list/create/retrieve/update/delete plus cached query create/results/delete.
- Important semantics: view query creates a cached result set, expires after about 15 minutes, cannot accept ad-hoc filters/sorts, and shares the 10k-ish query-depth limit.
- Future behavior: use views when a saved Notion view is the semantic truth; use data-source query when Hermes owns ad-hoc filters/sorts.

### Position objects differ by endpoint

- Sources: `patch-block-children`, page create/update markdown docs, views create docs, OpenAPI schemas.
- Direct fact: block append `position` supports `start`, `end`, and `after_block`; create-page and view placement use related but not identical variants (`page_start`/`page_end`, `after_view`, dashboard/widget placement, and `create_database.position`).
- Future behavior: do not reuse one generic `Position` type blindly across pages, blocks, markdown, and views.

### `in_trash` wins for REST, but schema/prose drift still exists

- Sources: `2026-03-11` upgrade guide, trash-page docs, OpenAPI page schemas.
- Direct fact: current migration docs say `archived` was removed/replaced by `in_trash`; official schemas still expose archive-ish names in some places (`is_archived` observed in scout output).
- Future behavior: write `in_trash` for REST; tolerate old/read-only archive-ish fields in responses/webhooks where docs require compatibility.

### Page templates are asynchronous and can erase content

- Sources: page create/update docs, data source templates endpoint.
- Direct fact: `GET /v1/data_sources/{data_source_id}/templates` lists template pages; page create/update can apply default or specific template; template application is async; update can use `erase_content`.
- Future behavior: after applying a template, read back or wait for webhooks before depending on content; treat `erase_content` as destructive.

### Page property values are not complete in retrieve-page responses

- Sources: page-property-values and property-item docs.
- Direct fact: relation/people/rich_text/title references can truncate around 25 refs in `GET /v1/pages/{page_id}`.
- Future behavior: for correctness, call `GET /v1/pages/{page_id}/properties/{property_id}` and paginate property items; final rollup values may only be reliable after `has_more: false`.

### Data-source query can be successful but incomplete

- Sources: data-source query docs, request-limits docs, views docs.
- Direct fact: data-source and view queries can cap around 10,000 matching results and expose `request_status.type == "incomplete"` with reason like `query_result_limit_reached`.
- Future behavior: sync code must inspect `request_status`; use filters/sharding/webhooks instead of treating `has_more: false` alone as complete inventory.

### Property/schema enum coverage differs between prose docs and OpenAPI

- Sources: property object docs, page property values docs, OpenAPI component schemas.
- Direct fact: OpenAPI exposes variants and request/response shapes that prose docs may not emphasize (`button`, `location`, `last_visited_time`, `place`, `verification`); `place` may read as null/unsupported; `verification` is wiki-specific.
- Future behavior: implement discriminated unions with unknown fallback and nullable unsupported values; prefer stable property IDs over names.

### OAuth refresh rotates both secrets; redirect URI has a conditional rule

- Sources: public connections guide and OAuth token/refresh docs.
- Direct facts: refreshing returns a new access token and refresh token; token exchange `redirect_uri` is required if used at authorization or if multiple redirect URIs exist, and disallowed when exactly one redirect URI exists and the auth URL omitted it.
- Future behavior: update access+refresh atomically; store `bot_id` as authorization key; do not assume `expires_in` exists; treat `refresh_token` as single-owner.

### PATs are useful but operationally brittle

- Sources: personal access token docs, API key handling docs.
- Direct facts: PATs expire after one year; guests/restricted members cannot create them; admin policy/revocation can invalidate API access; PATs cannot list all workspace users.
- Future behavior: use PATs for trusted personal scripts/CLI only; use public OAuth or internal connections for products.

### Webhook signatures need raw-body verification and replay/dedupe

- Sources: webhook reference and event delivery docs.
- Direct facts: `X-Notion-Signature` is `sha256=<hex>` HMAC over the request body with `verification_token`; no signature timestamp/nonce is documented; delivery can retry and arrive out of order.
- Future behavior: verify raw bytes before parsing, compare timing-safely with length check, dedupe by event `id`, order by timestamp only as a hint, then fetch latest REST state.

### Webhook versioning is not the REST header

- Sources: webhook docs and upgrade guides.
- Direct facts: webhook subscriptions carry a Developer Portal API version; 2025 changed data-source event shapes; 2026 webhook payloads are documented as identical to 2025 even though REST renamed `archived` to `in_trash`.
- Future behavior: version webhook parsers separately from REST clients and preserve old webhook payload fields if docs say they remain.

### File uploads are a three-mode lifecycle, not one upload URL

- Sources: file upload reference/guides.
- Direct facts: modes are `single_part`, `multi_part`, and `external_url`; multi-part uses `/send` parts then `/complete`; external import is async; attach only when `status == uploaded`.
- Future behavior: model File Upload as a lifecycle object; no public delete/revoke API found; do not cache signed download URLs.

### MCP/Workers/CLI are moving surfaces

- Sources: Notion MCP docs/well-known metadata, `ntn` docs/npm, Workers docs/npm, SDK repo/npm.
- Direct facts from second pass package/source scouts: hosted MCP is OAuth-user and no headless bearer auth; file uploads are not hosted-MCP-supported; `@notionhq/client` latest observed `5.21.0` but default API version remains `2025-09-03`; `ntn` latest observed `0.14.0`; `@notionhq/workers` latest observed `0.4.0` and requires Node 22/npm 10; local `@notionhq/notion-mcp-server` package and repo/release versions can drift.
- Future behavior: treat non-REST developer-platform surfaces as monitor-first/beta-ish; re-check docs/package metadata before production implementation.

## Codegen rules

1. Pin request header `Notion-Version: 2026-03-11` for new generated clients.
2. Generate path/method APIs from OpenAPI, but override stale operation IDs that still say database for data-source paths.
3. Keep `object`/`type` discriminators, but add unknown fallbacks for future resource/property/block/view types.
4. Model every list as cursor-paginated unless proven otherwise; treat cursors as opaque.
5. Model incompleteness separately from pagination: `request_status`, markdown `truncated`, and `unknown_block_ids` are correctness signals.
6. Split similar-but-different position types by endpoint family.
7. Use stable property IDs internally; allow names only at UX/boundary level.
8. Make create/retry paths dedupe-aware because Notion documents no idempotency key.
9. Treat OAuth token storage as an atomic refresh-token-family update, not a stateless bearer refresh.
10. Keep REST, webhook, MCP, Workers, CLI, and SDK versions as separate monitored surfaces.

## Monitor surface snapshot

Official/no-credential inputs suitable for a quiet monitor:

- OpenAPI: `https://developers.notion.com/openapi.json`
- Docs index: `https://developers.notion.com/llms.txt`
- Changelog: `https://developers.notion.com/page/changelog.md`
- Version changes: `https://developers.notion.com/reference/changes-by-version.md`
- SDK README: `https://raw.githubusercontent.com/makenotion/notion-sdk-js/main/README.md`
- NPM packages: `@notionhq/client`, `ntn`, `@notionhq/workers`, `@notionhq/notion-mcp-server`
- MCP discovery: `https://www.notion.com/.well-known/mcp.json`, `https://mcp.notion.com/.well-known/oauth-authorization-server`

Use `scripts/notion_api_surface_snapshot.py` to produce a JSON snapshot for diffing. Do not schedule chat-alerting cron until Notion is operationally important enough to warrant noise.

## Observed package/source metadata in this pass

- `@notionhq/client`: latest `5.21.0`; source `https://registry.npmjs.org/@notionhq%2fclient`
- `ntn`: latest `0.14.0`; source `https://registry.npmjs.org/ntn`
- `@notionhq/workers`: latest `0.4.0`; source `https://registry.npmjs.org/@notionhq%2fworkers`
- `@notionhq/notion-mcp-server`: latest `2.2.1`; source `https://registry.npmjs.org/@notionhq%2fnotion-mcp-server`

Extra source hashes:

- `llms.txt`: SHA-256 `76afca5f0a061e72ee5da364a03660860b4df910ca8e35ada074c9f752d2efde`, bytes `27627`, URL `https://developers.notion.com/llms.txt`
- `changelog.md`: SHA-256 `321d9c59fede7eabce45fd4ebe90af3cc7254ace1f657807047a735b8978f197`, bytes `32532`, URL `https://developers.notion.com/page/changelog.md`
- `changes-by-version.md`: SHA-256 `22f6573f8e3f8121bbe2f717c6cdbbf5fd727cdbec259a0c45c9edb4af24adcd`, bytes `4990`, URL `https://developers.notion.com/reference/changes-by-version.md`
- `sdk-readme`: SHA-256 `a4941bcdcddfd114c298c3d112e3fce0b39debb4dc353312ab26731f06eb4d7d`, bytes `17034`, URL `https://raw.githubusercontent.com/makenotion/notion-sdk-js/main/README.md`
- `mcp-well-known`: SHA-256 `6976115d9b63114c1e89c999a35702ef84e3ce578862bf1d201ae9dab10bafb9`, bytes `237`, URL `https://www.notion.com/.well-known/mcp.json`
