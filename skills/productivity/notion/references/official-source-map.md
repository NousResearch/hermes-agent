# Notion API Official Source Map

Retrieved: 2026-05-18T05:49:54+02:00

## Authority order

1. Public OpenAPI spec: `https://developers.notion.com/openapi.json`
   - Retrieved to `/tmp/notion-api-official-md/openapi.json`.
   - OpenAPI: 3.1.0.
   - Title: Notion API.
   - Server: `https://api.notion.com`.
   - Auth schemes: `bearerAuth`, `basicAuth`.
   - Public documented paths in retrieved spec: 32.
2. Official docs index and markdown corpus:
   - `https://developers.notion.com/llms.txt`
   - `https://developers.notion.com/llms-full.txt`
   - Individual `.md` pages under `developers.notion.com`.
3. Official SDK/repo docs:
   - `https://github.com/makenotion/notion-sdk-js`
   - `https://raw.githubusercontent.com/makenotion/notion-sdk-js/main/README.md`
   - `https://www.npmjs.com/package/@notionhq/client`
4. Official CLI/MCP/Workers docs and Notion product/developer platform pages.
5. Existing local Hermes Notion skill and plugin usage.
6. Lower-authority hints only: third-party blog posts, examples, random package wrappers.

## Machine-readable specs

Public spec:

```text
https://developers.notion.com/openapi.json
```

Paths observed in the 2026-05-18 fetch:

```text
GET    /v1/users/me
GET    /v1/users/{user_id}
GET    /v1/users
POST   /v1/pages
GET    /v1/pages/{page_id}
PATCH  /v1/pages/{page_id}
POST   /v1/pages/{page_id}/move
GET    /v1/pages/{page_id}/properties/{property_id}
GET    /v1/pages/{page_id}/markdown
PATCH  /v1/pages/{page_id}/markdown
GET    /v1/blocks/{block_id}
PATCH  /v1/blocks/{block_id}
DELETE /v1/blocks/{block_id}
GET    /v1/blocks/{block_id}/children
PATCH  /v1/blocks/{block_id}/children
GET    /v1/data_sources/{data_source_id}
PATCH  /v1/data_sources/{data_source_id}
POST   /v1/data_sources/{data_source_id}/query
POST   /v1/data_sources
GET    /v1/data_sources/{data_source_id}/templates
GET    /v1/databases/{database_id}
PATCH  /v1/databases/{database_id}
POST   /v1/databases
POST   /v1/search
GET    /v1/comments
POST   /v1/comments
GET    /v1/comments/{comment_id}
PATCH  /v1/comments/{comment_id}
DELETE /v1/comments/{comment_id}
GET    /v1/file_uploads
POST   /v1/file_uploads
POST   /v1/file_uploads/{file_upload_id}/send
POST   /v1/file_uploads/{file_upload_id}/complete
GET    /v1/file_uploads/{file_upload_id}
GET    /v1/custom_emojis
GET    /v1/views
POST   /v1/views
GET    /v1/views/{view_id}
PATCH  /v1/views/{view_id}
DELETE /v1/views/{view_id}
POST   /v1/views/{view_id}/queries
GET    /v1/views/{view_id}/queries/{query_id}
DELETE /v1/views/{view_id}/queries/{query_id}
POST   /v1/blocks/meeting_notes/query
POST   /v1/oauth/token
POST   /v1/oauth/revoke
POST   /v1/oauth/introspect
```

Undocumented spec:

```text
https://developers.notion.com/openapi-undocumented.json
```

Treat this as lower-authority and unstable. It includes agent/teamspace/export/tool endpoints and duplicate/experimental data-source paths. Do not build production integrations against undocumented paths without explicit approval and fresh validation.

## Official docs corpus fetched

The local research run saved official markdown pages to:

```text
/tmp/notion-api-official-md/
```

Important files:

- `llms.txt` — docs index; includes OpenAPI URLs and Worker/MCP/webhook pages.
- `llms-full.txt` — full official markdown docs corpus.
- `INDEX.tsv` — URL to local file mapping for selected pages.
- `developers-notion-com-reference-versioning-md.md` — version policy.
- `developers-notion-com-reference-changes-by-version-md.md` — breaking versions.
- `developers-notion-com-guides-get-started-upgrade-guide-2026-03-11-md.md` — latest version migration.
- `developers-notion-com-guides-get-started-upgrade-guide-2025-09-03-md.md` — data-source migration.
- `developers-notion-com-reference-request-limits-md.md` — rate and payload limits.
- `developers-notion-com-reference-status-codes-md.md` — status/error codes.
- `developers-notion-com-reference-webhooks-md.md` and `developers-notion-com-reference-webhooks-events-delivery-md.md` — webhooks.
- `raw-githubusercontent-com-makenotion-notion-sdk-js-main-readme-md.md` — official JS SDK README.

## Current version facts

Latest REST API version in the fetched docs: `2026-03-11`.

Breaking versions listed by Notion docs:

- `2026-03-11`
- `2025-09-03`
- `2022-06-28`
- `2022-02-22`
- `2021-08-16`
- `2021-05-13`

Version policy:

- `Notion-Version` header is mandatory.
- URL path `/v1` is not date-versioned and Notion says it does not intend to change those URLs.
- Version bumps cover backwards-incompatible changes.
- Additive changes can happen without a new API version.
- Cursor format is explicitly opaque and may change without a new version.

## Known conflicts and stale doc spots

- Some 2025 FAQ/data-source pages still say view management is unsupported; March 2026 changelog and current views docs define `/v1/views`. Treat views guide/changelog/OpenAPI as newer.
- One 2025 upgrade guide snippet labels data-source query as `PATCH`; OpenAPI/reference use `POST /v1/data_sources/{data_source_id}/query`.
- File upload expiry wording conflicts: guide mentions `archived`, object enum says `expired`. Treat `expired` as API status.
- Multipart upload count conflict: create schema allows `number_of_parts` up to 10,000; send schema lists `part_number` 1-1,000. Test before relying on >1,000 parts.
- MCP supported-tools page and changelog disagree about `notion-get-user` and meeting-notes query tools. Monitor before hardcoding MCP tool inventory.
- Worker docs were listed in `llms.txt`, but the focused local fetch did not include detailed Worker docs. Treat Worker SDK details as beta/active and re-fetch before production work.

## Existing Stockitup evidence

Read-only scout found no actual Stockitup Notion integration under `/home/snaz/stockitup`:

- no `api.notion.com`
- no `developers.notion.com`
- no `@notionhq`
- no `NOTION_API_KEY` / `NOTION_API_TOKEN`
- no Notion-specific integration directory or code path

Only product/documentation mentions were found, such as generic Notion references in Stockitup web/blog material. If Stockitup later needs a Notion integration, choose the owner path deliberately instead of assuming one exists.

## Local Hermes state found during learning

Before this refresh:

- `productivity/notion` existed but was disabled in `~/.hermes/config.yaml`.
- Bare `skill_view("notion")` collided with `creative/popular-web-designs/templates/notion.md`, because legacy flat-skill lookup counted support-file templates as skill candidates.
- The old Notion skill used `2025-09-03`, malformed curl Authorization snippets, dotenv-incompatible `export` guidance, and an unlinked `block-types.md` reference.

This run patched the loader collision class and refreshed the Notion skill/support files.

## Monitor candidates

Good low-noise public monitor targets:

- `https://developers.notion.com/openapi.json`
- `https://developers.notion.com/llms.txt`
- `https://developers.notion.com/reference/changes-by-version.md`
- `https://developers.notion.com/reference/versioning.md`
- `https://developers.notion.com/page/changelog.md`
- `https://developers.notion.com/reference/request-limits.md`
- `https://developers.notion.com/reference/webhooks.md`
- `https://raw.githubusercontent.com/makenotion/notion-sdk-js/main/README.md`

Monitor reasons:

- API versions and breaking changes are active.
- SDK default version lags latest REST API version.
- Webhook/MCP/Workers surfaces show catalog drift.
- Rate limits explicitly may change by plan or future policy.
