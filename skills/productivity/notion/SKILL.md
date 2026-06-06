---
name: notion
description: "Use when reading, writing, integrating, or troubleshooting Notion through the REST API, ntn CLI, MCP, webhooks, pages, data sources, markdown, blocks, comments, files, or official JS SDK."
version: 2.1.0
author: community + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  env_vars: [NOTION_API_KEY]
metadata:
  hermes:
    tags: [Notion, Productivity, Notes, Data Sources, API, CLI, Markdown, Files, Webhooks, MCP]
    homepage: https://developers.notion.com
    related_skills: [web-apis, oauth-sota, webhook-subscriptions]
---

# Notion

## Overview

Use this skill for Notion's developer surfaces: the Notion REST API, `ntn` CLI, official JS/TS SDK, webhooks, MCP, page markdown, blocks, data sources, comments, views, and file uploads.

This skill was refreshed from official Notion docs on 2026-05-18. Highest-authority sources are:

1. `https://developers.notion.com/openapi.json` — public OpenAPI 3.1 spec for the documented REST API.
2. `https://developers.notion.com/llms.txt` and official `.md` pages under `developers.notion.com`.
3. `https://github.com/makenotion/notion-sdk-js` / `@notionhq/client` README.
4. `ntn` CLI docs and Notion product/dev pages for Workers, MCP, and alpha/beta surfaces.

Default REST API version for new code: **`2026-03-11`**.

Important version cliffs:

- `2025-09-03`: Notion databases became containers; rows/schema live in **data sources**.
- `2026-03-11`: use `position` instead of legacy `after` for block insertion; use `in_trash` instead of `archived`; use `meeting_notes` instead of `transcription`.

## When to Use

Use this skill when the task mentions:

- Notion pages, databases, data sources, views, blocks, comments, users, search, files, markdown, webhooks, MCP, Workers, or `ntn`.
- `api.notion.com`, `Notion-Version`, `@notionhq/client`, `@notionhq/workers`, `mcp.notion.com`, or `NOTION_API_KEY` / `NOTION_API_TOKEN`.
- Building a Notion integration, syncing Notion data, exporting/importing content, receiving Notion events, or troubleshooting Notion permissions.

Do not use this skill for generic note-taking advice unrelated to the API.

## Credentials and Setup

### Hermes env variable

Hermes currently treats `NOTION_API_KEY` as the configured secret name. Keep that as the primary Hermes prerequisite.

`~/.hermes/.env` uses dotenv-style literal `KEY=value` lines. Do **not** put `export ...` lines in it, and do **not** rely on `$NOTION_API_KEY` expansion inside the file.

```dotenv
NOTION_API_KEY=ntn_xxx_or_secret_xxx
```

For `ntn`, Notion's own CLI reads `NOTION_API_TOKEN`. If you need both Hermes and `ntn`, duplicate the literal token:

```dotenv
NOTION_API_KEY=ntn_xxx_or_secret_xxx
NOTION_API_TOKEN=ntn_xxx_or_secret_xxx
```

Tokens are opaque strings. Newer public API tokens use the `ntn_` prefix; old `secret_` tokens can still work. Do not regex-validate token formats.

### Choose a client path

Prefer by availability:

1. **`ntn` CLI** on macOS/Linux for one-shot terminal work, file uploads, endpoint inspection, and Workers commands.
2. **HTTP + curl** everywhere, including Windows and minimal environments.
3. **Official JS/TS SDK** for TypeScript/Node integrations.
4. **Hosted Notion MCP** for interactive AI-user access, not headless bearer-token automation.

`ntn` install:

```bash
curl -fsSL https://ntn.dev | bash
# or, with Node 22+ and npm 10+
npm install --global ntn
ntn --version
```

`ntn` is documented for macOS/Linux x64/arm64; Windows support was still listed as “coming soon” in the 2026-05-18 docs. Use curl on Windows, or use `ntn` in WSL2.

### Minimal HTTP smoke test

Only run this if a real token is intentionally configured:

```bash
curl -sS "https://api.notion.com/v1/users/me" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2026-03-11"
```

For JSON bodies add:

```bash
  -H "Content-Type: application/json"
```

## Auth and Access Model

REST requests use bearer-token auth plus a mandatory `Notion-Version` header.

Token types:

- **Internal connection token**: static bot token for one workspace. It has no content access by default. Grant access from the Developer Portal Content access tab or in Notion UI by adding the connection to a page/database. Parent access flows to children.
- **Public connection token**: OAuth 2.0 access token per authorizing user/workspace. Use `state` for CSRF/app-state. Store `bot_id` as the primary authorization key. Refreshing returns a new access token and a new refresh token.
- **Personal access token (PAT)**: static user-scoped token. Acts as the user who created it, expires after one year, and follows that user's workspace/page permissions. Good for trusted scripts and CLI; use OAuth for multi-user products.

Capabilities matter and do not override page/workspace permissions:

- read content: retrieve/query pages, blocks, data sources, markdown.
- insert content: create pages/data sources, append blocks, attach uploaded files.
- update content: update/trash/restore pages/blocks, markdown updates, schema changes where permitted.
- read/insert comments: comment APIs.
- user info capabilities: user list/retrieve and optional email visibility.

Gotchas:

- `404 object_not_found` often means “not shared with this token/connection,” not true absence.
- `403 restricted_resource` means missing capability or permission.
- Relation targets and linked data sources usually must also be shared with the connection.
- PATs cannot list all workspace users; retrieve the token's bot/current user instead.

## API Basics

Base URL:

```text
https://api.notion.com
```

Public OpenAPI spec:

```text
https://developers.notion.com/openapi.json
```

Undocumented/lower-authority spec:

```text
https://developers.notion.com/openapi-undocumented.json
```

Rules:

- Use HTTPS and JSON unless the endpoint explicitly says multipart/form-data.
- `Notion-Version: 2026-03-11` for new REST work.
- The URL namespace remains `/v1`; date-versioning is only the header.
- IDs are UUIDs; Notion accepts dashed or undashed forms.
- Empty strings are not supported. Use `null` to unset nullable strings.
- Ignore unknown response fields; additive changes can occur without version bumps.
- Treat cursors as opaque. Pass `next_cursor` back as `start_cursor`; never parse it.
- For full endpoint/current-schema inventory, load `references/openapi-generated-inventory-2026-05-18.md`; for drift/codegen traps, load `references/deep-edge-cases-and-codegen.md`.

## Common Task Recipes

### Search for pages or data sources by title

Use search as discovery, not authoritative inventory. Search is eventually consistent and not exhaustive.

```bash
curl -sS -X POST "https://api.notion.com/v1/search" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2026-03-11" \
  -H "Content-Type: application/json" \
  -d '{"query":"roadmap","filter":{"property":"object","value":"page"},"page_size":10}'
```

For database-like objects in `2025-09-03+`, search/filter results use `data_source`, not `database`.

### Read a page for an agent

Prefer markdown for model-readable page content:

```bash
curl -sS "https://api.notion.com/v1/pages/${PAGE_ID}/markdown" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2026-03-11"
```

If `truncated` is true or `unknown_block_ids` are returned, fetch those block/page IDs or fall back to structured block traversal.

### Query a data source

Database containers are not row tables anymore. Discover the data source ID first from `GET /v1/databases/{database_id}`, then query:

```bash
curl -sS -X POST "https://api.notion.com/v1/data_sources/${DATA_SOURCE_ID}/query" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2026-03-11" \
  -H "Content-Type: application/json" \
  -d '{"page_size":50}'
```

For very large data sources, avoid blind full polling. Query pagination can cap around 10,000 results; use filters and webhooks.

### Work with saved views

Use views when the user's saved filter/sort/layout is the desired truth. Use data-source query when you need ad-hoc filters/sorts.

- List views with `GET /v1/views?database_id=...` or `GET /v1/views?data_source_id=...`; retrieve full configuration with `GET /v1/views/{view_id}`.
- Create views with `POST /v1/views` using `data_source_id` plus exactly one placement parent: `database_id`, dashboard `view_id`, or `create_database`.
- View queries are cached result sets: create `/queries`, paginate, then delete/free them. They expire after about 15 minutes and cannot accept extra filters/sorts.
- Dashboard view `configuration.rows` is read-only; change dashboard layout by creating/deleting widget views, respecting the documented widget-row limits.

### Create a page from markdown

Under a normal page:

```bash
curl -sS -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2026-03-11" \
  -H "Content-Type: application/json" \
  -d '{"parent":{"type":"page_id","page_id":"PAGE_ID"},"markdown":"# Notes\n\n- Decision: ship"}'
```

Under a data source, use `data_source_id` as parent and provide properties matching the schema.

### Create or apply page templates

- List data-source templates with `GET /v1/data_sources/{data_source_id}/templates`; templates are ordinary Notion pages surfaced for that data source.
- For page create/update, `template` is `{ "type": "default" }` or `{ "type": "template_id", "template_id": "..." }`; include `timezone` when `@now` / `@today` should resolve predictably.
- Template application is asynchronous: read the page after the request before depending on merged content or properties.
- Do not combine `template` with `children`; avoid mixing template with markdown/content modes unless the current schema docs prove that exact combination.
- Updating an existing page with `erase_content: true` is destructive replacement before template application.

### Update page markdown

Prefer exact search/replace updates with `update_content`, or whole-page replacement with `replace_content`. Legacy `insert_content` and `replace_content_range` still exist but are not preferred.

```bash
curl -sS -X PATCH "https://api.notion.com/v1/pages/${PAGE_ID}/markdown" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2026-03-11" \
  -H "Content-Type: application/json" \
  -d '{"command":{"type":"update_content","content_updates":[{"old_str":"Status: draft","new_str":"Status: final"}]}}'
```

If the old string matches multiple places, set `replace_all_matches: true` deliberately. If the page changed, exact matching fails instead of silently editing the wrong text.

### Append blocks

Use `position`, not legacy `after`:

```bash
curl -sS -X PATCH "https://api.notion.com/v1/blocks/${BLOCK_ID}/children" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2026-03-11" \
  -H "Content-Type: application/json" \
  -d '{"position":{"type":"end"},"children":[{"object":"block","type":"paragraph","paragraph":{"rich_text":[{"text":{"content":"Hello from Hermes"}}]}}]}'
```

Limits: max 100 block children per append request, max two nesting levels in one request.

### File uploads

Prefer `ntn` when installed:

```bash
ntn files create < ./photo.png
```

HTTP flow is not a PUT-to-presigned-URL flow. It is:

1. `POST /v1/file_uploads` to create a File Upload object.
2. `POST /v1/file_uploads/{file_upload_id}/send` with multipart form field `file`.
3. Attach `{ "type": "file_upload", "file_upload": { "id": "..." } }` in a supported page/block/property API.

Attach uploaded files within one hour. File download URLs expire after one hour; re-fetch the file/page/block object to refresh URLs.

## Webhooks, MCP, Workers, SDK

Webhooks:

- Created in the connection's Developer Portal Webhooks tab, not by a REST endpoint in the public OpenAPI spec.
- Target must be public HTTPS; localhost is not reachable.
- Verification POST includes `verification_token`; paste it into the portal to activate.
- Validate event bodies with `X-Notion-Signature`: HMAC-SHA256 over the exact raw JSON body using the subscription verification token.
- Events are signals. Fetch latest state by REST API; ordering is not guaranteed.
- Notion retries failed deliveries up to 8 times with exponential backoff, with final retry around 24 hours after the first event.
- Webhook subscriptions have their own Developer Portal API version. Upgrade handlers deliberately: `2025-09-03` changes database/data-source event shapes, while `2026-03-11` webhook payloads are documented as identical to `2025-09-03`; REST `archived` → `in_trash` does not apply to webhook payload fields.

MCP:

- Hosted endpoints: `https://mcp.notion.com/mcp` and legacy `https://mcp.notion.com/sse`.
- Requires user OAuth; it does not support bearer-token headless auth.
- MCP access equals the authorizing Notion user's access.
- File uploads are not currently supported by hosted Notion MCP; use File Upload API.

Workers / Developer Platform:

- `ntn` has Workers commands for scaffold/deploy/list/exec/sync/env/oauth/runs/webhooks.
- Workers are hosted TypeScript programs with syncs, tools, and incoming webhooks. Treat deeper Worker/Agent SDK docs as active beta/alpha surfaces; re-check official docs before production work.

JS/TS SDK:

```bash
npm install @notionhq/client
```

SDK v5+ supports `2025-09-03` and `2026-03-11`, but defaults to `2025-09-03`. Opt into latest explicitly:

```javascript
const { Client } = require("@notionhq/client");
const notion = new Client({
  auth: process.env.NOTION_API_KEY,
  notionVersion: "2026-03-11",
});
```

SDK retries 429 for all methods and 500/503 for idempotent GET/DELETE by default; still design caller-level idempotency for creates.

## Request Mechanics

- Rate limit: average 3 requests/second per connection; respect `Retry-After` on 429.
- Payload limit: 500KB overall and 1000 block elements per request.
- Array limits: many block/rich-text arrays cap at 100 elements.
- Rich text content/link URL: 2000 chars; equation: 1000 chars; URL: 2000 chars; email/phone: 200 chars; relation/people: 100 entries.
- `GET` paginated endpoints take query params; `POST` paginated endpoints take JSON body params.
- `page_size` max is generally 100.
- Error response programmatic field is `code`; message text may change without version bump.
- Retry transient 502/503/504 with backoff and jitter. For data-source query 503, reduce `page_size` and narrow filters/sorts.
- Do not blind-retry `POST /v1/pages` or `POST /v1/file_uploads` after ambiguous network failures; no idempotency-key is documented.

## References

Load these support files for deeper work:

- `references/api-knownness-packet.md` — web-apis knownness packet and handoff summary.
- `references/official-source-map.md` — source ranking, OpenAPI/spec receipts, monitor candidates, Stockitup/local state.
- `references/setup-auth-and-cli.md` — tokens, OAuth/PAT/internal connections, `ntn`, curl, SDK setup.
- `references/api-2026-03-11.md` — version cliffs, endpoint map, pagination/errors/limits.
- `references/data-sources-and-pages.md` — data source/database/page/block/property/view model.
- `references/markdown-workflows.md` — page markdown create/read/update and enhanced markdown syntax.
- `references/block-types.md` — block payload examples and traversal/update rules.
- `references/file-uploads.md` — File Upload API and `ntn files` workflows.
- `references/webhooks-mcp-workers-sdk.md` — webhooks, MCP, Workers, JS SDK, monitor gaps.
- `references/openapi-generated-inventory-2026-05-18.md` — generated endpoint/operation/webhook/schema inventory from official OpenAPI.
- `references/deep-edge-cases-and-codegen.md` — second-pass edge cases, docs drift, codegen rules, and monitor inputs.
- `scripts/notion_api_surface_snapshot.py` — no-credential public docs/spec/package snapshot tool for drift monitoring.

## Common Pitfalls

1. **Using stale `2025-09-03` examples for new code.** Use `2026-03-11` unless compatibility requires an older version.
2. **Calling databases rows.** Databases are containers; data sources hold schema and rows.
3. **Using `database_id` for new row parents/relations.** Use `data_source_id` after `2025-09-03`.
4. **Using `archived`, `after`, or `transcription` in new REST payloads.** Use `in_trash`, `position`, and `meeting_notes`; treat meeting-notes blocks as read/query-only, not create/update payload targets.
5. **Assuming 404 means absent.** It often means the page/data source is not shared with the token owner/connection.
6. **Forgetting capabilities.** Token page access is not enough if the connection lacks read/insert/update/comment/user capability.
7. **Using search as inventory.** Search is eventually consistent and not exhaustive. Prefer known IDs or data-source queries.
8. **Pasting shell exports into `.env`.** Hermes `.env` wants literal `KEY=value` lines.
9. **Copying malformed docs/examples blindly.** Normalize curl Authorization headers and version headers yourself.
10. **Caching signed file URLs.** They expire; re-fetch objects for fresh URLs.
11. **Treating MCP as headless integration auth.** Hosted MCP is OAuth-user oriented; use REST/SDK/Workers for unattended automation.
12. **Retrying creates without dedupe.** Notion does not document idempotency keys for create APIs.

## Verification Checklist

- [ ] Chosen source: OpenAPI/spec, official `.md` docs, SDK README, CLI docs, or explicitly lower-authority product/beta docs.
- [ ] REST calls include `Authorization: Bearer …` and `Notion-Version: 2026-03-11`.
- [ ] Content access was granted/shared to the internal/public connection, or a PAT is intentionally used.
- [ ] Required capabilities match the endpoint.
- [ ] Data-source work uses `data_source_id`; old database endpoints are treated as deprecated/compat paths.
- [ ] Pagination follows `has_more` / `next_cursor`; cursors are opaque.
- [ ] 429 honors `Retry-After`; transient 502/503/504 use backoff; unsafe creates are deduped.
- [ ] File uploads attach within one hour and signed file URLs are not cached long-term.
- [ ] Webhook receiver validates `X-Notion-Signature` before production use.
- [ ] No credentials, webhook tokens, page customer data, or file URLs are pasted into durable public artifacts.
