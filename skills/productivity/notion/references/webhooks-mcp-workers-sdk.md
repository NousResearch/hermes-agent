# Notion Webhooks, MCP, Workers, CLI Automation, and SDK

Sources:

- `https://developers.notion.com/reference/webhooks.md`
- `https://developers.notion.com/reference/webhooks-events-delivery.md`
- `https://developers.notion.com/cli/reference/commands.md`
- `https://developers.notion.com/cli/get-started/*.md`
- `https://developers.notion.com/guides/mcp/*.md`
- `https://raw.githubusercontent.com/makenotion/notion-sdk-js/main/README.md`
- `https://www.notion.com/product/dev`

## Webhooks

Purpose: receive near-real-time signals from Notion instead of polling.

Setup:

1. In the Notion Developer Portal, open connection settings.
2. Webhooks tab → Create subscription.
3. Enter a public HTTPS endpoint; localhost is not reachable.
4. Choose event types.
5. Notion sends verification POST with `verification_token`.
6. Paste token into the Webhooks UI to activate.

Changing target URL after verification requires deleting/recreating the subscription. Event type selection can be changed.

### Signature validation

Each event includes:

```text
X-Notion-Signature: sha256=<hex>
```

Compute HMAC-SHA256 over the exact raw request body using the subscription `verification_token`, then compare timing-safely.

Validation is optional in docs for no-code platforms but should be required for production custom receivers.

### Event envelope

Common fields:

- `id`
- `timestamp`
- `workspace_id`
- `subscription_id`
- `integration_id`
- `type`
- `authors`
- `accessible_by` for public connections
- `attempt_number`
- `entity`
- `data`

Events are signals. Fetch latest state by REST API; delivered payload may not be the final state by the time you process it.

### Event families

Page events:

- `page.content_updated`
- `page.created`
- `page.deleted`
- `page.moved`
- `page.properties_updated`
- `page.undeleted`
- `page.locked`
- `page.unlocked`

Database events:

- `database.content_updated`
- `database.created`
- `database.deleted`
- `database.moved`
- `database.schema_updated`
- `database.undeleted`

Data-source events, new after `2025-09-03`:

- `data_source.content_updated`
- `data_source.created`
- `data_source.deleted`
- `data_source.moved`
- `data_source.schema_updated`
- `data_source.undeleted`

Comment events:

- `comment.created`
- `comment.deleted`
- `comment.updated`

Additional event families exposed by the official OpenAPI/event reference surface:

- File uploads: `file_upload.created`, `file_upload.completed`, `file_upload.expired`, `file_upload.upload_failed`.
- Views: `view.created`, `view.updated`, `view.deleted`.
- Transcript deletion: `page.transcription_block.transcript_deleted` can still appear in event naming even though REST block type is `meeting_notes` in `2026-03-11`.

Use the generated OpenAPI inventory for the current event-key list before hardcoding allowlists.

Delivery facts:

- Target delivery is within 5 minutes; most events within 1 minute.
- Ordering is not guaranteed; use timestamps and fetch current state.
- High-frequency events are aggregated by entity and delayed briefly.
- Notion retries failed delivery up to 8 times with exponential backoff; final retry is about 24 hours after initial event.

### Webhook API-version cliffs

Connection webhook subscriptions carry an API version selected in the Developer Portal, separately from REST request `Notion-Version` headers.

- Upgrade handlers deliberately. `2025-09-03` introduces data-source event/entity shapes and database/data-source parent changes. Source: `https://developers.notion.com/guides/get-started/upgrade-guide-2025-09-03.md`.
- `2026-03-11` is available for webhooks and database-automation webhooks, but the docs say its webhook payloads are identical to `2025-09-03`. Source: `https://developers.notion.com/guides/get-started/upgrade-guide-2026-03-11.md`.
- REST `archived` → `in_trash` does **not** apply to webhook payload fields; keep webhook payload parsers compatible with documented webhook shapes.

Monitor gaps:

- `llms.txt` and changelog list file-upload, view, and page-transcript webhook references not fully reflected in the focused delivery table.
- Envelope prose listed entity types narrower than examples. Re-check webhook docs before hardcoding event schema enums.

## MCP

Hosted endpoints:

```text
https://mcp.notion.com/mcp
https://mcp.notion.com/sse       # legacy
```

For stdio-only clients, docs show `mcp-remote` bridge:

```bash
npx -y mcp-remote https://mcp.notion.com/mcp
```

Auth model:

- OAuth-user based.
- Does not support bearer-token auth in hosted MCP.
- Access equals authorizing user's access.
- Good for interactive AI clients; not ideal for headless automation.

Security:

- Use official domains only.
- Use trusted MCP clients.
- Watch for prompt injection because MCP tools can read/write workspace data.
- Enable human confirmation in external workflows.

Limitations:

- Hosted Notion MCP does not currently support file uploads; use File Upload API.
- Search has stricter MCP rate limit than general API: 30/minute in focused docs.

Supported tool families in 2026 docs:

- Search/fetch: `notion-search`, `notion-fetch`.
- Pages/content: create/update/move/duplicate pages.
- Data model: create database, update data source, create/update views.
- Query: query data sources or database views depending on plan/features.
- Comments: create/get comments.
- Workspace/users: teams/teamspaces/users/self.

MCP catalog drift exists: changelog and current supported tools page disagree on some user/meeting-notes tools. Re-check current docs before writing client allowlists.

## Workers / Developer Platform

Official docs list Worker pages, and `ntn` exposes Worker commands. Product page describes Workers as hosted TypeScript programs that can provide:

- syncs: scheduled external API → Notion data source updates;
- tools: callable actions inside Notion/custom agents;
- webhooks: incoming external HTTP events that can update Notion or trigger workflows.

CLI command families:

```text
ntn workers new
ntn workers deploy
ntn workers list / ls
ntn workers get
ntn workers create
ntn workers delete / rm
ntn workers exec
ntn workers capabilities list
ntn workers tui / ui
ntn workers sync status/trigger/pause/resume/state
ntn workers env set/list/unset/pull/push
ntn workers oauth start/token/show-redirect-url
ntn workers runs list/logs
ntn workers webhooks list
```

Treat Workers and Agent SDK surfaces as beta/alpha unless fresh docs confirm stability. Re-fetch official Worker docs before implementing production Workers.

## CLI automation surfaces

`ntn api`:

- Adds Authorization and Notion-Version headers automatically.
- Uses keychain auth or `NOTION_API_TOKEN`.
- No body means GET; inline body/stdin means POST unless `-X` overrides.
- Inline values use path syntax; `:=` supplies JSON-typed values.
- Query params use `name==value`.
- Header overrides use `Header:Value`.
- Use `--notion-version 2026-03-11` or `NOTION_API_VERSION=2026-03-11` to force version.

`ntn datasources`:

- `ntn datasources query <data-source-id>` shortcut.
- `ntn datasources resolve <database-id>` to find data source IDs.

`ntn pages`:

- markdown get/create/update/trash helpers.

`ntn files`:

- create/get/list File Upload helpers.

Diagnostics:

- `ntn doctor`
- `ntn update`
- `ntn --verbose api ...` redacts Authorization; `--unsafe-verbose` does not.

## JS/TS SDK

Package:

```text
@notionhq/client
```

Key methods and helpers from README/changelog:

- `notion.users.list()`
- `notion.users.retrieve()`
- `notion.users.me()` / self equivalent in SDK docs
- `notion.pages.create()`, retrieve/update, markdown retrieve/update
- `notion.blocks.children.list()` and append/update/delete block methods
- `notion.dataSources.query()`
- `notion.dataSources.retrieve/update/create`
- `notion.views.*` and `notion.views.queries.*`
- `notion.comments.create/update/delete`
- `notion.customEmojis.list()`
- `notion.blocks.meetingNotes.query()`
- `notion.request({ path, method, body })` for custom endpoints
- `iteratePaginatedAPI()` and `collectPaginatedAPI()`

Versioning:

- SDK v5+ supports API versions `2025-09-03` and `2026-03-11`.
- Default SDK version was `2025-09-03` in the 2026-05-18 README fetch.
- Pass `notionVersion: "2026-03-11"` explicitly for new work.

Retries:

- Defaults to up to 2 retries.
- Retries 429 for all HTTP methods.
- Retries 500/503 for idempotent GET/DELETE.
- Respects Retry-After.
- Caller still owns create dedupe/idempotency.

## Monitor candidates

- Webhook event catalog and per-event pages.
- Hosted MCP supported tool list and rate limits.
- Workers docs and package APIs.
- SDK default API version and supported version matrix.
- Developer Platform alpha/beta package names and pricing/gating.
