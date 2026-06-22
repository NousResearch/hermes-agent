---
name: notion
description: "Notion API + ntn CLI: pages, databases, markdown, Workers."
version: 2.1.0
author: community
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  env_vars: [NOTION_API_KEY]
metadata:
  hermes:
    tags: [Notion, Productivity, Notes, Database, API, CLI, Workers]
    homepage: https://developers.notion.com
---

# Notion

Talk to Notion two ways. Same integration token works for both — pick by what's available.

◆ **`ntn` CLI** — Notion's official CLI. Shorter syntax, one-line file uploads, required for Workers. macOS + Linux only as of May 2026 (Windows support "coming soon"). **Default when installed.**
◆ **HTTP + curl** — works everywhere including Windows. **Default fallback** when `ntn` isn't installed.

## Setup

### 1. Get an integration token (required for both paths)

1. Create an integration at https://notion.so/my-integrations
2. Copy the API key (starts with `ntn_` or `secret_`)
3. Store in `${HERMES_HOME:-~/.hermes}/.env`:
   ```
   NOTION_API_KEY=ntn_your_key_here
   ```
4. **Share target pages/databases with the integration** in Notion: page menu `...` → `Connect to` → your integration name. Without this, the API returns 404 for that page even though it exists.

### 2. Install `ntn` (preferred path on macOS / Linux)

```bash
# Recommended
curl -fsSL https://ntn.dev | bash

# Or via npm (needs Node 22+, npm 10+)
npm install --global ntn

ntn --version    # verify
```

**Skip `ntn login` — use the integration token instead.** This works headlessly, no browser needed:
```bash
export NOTION_API_TOKEN=$NOTION_API_KEY      # ntn reads NOTION_API_TOKEN
export NOTION_KEYRING=0                       # don't try to use the OS keychain
```

Add those exports to your shell profile (or to `${HERMES_HOME:-~/.hermes}/.env`) so every session inherits them.

### 3. Choose path at runtime

```bash
if command -v ntn >/dev/null 2>&1; then
  # use ntn
else
  # fall back to curl
fi
```

Windows users: skip step 2 entirely until native `ntn` ships — Path B works fine. If you want CLI ergonomics now, install `ntn` inside WSL2.

## API Basics

`Notion-Version: 2025-09-03` is required on all HTTP requests. `ntn` handles this for you. In this version, what users call "databases" are called **data sources** in the API.

## Path A — `ntn` CLI (preferred, macOS / Linux)

### Raw API calls (shorthand for curl)
```bash
ntn api v1/users                                  # GET
ntn api v1/pages parent[page_id]=abc123 \         # POST with inline body
  properties[title][0][text][content]="Notes"
ntn api v1/pages/abc123 -X PATCH archived:=true   # PATCH; := is non-string (bool/num/null)
```

Syntax notes:
- `key=value` — string fields
- `key[nested]=value` — nested object fields
- `key:=value` — typed assignment (booleans, numbers, null, arrays)

### Passing JSON bodies with `-d` (preferred for complex payloads)
```bash
# Use -d for any JSON body that's too complex for inline key=value syntax
ntn api v1/data_sources/{id} -X PATCH -d '{"properties":{"Thématique":{"select":{"options":[...]}}}}'

# Or from a file
ntn api v1/data_sources/{id}/query -X POST --file /tmp/query.json
```
The `-d` / `--data <JSON>` flag is the cleanest way to pass structured bodies. It avoids all shell escaping issues that the inline `key=value` syntax can have with nested arrays, unicode, or special characters.

### Search
```bash
ntn api v1/search query="page title"
```

### Read page metadata
```bash
ntn api v1/pages/{page_id}
```

### Read page as Markdown (agent-friendly)
```bash
ntn api v1/pages/{page_id}/markdown
```

### Read page content as blocks
```bash
ntn api v1/blocks/{page_id}/children
```

### Create page from Markdown
```bash
ntn api v1/pages \
  parent[page_id]=xxx \
  properties[title][0][text][content]="Notes from meeting" \
  markdown="# Agenda

- Q3 roadmap
- Hiring"
```

### Patch a page with Markdown
```bash
ntn api v1/pages/{page_id}/markdown -X PATCH \
  markdown="## Update

Shipped the prototype."
```

### Query a database (data source)
```bash
ntn api v1/data_sources/{data_source_id}/query -X POST \
  filter[property]=Status filter[select][equals]=Active
```

For complex queries with `sorts`, multiple filter clauses, or compound logic, pipe JSON in:
```bash
echo '{"filter": {"property": "Status", "select": {"equals": "Active"}}, "sorts": [{"property": "Date", "direction": "descending"}]}' | \
  ntn api v1/data_sources/{data_source_id}/query -X POST --json -
```

### File uploads (one-liner — biggest CLI win)
```bash
ntn files create < photo.png
ntn files create --external-url https://example.com/photo.png
ntn files list
```

Compare to the 3-step HTTP flow (create upload → PUT bytes → reference).

### Shorthand: `ntn datasources query`

For simple queries without complex filters, use the dedicated subcommand (cleaner than `ntn api`):

```bash
ntn datasources query <data-source-id> --limit 50
ntn datasources query <data-source-id> --filter '{"property":"Status","select":{"equals":"Open"}}'
ntn datasources query <data-source-id> --json   # machine-readable output
```

For sorts, compound filters, or `filter_properties`, use `ntn api v1/data_sources/{id}/query -X POST` with `-d` or inline `filter:='{...}'` syntax instead.

### Self-discovery: inspect endpoints without leaving the terminal

```bash
ntn api ls                          # list all public API endpoints
ntn api v1/data_sources --spec      # reduced OpenAPI fragment for this endpoint
ntn api v1/data_sources --docs      # full official markdown docs for this endpoint
ntn api v1/data_sources --docs -X POST   # specify method when ambiguous
```

Use these when unsure about request/response shapes — faster than guessing or searching the web.

### Useful env vars
| Var | Effect |
|---|---|
| `NOTION_API_TOKEN` | Auth token (overrides keychain) — set this to your integration token (NOT a `$VAR` reference) |
| `NOTION_KEYRING=0` | File-based creds at `~/.config/notion/auth.json` instead of OS keychain |
| `NOTION_WORKSPACE_ID` | Skip the workspace picker prompt |

## Path B — HTTP + curl (cross-platform, default on Windows)

All requests share this pattern:

```bash
curl -s -X GET "https://api.notion.com/v1/..." \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json"
```

On Windows the `curl` shipped with Windows 10+ works as-is. PowerShell users can also use `Invoke-RestMethod`.

### Search
```bash
curl -s -X POST "https://api.notion.com/v1/search" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"query": "page title"}'
```

### Read page metadata
```bash
curl -s "https://api.notion.com/v1/pages/{page_id}" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### Read page as Markdown (agent-friendly)

Easier to feed to a model than block JSON.

```bash
curl -s "https://api.notion.com/v1/pages/{page_id}/markdown" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### Read page content as blocks (when you need structure)
```bash
curl -s "https://api.notion.com/v1/blocks/{page_id}/children" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### Create page from Markdown

`POST /v1/pages` accepts a `markdown` body param.

```bash
curl -s -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"page_id": "xxx"},
    "properties": {"title": [{"text": {"content": "Notes from meeting"}}]},
    "markdown": "# Agenda\n\n- Q3 roadmap\n- Hiring\n\n## Decisions\n- Ship MVP Friday"
  }'
```

### Patch a page with Markdown
```bash
curl -s -X PATCH "https://api.notion.com/v1/pages/{page_id}/markdown" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"markdown": "## Update\n\nShipped the prototype."}'
```

### Create page in a database (typed properties)
```bash
curl -s -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"database_id": "xxx"},
    "properties": {
      "Name": {"title": [{"text": {"content": "New Item"}}]},
      "Status": {"select": {"name": "Todo"}}
    }
  }'
```

### Query a database (data source)
```bash
curl -s -X POST "https://api.notion.com/v1/data_sources/{data_source_id}/query" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {"property": "Status", "select": {"equals": "Active"}},
    "sorts": [{"property": "Date", "direction": "descending"}]
  }'
```

### Create a database
```bash
curl -s -X POST "https://api.notion.com/v1/data_sources" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"page_id": "xxx"},
    "title": [{"text": {"content": "My Database"}}],
    "properties": {
      "Name": {"title": {}},
      "Status": {"select": {"options": [{"name": "Todo"}, {"name": "Done"}]}},
      "Date": {"date": {}}
    }
  }'
```

### Update page properties
```bash
curl -s -X PATCH "https://api.notion.com/v1/pages/{page_id}" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"properties": {"Status": {"select": {"name": "Done"}}}}'
```

### Append blocks to a page
```bash
curl -s -X PATCH "https://api.notion.com/v1/blocks/{page_id}/children" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "children": [
      {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "Hello from Hermes!"}}]}}
    ]
  }'
```

### File uploads (3-step flow)
```bash
# 1. Create upload
curl -s -X POST "https://api.notion.com/v1/file_uploads" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"filename": "photo.png", "content_type": "image/png"}'

# 2. PUT bytes to the upload_url returned above
curl -s -X PUT "{upload_url}" --data-binary @photo.png

# 3. Reference {file_upload_id} in a page/block payload
```

## Property Types

Common property formats for database items:

- **Title:** `{"title": [{"text": {"content": "..."}}]}`
- **Rich text:** `{"rich_text": [{"text": {"content": "..."}}]}`
- **Select:** `{"select": {"name": "Option"}}`
- **Multi-select:** `{"multi_select": [{"name": "A"}, {"name": "B"}]}`
- **Date:** `{"date": {"start": "2026-01-15", "end": "2026-01-16"}}`
- **Checkbox:** `{"checkbox": true}`
- **Number:** `{"number": 42}`
- **URL:** `{"url": "https://..."}`
- **Email:** `{"email": "user@example.com"}`
- **Relation:** `{"relation": [{"id": "page_id"}]}`

## API Version 2025-09-03 — Databases vs Data Sources

- **Databases became data sources.** Use `/data_sources/` endpoints for queries and retrieval.
- **Two IDs per database:** `database_id` and `data_source_id`.
  - `database_id` when creating pages: `parent: {"database_id": "..."}`
  - `data_source_id` when querying: `POST /v1/data_sources/{id}/query`
- Search returns databases as `"object": "data_source"` with the `data_source_id` field.

## Notion Workers (advanced, requires `ntn`)

Workers are TypeScript programs Notion hosts for you. One worker can expose any combination of:
- **Syncs** — pull data from external APIs into a Notion database on a schedule (default 30 min).
- **Tools** — appear as callable tools inside Notion's Custom Agents.
- **Webhooks** — receive HTTP events from external services (GitHub, Stripe, etc.) and act in Notion.

**Plan / platform gating:**
- CLI works on all plans. **Deploying Workers requires Business or Enterprise.**
- `ntn` is macOS/Linux only as of May 2026. Windows users need WSL2 or to wait for native support.
- Free through August 11, 2026; metered on Notion credits after.

### Minimal Worker

```bash
ntn workers new my-worker      # scaffold
cd my-worker
# Edit src/index.ts
ntn workers deploy --name my-worker
```

`src/index.ts`:
```typescript
import { Worker } from "@notionhq/workers";

const worker = new Worker();
export default worker;

worker.tool("greet", {
  title: "Greet a User",
  description: "Returns a friendly greeting",
  inputSchema: { type: "object", properties: { name: { type: "string" } }, required: ["name"] },
  execute: async ({ name }) => `Hello, ${name}!`,
});
```

### Webhook capability

```typescript
worker.webhook("onGithubPush", {
  title: "GitHub Push Handler",
  execute: async (events, { notion }) => {
    for (const event of events) {
      // event.body, event.rawBody (for signature verification), event.headers
      console.log("got delivery", event.deliveryId);
    }
  },
});
```

After deploy: `ntn workers webhooks list` shows the URL Notion generates. Treat that URL as a secret — anyone with it can POST events unless you add signature verification.

### Worker lifecycle commands

```bash
ntn workers deploy
ntn workers list
ntn workers exec <capability-key> -d '{"name": "world"}'
ntn workers sync trigger <key>            # run a sync now
ntn workers sync pause <key>
ntn workers env set GITHUB_WEBHOOK_SECRET=...
ntn workers runs list                     # recent invocations
ntn workers runs logs <run-id>
ntn workers webhooks list
```

When asked to build a Worker, scaffold with `ntn workers new`, write the code in `src/index.ts`, set any secrets with `ntn workers env set`, and deploy. Notion's docs at https://developers.notion.com/workers cover the full API surface.

## Notion-Flavored Markdown (used by `/markdown` endpoints)

Standard CommonMark plus XML-like tags for Notion-specific blocks. Use **tabs** for indentation.

**Blocks beyond CommonMark:**
```
<callout icon="🎯" color="blue_bg">
	Ship the MVP by **Friday**.
</callout>

<details color="gray">
<summary>Toggle title</summary>
	Children indented one tab
</details>

<columns>
	<column>Left side</column>
	<column>Right side</column>
</columns>

<table_of_contents color="gray"/>
```

**Inline:**
- Mentions: `<mention-user url="..."/>`, `<mention-page url="...">Title</mention-page>`, `<mention-date start="2026-05-15"/>`
- Underline: `<span underline="true">text</span>`
- Color: `<span color="blue">text</span>` or block-level `{color="blue"}` on the first line
- Math: inline `$x^2$`, block `$$ ... $$`
- Citations: `[^https://example.com]`

**Colors:** `gray brown orange yellow green blue purple pink red`, plus `*_bg` variants for backgrounds.

Headings 5/6 collapse to H4. Multiple `>` lines render as separate quote blocks — use `<br>` inside a single `>` for multi-line quotes.

## Choosing the Right Path

| Task | mac / Linux | Windows |
|---|---|---|
| Read/write pages, search, query databases | `ntn api ...` | curl |
| Read a page for an agent to summarize | `ntn api v1/pages/{id}/markdown` | curl `/markdown` endpoint |
| Upload a file | `ntn files create < file` | 3-step HTTP flow |
| One-off API exploration | `ntn api ...` | curl |
| Build a sync / webhook / agent tool hosted by Notion | `ntn workers ...` | WSL2 + `ntn workers ...` |

## Notes

- Page/database IDs are UUIDs (with or without dashes — both accepted).
- Rate limit: ~3 requests/second average. The CLI doesn't bypass this.
- The API cannot set database **view** filters — that's UI-only.
- Use `"is_inline": true` when creating data sources to embed them in a page.
- Always pass `-s` to curl to suppress progress bars (cleaner agent output).
- Pipe JSON through `jq` when reading: `... | jq '.results[0].properties'`.
- `ntn api` supports `--file /path/to/body.json` as an alternative to `-d` for large JSON payloads.
- Notion also ships an MCP server now (`Notion MCP`, ~91% more token-efficient on DB ops than the previous version) — wire it via Hermes' MCP support if you want streaming Notion access from inside a session, but the paths above are enough for most one-shot tasks.

## Official Documentation

**When in doubt about Notion CLI behavior, API shapes, or syntax — check the official docs first before guessing.** The `ntn` CLI is also self-documenting via `ntn api ls`, `ntn api <path> --spec`, and `ntn api <path> --docs`.

- **CLI overview:** https://developers.notion.com/cli/get-started/overview
- **API requests syntax (inline, -d, --file, query params):** https://developers.notion.com/cli/guides/api-requests
- **Data sources (query, create, update, templates):** https://developers.notion.com/cli/guides/data-sources
- **Command reference (all ntn commands):** https://developers.notion.com/cli/reference/commands
- **Authentication (keychain, --no-browser, env vars):** https://developers.notion.com/cli/get-started/authentication

## Common Pitfalls

> **Reference:** See `references/select-option-operations.md` for detailed recipes on renaming select options, deleting properties, and avoiding emoji-duplicate options.

1. **ntn fails with "Failed to build public API request" in Hermes.** `NOTION_API_TOKEN` and `NOTION_KEYRING=0` must be in `~/.hermes/.env` (not just `NOTION_API_KEY`). Without `NOTION_API_TOKEN`, ntn can't find the token. Without `NOTION_KEYRING=0`, it tries the OS keychain which doesn't exist in a headless environment. If only `NOTION_API_KEY` is set, add the other two with the same token value.

2. **ntn shell escaping in terminal tool.** Inline `export VAR=$(...)` with parentheses and quotes gets mangled by the Hermes terminal shell wrapper. **Use `ntn -d '{"key":"value"}'` to pass JSON bodies directly** — this avoids all shell escaping issues and keeps commands as one-liners. Do NOT wrap ntn calls in Python `subprocess.run()` — the user has corrected this reflex multiple times. The CLI is designed to be called directly from the terminal. For complex JSON bodies, write the JSON to a temp file and use `ntn api ... -d @/tmp/body.json`, or use heredoc with `--file`.

3. **ntn fails with "No workspace selected" despite correct token/keyring env vars.** `NOTION_API_TOKEN` and `NOTION_KEYRING=0` are set correctly but ntn still can't resolve the workspace. Fix: set `NOTION_WORKSPACE_ID` in `~/.hermes/.env` (find it in the Notion URL or via `ntn api v1/users` if you can get past the error). If you can't get the workspace ID easily, **fall back to curl immediately** — the curl path (Path B) works perfectly with just `NOTION_API_KEY` and no workspace selection. Don't waste turns trying to fix ntn env when curl works.

4. **DEFAULT TO ntn CLI, NOT MCP TOOLS.** Agents reach for the MCP Notion tools (notion_find, notion_query_data_source, etc.) by reflex because they're function-shaped, but ntn CLI returns compact JSON (~500 chars) vs MCP's 3-5KB wrappers. **Use ntn for ALL standard operations**: search, query, read pages, create/update/delete items, archive blocks. MCP tools should only be used for operations ntn genuinely cannot do (e.g. typed relation creation via notion_create_data_source_item_from_values, schema inspection with notion_inspect_data_source). When in doubt, use ntn.

   Key ntn operations agents forget exist:
   ```bash
   # Query a data source
   ntn api v1/data_sources/{data_source_id}/query -X POST

   # Delete/archive a block (page item)
   ntn api v1/blocks/{block_id} -X PATCH archived:=true

   # Inspect a data source schema
   ntn api v1/data_sources/{data_source_id}

   # Search
   ntn api v1/search query="Objets"

   # Delete a property from a data source (set to null)
   ntn api v1/data_sources/{data_source_id} -X PATCH 'properties[PropertyName]:=null'

   # Or with -d for clarity:
   ntn api v1/data_sources/{data_source_id} -X PATCH -d '{"properties":{"PropertyName":null}}'
   ```

5. **Renaming a select option by ID does NOT work via data_source PATCH.** The Notion API silently ignores name changes to existing select options when you PATCH `/v1/data_sources/{id}` with the option ID + new name. The workaround:
   1. Add a new option with the desired name: `ntn api v1/data_sources/{id} -X PATCH -d '{"properties":{"Thématique":{"select":{"options":[{"name":"🏙️ Vie terrestre","color":"gray"}]}}}}'`
   2. Reassign all pages that had the old option to the new one: `ntn api v1/pages/{page_id} -X PATCH -d '{"properties":{"Thématique":{"select":{"name":"🏙️ Vie terrestre"}}}}'`
   3. Remove the old option by PATCHing the data source with only the desired options list (omitting the old option ID).
   
   **WARNING**: If you PATCH the options list to remove an option that pages still reference, those pages lose their select value (it becomes null). Always reassign pages BEFORE removing the old option.

6. **`.env` variable references don't resolve.** If `~/.hermes/.env` contains `NOTION_API_TOKEN=*** `, the `$NOTION_API_KEY` is a literal string, not a variable reference. `.env` files are not shell scripts — they don't expand variables. Always write the actual token value: `NOTION_API_TOKEN=ntn_abc123...`. If `NOTION_API_TOKEN` is missing or invalid, ntn fails with "API token is invalid" even when `NOTION_API_KEY` is correct.

7. **`ntn api -d` for JSON bodies, NOT Python wrappers.** Do NOT wrap `ntn` calls in Python `subprocess.run()` scripts. Call `ntn` directly from the terminal. For complex JSON, use `-d '{"...": "..."}'` or `--file /tmp/body.json`. Python wrappers around ntn are unnecessary indirection — the CLI is designed for direct terminal use.
