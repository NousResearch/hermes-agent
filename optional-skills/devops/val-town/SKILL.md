---
name: val-town
description: Deploy HTTP endpoints, cron jobs, and SQLite-backed utilities on Val Town — a serverless TypeScript platform. Use when a task needs a persistent deployed capability that outlives the session: webhooks, scheduled jobs, shared URLs, or lightweight stateful APIs.
version: 1.0.0
author: Hermes Agent
license: MIT
required_environment_variables:
  - name: VAL_TOWN_API_KEY
    prompt: Val Town API token
    help: "Create one at https://www.val.town/settings/api — enable val:read and val:write scopes. Not needed when using the Val Town MCP."
    required_for: REST API fallback
metadata:
  hermes:
    tags: [Serverless, TypeScript, Webhooks, Cron, Deployment, Persistence, Val Town]
    homepage: https://val.town
    requires_toolsets: [terminal]
---

# Val Town

Deploy serverless TypeScript vals — HTTP endpoints, cron jobs, and SQLite-backed utilities
— directly from the agent. Vals are versioned, permanently addressable, and run on Deno.

## When to Use

Reach for Val Town when the task needs:

- **A webhook receiver** — an external service (GitHub, Stripe, Zapier) needs a URL to POST events to
- **A cron job** — recurring work (hourly monitor, daily digest, cleanup sweep) that must run when the user's machine is off
- **A shared utility endpoint** — a small API the user wants to hand to a colleague or embed in a workflow
- **Persistent state** — lightweight data that must survive between agent sessions (via the per-val SQLite)
- **Offloaded computation** — a periodic task that should not block or require the user's machine

## When NOT to Use

- One-time local scripts or data transformations — use the `terminal` tool instead
- The user has no Val Town account and the task is time-sensitive — sign-up is fast but requires a browser step
- Outputs are only used locally (file processing, local DB migrations)
- Strict latency requirements where an extra network hop to Val Town is unacceptable

---

## Setup

### Option 1: Val Town MCP (preferred)

Check if the Val Town MCP is already connected by listing available tools for names prefixed
with `valtown` or `val_town`. If present, use the MCP directly for all operations — it handles
auth, file uploads, and trigger wiring without manual `curl` commands.

To connect the MCP to Hermes, add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  val-town:
    url: https://api.val.town/v3/mcp
    transport: sse
    headers:
      Authorization: "Bearer $VAL_TOWN_API_KEY"
```

Then restart Hermes. Verify with: `hermes tools list | grep -i val`

### Option 2: REST API via terminal (fallback)

All playbooks below include `curl` commands using `VAL_TOWN_API_KEY`. Verify the key works:

```bash
curl -s https://api.val.town/v2/vals \
  -H "Authorization: Bearer $VAL_TOWN_API_KEY" | jq '.data[0].name // "no vals yet"'
```

---

## API Quick Reference

Base URL: `https://api.val.town`  Auth header: `Authorization: Bearer $VAL_TOWN_API_KEY`

| Operation | Endpoint |
|-----------|----------|
| List my vals | `GET /v2/vals` |
| Create val (project container) | `POST /v2/vals` |
| Get val by ID | `GET /v2/vals/{val_id}` |
| Delete val | `DELETE /v2/vals/{val_id}` |
| **Create** a file with trigger | `POST /v2/vals/{val_id}/files?path=<filename>` |
| **Update** an existing file | `PUT /v2/vals/{val_id}/files?path=<filename>` |
| List files | `GET /v2/vals/{val_id}/files?path=&recursive=true` |
| Get file content | `GET /v2/vals/{val_id}/files/content?path=<filename>` |
| Set env var | `POST /v2/vals/{val_id}/environment_variables` |
| Update env var | `PUT /v2/vals/{val_id}/environment_variables/{key}` |

**File trigger types** (set via `type` in the POST body):

| `type` value | Trigger |
|---|---|
| `"http"` | HTTP endpoint |
| `"interval"` | Cron / scheduled job |
| `"email"` | Email handler |
| `"script"` | Manually-run script |
| `"file"` | Static asset (no trigger) |

Val names: `kebab-case`, ≤48 chars, pattern `^[a-zA-Z][a-zA-Z0-9\-_]*$`.

---

## Playbook A: HTTP Val (Webhook or Utility Endpoint)

**Trigger signature** — the export determines the trigger type:

```typescript
// http.ts
export default async function(req: Request): Promise<Response> {
  try {
    const body = await req.json().catch(() => null);
    console.log("Received:", JSON.stringify(body));
    return Response.json({ ok: true, received: body });
  } catch (err) {
    return Response.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
```

### Deploy via MCP

Ask the MCP to create a val, add an HTTP trigger file with the above content, and return the
endpoint URL.

### Deploy via REST API

```bash
# 1. Create the val (project container)
VAL=$(curl -s -X POST https://api.val.town/v2/vals \
  -H "Authorization: Bearer $VAL_TOWN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"my-webhook","privacy":"unlisted"}')

VAL_ID=$(echo "$VAL" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "Val ID: $VAL_ID"

# 2. Create the HTTP trigger file — type:"http" registers the endpoint
HANDLER='export default async function(req: Request): Promise<Response> {
  try {
    const body = await req.json().catch(() => null);
    console.log("Received:", JSON.stringify(body));
    return Response.json({ ok: true, received: body });
  } catch (err) {
    return Response.json({ ok: false, error: String(err) }, { status: 500 });
  }
}'

RESULT=$(curl -s -X POST "https://api.val.town/v2/vals/$VAL_ID/files?path=http.ts" \
  -H "Authorization: Bearer $VAL_TOWN_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"type\": \"http\", \"content\": $(echo "$HANDLER" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")}")

# 3. The endpoint URL is in the create response — no second request needed
ENDPOINT=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['links']['endpoint'])")
echo "Endpoint: $ENDPOINT"

# 4. Smoke-test it
curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"test": "hello from hermes"}'
```

**Always return the endpoint URL to the user.** They pass it to the external service as the
webhook URL. Use `unlisted` privacy for webhooks — they don't appear on the profile but are
accessible to anyone with the URL.

> **Key rule:** `POST /v2/vals/{id}/files` creates a new file (with trigger type).
> `PUT /v2/vals/{id}/files` updates an existing file's content. Do not use PUT to create.

---

## Playbook B: Cron Val (Recurring Job)

**Trigger signature:**

```typescript
// cron.ts
export default async function(interval: Interval) {
  console.log("Fired at:", new Date().toISOString());
  console.log("Last run:", interval.lastRunAt ?? "first run");
  // your recurring work here
}
```

### Deploy via REST API

```bash
VAL=$(curl -s -X POST https://api.val.town/v2/vals \
  -H "Authorization: Bearer $VAL_TOWN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"daily-digest","privacy":"private"}')

VAL_ID=$(echo "$VAL" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

HANDLER='export default async function(interval: Interval) {
  console.log("Fired at:", new Date().toISOString());
  console.log("Last run:", interval.lastRunAt ?? "first run");
  // TODO: add recurring logic here
}'

# type:"interval" registers the cron trigger
curl -s -X POST "https://api.val.town/v2/vals/$VAL_ID/files?path=cron.ts" \
  -H "Authorization: Bearer $VAL_TOWN_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"type\": \"interval\", \"content\": $(echo "$HANDLER" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")}"

echo "Val created. Open https://www.val.town to set the cron schedule."
```

> **Schedule setup:** Cron trigger intervals must be configured in the Val Town web UI
> (click the file → trigger settings) or via the MCP. The REST API file endpoint does not
> yet expose a schedule field. Free plan: minimum 15-minute interval. Pro: down to 1 minute.
> All cron times are UTC — use [crongpt.com](https://crongpt.com) for timezone conversion.

**Cron best practices:**
- Use `interval.lastRunAt` to fetch only data since the last run (avoid re-processing)
- Keep execution under 15 seconds; long-running work should be async or split across runs
- `console.log` key events — visible in Val Town's log viewer
- Keep the handler idempotent: safe to run twice if Val Town retries after a failure

---

## Playbook C: Persistent State with Val Town SQLite

Every val has its own private SQLite database. Access it from any file in that val:

```typescript
// Any .ts file in the val
import { sqlite } from "https://esm.town/v/std/sqlite";

// Initialize schema
await sqlite.execute(`
  CREATE TABLE IF NOT EXISTS events (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    kind TEXT    NOT NULL,
    data TEXT
  )
`);

// Insert
await sqlite.execute({
  sql: "INSERT INTO events (kind, data) VALUES (?, ?)",
  args: ["webhook_received", JSON.stringify({ source: "github", ref: "main" })]
});

// Query
const result = await sqlite.execute(
  "SELECT * FROM events ORDER BY id DESC LIMIT 10"
);
console.log(result.rows);
```

**Combine with an HTTP val** to expose a read endpoint:

```typescript
// http.ts — serves recent events as JSON
import { sqlite } from "https://esm.town/v/std/sqlite";

export default async function(req: Request): Promise<Response> {
  await sqlite.execute(`CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    kind TEXT NOT NULL, data TEXT
  )`);

  if (req.method === "POST") {
    const body = await req.json();
    await sqlite.execute({
      sql: "INSERT INTO events (kind, data) VALUES (?, ?)",
      args: [body.kind ?? "unknown", JSON.stringify(body)]
    });
    return Response.json({ ok: true });
  }

  const rows = await sqlite.execute("SELECT * FROM events ORDER BY id DESC LIMIT 50");
  return Response.json(rows.rows);
}
```

> SQLite limits: 10 MB free plan, 1 GB Pro. Each val has an isolated database — data
> written in one val is not visible in another.

---

## Conventions

| Convention | Rule |
|-----------|------|
| **Privacy** | Default `unlisted` for HTTP vals, `private` for crons and internal tools. Use `public` only when the user explicitly wants the code to be discoverable. |
| **Naming** | `kebab-case`, descriptive, ≤48 chars. Prefix by domain: `github-pr-notifier`, `stripe-payment-log`. |
| **Logging** | Always `console.log` key events. Logs are visible in the Val Town UI and are the primary debugging surface. |
| **Secrets** | Store via `POST /v2/vals/{id}/environment_variables`. Access in code via `Deno.env.get("KEY")` or `process.env.KEY`. Never hardcode tokens in val files. |
| **Error handling** | Wrap HTTP handlers in try/catch. Return structured error responses (`Response.json({ error: ... }, { status: 500 })`) instead of letting exceptions propagate. |
| **Branching** | For iterative development, create a `dev` branch (`POST /v2/vals/{id}/branches`), iterate there, then merge via the web UI when stable. |
| **Return URL** | Always surface the deployed endpoint URL to the user after creating an HTTP val. |

### Setting a secret via REST API

```bash
curl -s -X POST "https://api.val.town/v2/vals/$VAL_ID/environment_variables" \
  -H "Authorization: Bearer $VAL_TOWN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key": "WEBHOOK_SECRET", "value": "your-secret-here"}'
```

Access in the val: `const secret = Deno.env.get("WEBHOOK_SECRET");`

---

## Safety

**Prompt injection on public HTTP endpoints:** Any caller can POST arbitrary content. Never
pass request body fields to `eval()`, template strings fed to shell commands, or LLM prompts
without sanitizing. Treat all incoming JSON as untrusted.

**Code review before deploy:** Show the generated TypeScript to the user before uploading it.
A brief "here's what I'll deploy — shall I proceed?" prevents accidents and builds trust.

**Key scope:** Use `val:read` and `val:write` scopes only. Avoid requesting `user:write`
unless explicitly needed — it's excluded from defaults for security reasons.

**Public vs unlisted:** Default to `unlisted`. Unlisted vals are not indexed or listed on
profiles but are accessible to anyone with the URL. Only use `public` when discoverability
is the point.

**Rate limiting:** Val Town enforces execution quotas. For high-frequency public webhooks,
use Val Town SQLite to track request counts per time window and return 429 if exceeded.

---

## Cleanup

```bash
# List all owned vals
curl -s "https://api.val.town/v2/vals" \
  -H "Authorization: Bearer $VAL_TOWN_API_KEY" | jq '[.data[] | {name, id, privacy}]'

# Delete a val and all its files
curl -s -X DELETE "https://api.val.town/v2/vals/$VAL_ID" \
  -H "Authorization: Bearer $VAL_TOWN_API_KEY"
```

Always delete test vals after validation runs.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| 401 on API calls | Invalid or expired token | Regenerate at `https://www.val.town/settings/api` |
| No `links.endpoint` in file response | File created with `type:"file"` instead of `type:"http"` | Delete and recreate the file with `"type": "http"` |
| "Val not found" at endpoint URL | File trigger type not set to HTTP | Same as above |
| PUT returns 404 "File not found" | Trying to update a file that doesn't exist yet | Use POST to create first, then PUT to update |
| Cron not firing | Schedule not configured | Set interval in Val Town web UI (click trigger settings on the file) |
| Val name rejected (400) | Name violates pattern | Must match `^[a-zA-Z][a-zA-Z0-9\-_]*$`, ≤48 chars |
| 409 on val creation | Name already in use | Choose a different name or delete the existing val first |
| SQLite import error | Wrong import URL | Use `https://esm.town/v/std/sqlite` (not `npm:`) |

## References

- [Val Town docs](https://docs.val.town)
- [REST API reference](https://docs.val.town/openapi)
- [Val Town MCP guide](https://docs.val.town/guides/prompting)
- [std/sqlite](https://docs.val.town/reference/std/sqlite)
- [Deno runtime](https://docs.val.town/reference/runtime)
- [crongpt.com](https://crongpt.com) — timezone-aware cron expression generator
