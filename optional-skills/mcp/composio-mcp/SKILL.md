---
name: composio-mcp
description: "Connect agents to external apps via Composio."
license: MIT
version: 1.1.1
author: afonsoft
platforms: [linux, macos, windows]
compatibility: CLI mode needs Node.js + `composio` on PATH (npm i -g composio-core
  @composio/cli or via `composio setup`). MCP fallback mode needs an HTTP-capable
  MCP client and a `ck_*` consumer key from the Composio dashboard. Works on macOS/Linux/Windows.
metadata:
  hermes:
    tags: [MCP, Composio, Integrations, Apps]
  version: 1.1.1
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
  homepage: https://docs.composio.dev
  sources: https://docs.composio.dev/docs/composio-connect, https://github.com/ComposioHQ/composio,
    https://docs.composio.dev/kb/guide/consumer-project-boundaries-and-auth-selection
  openclaw: '{"envVars":[{"name":"COMPOSIO_API_KEY","required":false,"description":"Project
    API key (ak_*) for the CLI path."},{"name":"COMPOSIO_CONSUMER_KEY","required":false,"description":"Consumer
    key (ck_*) for the MCP fallback path."}]}'
---

# Composio — CLI (primary) + MCP (fallback)

Connect AI agents to 1000+ external apps through Composio. This skill covers **two execution paths** and the full tool surface.

| Path | Transport | Auth | When to use |
|------|-----------|------|-------------|
| **A. CLI (primary)** | `composio` binary on PATH | `ak_*` project API key | Headless servers, scripts, any agent with shell access. Works once `composio login` succeeds. |
| **B. MCP (fallback)** | HTTP `https://connect.composio.dev/mcp` | `ck_*` consumer key in `x-consumer-api-key` | MCP-native agents that prefer tool calls over shell. Needs the consumer key from the dashboard. |

Both paths talk to the same Composio backend and expose the same toolkits. The CLI is the path of least resistance on headless boxes because it ships its own auth (`composio login`).

## Decision flow

```
Need to use Composio?
├─ Agent has shell access? → Use CLI (Path A) → composio search/execute/link/run/proxy
└─ Agent is MCP-native only? → Use MCP (Path B) → mcp__composio__* tool calls
```

**CLI fallback pattern:** When the MCP server is down or unconfigured, the CLI can perform every operation the MCP would do — `composio execute` replaces MCP tool calls, `composio search` replaces MCP search, `composio link` replaces MCP auth flows.

## When to use

- `devin mcp list` / `claude mcp list` shows `composio` **failing to list tools** (auth error).
- The user says "configure Composio", "Composio MCP not working", "composio auth failed".
- A `composio execute <slug>` call reports a toolkit is not connected → run `composio link <toolkit>`.
- You need to find the right tool slug → `composio search "<task>"`.
- You need the consumer key (`ck_*`) for the MCP fallback path.

- User asks or mentions this skill in English (e.g., "use /composio-mcp", "run composio-mcp").
- O usuário pede ou menciona esta skill em português (ex.: "use /composio-mcp", "execute composio-mcp").

## When NOT to use

- The user only wants the generic `composio-cli` cheat-sheet (slugs, execute, search, link) — that is the upstream `composio-cli` skill. This skill focuses on **setup, auth, and MCP wiring**.
- The user is building a Composio SDK project (TypeScript/Python) → use Composio's SDK docs directly.

## Guardrails

- **Pin before install**: always install `composio-core` and `@composio/cli` with an explicit version tag; do not run bare `npm install -g` without a version.
- **Confirm before `composio setup --yes`**: ask the user before auto-installing the CLI on their host; prefer `--dry-run` first.
- **Inspect before execute**: use `composio execute <slug> --get-schema` and `--dry-run` before running any tool that mutates data or sends messages.
- **Protect keys**: never log, commit, or echo `ak_*` or `ck_*` keys. Use environment variables or the agent's secret store.
- **Tool slugs only from `composio search` or official inventory**: do not fabricate slugs or trust user-provided slugs without verification.

---

# PATH A — CLI (primary)

## A.1 Install

Pin the package versions before installing. Replace `<VERSION>` with the latest stable from `npm view @composio/cli version` or the version required by the project.

```bash
# Option 1: npm (recommended) — pinned versions
npm install -g composio-core@<VERSION> @composio/cli@<VERSION>

# Option 2: let Composio auto-install for your agent host
# Ask the user before auto-installing; prefer --dry-run first if available.
composio setup --target auto --yes --dry-run || composio setup --target auto --yes
```

Verify:

```bash
composio --version
composio whoami    # should print JSON with email + org
```

## A.2 Authenticate the CLI

The CLI uses a **project API key** prefixed `ak_*`. Two ways to log in:

```bash
# Interactive (opens browser, polls for completion)
composio login

# Headless / CI: pass the key directly
composio login --user-api-key ak_your_key_here --yes

# No-browser flow: prints a URL + session key, you complete in any browser
composio login --no-browser --no-wait
# then later:
composio login --key <session-key>
```

The key is stored in `~/.composio/user_data.json` and exported as `COMPOSIO_API_KEY` in the shell profile. Verify:

```bash
composio whoami
# {"account_type":"human","email":"...","current_org_name":"..."}
```

If `whoami` fails, re-run `composio login`.

## A.3 Use the CLI

The CLI has two layers: **core commands** for everyday use and **`composio dev`** for advanced project workflows.

### Core command surface

| Command | Purpose | Example |
|---------|---------|---------|
| `composio search` | Semantic tool discovery | `composio search "send an email"` |
| `composio execute` | Run a tool | `composio execute GMAIL_SEND_EMAIL -d '{...}'` |
| `composio link` | Connect an app account | `composio link github` |
| `composio listen` | Subscribe to trigger events | `composio listen --toolkit github --trigger GITHUB_COMMIT_EVENT` |
| `composio run` | Inline TS/JS with helpers | `composio run 'await execute("...")'` |
| `composio proxy` | Raw API access | `composio proxy https://api.github.com/user --toolkit github` |
| `composio login` | Authenticate | `composio login --user-api-key ak_...` |
| `composio whoami` | Check auth | `composio whoami` |
| `composio setup` | Install plugins | `composio setup --target auto` |
| `composio orgs` | Manage orgs | `composio orgs list` |
| `composio config` | CLI config | `composio config` |

### Default workflow (execute → search → link)

1. **Start with `composio execute <slug>`** whenever the slug is known.
2. **Parallel calls** → `composio execute -p/--parallel` with repeated `<slug> -d <json>` groups.
3. **Toolkit not connected?** → `composio link <toolkit>` and retry.
4. **Arguments unclear?** → `composio execute <slug> --get-schema` or `--dry-run` before guessing.
5. **Slug unknown?** → `composio search "<task>"` (batch related queries into one call).

```bash
# Known slug — just execute
composio execute GITHUB_GET_THE_AUTHENTICATED_USER -d '{}'

# Unknown slug — search first
composio search "create a github issue"
composio search "send an email" --toolkits gmail

# Inspect before executing
composio execute GITHUB_CREATE_ISSUE --get-schema
composio execute GITHUB_CREATE_ISSUE --dry-run -d '{ owner: "acme", repo: "app", title: "Bug" }'

# Pass data from file or stdin
composio execute GITHUB_CREATE_ISSUE -d @issue.json
cat issue.json | composio execute GITHUB_CREATE_ISSUE -d -

# Upload a local file
composio execute SLACK_UPLOAD_OR_CREATE_A_FILE_IN_SLACK \
  --file ./image.png \
  -d '{ channels: "C123" }'

# Parallel independent calls
composio execute --parallel \
  GMAIL_SEND_EMAIL -d '{ recipient_email: "a@b.com", subject: "Hi" }' \
  GITHUB_CREATE_ISSUE -d '{ owner: "acme", repo: "app", title: "Bug" }'

# Skip connection check (when you know the account is linked)
composio execute GITHUB_CREATE_ISSUE --skip-connection-check -d '{...}'
```

**Key flags:**

| Flag | Purpose |
|------|---------|
| `--get-schema` | Inspect required arguments without executing |
| `--dry-run` | Preview the request shape without performing the action |
| `--file <path>` | Inject a local file into a tool with exactly one uploadable file input |
| `--account <alias>` | Pick a connected account when multiple exist for the same toolkit |
| `--parallel` / `-p` | Execute multiple independent tool calls in the same invocation |
| `--skip-connection-check` | Skip the connected-account check |
| `--skip-tool-params-check` | Skip input validation against cached schema |
| `--skip-checks` | Skip both checks above |

### `composio run` — scripting without SDK

`composio run` executes an inline ESM JavaScript/TypeScript snippet with authenticated `execute()`, `search()`, `proxy()`, and the experimental `experimental_subAgent()` helper pre-injected. No SDK setup required.

```bash
composio run '
  const me = await execute("GITHUB_GET_THE_AUTHENTICATED_USER");
  console.log(me.data.login);
'

composio run '
  const [emails, issues] = await Promise.all([
    execute("GMAIL_FETCH_EMAILS", { max_results: 5 }),
    execute("GITHUB_LIST_REPOSITORY_ISSUES", { owner: "acme", repo: "app", state: "open" }),
  ]);
  const brief = await experimental_subAgent(`Summarize:\n${emails.prompt()}\n${issues.prompt()}`);
  console.log(brief);
'
```

**Structured output with zod:**

```bash
composio run --logs-off '
  const emails = await execute("GMAIL_FETCH_EMAILS", { max_results: 5 });
  const brief = await experimental_subAgent(
    `Summarize these emails and count them.\n\n${emails.prompt()}`,
    { schema: z.object({ summary: z.string(), count: z.number() }) }
  );
  console.log(brief.structuredOutput);
'
```

**Injected helpers:** `execute()`, `search()`, `proxy()`, `experimental_subAgent()`, `result.prompt()`, `z` (zod).

### `composio listen` — subscribe to trigger events

```bash
composio listen --toolkit github --trigger GITHUB_COMMIT_EVENT
```

Subscribe to toolkit trigger events and stream them to stdout. Useful for reactive workflows and automation.

### `composio dev` — advanced workflows

Developer-scoped management: scaffolding, playground execution, logs, connected accounts, triggers, projects.

```bash
composio dev init                          # scaffold a project
composio dev toolkits list                 # browse all toolkits
composio dev toolkits info github          # inspect a toolkit
composio dev auth-configs list             # auth configs
composio dev connected-accounts list       # connected accounts
composio dev triggers list                 # trigger types
composio dev playground-execute            # playground execution
composio dev logs tools                    # browse tool logs
```

### `composio generate` — type stubs

```bash
composio generate --toolkits github gmail --output-dir ./src/types
```

See **`references/cli-reference.md`** for the full command surface.

---

# PATH B — MCP fallback

## B.1 Get the consumer key (`ck_*`)

The MCP endpoint `https://connect.composio.dev/mcp` does **not** accept the `ak_*` project API key. It requires a **consumer key** prefixed `ck_*`, which is a separate credential.

> **`ck_*` vs `ak_*`** — Consumer keys (`ck_*`) authenticate MCP clients connecting to the Connect endpoint. Project API keys (`ak_*`) authenticate backend API calls (`backend.composio.dev/api/v3/...`) and the CLI. They are distinct; one cannot substitute for the other.

How to get the consumer key:

1. Open the dashboard: **https://dashboard.composio.dev/**
2. Go to **For You → Connect Settings → Sessions & API Key**
3. Copy the consumer key (starts with `ck_`)
4. (Optional) Rotate it with **Regenerate** — this immediately invalidates the old key across all MCP clients.

## B.2 Configure the MCP server

The MCP server is a **streamable HTTP** server at `https://connect.composio.dev/mcp`. It needs the `x-consumer-api-key` header.

### Universal config (any stdio/http MCP client)

```json
{
  "mcpServers": {
    "composio": {
      "type": "http",
      "url": "https://connect.composio.dev/mcp",
      "headers": {
        "x-consumer-api-key": "ck_your_consumer_key_here"
      }
    }
  }
}
```

### Env-substituted (recommended — avoids committing the key)

```json
{
  "mcpServers": {
    "composio": {
      "type": "http",
      "url": "https://connect.composio.dev/mcp",
      "headers": {
        "x-consumer-api-key": "${COMPOSIO_CONSUMER_KEY}"
      }
    }
  }
}
```

Then export in your shell profile (`~/.bashrc` / `~/.zshrc`):

```bash
export COMPOSIO_CONSUMER_KEY="ck_your_consumer_key_here"
```

### Per-platform config — critical gotchas

Each MCP client platform has its own config format. Getting field names wrong causes the server to be **silently ignored** (no error, just no tools). See **`references/platform-quirks.md`** for the full matrix.

| Platform | Config file | Root key | URL field | Gotcha |
|----------|-------------|----------|-----------|--------|
| Claude Code | `~/.claude.json` | `mcpServers` | `url` | `type: "http"` |
| Claude Desktop | `claude_desktop_config.json` | `mcpServers` | `url` | — |
| Cursor | `~/.cursor/mcp.json` | `mcpServers` | `url` | — |
| Devin CLI | `~/.config/devin/mcp_config.json` | `mcpServers` | `url` | `devin mcp add` CLI |
| Devin Desktop | `~/.devin/mcp_config.json` | `mcpServers` | **`serverUrl`** | NOT `url`! |
| OpenCode | `~/.config/opencode/opencode.json` | **`mcp`** | `url` | `type: "remote"`, `environment` not `env` |
| Antigravity IDE/CLI | `~/.gemini/config/mcp_config.json` | `mcpServers` | **`serverUrl`** | NOT `url`! Clear cache on uninstall |
| OpenClaw | OpenClaw config | **`mcp.servers`** | `url` | `transport: "streamable-http"`, `openclaw mcp add` CLI |

> **Top 3 silent-failure traps:**
> 1. **Devin Desktop / Antigravity** use `serverUrl` (not `url`) — using `url` = silently ignored.
> 2. **OpenCode** uses `mcp` (not `mcpServers`), `environment` (not `env`), `command` as single array.
> 3. **OpenCode** env substitution uses `{env:VAR}` not `${VAR}`.

See **`references/mcp-config.md`** for the exact JSON block per platform.

### Automated setup helper

Run the bundled helper to detect **all** installed platforms and patch each one with the correct format (handles `serverUrl` vs `url`, `mcp` vs `mcpServers`, `environment` vs `env`, and OpenClaw CLI):

```bash
bash skills/composio-mcp/scripts/setup_composio_mcp.sh
# or with the key inline:
COMPOSIO_CONSUMER_KEY=ck_xxx bash skills/composio-mcp/scripts/setup_composio_mcp.sh
# dry-run (show what would change):
COMPOSIO_CONSUMER_KEY=ck_xxx bash skills/composio-mcp/scripts/setup_composio_mcp.sh --dry-run
# target one platform:
bash skills/composio-mcp/scripts/setup_composio_mcp.sh --platform cursor
# remove:
bash skills/composio-mcp/scripts/setup_composio_mcp.sh --remove
```

## B.3 Verify the MCP server

After configuring, restart the agent and check:

```bash
# From the agent (Claude Code / Devin):
#   mcp_list_tools for composio should return tools

# From the shell, test the endpoint directly:
curl -sS -X POST "https://connect.composio.dev/mcp" \
  -H "Content-Type: application/json" \
  -H "x-consumer-api-key: $COMPOSIO_CONSUMER_KEY" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
# Expect: a JSON-RPC InitializeResult (not an "Authorization required" error)
```

If you see `{"error":"Authorization required"}`, the consumer key is missing, wrong, or revoked. Re-check the dashboard and rotate if needed.

## B.4 MCP tool surface

The MCP endpoint exposes the same toolkits as the CLI. When `mcp_list_tools` succeeds, tools appear as `mcp__composio__<SLUG>` (e.g. `mcp__composio__GMAIL_SEND_EMAIL`). The underlying slugs are identical to the CLI — `composio search` and `composio execute` use the same names.

## B.5 Optional: enforce API key on the MCP server (org-level)

Orgs can require that **every** MCP request carry a valid project API key (`ak_*`) in addition to the consumer key:

```bash
curl -X PATCH "https://backend.composio.dev/api/v3/org/project/config" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $COMPOSIO_API_KEY" \
  -d '{"require_mcp_api_key": true}'
```

When enabled, MCP requests must include **both** `x-consumer-api-key` (ck_) and `x-api-key` (ak_). Default is disabled.

---

# Authentication reference

| Credential | Prefix | Where it lives | What it authenticates |
|-----------|--------|----------------|----------------------|
| Project API key | `ak_*` | `~/.composio/user_data.json`, `$COMPOSIO_API_KEY` | CLI commands, `backend.composio.dev` API |
| Consumer key | `ck_*` | dashboard only; you paste into MCP config | MCP Connect endpoint (`connect.composio.dev/mcp`) |
| Connected account (per toolkit) | — | browser OAuth via `composio link <toolkit>` | Individual app access (Gmail, GitHub, Slack…) |
| AuthKit JWT | — | OAuth flow | Alternative MCP bearer auth (rare; for OAuth-based deployments) |

## Auth flow diagram

```
┌─────────────────────────────────────────────────────────────┐
│  CLI path (ak_*)                                            │
│  composio login ──► ~/.composio/user_data.json              │
│  composio whoami ──► verify                                 │
│  composio link <toolkit> ──► browser OAuth per app          │
│  composio execute <slug> ──► uses ak_* + connected account  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  MCP path (ck_*)                                            │
│  dashboard → Connect Settings → copy ck_*                   │
│  patch mcp config: headers.x-consumer-api-key = ck_*        │
│  restart agent → mcp_list_tools(composio) → tools            │
│  (optional) require_mcp_api_key=true → also send x-api-key  │
└─────────────────────────────────────────────────────────────┘
```

---

# Tool discovery & consultation

## How to find the right tool

1. **Semantic search** — `composio search "<describe the task>"` uses semantic search across all toolkits. This is the primary discovery mechanism.
2. **Narrow by toolkit** — when you know the app, filter: `--toolkits <slug>`.
3. **Inspect the schema** — `composio execute <slug> --get-schema` shows required inputs before running.
4. **Dry-run** — `composio execute <slug> --dry-run -d '{...}'` validates without side effects.

## Tool inventory

The live catalog is 30 toolkits / ~3300 tools / ~170 triggers. See **`references/tools-inventory.md`** for the full toolkit breakdown and common tool patterns.

| Toolkit | Slug | Tools | Triggers |
|---------|------|-------|----------|
| Gmail | `gmail` | 61 | 2 |
| GitHub | `github` | 871 | 46 |
| Slack | `slack` | 158 | 9 |
| Google Calendar | `googlecalendar` | 45 | 7 |
| Notion | `notion` | 53 | 8 |
| Jira | `jira` | 97 | 17 |
| HubSpot | `hubspot` | 244 | 2 |
| … | … | … | … |

## Tags

Tools are tagged for filtering. Common tags:

| Tag | Meaning |
|-----|---------|
| `important` | Core/high-value tools |
| `destructiveHint` | Irreversible or data-destroying |
| `idempotentHint` | Safe to retry |
| `createHint` | Creates a resource |
| `updateHint` | Modifies a resource |
| `deleteHint` | Deletes a resource |
| `readOnlyHint` | No side effects |
| `openWorldHint` | Results depend on external state |
| `batch` | Bulk operation |

---

# CLI as MCP fallback

When the MCP endpoint is unreachable or unconfigured, the CLI can perform every operation the MCP would do:

| MCP operation | CLI equivalent |
|---------------|----------------|
| `mcp__composio__GMAIL_SEND_EMAIL` | `composio execute GMAIL_SEND_EMAIL -d '{...}'` |
| Tool discovery | `composio search "<task>"` |
| Account linking | `composio link <toolkit>` |
| Trigger subscription | `composio listen --toolkit <toolkit> --trigger <slug>` |
| Raw API calls | `composio proxy <url> --toolkit <toolkit>` |
| Multi-step workflows | `composio run '<code>'` |

The CLI path is more reliable on headless servers because it handles auth locally (`ak_*` key in `~/.composio/user_data.json`) and does not depend on the MCP endpoint or consumer key.

---

# Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `composio whoami` fails / empty | Not logged in | `composio login` (or `--user-api-key ak_...`) |
| `composio execute` says "toolkit not connected" | App account not linked | `composio link <toolkit>` then retry |
| MCP `Failed to list tools for composio` | Missing/wrong `x-consumer-api-key` header | Get `ck_*` from dashboard, patch config (B.2), restart agent |
| MCP `Authorization required: Bearer token rejected` | Sent `ak_*` where `ck_*` expected | Use consumer key (`ck_*`), not project API key |
| MCP `Authorization required: No Authorization header` | No header at all | Add `x-consumer-api-key` header to the MCP config |
| `composio login` hangs on headless box | Browser flow needs a display | Use `--no-browser --no-wait` then `--key <session>` or `--user-api-key ak_...` |
| Tools appear but execute returns 401 | `require_mcp_api_key` enabled, no `x-api-key` | Add `x-api-key: ak_*` header alongside `x-consumer-api-key` |
| `composio search` returns no results | Cache stale or org not set | `rm -rf ~/.composio/toolkits.json` then retry |

---

# References

- **`references/cli-reference.md`** — Full `composio` CLI command reference (core + `dev` + `generate` + `setup`).
- **`references/tools-inventory.md`** — Live toolkit inventory (~3300 tools across 30 toolkits) and common tool patterns.
- **`references/mcp-config.md`** — Full per-platform JSON config blocks (Claude Code/Desktop, Cursor, Devin CLI/Desktop, OpenCode, Antigravity IDE/CLI, OpenClaw).
- **`references/platform-quirks.md`** — Cross-platform MCP config quirks matrix (serverUrl vs url, mcp vs mcpServers, environment vs env, env substitution syntax, OpenClaw CDP ports).
- **`references/troubleshooting.md`** — Extended troubleshooting (CLI cache, pending-login, org picker, `composio dev` projects).
- **`scripts/setup_composio_mcp.sh`** — Detects all installed platforms and patches each with the correct format (handles serverUrl/url, mcp/mcpServers, environment/env, OpenClaw CLI).
- **`scripts/verify_composio.sh`** — Runs `whoami` + curl initialize probe + lists a few tools to confirm end-to-end.
- [Composio Connect docs](https://docs.composio.dev/docs/composio-connect)
- [Consumer vs project key boundaries](https://docs.composio.dev/kb/guide/consumer-project-boundaries-and-auth-selection)
- [Devin CLI MCP configuration](https://docs.devin.ai/cli/extensibility/mcp/configuration)
- [Antigravity MCP docs](https://antigravity.google/docs/mcp/)
- [OpenCode MCP servers](https://opencode.ai/docs/mcp-servers/)
- [OpenClaw MCP tools](https://docs.openclaw.ai/tools/mcp)
- [Cursor MCP docs](https://cursor.com/docs/mcp)
- Upstream `composio-cli` skill for the full command cheat-sheet.
