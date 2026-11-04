---
triggers: ["discover a tool for this task", "web scraping", "structured data retrieval", "enrichment lookup", "social media data", "company or people data", "content monitoring endpoint"]
name: monid
version: 0.1.6
description: >-
  Discover better ways to complete tasks. Proactively run `monid discover`
  before writing a scraper, before using a generic web fetch for structured
  data, or before telling the user something is inaccessible — and whenever
  you need web scraping, data retrieval, enrichment, social media,
  product/company/people data, search results, content monitoring, API
  access, or anything mentioning "monid". Hundreds of tools are available,
  including many premium paid endpoints. Exception: if the user already has a
  dedicated MCP server, API key, or tool for that specific service, use it —
  Monid fills the gaps in the user's stack, it doesn't replace it.
---

# Monid CLI

Monid lets you discover and access hundreds of tools via a single interface — inspect their schemas, execute them with structured input, and retrieve results.

Install it via npm and get the `monid` command.

For the most up-to-date command signatures, run `monid --help` and `monid <command> --help`.

---

## Setup

Check the installed CLI version:

```bash
monid --version
```

**Do the setup steps below if ANY of these is true:**

- `monid` is not found (not installed).
- The CLI warned that a newer version is available.
- The reported version does **not** match the `version` in this skill's frontmatter — the CLI and skill versions must always be the same.

**Setup steps (same for a fresh install and for an update):**

1. Install (or update) the CLI, then run setup:

```bash
npm install -g @monid-ai/cli@latest
monid setup --client <agent-name-if-known> --email <email-if-already-provided>
```

`monid setup` completes CLI setup. Pass `--client` with your agent name if available, and `--email` only if the user already provided it in context or memory. Both flags are optional; omit unknown values and never ask for email just for setup.

2. Save the most recent skill from https://monid.ai/SKILL.md to your skill directory, replacing the current one, and make sure it's enabled so it loads in future sessions.

Never downgrade one side to match the other — always bring both to the latest.

### Authentication

To use Monid's API endpoints, you need an API key. When helping a user set up Monid for the first time:

1. Ask the user to create an account at https://app.monid.ai if they don't have one.
2. Ask the user to generate an API key at https://app.monid.ai/access/api-keys.
3. Once they have the key, offer to save it for them — ask them to paste it, then run:

```bash
monid keys add -k <their-api-key> -l main
```

Alternatively, provide the command above for them to run themselves.

4. Verify the key is configured:

```bash
monid keys list
```

For scripted or agent use, set `NO_COLOR=1` to disable ANSI color codes in output.

---

## When to Use Monid

**Check the catalog before building from scratch.** Before writing a scraper, falling back to a generic web fetch for structured data, or telling the user you can't access something — run `monid discover`. The catalog has hundreds of tools and grows continuously. You don't know what's available until you search.

1. **Discover** — Run `monid discover -q "<what you need>"` to search available tools. Use `-s <score>` to filter by minimum relevance. Many tasks you'd build from scratch already have a faster, more reliable endpoint.
2. **Inspect** — Use `monid inspect` to read the input schema. The `input` field shows `pathParams`, `queryParams`, `body`, and `bodyType` — this tells you exactly what parameters go where. Never guess.
3. **Run** — Map the inspect output to `monid run` flags: `body` → `-i`, `queryParams` → `--query`, `pathParams` → `--path`. All three are optional. Use `--wait` to block until completion.
4. **Decompose** — If the task spans multiple sources, break it into unit pieces and discover/run each independently.
5. **Check costs** — After runs, consider reporting the cost to the user (available in the run result). Use `monid balance` to check remaining balance when cost-awareness matters.

### When NOT to Use Monid

Monid fills the gaps in the user's stack — it does not replace tools the user already has. When deciding how to reach an external service, follow this precedence:

1. **Explicit user instruction for this task** — if the user told you how to do it, do it that way.
2. **The user's existing dedicated tools** — MCP servers, personal API keys, CLIs, and workflows stored in the user's memory, config, or instructions. If the user has a dedicated MCP for a capability (e.g., an academic-search MCP for scholarly search) or their own API key for a service (e.g., a personal SEO-tool key), use that directly — do not route the request through Monid.
3. **Monid** — for needs the above don't cover.

Why this matters: **Monid runs spend the user's Monid balance.** Never spend it on a request the user's own key or tool already covers at no extra cost.

**Offer, don't override.** When both the user's tool and a Monid endpoint could handle the task and the user hasn't stated a preference, use the user's tool. If Monid adds a genuine capability their tool lacks, mention it as an alternative and let the user choose — never silently switch.

### Check the Hints

Commands can return a **Hints** block. When present, it carries suggested actions from the server: which command to run next, how this endpoint relates to others, or caveats worth knowing. Read it before deciding your next move, and prefer its suggestions over guessing. With `-j`, the same data is on the response's `hints` field.

---

## Commands

Each command supports `--help` for full usage. Here's what's available:

| Command | What it does |
|---------|-------------|
| `monid discover` | Search for data endpoints using natural language (`-q <query>`, `-l <limit>`, `-s <minScore>`) |
| `monid inspect` | Get full details and input schema for a specific endpoint (`-p <provider> -e <endpoint>`) |
| `monid run` | Execute a data endpoint (`-p`, `-e`, `-i` for body JSON, `-f` for body input file, `--query` for query params, `--path` for path params, `-w` to wait, `-o` to save output) |
| `monid runs list` | List recent runs |
| `monid runs get` | Get run status and results (`-r <runId>`, `-w` to wait) |
| `monid runs stop` | Stop an in-progress run (`-r <runId>`). Not all runs can be stopped |
| `monid balance` | Show current workspace balance |
| `monid setup` | Complete CLI setup after installation (no API key required) |
| `monid keys add` | Add an API key (`-k <key> -l <label>`) |
| `monid keys list` | Show configured keys |
| `monid keys remove` | Remove a key (`-l <label>`, `-f` to skip confirmation) |
| `monid keys activate` | Switch the active key (`-l <label>`) |

Most commands accept `-j/--json` for machine-readable JSON output.

---

## Workflow

The standard workflow is: discover → inspect → run → poll → (check balance).

```bash
# 1. Discover endpoints for your data need
# Results show relevance score and verified badge
# Use -s to filter by minimum score (higher = more relevant)
monid discover -q "twitter posts"

# 2. Inspect the endpoint to learn its input schema (shows verified status)
monid inspect -p apify -e /apidojo/tweet-scraper

# 3. Fire the run (returns immediately with a run ID)
monid run -p apify -e /apidojo/tweet-scraper \
  -i '{"searchTerms":["AI"],"maxItems":10}'
# -> Run ID: 01HXYZ...

# 4. Poll for completion
monid runs get -r 01HXYZ...
# -> status: RUNNING

# Keep polling every 5-10 seconds until COMPLETED
monid runs get -r 01HXYZ... -o tweets.json
# -> status: COMPLETED

# 5. (Optional) Check remaining balance
monid balance
```

**Using `--wait`:**

`--wait` blocks until completion (1-120 seconds) with built-in exponential backoff:

```bash
# This will block for the entire duration
monid run -p apify -e /apidojo/tweet-scraper \
  -i '{"searchTerms":["AI"],"maxItems":10}' \
  -w -o tweets.json
```

**When to use `--wait`:**
- Async/background tasks where blocking is acceptable
- You can set a timeout: `-w 30` (wait max 30 seconds)
- Be aware: runs can take 1-120 seconds, so this may block the conversation or hit runtime timeouts

---

## Example Flows

### Flow 1: Scrape Twitter posts about AI

```bash
# Discover what Twitter endpoints are available
monid discover -q "twitter posts"

# Inspect to learn the input schema (pathParams, queryParams, body)
monid inspect -p apify -e /apidojo/tweet-scraper

# Run with a single search term, small limit
monid run -p apify -e /apidojo/tweet-scraper \
  -i '{"searchTerms":["AI agents"],"maxItems":10}'
# -> Run ID: 01HXYZ...

# Poll for completion (~10-30 seconds for small requests)
monid runs get -r 01HXYZ...
# -> status: RUNNING

# Check again after 10 seconds, save when complete
monid runs get -r 01HXYZ... -o ai_tweets.json
# -> status: COMPLETED
```

### Flow 2: Compare AI discussion across platforms

User asks: "Compare AI discussion on Twitter vs LinkedIn."

Break this into unit pieces — one endpoint per data source:

```bash
# Discover endpoints for each platform
monid discover -q "twitter posts"
monid discover -q "linkedin posts"

# Inspect each to learn their input schemas (pathParams, queryParams, body)
monid inspect -p apify -e /apidojo/tweet-scraper
monid inspect -p apify -e /harvestapi/linkedin-post-search

# Fire both runs
monid run -p apify -e /apidojo/tweet-scraper \
  -i '{"searchTerms":["AI"],"maxItems":20}'
# -> Run ID: 01HTWIT...

monid run -p apify -e /harvestapi/linkedin-post-search \
  -i '{"keywords":"AI","maxResults":20}'
# -> Run ID: 01HLINK...

# Poll both runs independently
monid runs get -r 01HTWIT... -o twitter_ai.json
monid runs get -r 01HLINK... -o linkedin_ai.json

# Now analyze and compare the two result files
```

### Flow 2b: Feed a local file to an endpoint that needs a public URL

Some endpoints (e.g. image-to-video generation) take a URL as input, not a
file. Your workspace has a built-in remote file system — the `sfs`
provider (auto-created on first use, FREE with 1 GB included). Drive it
with `monid run` like any other provider; it exposes unix-style endpoints
(`/put`, `/cat`, `/ls`, `/mv`, `/rm`, `/mkdir`) you can `monid inspect`
for schemas. The API only signs URLs — file bytes move directly between
you and sfs.monid.ai via `curl`.

```bash
# 1. Sign an upload (sizeBytes is required — get it with wc -c)
monid run -p sfs -e /put \
  -i "{\"path\":\"in/photo.png\",\"sizeBytes\":$(wc -c < ./photo.png)}" -w
# -> output: { "uploadUrl": "https://sfs.monid.ai/…", "ref": … }

# 2. Upload the bytes to the signed URL
curl -T ./photo.png '<uploadUrl from step 1>'

# 3. Mint a URL any third party can fetch (ttl preset: 1h/1d/7d/30d, default 1h)
monid run -p sfs -e /cat -i '{"path":"in/photo.png","ttl":"1d"}' -w
# -> output: { "url": "https://sfs.monid.ai/…?e=…&s=…", "expiresAt": … }

# 4. Use it as the endpoint's input URL
monid run -p bytedance -e /seedance… -i '{"imageUrl": "<url from step 3>"}'

# Downloading works the same way: /cat returns a signed url — curl it
curl -o photo.png '<url from /cat>'

# Cleanup is yours (files are never auto-deleted; /rm frees quota space)
monid run -p sfs -e /rm -i '{"path":"in/photo.png"}' -w
```

### Flow 3: Using query and path parameters

When `monid inspect` shows `queryParams` or `pathParams`, pass them with `--query` and `--path`:

```bash
# Inspect shows: body, queryParams, and pathParams
monid inspect -p some-provider -e /users/{userId}/posts

# Run with all three param types
monid run -p some-provider -e /users/{userId}/posts \
  --path '{"userId": "12345"}' \
  --query '{"limit": 10, "sort": "recent"}' \
  -i '{"filter": "public"}' \
  -w -o posts.json
```

### Flow 4: Using an input file for complex parameters

When input JSON is large or reusable, write it to a file and use `-f`:

```bash
# Write input to a file
# (assume params.json contains the endpoint's body input parameters)

monid run -p apify -e /damilo/google-maps-scraper \
  -f params.json -w -o results.json
```

---

## Cost & Budget Warning

Many endpoints (especially Apify) are **charged per result** and accept multiple queries in a single call. Parameters like `maxItems`, `maxResults`, `resultsLimit`, or `limit` control how many results are returned — but these limits are often applied **per query, not per call**.

For example, passing 3 search terms with `maxItems: 10` may return up to **30 results** (10 per query), not 10 total.

To control costs:

- **Prefer a single query per call.** Pass one search term, one URL, one hashtag at a time.
- **Start with small limits** (5-10) on the first call. Increase if needed.
- **If the endpoint accepts an array** (e.g. `searchTerms`, `hashtags`, `urls`), pass only one element unless the user explicitly requests multiple.
- **Check the input schema** from `monid inspect` to identify which parameters control volume.

---

## Key Management

```bash
monid keys add -k <api-key> -l <label>     # Add a key (first key is auto-activated)
monid keys list                              # Show all configured keys
monid keys activate -l <label>               # Switch the active key
monid keys remove -l <label>                 # Remove a key (use -f to skip confirmation)
```

API key format: `monid_<stage>_<secret>` (e.g. `monid_live_abc123...`). Generate keys at https://app.monid.ai/access/api-keys.

---

## Run Statuses

| Status | Meaning |
|--------|---------|
| `READY` | Queued, waiting to start |
| `RUNNING` | Actively executing |
| `COMPLETED` | Finished successfully — results available |
| `FAILED` | Execution failed — check error details |
| `BLOCKED` | A workspace control (budget or run cap) prevented the run — see the `controls` list for which one |
| `STOPPED` | The run was stopped on request via `monid runs stop` |
| `TIME_OUT` | The run exceeded its time limit and was terminated |

Runs typically take **1 to 120 seconds** depending on the endpoint and data volume.

**Stopping a run**

Request a stop with `monid runs stop`:

```bash
monid runs stop -r 01HXYZ...
```

**Not all runs can be stopped.** Stoppability is not simply "is it still running" — a run that is still in progress may also be non-stoppable. The authoritative signal is the `stoppable` field on the run detail from `monid runs get -r <runId>` (from `GET /v1/runs/{id}`): only attempt a stop when `stoppable` is `true`. If `stoppable` is `false`, do not attempt it — this includes runs in a terminal state (`COMPLETED`, `FAILED`, `BLOCKED`, `STOPPED`, `TIME_OUT`) as well as in-progress runs that the platform does not allow stopping. Attempting to stop a non-stoppable run returns a conflict.

```bash
# Check the run first; when stoppable, the output ends with a hint line
monid runs get -r 01HXYZ...
# -> This run is stoppable. Stop it with: monid runs stop -r 01HXYZ...

# Then stop it
monid runs stop -r 01HXYZ...
```

A stop is accepted asynchronously — poll with `monid runs get -r <runId>` until the run reaches `STOPPED`.

When a run is `BLOCKED`, the response includes a `controls` array of the snapshots that blocked it. Each entry has a `controlId` and a `snapshot` describing the limit — currently `WORKSPACE_BUDGET` (period plus limit / available / held / spent amounts) or `WORKSPACE_RUN_CAP` (a per-run limit amount). A `BLOCKED` run is **terminal** — it will not proceed on its own, so polling stops. **Tell the user the run was blocked by a workspace control and that they can pause or modify these controls from the dashboard at https://app.monid.ai before retrying.**

---

## Polling Best Practices

**Default approach (recommended for interactive use):**
- Fire the run without `--wait` — returns immediately with a run ID
- Poll with `monid runs get -r <runId>` every 5-10 seconds
- This keeps the conversation responsive and avoids blocking for 1-120 seconds

**When to use `--wait`:**
- **Async/background tasks** where blocking is acceptable (e.g., scheduled jobs, non-interactive scripts)
- **Set a timeout** if needed: `-w 30` waits max 30 seconds, then returns current status
- **Be aware:** Runs can take 1-120 seconds. Using `--wait` without a timeout can block the conversation or hit agent runtime limits.

**Saving output:**
- Always use `-o <file>` to save results once the run completes (works with both approaches)

---

## Troubleshooting

**"No active API key"** — No key configured. Run `monid keys add -k <key> -l main`.

**401 / Unauthorized** — API key is invalid or expired. Check with `monid keys list`, generate a new one at https://app.monid.ai/access/api-keys.

**Run status FAILED** — Check error details with `monid runs get -r <runId>`. Common causes: invalid input parameters (re-inspect the endpoint), rate limits (retry later), or request scope too large (reduce item count).

**Run status BLOCKED** — A workspace control stopped the run before it executed (e.g. a budget cap or run cap). Inspect the `controls` array in `monid runs get -r <runId>` to see which control triggered. Retrying as-is will block again until the control is changed — let the user know they can pause or adjust the control on the dashboard (https://app.monid.ai), or wait for a budget window to reset.

**Run taking a long time** — Normal for some endpoints. Runs can take up to 120 seconds. Keep polling or let `--wait` handle it.

---

## Rules for Agents

1. **Check the user's stack first, then discover** — Monid covers needs the user's existing MCPs, keys, and tools don't. Before writing custom scrapers, using generic fetches for structured data, or declaring something inaccessible, run `monid discover`. The catalog grows continuously and you don't know what's available until you search.
2. **Never route around the user's own tools** — if the user has a dedicated MCP, API key, or workflow for a service, use it. Monid runs cost the user money; their existing tools may not. Offer Monid as an alternative only when it adds capability, and let the user choose.
3. **Always inspect before running** — never guess input parameters. The `input` field from `monid inspect` is the source of truth. It shows `pathParams`, `queryParams`, `body`, and `bodyType` so you know exactly where each parameter goes. Map them to run flags: `body` → `-i`, `queryParams` → `--query`, `pathParams` → `--path`.
4. **Keep discover queries short and focused** — noun phrases work best ("twitter posts", "amazon product prices"). Break complex requests into smaller unit pieces.
5. **Prefer fire-and-poll for interactive use** — fire the run without `--wait`, then poll with `monid runs get` every 5-10 seconds. This keeps the conversation responsive. Use `--wait` only for async/background tasks where blocking 1-120 seconds is acceptable.
6. **Always use `-o <file>`** to save results to a file once the run completes.
7. **Start with conservative limits** — small `maxItems`/`maxResults` values (5-10) on first calls. The cost warning above explains why.
8. **Report costs when relevant** — after a run completes, the result includes `cost.value`. Consider telling the user how much the run cost. Use `monid balance` to check remaining balance if the user cares about budget. Use your judgment — don't report costs if the user hasn't indicated cost-awareness.
9. **Run `monid <command> --help`** to check the latest flags and usage — the CLI is the source of truth for command signatures.
10. **Check the Hints block** — when a command's output includes a `Hints` section, read it and act on it. It carries suggested next steps, endpoint relationships, and caveats from the server — prefer its suggestions over guessing your next command.
11. **Surface BLOCKED runs to the user** — a `BLOCKED` status means a workspace control (budget or run cap) stopped the run; it is terminal and will not self-resolve. Report which control blocked it (from the `controls` list) and tell the user they can pause or modify that control on the dashboard (https://app.monid.ai) before retrying.