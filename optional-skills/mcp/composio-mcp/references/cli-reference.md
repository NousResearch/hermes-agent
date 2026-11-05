# Composio CLI Reference

Complete command reference for the `composio` binary. Covers the core execution surface and the `dev` namespace for advanced workflows.

## Core commands

| Command | Purpose | Key flags |
|---------|---------|-----------|
| `composio search <query...>` | Find tools by semantic use case | `--toolkits`, `--limit`, `--human` |
| `composio execute <slug>` | Run a tool by slug | `-d/--data`, `--account`, `--file`, `--dry-run`, `--get-schema`, `-p/--parallel`, `--skip-*` |
| `composio link <toolkit>` | Connect an external account | `--alias`, `--no-browser`, `--no-wait`, `--list` |
| `composio listen` | Subscribe to trigger events | `--toolkit`, `--trigger` |
| `composio run <code>` | Run inline TS/JS with injected helpers | `-f/--file`, `--dry-run`, `--debug`, `--logs-off` |
| `composio proxy <url>` | Raw API access through a toolkit | `--toolkit`, `--account`, `-X`, `-H`, `-d` |
| `composio login` | Authenticate the CLI | `--user-api-key ak_*`, `--no-browser`, `--no-wait`, `--key <session>`, `--yes`, `--org` |
| `composio whoami` | Show current account | — |
| `composio setup` | Install plugins for agent hosts | `--target`, `--yes`, `--uninstall` |
| `composio orgs` | Manage org context | `list`, `switch` |
| `composio config` | View/manage CLI config | — |
| `composio version` | Show version | — |
| `composio upgrade` | Upgrade CLI | — |

## `composio search`

Semantic search across all toolkits. Returns tool slugs, descriptions, and metadata.

```bash
composio search "send an email"
composio search "create issue" --toolkits github
composio search "send email" "create issue" --toolkits gmail,github --limit 5
```

**Output:** JSON with `slug`, `name`, `description`, `toolkit`, `version`, `schemaPath`, `inputSchema`, `human_parameter_name`, `human_parameter_description`, `tags`.

**When to use:** Before every `execute` — `search` returns the exact slug and schema you need.

## `composio execute`

Runs a tool. Validates inputs against cached schemas and checks connections automatically.

```bash
composio execute GITHUB_CREATE_ISSUE -d '{ owner: "acme", repo: "app", title: "Bug" }'
composio execute GMAIL_SEND_EMAIL -d '{ recipient_email: "a@b.com", subject: "Hi" }'
composio execute --parallel \
  GMAIL_SEND_EMAIL -d '{ recipient_email: "a@b.com", subject: "Hi" }' \
  SLACK_SEND_A_MESSAGE_TO_A_SLACK_CHANNEL -d '{ channel: "general", text: "Hello" }'
```

**Key flags:**

- `-d, --data`: JSON/JS-style object, `@file`, or `-` (stdin)
- `--account <alias|word_id|id>`: pick a connected account
- `--file <path>`: inject a local file into a file_uploadable input (only works when the tool exposes exactly one uploadable file argument)
- `--dry-run`: validate and preview without executing
- `--get-schema`: print the input schema only
- `-p, --parallel`: run independent calls concurrently
- `--skip-connection-check`: skip the connected-account check
- `--skip-tool-params-check`: skip input validation against cached schema
- `--skip-checks`: bypass both checks above

**Flow:** `search` → `execute` (with `link` when the toolkit is not connected).

## `composio link`

Connects an external account for a toolkit. Browser OAuth by default; `--no-browser` prints a URL for headless boxes.

```bash
composio link github
composio link gmail --alias work
composio link gmail --list            # list existing connections
composio link gmail --no-browser      # headless
```

## `composio listen`

Subscribe to toolkit trigger events and stream them to stdout.

```bash
composio listen --toolkit github --trigger GITHUB_COMMIT_EVENT
```

## `composio run`

Executes inline Bun ESNext code with injected helpers. No SDK setup needed.

```bash
composio run 'const me = await execute("GITHUB_GET_THE_AUTHENTICATED_USER"); console.log(me)'
composio run '
  const [emails, issues] = await Promise.all([
    execute("GMAIL_FETCH_EMAILS", { max_results: 5 }),
    execute("GITHUB_LIST_REPOSITORY_ISSUES", { owner: "acme", repo: "app", state: "open" }),
  ]);
  const brief = await experimental_subAgent(`Summarize:\n${emails.prompt()}\n${issues.prompt()}`);
  console.log(brief);
'
```

**Injected helpers:**

| Helper | Behavior |
|--------|----------|
| `execute(slug, data?)` | Run a tool — same as `composio execute` |
| `search(query, opts?)` | Find tools — same as `composio search` |
| `proxy(toolkit)` | Returns a `fetch()` bound to the connected account |
| `experimental_subAgent(prompt, opts?)` | Structured LLM sub-agent (optional zod schema) |
| `result.prompt()` | Serialize helper output into LLM-friendly text |
| `z` | Global zod for structured output schemas |

## `composio proxy`

Direct HTTP access to any toolkit API through the connected account.

```bash
composio proxy https://api.github.com/user --toolkit github --method GET
composio proxy https://gmail.googleapis.com/gmail/v1/users/me/profile --toolkit gmail
```

## `composio dev` — developer workflows

Developer mode for building agent projects with the Composio SDK. Scaffolding, playground execution, logs, and developer-scoped management.

| Command | Purpose |
|---------|---------|
| `dev init` | Initialize a directory with a developer project |
| `dev playground-execute` | Execute a tool against a playground user |
| `dev listen` | Listen to developer-project trigger events |
| `dev logs tools` | Browse tool logs |
| `dev logs triggers` | Browse trigger logs |
| `dev toolkits list` | Browse available toolkits |
| `dev toolkits info` | Inspect a toolkit |
| `dev toolkits search` | Search toolkits |
| `dev toolkits version` | Inspect toolkit versions |
| `dev auth-configs list` | List auth configs |
| `dev auth-configs info` | View one auth config |
| `dev auth-configs create` | Create an auth config |
| `dev connected-accounts link` | Link a connected account |
| `dev connected-accounts list` | List connected accounts |
| `dev connected-accounts info` | View one connected account |
| `dev connected-accounts whoami` | Resolve an account identity |
| `dev triggers list` | List trigger types |
| `dev triggers info` | View one trigger type |
| `dev triggers status` | Inspect trigger instances |
| `dev triggers create` | Create a trigger instance |
| `dev triggers enable` | Enable a trigger instance |
| `dev triggers disable` | Disable a trigger instance (guarded) |
| `dev projects list` | List projects |
| `dev projects switch` | Switch default project |

## `composio generate`

Generate type stubs for toolkits, tools, and triggers. Auto-detects TypeScript or Python.

```bash
composio generate --toolkits github gmail --output-dir ./src/types
```

## `composio setup`

Install or uninstall agent plugins for supported hosts.

```bash
composio setup --target auto --yes
composio setup --uninstall --target auto --yes
composio setup --target all
```

## Data files

| Path | Content |
|------|---------|
| `~/.composio/user_data.json` | `ak_*` API key, base URL, org id |
| `~/.composio/config.json` | Developer flags, security mode |
| `~/.composio/pending-login-session.json` | In-flight `--no-browser` session |
| `~/.composio/toolkits.json` | Known toolkit slugs cache |
| `~/.composio/tool_definitions/` | Cached tool schemas |
| `~/.composio/tool-permissions-cache.json` | Per-tool permission grants |
| `/tmp/composio/` | Session artifacts (downloads, generated outputs) |

## Files

`composio files --help` — manage local files for tools that accept uploads.
