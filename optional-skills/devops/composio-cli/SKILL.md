---
name: composio-cli
description: Find and run tools for 1000+ apps via the Composio CLI.
version: 1.0.0
author: Shawn Esquivel (shawnesquivel), Composio
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Developer, Automation, CLI, Integrations, Composio]
    category: devops
    requires_toolsets: [terminal]
---

# Composio CLI

Drive the local `composio` CLI through the `terminal` tool to act in third-party apps such as GitHub, Gmail, Slack, Notion, Linear, Jira, and Google Calendar. The CLI manages its own auth, so it is the integration surface for connected apps. Use native Hermes tools for local files, shell work, code edits, browser work, and ordinary web research.

## When to Use

- Find or run Composio tools for external apps.
- Connect or inspect app accounts.
- Inspect a tool or trigger schema before calling it.
- Chain multiple Composio tool calls.
- Call an authenticated provider API through Composio.
- Listen for Composio trigger events.
- Debug Composio developer-project auth configs, connected accounts, logs, toolkits, or triggers.

## Prerequisites

The `composio` binary must be installed and authenticated. Check first, and install only if missing:

```bash
composio --version
composio whoami
```

If `whoami` fails, run `composio login`. If the binary is missing, install it:

```bash
curl -fsSL https://composio.dev/install | bash
```

No Hermes-managed API key is needed — the CLI owns its own credentials.

**Headless setup (servers, gateway, cron):** `composio login` defaults to an interactive browser flow, which won't work where there's no GUI. Authenticate one of these ways instead, before Hermes runs there:

```bash
# Option A — paste an API key from https://dashboard.composio.dev (no browser)
composio login --user-api-key <KEY>

# Option B — open the browser on another machine
composio login --no-browser --no-wait   # prints a login URL + session key, then exits
# open the URL on your laptop, authenticate, then finish on the server:
composio login --key <session-key>
```

## How to Run

Invoke every `composio` command through the `terminal` tool. Start each session by checking auth and what is already connected:

```bash
composio whoami
composio connections list
```

If `connections list` returns no active accounts, the user has nothing linked yet — follow **First Run: Suggesting Apps to Connect** before anything else.

## Quick Reference

| Task | Command |
|---|---|
| Check auth | `composio whoami` |
| List connected accounts | `composio connections list` |
| Discover tools | `composio search "create a github issue" --toolkits github --limit 3` |
| List toolkit tools | `composio tools list github --limit 5` |
| Inspect tool | `composio tools info GITHUB_CREATE_AN_ISSUE` |
| Get input schema | `composio execute GITHUB_CREATE_AN_ISSUE --get-schema` |
| Validate safely | `composio execute GITHUB_CREATE_AN_ISSUE --dry-run --skip-connection-check -d '{ owner: "acme", repo: "app", title: "Bug" }'` |
| Execute a known tool | `composio execute GITHUB_GET_THE_AUTHENTICATED_USER -d '{}'` |
| Parallel independent calls | `composio execute -p TOOL_A -d '{}' TOOL_B -d '{}'` |
| Connect an app | `composio link github` |
| Raw authenticated API call | `composio proxy https://api.github.com/user --toolkit github --method GET` |
| Script workflow | `composio run 'const me = await execute("GITHUB_GET_THE_AUTHENTICATED_USER"); console.log(me);'` |
| List trigger types | `composio triggers list github --limit 5` |
| Listen for events | `composio listen GMAIL_NEW_GMAIL_MESSAGE --timeout 5m --stream` |
| Find artifact directory | `composio artifacts cwd` |

## Procedure

1. Check auth and connections (see **How to Run**). Resolve missing auth or an empty connection list before continuing.

2. If the tool slug is already known, start with `execute`, `tools info`, or `--get-schema`. Do not search unnecessarily.

   ```bash
   composio execute GITHUB_CREATE_AN_ISSUE --get-schema
   composio execute GITHUB_CREATE_AN_ISSUE --dry-run --skip-connection-check -d '{ owner: "acme", repo: "app", title: "Bug" }'
   ```

3. If the slug is unknown, search by the user goal and read the returned plan, pitfalls, primary tool slugs, related tool slugs, connected toolkits, and schema paths.

   ```bash
   composio search "create a github issue" --toolkits github --limit 3
   ```

4. Inspect schemas before guessing arguments. Prefer `composio execute <SLUG> --get-schema` for exact required fields. `composio tools info <SLUG>` gives a compact summary and the cached schema path.

5. For a real action, execute only after the target app, object, and write intent are clear. If the CLI reports no active connection, run the exact `composio link <toolkit>` command it suggests, then retry.

6. For independent calls, use `execute -p/--parallel`. For conditionals, loops, fan-out over arrays, output transformation, or sub-agent summarization, use `composio run`.

   ```bash
   composio run --dry-run --skip-connection-check '
     const [issue, me] = await Promise.all([
       execute("GITHUB_CREATE_AN_ISSUE", { owner: "acme", repo: "app", title: "Bug" }),
       execute("GITHUB_GET_THE_AUTHENTICATED_USER"),
     ]);
     console.log({ issue, me });
   '
   ```

7. Use `proxy` when the provider API endpoint is known and a dedicated tool is harder to find than the raw call.

8. Use top-level `composio listen` for temporary consumer trigger subscriptions, always bounded with `--timeout` or `--max-events`. Use `composio triggers info <SLUG>` first when trigger params are unclear. Use `composio artifacts cwd` to find persisted payloads.

9. Use `composio dev ...` only for developer-project tasks: project setup, auth configs, developer connected accounts, playground execution, logs, toolkit versions, trigger configuration, or generated type stubs. Do not make users create auth configs for ordinary personal app actions.

## First Run: Suggesting Apps to Connect

When `composio connections list` returns no active accounts, the user hasn't connected anything yet. Don't dead-end — pitch a few concrete, outcome-focused things they could do once apps are linked to their Hermes agent, then offer to connect one:

- **Social media** — draft and publish posts, or read engagement across X, LinkedIn, Instagram, TikTok, Reddit.
- **Fitness** — read your Strava activity and build a training plan for an upcoming race.
- **DevOps** — read Linear tickets and open the matching GitHub PRs.
- **AI generation** — connect an image/audio provider and generate media on demand.
- **Google Workspace** — triage Gmail, manage Calendar, and edit Docs/Sheets/Drive.
- **Cross-app workflows** — chain the above, e.g. summarize unread Gmail, file action items as Linear tickets, post a digest to Slack.

These apps are all supported by Composio — suggest them directly. Do not run `composio search` here; first run is a pitch, not a lookup. Save `search` for later, once the user picks a concrete task and you need the exact tool slug. When the user picks an app, connect it and continue:

```bash
composio link strava
```

## Pitfalls

- Do not fabricate tool slugs. Search or inspect first.
- Do not ask users to manually create auth configs for normal user-level actions; use `composio link <toolkit>`.
- `composio login` and `composio link` open an interactive browser/OAuth flow that needs a live user. In headless contexts (gateway, cron), don't block on them or wait on the user's app choice — report the setup step and stop.
- `composio run` throws when an inner `execute()` returns `successful: false`; read the CLI error and fix the connection or arguments.
- `--skip-connection-check` is useful with `--dry-run` during outages or when validating schemas, but do not use it to pretend a real action succeeded.
- Missing required fields fail locally with `ToolInputValidationError` and point at the schema path.
- Large results may be written to an artifact file instead of printed inline; if the output says `storedInFile` or includes an `outputFilePath`, use `read_file` to read that JSON before summarizing.
- Write actions are often not idempotent. Avoid blind retries after ambiguous failures.
- `search` may return recommended prerequisite checks and known pitfalls; use them for risky writes.
- The CLI evolves quickly. If behavior looks different, run `composio <command> --help full`.

## Verification

For a safe smoke test, use read-only or dry-run commands:

```bash
composio whoami
composio connections list
composio search "create a github issue" --toolkits github --limit 1
composio execute GITHUB_CREATE_AN_ISSUE --get-schema
composio execute GITHUB_CREATE_AN_ISSUE --dry-run --skip-connection-check -d '{ owner: "acme", repo: "app", title: "Bug" }'
```

When testing through Hermes, ask it to use this skill and verify that it checks auth, lists connections, inspects schemas, dry-runs uncertain writes, links accounts only when the CLI reports no active connection, and uses `run` only for real scripting needs.
