---
name: composio
description: "Reach 1000+ SaaS apps (Gmail, Notion, Slack) via Composio."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [composio, connectors, saas, oauth, gmail, notion, slack, linear, github]
    category: productivity
---

# Composio Connectors Skill

Execute actions on 1000+ SaaS services — Gmail, Google Calendar/Drive, GitHub,
Slack, Notion, Linear, Airtable, HubSpot, Jira, and more — through Composio's
hosted connector platform. Composio manages the OAuth flows and stores the
connected accounts; Hermes only holds one `COMPOSIO_API_KEY`. This skill does
NOT archive or index data — it performs live reads and actions.

## When to Use

- The user asks to read/send email, manage calendar events, query Notion,
  file Linear/Jira issues, post to Slack, or touch any workplace SaaS app
  for which no dedicated Hermes skill exists.
- Prefer a dedicated skill when one exists (e.g. `himalaya` for raw IMAP,
  `xurl` for X) — Composio is the broad fallback, not the specialist.

## Prerequisites

- `COMPOSIO_API_KEY` in `~/.hermes/.env` — create at https://app.composio.dev
- `composio` Python package: `pip install composio` (auto-prompted by the
  helper script when missing)
- Per-service accounts are connected once via OAuth (see Connecting below)

## How to Run

All operations go through the helper script via `terminal`:

```bash
python3 scripts/composio_cli.py toolkits                    # what's connected
python3 scripts/composio_cli.py search "send an email"      # find a tool
python3 scripts/composio_cli.py schema GMAIL_SEND_EMAIL     # its arguments
python3 scripts/composio_cli.py execute GMAIL_SEND_EMAIL \
    --args '{"recipient_email": "a@b.com", "subject": "Hi", "body": "..."}'
```

Every command prints one JSON object. `"successful": false` + `"error"` on
failure (exit code 1).

## Quick Reference

| Command | Purpose |
|---------|---------|
| `toolkits` | List connected accounts (what you can use right now) |
| `tools <toolkit>` | List tools in one toolkit (`gmail`, `github`, `notion`, ...) |
| `search <query>` | Free-text search for tools across all toolkits |
| `schema <TOOL_SLUG>` | Input schema for one tool |
| `execute <TOOL_SLUG> --args '<json>'` | Run the tool |
| `connect <toolkit>` | Start OAuth for a new service (prints a URL) |
| `wait <request_id>` | Block until a pending connection completes |

Tool slugs look like `GMAIL_SEND_EMAIL`, `GITHUB_LIST_REPOS_FOR_USER`,
`NOTION_SEARCH_NOTION_PAGE`. The `--user` flag (default `hermes`, override
with `COMPOSIO_USER_ID`) scopes connected accounts; use distinct user ids if
multiple people's accounts must stay isolated.

## Procedure

1. **Check connections first**: run `toolkits`. If the needed service is
   missing, run `connect <toolkit>`, deliver the printed `redirect_url` to
   the user, and ask them to open it and grant access. Then `wait <id>` or
   simply retry.
2. **Discover the right tool**: `search` with a natural-language phrase, or
   `tools <toolkit>` to browse. Don't guess slugs.
3. **Get the schema** before executing anything with non-obvious arguments.
4. **Execute** with exact JSON arguments. Read `data` from the response.
5. For multi-step jobs (e.g. "summarize today's emails then post to Slack"),
   chain execute calls; keep intermediate JSON in files if large.

## Pitfalls

- `connect` requires the USER to open the OAuth URL in their browser — the
  agent cannot complete OAuth itself. Always hand the URL to the user and
  wait for confirmation.
- Search returns at most 20 results; refine the query rather than paging.
- Composio slugs are versioned server-side; the helper executes latest.
  If a tool's behavior seems off, re-fetch the `schema` — arguments drift.
- A 4xx from `execute` usually means a missing/expired connected account
  for that toolkit — re-run `toolkits` and reconnect, don't retry blindly.
- Do not fabricate slugs from memory of other deployments; always confirm
  with `search`/`tools` output from THIS account.

## Verification

- `toolkits` returns `"successful": true` with your API key → wiring works.
- After `connect` + OAuth grant, the toolkit appears in `toolkits` output.
- After `execute`, check `"successful": true` and inspect `data`; treat
  `error` text as the source of truth, not the exit code alone.
