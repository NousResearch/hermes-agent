# Browser Agent Profile Distribution

A shareable Hermes profile that wires **Browserbase cloud browser** as the backend for all browser-based Kanban tasks.

## What this is

Hermes supports multiple profiles, each with its own config, skills, and SOUL. This distribution defines a `browser` profile that:

- Routes all `browser_navigate` / `browser_click` / etc. through **Browserbase** (no local Chromium)
- Is registered as a **Kanban assignee** — the dispatcher routes tasks tagged `browser` to it
- Has a minimal toolset (browser, web, vision, file, delegation) — no terminal
- Has its own SOUL that keeps it focused on browser automation

## Architecture

```
default profile (you, Kyle)          browser profile (this distribution)
┌─────────────────────────┐          ┌──────────────────────────────────┐
│ Kanban dispatcher       │  assigns │ Kanban worker                    │
│ kanban.dispatch_in_     │ ───────► │ engine: browserbase              │
│ gateway: true           │  browser │ plugin: browser-browserbase      │
│                         │  tasks   │ SOUL: browser-focused agent      │
│ default_assignee: ""    │          │ skills: kanban-browserbase-      │
│ (orchestrator)          │          │         worker                   │
└─────────────────────────┘          └──────────────────────────────────┘
                                              │
                                              ▼
                                     Browserbase Cloud
                                     (stealth, proxies,
                                      keep-alive sessions)
```

## Install

```bash
# Install the browser profile from this distribution
hermes profile install github.com/NousResearch/hermes-agent#feat/browser-agent-profile-distribution \
  --name browser

# Or from a local checkout during development
hermes profile install ./distributions/browser-agent --name browser
```

## Configure

```bash
# Add your Browserbase credentials to the browser profile's .env
# (never committed — user-owned)
cp distributions/browser-agent/.env.EXAMPLE ~/.hermes/profiles/browser/.env
# then edit ~/.hermes/profiles/browser/.env and fill in the values

# Enable the browserbase plugin
hermes -p browser plugins enable browser-browserbase

# Verify setup
hermes -p browser doctor
```

## Wire into Kanban

In your **default** profile's `config.yaml`, add:

```yaml
kanban:
  dispatch_in_gateway: true
  default_assignee: ""        # keep blank — default profile is orchestrator
  # optional: tag-based routing (future feature)
```

Then create Kanban tasks assigned to the `browser` profile:

```bash
# Assign directly
hermes kanban create "Scrape Amazon return status" \
  --assignee browser \
  --body "URL: https://amazon.com/returns\nAction: extract all pending returns\nOutput: JSON list"

# Or let the orchestrator assign it (tag-based routing coming soon)
hermes kanban create "Check Airbnb listing prices" --label browser
```

## File layout

```
distributions/browser-agent/
├── distribution.yaml          # manifest — name, version, env_requires
├── config.yaml                # profile config — engine: browserbase, plugins
├── SOUL.md                    # personality — focused browser worker
├── .env.EXAMPLE               # credential template (copy → .env, never commit)
└── skills/
    └── browser/
        └── kanban-browserbase-worker/
            └── SKILL.md       # execution loop for Kanban browser tasks
```

## Update

```bash
hermes profile update browser
```

Distribution-owned files (config.yaml, SOUL.md, skills/) are refreshed.
Your `.env`, `memories/`, and `sessions/` are never touched.

## Credentials

Never commit real keys. The `.env.EXAMPLE` file shows what's needed:

- `BROWSERBASE_API_KEY` — from https://browserbase.com
- `BROWSERBASE_PROJECT_ID` — your project ID

These live in `~/.hermes/profiles/browser/.env` (user-owned, excluded from distribution updates).
