---
name: keenable-cli
description: Search the web and fetch pages as markdown via Keenable.
version: 1.0.0
author: Ilya Gusev (Keenable)
license: MIT
platforms: [linux, macos, windows]
required_environment_variables:
  - name: KEENABLE_API_KEY
    prompt: Keenable API key (optional)
    help: Create one at https://keenable.ai/signup. Skippable — the free tier works without a key; a key raises rate limits.
    required_for: higher rate limits
metadata:
  hermes:
    tags: [Research, Web, Search, Fetch, Markdown, CLI]
    related_skills: [parallel-cli, searxng-search, duckduckgo-search]
---

# Keenable CLI

`keenable` is a single-binary CLI over Keenable's web search and content APIs. It returns search hits with snippets/dates and fetches any page as clean markdown — both work on a free tier without login, and an API key only raises rate limits.

This is an optional third-party vendor workflow, not a Hermes core capability. It overlaps native `web_search` / `web_extract`, so don't prefer it for ordinary lookups — reach for it when the user names Keenable, when a terminal-native search/fetch flow is cleaner, or as a fallback when the `web` toolset is unconfigured.

## When to Use

- The user explicitly mentions Keenable or `keenable`.
- You want index-date / publication-date filtering on search results.
- You need a page converted to clean markdown from the terminal.
- Native `web_search` / `web_extract` are unavailable and you need a search/fetch fallback.

Prefer native `web_search` / `web_extract` for one-off lookups when Keenable isn't specifically requested.

## Prerequisites

Install the binary, then verify with `keenable --version`. Run every command below through the `terminal` tool.

Homebrew (macOS / Linux):
```bash
brew install keenableai/tap/keenable-cli
```

Installer script (macOS / Linux — two steps, no pipe; updates PATH for future shells only):
```bash
curl --proto '=https' --tlsv1.2 -LsSf -o /tmp/keenable-install.sh \
  https://github.com/keenableai/keenable-cli/releases/latest/download/keenable-cli-installer.sh
sh /tmp/keenable-install.sh
# Source the env file so the current shell sees it (either dir may exist):
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
```

Windows (PowerShell):
```powershell
Invoke-WebRequest -OutFile $env:TEMP\keenable-install.ps1 `
  https://github.com/keenableai/keenable-cli/releases/latest/download/keenable-cli-installer.ps1
& $env:TEMP\keenable-install.ps1
```

From source: `cargo install --git https://github.com/keenableai/keenable-cli`.
Update with `brew upgrade keenable-cli` or by re-running the installer.

**Auth is optional.** The free tier needs nothing. To raise rate limits, either run `keenable login` (device-code flow — prints a link + code, works headless) or pass `--api-key keen_***` on any command. `KEENABLE_API_KEY` from `~/.hermes/.env` is picked up as the key for REST calls.

## How to Run

```bash
# Search (YAML output, designed for agents)
keenable search "rust async patterns"
# Fetch a page as markdown
keenable fetch https://example.com
```

## Quick Reference

| Command | Purpose |
|---|---|
| `keenable search "<query>"` | Web search, YAML results |
| `keenable search "<q>" -p` | Pretty output (for humans) |
| `keenable search "<q>" --site techcrunch.com` | Restrict to one site |
| `keenable search "<q>" --published-after 2026-05-01` | Filter by publication date |
| `keenable search "<q>" --acquired-after 7d` | Filter by index date |
| `keenable fetch <url>` | Page content as markdown (YAML: content, title, url) |
| `keenable login` / `keenable logout` | Manage credentials in `~/.keenable/` |
| `keenable configure-mcp --all` | Wire Keenable MCP into detected AI clients |
| `keenable config` | View CLI settings |

Date filters (`--published-after/-before`, `--acquired-after/-before`) accept `YYYY-MM-DD`, ISO 8601 datetimes, or relative values (`12h`, `7d`, `3mo`, `1y`).

Each search result is YAML with: `title`, `url`, `description`, `snippet`, `published_at`, `acquired_at`. `fetch` returns `content` (markdown), `title`, `url`.

## Procedure

1. Confirm the binary is present: `keenable --version`. If missing, install per **Prerequisites**.
2. Search: `keenable search "<query>"` — add `--site` / date filters to narrow. Parse the YAML; pick the relevant `url`s.
3. Read a result in full: `keenable fetch <url>` and work from the returned markdown `content`.
4. For tighter rate limits or many calls, authenticate once with `keenable login` (or set `KEENABLE_API_KEY`).

## Pitfalls

- **Free-tier rate limits.** Unauthenticated use is throttled; authenticate before batch/loop calls.
- **Default output is YAML, not JSON.** Don't add `-p` (pretty) when parsing programmatically — parse the default YAML.
- **PATH after the installer.** The installer only updates future shells; source the printed env file (shown above) or the `keenable` call will fail with "command not found" in the same session.
- **`configure-mcp` targets other clients.** It supports Claude Code, Claude Desktop, Cursor, Windsurf, Codex, and OpenCode — not Hermes. To use Keenable's MCP inside Hermes, install the catalog entry: `hermes mcp install official/keenable`.
- **Overlaps native tools.** Don't default to this over `web_search` / `web_extract` for simple lookups.

## Verification

```bash
keenable --version && keenable search "hello world" | head -20
```
Prints a version string followed by YAML search results.
