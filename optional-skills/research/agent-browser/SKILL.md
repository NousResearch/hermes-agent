---
name: agent-browser
description: >
  CLI browser automation for AI agents via terminal. Navigate, snapshot
  accessible elements, click, fill forms, screenshot, scrape — all from shell
  commands. Supports Chrome and Lightpanda engines. Use when Browserbase or
  browser toolset is unavailable.
version: 1.0.0
author: Hermes Agent
license: MIT
prerequisites:
  commands: [agent-browser]
metadata:
  hermes:
    tags: [Browser, Automation, Headless, Lightpanda, Chrome, Web, Scraping, CDP]
    related_skills: [duckduckgo-search, dogfood]
    fallback_for_toolsets: [browser]
    requires_toolsets: [terminal]
---

# agent-browser

CLI browser automation designed for AI agents. Compact accessibility-tree
output minimizes token usage. 100% native Rust binary. Works with Chrome
(default) or Lightpanda (lightweight Zig-based engine, ~2x faster).

- Docs: https://agent-browser.dev
- GitHub: https://github.com/vercel-labs/agent-browser
- npm: https://www.npmjs.com/package/agent-browser

## When to Use

- Need to interact with web pages (click, fill, navigate) and the `browser`
  toolset (Browserbase) is unavailable
- Scraping JS-rendered content that `web_extract` can't handle
- Testing web apps from the terminal
- Lightweight alternative to Playwright/Puppeteer for agent workflows
- Need parallel browser sessions with isolated auth

## Setup

```bash
# Install agent-browser
npm install -g agent-browser
agent-browser install              # downloads Chrome on first run

# Optional: install Lightpanda for faster, lighter browsing
# Linux x86_64:
curl -L -o lightpanda https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-x86_64-linux
chmod +x lightpanda && mv lightpanda ~/.local/bin/

# macOS Apple Silicon:
curl -L -o lightpanda https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-aarch64-macos
chmod +x lightpanda && mv lightpanda ~/.local/bin/
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `agent-browser open <url>` | Navigate to URL |
| `agent-browser snapshot -i` | Interactive elements with refs (primary AI output) |
| `agent-browser click @e2` | Click element by ref |
| `agent-browser fill @e3 "text"` | Clear field and type |
| `agent-browser press Enter` | Press keyboard key |
| `agent-browser get text @e1` | Extract text content |
| `agent-browser get url` | Current page URL |
| `agent-browser screenshot page.png` | Take screenshot |
| `agent-browser eval "js"` | Run JavaScript |
| `agent-browser close` | Close browser |
| `agent-browser --engine lightpanda ...` | Use Lightpanda engine |

## Core Workflow

The fundamental pattern — always snapshot after navigation or DOM changes:

```bash
agent-browser open <url>            # 1. Navigate
agent-browser snapshot -i           # 2. Get interactive elements with refs
agent-browser click @e2             # 3. Interact using refs
agent-browser snapshot -i           # 4. Re-snapshot (refs invalidated by DOM changes)
agent-browser close                 # 5. Done
```

Chain commands for efficiency:

```bash
agent-browser open example.com && agent-browser snapshot -i
```

## Snapshot Flags

| Flag | Meaning |
|------|---------|
| `-i` | Interactive elements only (always use this) |
| `-c` | Compact output |
| `-C` | Cursor-interactive (detects cursor:pointer, onclick, tabindex) |
| `-d N` | Depth limit |
| `-s selector` | Scope to CSS selector |

Output: compact text with refs (`@e1`, `@e2`), role, name, attributes.
Refs are **invalidated** after any DOM change — always re-snapshot.

## Interaction Commands

```bash
# Forms
agent-browser fill @e3 "search text"     # clear + type
agent-browser type @e3 "append"          # type without clearing
agent-browser select @e6 "option-value"
agent-browser check @e7
agent-browser upload @e9 /path/to/file

# Navigation
agent-browser click @e2
agent-browser press Enter
agent-browser scroll down 500
agent-browser back
agent-browser reload

# Data extraction
agent-browser get text @e1
agent-browser get html @e1
agent-browser get attr @e1 href
agent-browser get url
agent-browser get title

# State checking
agent-browser is visible @e1
agent-browser is enabled @e2

# Waiting
agent-browser wait for "selector"
agent-browser wait for networkidle
agent-browser wait 2000                  # milliseconds
```

## Selectors

Priority order:
1. **Refs** (always preferred): `@e1`, `@e2` from snapshot output
2. **CSS**: `#id`, `.class`, `[data-testid='submit']`
3. **Text/XPath**: `text=Submit`, `xpath=//button[@type='submit']`
4. **Semantic**: `find role button click --name Submit`

## Lightpanda Engine

Lightweight headless browser built in Zig. ~2x faster than Chrome, 10x less
memory. Best for scraping, data extraction, and simple interactions.

```bash
# Per-command
agent-browser --engine lightpanda open example.com

# Set as default
export AGENT_BROWSER_ENGINE=lightpanda

# Or in agent-browser.json config:
# { "engine": "lightpanda" }
```

### Lightpanda standalone (fastest for simple scraping)

```bash
# Direct markdown extraction — no agent-browser needed
lightpanda fetch --dump markdown https://example.com

# Direct semantic tree
lightpanda fetch --dump semantic_tree https://example.com

# Lightpanda also has an MCP server mode
lightpanda mcp
```

### Lightpanda Limitations

- No JS-driven form submissions (SPA search forms, React routers)
- No extensions, persistent profiles, or storage state
- No headed mode (headless only)
- Screenshot support depends on CDP implementation maturity
- Click-to-navigate works; JS event handler navigation may not

Use Chrome engine when you need full JS fidelity or SPA interaction.

## Sessions

Multiple isolated browser instances with separate auth:

```bash
agent-browser --session s1 open site-a.com
agent-browser --session s2 open site-b.com
agent-browser --session s1 snapshot -i
agent-browser session list
```

## CDP Mode

Connect to existing browsers via Chrome DevTools Protocol:

```bash
agent-browser --cdp 9222 snapshot -i         # local Chrome with --remote-debugging-port
agent-browser --cdp wss://host/devtools open  # remote WebSocket
```

## Configuration

Priority: CLI flags > env vars > `./agent-browser.json` > `~/.agent-browser/config.json`

Key environment variables:

```
AGENT_BROWSER_ENGINE             # chrome or lightpanda
AGENT_BROWSER_HEADED             # true for visible browser
AGENT_BROWSER_EXECUTABLE_PATH    # custom browser binary
AGENT_BROWSER_DEFAULT_TIMEOUT    # default 25000ms
AGENT_BROWSER_ALLOWED_DOMAINS    # restrict navigation
```

## Pitfalls

- **Refs invalidate on DOM changes.** Always re-snapshot after click, navigation,
  or any interaction that modifies the page.
- **Use `snapshot -i` not bare `snapshot`.** Interactive-only saves massive tokens.
  Full snapshots can be thousands of lines on complex pages.
- **Lightpanda can't submit JS-only forms.** DuckDuckGo search, React SPAs, and
  similar sites that rely on JavaScript event handlers for form submission won't
  navigate. Use Chrome engine for these.
- **Default timeout is 25 seconds.** Override with `AGENT_BROWSER_DEFAULT_TIMEOUT`
  for slow-loading pages.
- **On Linux, install system deps.** Run `agent-browser install --with-deps`
  for Chrome's system library requirements.
- **The daemon persists.** agent-browser runs a background daemon — commands are
  fast after first launch. Use `agent-browser close` to shut down.

## Verification

```bash
# Verify agent-browser works
agent-browser open example.com && agent-browser snapshot -i && agent-browser close

# Verify Lightpanda works
agent-browser --engine lightpanda open example.com && agent-browser --engine lightpanda snapshot -i && agent-browser --engine lightpanda close
```
