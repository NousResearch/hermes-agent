---
name: agent-browser
description: Browser automation via Vercel's agent-browser CLI — native Rust, 50+ commands, ref-based snapshots, diffing, multi-engine support (Chrome/Lightpanda), multi-session isolation, dogfood QA mode, Electron app automation, Slack automation without API tokens, and auth import from any Chromium browser.
version: 1.1.0
metadata:
  hermes:
    tags: [browser, automation, cdp, vercel, local, scraping, agent-browser, dogfood, electron, slack, qa]
    related_skills: [chrome-cdp, dogfood]
---

# agent-browser — Vercel's Browser Automation CLI

Native Rust CLI for AI-agent browser automation. 22k+ GitHub stars. Compact text output designed for minimal token usage. Works locally without cloud services.

**Agent-first design:** ref-based snapshots, semantic finders, diffing, profiling, multi-engine (Chrome + Lightpanda), session isolation, auth state import from any Chromium browser, dogfood QA testing, Electron app automation, and Slack automation — all from one CLI.

Docs: https://agent-browser.dev/
GitHub: https://github.com/vercel-labs/agent-browser
npm: https://www.npmjs.com/package/agent-browser

---

## Installation

```bash
npm install -g agent-browser          # All platforms (recommended)
brew install agent-browser            # macOS
cargo install agent-browser           # Rust users
npx agent-browser open example.com    # Try without installing
```

First-time Chrome download:
```bash
agent-browser install
```

Linux system deps:
```bash
agent-browser install --with-deps
```

---

## Quick Start

```bash
agent-browser open example.com
agent-browser snapshot                    # Accessibility tree with refs (@e1, @e2)
agent-browser click @e2                   # Click by ref
agent-browser fill @e3 "user@example.com" # Fill by ref
agent-browser get text @e1                # Get text by ref
agent-browser screenshot page.png
agent-browser close
```

---

## Core Commands

### Navigation & Interaction

```bash
agent-browser open <url>              # Navigate (aliases: goto, navigate)
agent-browser click <sel>             # Click (--new-tab for new tab)
agent-browser dblclick <sel>          # Double-click
agent-browser fill <sel> <text>       # Clear and fill
agent-browser type <sel> <text>       # Type into element
agent-browser press <key>             # Press key (Enter, Tab, Control+a)
agent-browser keyboard type <text>    # Type at current focus (no selector)
agent-browser keyboard inserttext <text>  # Insert text without key events
agent-browser keydown <key>           # Hold key down
agent-browser keyup <key>             # Release key
agent-browser hover <sel>             # Hover element
agent-browser focus <sel>             # Focus element
agent-browser select <sel> <val>      # Select dropdown option
agent-browser check <sel>             # Check checkbox
agent-browser uncheck <sel>           # Uncheck checkbox
agent-browser scroll <dir> [px]       # Scroll (up/down/left/right, --selector)
agent-browser scrollintoview <sel>    # Scroll element into view
agent-browser drag <src> <dst>        # Drag and drop
agent-browser upload <sel> <files>    # Upload files
agent-browser snapshot                # Accessibility tree with refs
agent-browser eval <js>               # Run JavaScript (-b for base64, --stdin)
agent-browser close                   # Close browser (aliases: quit, exit)
```

### Get Info

```bash
agent-browser get text <sel>          # Text content
agent-browser get html <sel>          # InnerHTML
agent-browser get value <sel>         # Input value
agent-browser get attr <sel> <attr>   # Attribute
agent-browser get title               # Page title
agent-browser get url                 # Current URL
agent-browser get cdp-url             # CDP WebSocket URL
agent-browser get count <sel>         # Count matching elements
agent-browser get box <sel>           # Bounding box
agent-browser get styles <sel>        # Computed styles
```

### State Checks

```bash
agent-browser is visible <sel>
agent-browser is enabled <sel>
agent-browser is checked <sel>
```

### Semantic Finders

Locate elements by accessibility semantics (not CSS):

```bash
agent-browser find role button click --name "Submit"
agent-browser find label "Email" fill "test@test.com"
agent-browser find text "Sign In" click
agent-browser find placeholder "Search..." type "query"
agent-browser find alt "Logo" click
agent-browser find testid "submit-btn" click
agent-browser find first ".item" click
agent-browser find last ".item" text
agent-browser find nth 2 ".card" hover
```

Options: `--name <name>` (filter by accessible name), `--exact` (exact match)

### Wait

```bash
agent-browser wait "#element"                 # Wait for element
agent-browser wait 2000                       # Wait 2 seconds
agent-browser wait --text "Welcome"           # Wait for text
agent-browser wait --url "**/dashboard"       # Wait for URL
agent-browser wait --load networkidle         # Wait for load state
agent-browser wait --fn "condition"           # Wait for JS condition
agent-browser wait --download [path]          # Wait for download
agent-browser wait "#spinner" --state hidden  # Wait for disappearance
```

### Screenshots & PDF

```bash
agent-browser screenshot [path]                           # Viewport
agent-browser screenshot --full                           # Full page
agent-browser screenshot --annotate                       # Numbered labels with refs
agent-browser screenshot --screenshot-dir ./shots         # Custom dir
agent-browser screenshot --screenshot-format jpeg --screenshot-quality 80
agent-browser pdf output.pdf
```

**Annotated screenshots** overlay numbered `[N]` labels on interactive elements. Each `[N]` maps to ref `@eN` — same refs work for both visual and text-based workflows. Useful for multimodal models reasoning about visual layout.

---

## Import Auth from Any Chromium Browser

**Pick up your existing web sessions** from Chrome, Brave, Edge, Arc, or any Chromium-based browser — and continue where you left off:

```bash
# 1. Launch your browser with remote debugging
# macOS:
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222
# Brave:
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" --remote-debugging-port=9222
# Or use --auto-connect to discover a running browser automatically

# 2. Log into your sites in the browser (Gmail, GitHub, Slack, etc.)

# 3. Save the authenticated state
agent-browser --auto-connect state save ./my-auth.json

# 4. Reuse in agent-browser — no re-login needed
agent-browser --state ./my-auth.json open https://app.example.com/dashboard
```

### Named Sessions (Auto-Persist)

```bash
# Auto-save/load state across restarts
agent-browser --session-name twitter open twitter.com
# Login once, state persists automatically in ~/.agent-browser/sessions/
```

### Persistent Profiles

```bash
# Full browser profile (cookies, IndexedDB, service workers, cache)
agent-browser --profile ~/.myapp-profile open myapp.com
```

### State Encryption

```bash
export AGENT_BROWSER_ENCRYPTION_KEY=$(openssl rand -hex 32)
agent-browser --session-name secure open example.com
```

### Auth Vault (Encrypted Credential Storage)

Store credentials locally (always encrypted). The LLM never sees passwords:

```bash
echo "pass" | agent-browser auth save github \
  --url https://github.com/login \
  --username user \
  --password-stdin

agent-browser auth login github    # Auto-fills login form
```

---

## Sessions (Multi-Instance Isolation)

```bash
# Isolated sessions with separate auth/cookies
agent-browser --session agent1 open site-a.com
agent-browser --session agent2 open site-b.com
agent-browser session list
```

Each session has its own browser, cookies, storage, navigation history, and auth state.

---

## Diffing (DOM & Visual)

Compare page states before/after changes:

```bash
# Snapshot diff (text-based)
agent-browser diff snapshot                              # Current vs last snapshot
agent-browser diff snapshot --baseline before.txt        # Compare vs saved file
agent-browser diff snapshot --selector "#main" --compact # Scoped diff

# Visual pixel diff
agent-browser diff screenshot --baseline before.png      # Pixel-level comparison
agent-browser diff screenshot --baseline b.png -o d.png  # Save diff image

# URL comparison (A/B testing, staging vs prod)
agent-browser diff url https://v1.com https://v2.com
agent-browser diff url https://v1.com https://v2.com --screenshot
```

---

## Profiler & Debugging

```bash
# Chrome DevTools Profiler
agent-browser profiler start
# ... perform actions ...
agent-browser profiler stop profile.json

# Trace recording
agent-browser trace start [path]
agent-browser trace stop [path]

# Console & errors
agent-browser console                 # View console messages
agent-browser console --clear         # Clear console
agent-browser errors                  # View JS exceptions
agent-browser errors --clear          # Clear errors

# Highlight & inspect
agent-browser highlight <sel>         # Highlight element
agent-browser inspect                 # Open Chrome DevTools
```

---

## Dogfood Mode (Exploratory QA Testing)

agent-browser ships with a built-in **dogfood skill** for systematic exploratory testing. It navigates an app like a real user, finds bugs and UX issues, and produces a structured report with screenshots and repro videos.

### Quick Usage

```
"dogfood vercel.com"
"QA http://localhost:3000 — focus on the billing page"
```

### What It Does

1. **Navigate** the app systematically (every section, every page)
2. **Interact** like a user (click buttons, fill forms, test edge cases)
3. **Document** every issue with:
   - Numbered repro steps
   - Step-by-step screenshots
   - Repro videos for interactive bugs
   - Severity classification (Critical/High/Medium/Low)
   - Category (Functional/Visual/Accessibility/Console/UX/Content)
4. **Produce** a markdown report ready to hand to the responsible team

### Key Commands for QA

```bash
# Video recording for repro steps
agent-browser record start ./repro.webm
# ... reproduce bug ...
agent-browser record stop

# Console/error checking at every page
agent-browser errors
agent-browser console

# Annotated screenshots for evidence
agent-browser screenshot --annotate ./evidence.png
```

---

## Electron App Automation

Automate any Electron desktop app (VS Code, Slack, Discord, Figma, Notion, Spotify, etc.) by connecting to its built-in Chrome DevTools Protocol port:

```bash
# Launch any Electron app with remote debugging
# macOS:
open -a "Slack" --args --remote-debugging-port=9222
open -a "Visual Studio Code" --args --remote-debugging-port=9223
open -a "Discord" --args --remote-debugging-port=9224
open -a "Figma" --args --remote-debugging-port=9225
open -a "Notion" --args --remote-debugging-port=9226
open -a "Spotify" --args --remote-debugging-port=9227

# Connect and automate
agent-browser connect 9222
agent-browser snapshot -i
agent-browser click @e5
agent-browser screenshot slack-desktop.png
```

---

## Slack Automation (No API Tokens)

Browser-based Slack automation — no API tokens or bot setup needed:

```bash
# Connect to existing Slack session
agent-browser connect 9222
# Or open in browser
agent-browser open https://app.slack.com

# Check unread channels
agent-browser snapshot -i
agent-browser click @e21     # "More unreads" button

# Navigate channels, search, extract data, send messages
# All via snapshot-interact workflow
```

---

## Network Interception

```bash
agent-browser network route <url>               # Intercept requests
agent-browser network route <url> --abort       # Block requests
agent-browser network route <url> --body <json> # Mock response
agent-browser network unroute [url]             # Remove routes
agent-browser network requests                  # View tracked requests
agent-browser network requests --filter api     # Filter requests
```

---

## Clipboard & Mouse

```bash
# Clipboard
agent-browser clipboard read
agent-browser clipboard write "Hello"
agent-browser clipboard copy      # Ctrl+C
agent-browser clipboard paste     # Ctrl+V

# Mouse
agent-browser mouse move <x> <y>
agent-browser mouse down [button]  # left/right/middle
agent-browser mouse up [button]
agent-browser mouse wheel <dy> [dx]
```

---

## Browser Settings

```bash
agent-browser set viewport <w> <h> [scale]  # Viewport size
agent-browser set device "iPhone 14"        # Device emulation
agent-browser set geo <lat> <lng>           # Geolocation
agent-browser set offline [on|off]          # Offline mode
agent-browser set headers <json>            # HTTP headers
agent-browser set credentials <u> <p>       # Basic auth
agent-browser set media [dark|light]        # Color scheme
```

---

## Tabs, Frames & Dialogs

```bash
# Tabs
agent-browser tab                     # List tabs
agent-browser tab new [url]           # New tab
agent-browser tab <n>                 # Switch to tab n
agent-browser tab close [n]           # Close tab

# Frames
agent-browser frame <sel>             # Switch to iframe
agent-browser frame main              # Back to main frame

# Dialogs
agent-browser dialog accept [text]    # Accept (with optional prompt text)
agent-browser dialog dismiss          # Dismiss
```

---

## Cookies & Storage

```bash
agent-browser cookies                 # Get all cookies
agent-browser cookies set <name> <val> # Set cookie
agent-browser cookies clear           # Clear cookies

agent-browser storage local           # Get all localStorage
agent-browser storage local <key>     # Get specific key
agent-browser storage local set <k> <v>
agent-browser storage local clear

agent-browser storage session         # Same for sessionStorage
```

---

## Downloads

```bash
agent-browser download <sel> <path>       # Click to trigger download
agent-browser wait --download [path]      # Wait for completion
```

---

## Navigation History

```bash
agent-browser back                    # Go back
agent-browser forward                 # Go forward
agent-browser reload                  # Reload page
```

---

## Snapshot Filtering

Reduce token usage by filtering snapshots:

```bash
agent-browser snapshot                    # Full tree
agent-browser snapshot -i                 # Interactive elements only
agent-browser snapshot -C                 # Include cursor-interactive (divs with onclick)
agent-browser snapshot -c                 # Compact (remove empty elements)
agent-browser snapshot -d 3               # Limit depth to 3
agent-browser snapshot -s "#main"         # Scope to selector
agent-browser snapshot -i -c -d 5         # Combine options
```

---

## Security Features (Opt-In)

All security features are opt-in — existing workflows unaffected:

```bash
# Content boundary markers (wrap output for LLM safety)
agent-browser --content-boundaries open example.com

# Domain allowlist (restrict navigation)
agent-browser --allowed-domains "example.com,*.example.com" open example.com

# Action policy (gate destructive actions)
agent-browser --action-policy ./policy.json open example.com

# Action confirmation (require approval for sensitive actions)
agent-browser --confirm-actions eval,download open example.com

# Output length limits (prevent context flooding)
agent-browser --max-output 50000 open example.com
```

---

## Multi-Engine Support

```bash
agent-browser --engine chrome open example.com        # Chrome (default)
agent-browser --engine lightpanda open example.com    # Lightpanda (fast, JS-limited)
```

---

## CDP Mode (Connect to Existing Browser)

```bash
agent-browser connect 9222              # By port
agent-browser connect ws://host:port    # By URL
agent-browser --auto-connect state save # Auto-discover running Chrome
```

---

## JSON Output (For Agents)

```bash
agent-browser --json get title
agent-browser --json snapshot
```

---

## iOS Simulator

```bash
agent-browser --device "iPhone 15 Pro" open example.com
```

---

## Proxy Support

```bash
agent-browser --proxy http://proxy:8080 open example.com
agent-browser --proxy http://user:pass@proxy:8080 open example.com
```

---

## Vercel Sandbox Integration

Run agent-browser inside ephemeral Vercel Sandbox microVMs for CI/CD and serverless automation:

- Sandbox snapshots for sub-second startup
- Multi-step workflows with persistent state
- Works with Next.js, SvelteKit, Nuxt, Remix, Astro
- See: https://agent-browser.dev/next

---

## Tips for AI Agents

1. **Always use `snapshot` before `click`/`type`** — the refs change after navigation
2. **Use `find` for semantic locators** — more robust than CSS selectors
3. **Use `wait` after navigation** — pages need time to load
4. **Use `--session-name` for persistent state** — avoids re-authentication
5. **Use `--annotate` screenshots** — for visual debugging with numbered refs
6. **Use `--auto-connect` to grab existing browser sessions** — continue from where you left off
7. **Use `diff snapshot` to detect changes** — compare before/after states
8. **Use `errors` and `console`** — catch silent JS failures at every page
9. **Use `record start/stop` for repro videos** — document bugs with evidence
10. **Use `--engine lightpanda` for speed** — when full JS isn't needed
