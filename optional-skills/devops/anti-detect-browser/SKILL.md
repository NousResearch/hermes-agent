---
name: anti-detect-browser
description: Launch and manage anti-detect browsers with unique real-device fingerprints for multi-account operations, web scraping, ad verification, and AI agent automation. Use when the user needs to run multiple browser sessions with distinct identities, manage persistent browser profiles, automate tasks across accounts, or build agentic workflows that require browser fingerprint isolation. Also use when the user mentions antibrow, anti-detect browser, fingerprint browser, or multi-account browser.
---

# Anti-Detect Browser — The First Fingerprint Browser Built for AI Agents

Launch Chromium with real-device fingerprints via standard Playwright APIs. Each browser gets a unique, consistent digital identity from 500+ parameters across 30+ categories (Canvas, WebGL, Audio, Fonts, WebRTC, WebGPU…) — collected from actual devices, not synthetic.

**What makes it different:**
- 🤖 **Agent-first design** — MCP server mode lets AI agents launch, navigate, screenshot, and interact with fingerprint browsers through tool calls. The first anti-detect browser built for agentic workflows.
- 👁️ **Live View** — Watch headless browsers in real time from the dashboard. Debug AI agent actions, share URLs with teammates, observe automation as it happens.
- 🎭 **Real fingerprints, not random noise** — Every fingerprint comes from an actual device. Internally consistent across all browser APIs, passing even advanced anti-bot checks.
- 🔌 **Zero learning curve** — Standard Playwright API. Drop in `anti-detect-browser` and your existing scripts work with fingerprint protection.

**Links:** npm: `anti-detect-browser` · Dashboard: https://antibrow.com · API: `https://antibrow.com/api/v1/` · Docs: https://antibrow.com/docs

## Setup

```bash
npm install anti-detect-browser
npx playwright install chromium
npx playwright install-deps chromium  # Linux: system libs
```

## Quick Start

```typescript
import { AntiDetectBrowser } from 'anti-detect-browser'
const ab = new AntiDetectBrowser({ key: 'your-api-key' })

const { browser, page } = await ab.launch({
  fingerprint: { tags: ['Windows 10', 'Chrome'] },
  profile: 'my-account-01',
  proxy: 'http://user:pass@host:port',  // optional
})

await page.goto('https://example.com')  // standard Playwright from here
await browser.close()  // auto-syncs cookies & storage to cloud
```

## Profiles — Persistent Browser Identities

Profiles save cookies, localStorage, and session data to the cloud. Same profile name = same logged-in state next time, even across machines.

```typescript
// First launch — login
const { browser, page } = await ab.launch({ profile: 'shop-01' })
await page.goto('https://shop.example.com/login')
// ... login ...
await browser.close()

// Later — already logged in, session restored from cloud
const { page: p2 } = await ab.launch({ profile: 'shop-01' })
await p2.goto('https://shop.example.com/dashboard')
```

## Fingerprint Tags

Filter by: `Microsoft Windows`, `Apple Mac`, `Android`, `Linux`, `iPad`, `iPhone`, `Edge`, `Chrome`, `Safari`, `Firefox`, `Desktop`, `Mobile`, `Windows 7`, `Windows 8`, `Windows 10`

```typescript
await ab.launch({ fingerprint: { tags: ['Apple Mac', 'Safari'] } })
await ab.launch({ fingerprint: { tags: ['Android', 'Mobile', 'Chrome'] } })
await ab.launch({ fingerprint: { tags: ['Windows 10', 'Chrome'], minBrowserVersion: 130 } })
```

## Live View — Watch Headless Browsers in Real Time

Stream any headless session to the antibrow.com dashboard. Share the URL — anyone can watch the browser screen live. Perfect for debugging AI agent actions or team monitoring.

```typescript
const { page, liveView } = await ab.launch({
  headless: true,
  liveView: true,
  profile: 'price-monitor',
  fingerprint: { tags: ['Windows 10', 'Chrome'] },
})
console.log('Watch live:', liveView.viewUrl)
```

## Visual Identification

Run many browsers at once — each gets a floating label, title prefix, and colored border:

```typescript
await ab.launch({ profile: 'twitter-main', label: '@myhandle', color: '#e74c3c' })
await ab.launch({ profile: 'twitter-alt', label: '@support', color: '#3498db' })
```

## Inject into Existing Playwright

Already have Playwright scripts? Add fingerprints without rewriting:

```typescript
import { chromium } from 'playwright'
import { applyFingerprint } from 'anti-detect-browser'

const browser = await chromium.launch()
const context = await browser.newContext()
await applyFingerprint(context, {
  key: 'your-api-key',
  fingerprint: { tags: ['Windows 10', 'Chrome'] },
  profile: 'my-profile',
})
const page = await context.newPage()
```

## MCP Server — AI Agent Browser Control

The first anti-detect browser with native MCP support. AI agents (Claude, GPT, etc.) can launch and control fingerprint browsers through tool calls.

```json
{
  "mcpServers": {
    "anti-detect-browser": {
      "command": "npx",
      "args": ["anti-detect-browser", "--mcp"],
      "env": { "ANTI_DETECT_BROWSER_KEY": "your-api-key" }
    }
  }
}
```

mcporter setup:
```bash
mcporter config add anti-detect-browser \
  --command 'npx anti-detect-browser --mcp' \
  --env ANTI_DETECT_BROWSER_KEY=your-key
```

### MCP Tools

| Tool | Purpose |
|------|---------|
| `launch_browser` | Start fingerprint browser (profile, tags, proxy, headless) |
| `close_browser` | Close session by ID |
| `navigate` | Go to URL |
| `screenshot` | Capture page screenshot |
| `click` / `fill` | Interact with elements by CSS selector |
| `evaluate` | Run JavaScript on page |
| `get_content` | Extract text from page or element |
| `start_live_view` / `stop_live_view` | Stream browser screen to dashboard |
| `list_sessions` | List running browsers |
| `list_profiles` / `create_profile` / `delete_profile` | Manage profiles |

**⚠️ MCP session note:** stdio transport spawns a new process per call. For multi-step workflows (launch → navigate → screenshot), use SDK scripts or configure a persistent MCP transport.

## REST API

Base: `https://antibrow.com/api/v1/` — all endpoints require `Authorization: Bearer <api-key>`.

**Profiles:**
```bash
GET    /profiles              # List all
POST   /profiles              # Create (body: {"name":"x","tags":["Windows 10","Chrome"]})
GET    /profiles/:name        # Get details + dataUrl for fingerprint download
DELETE /profiles/:name        # Delete
```

POST response includes `dataUrl` — a presigned URL (10 min) to the full fingerprint JSON (~9MB).

**Fingerprints:**
```bash
GET /fingerprints/fetch       # Fetch matching fingerprint (query: tags, minBrowserVersion, etc.)
GET /fingerprints/versions    # Available browser versions
```

## Workflow Examples

### Multi-Account Management
```typescript
const accounts = [
  { profile: 'twitter-1', label: '@brand', color: '#1DA1F2' },
  { profile: 'twitter-2', label: '@support', color: '#FF6B35' },
  { profile: 'twitter-3', label: '@personal', color: '#6C5CE7' },
]
for (const acct of accounts) {
  const { page } = await ab.launch({
    fingerprint: { tags: ['Windows 10', 'Chrome'] },
    proxy: getNextProxy(),
    ...acct,
  })
  await page.goto('https://twitter.com')
}
```

### Scraping with Fingerprint Rotation
```typescript
for (const url of urlsToScrape) {
  const { browser, page } = await ab.launch({
    fingerprint: { tags: ['Desktop', 'Chrome'], minBrowserVersion: 125 },
    proxy: rotateProxy(),
  })
  await page.goto(url)
  const data = await page.evaluate(() => document.body.innerText)
  saveData(url, data)
  await browser.close()
}
```

### AI Agent + Live View Monitoring
```typescript
const { page, liveView } = await ab.launch({
  headless: true, liveView: true,
  profile: 'price-monitor',
  fingerprint: { tags: ['Windows 10', 'Chrome'] },
})
console.log('Dashboard:', liveView.viewUrl)  // share with team
// Agent monitors prices, team watches live
```

## Known Limitations

1. **WebGL on GPU-less servers** — Reports SwiftShader renderer. Not an SDK issue; hardware limitation of headless environments.
2. **Free tier** — 2 browser profiles. Upgrade at https://antibrow.com for more.
