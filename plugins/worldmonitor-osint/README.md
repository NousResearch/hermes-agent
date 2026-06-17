# World Monitor OSINT (Hermes plugin)

Real-time OSINT for **Japan security** and **world affairs**, combining:

- [koala73/worldmonitor](https://github.com/koala73/worldmonitor) REST API (risk scores, regional briefs, news digest)
- [ShinkaEvolve-OSINT](../shinka-osint/) MILSPEC scenario scoring and evolution-style evaluation
- [e-Gov Law MCP](../../optional-mcps/egov-law/manifest.yaml) for Japanese **primary legal sources**

## Quick start

```powershell
hermes worldmonitor-osint setup-stack
hermes plugins enable shinka-osint worldmonitor-osint
hermes mcp install egov-law
```

## Authentication (3 paths — pick one)

World Monitor auth is **separate** from Hermes LLM keys (OpenRouter, xAI, Codex, etc.).

| Mode | When to use | Setup |
|------|-------------|-------|
| **OAuth MCP** (recommended) | World Monitor Pro, interactive Hermes | `hermes worldmonitor-osint setup-auth --mode oauth` then `hermes mcp test worldmonitor` |
| **wm_ API key** | PRO/API tier, REST plugin + scripts | `hermes worldmonitor-osint setup-auth --mode key --api-key wm_...` |
| **Local sidecar** | Desktop app installed, local-first | Install WM desktop → `hermes worldmonitor-osint setup-auth --mode sidecar` |
| **Free web crawl** | No Pro, no desktop — public web JSON only | `hermes worldmonitor-osint free-crawl` or `snapshot --tier free` |

### API key (PRO / API tier)

1. Subscribe at [worldmonitor.app](https://www.worldmonitor.app/) (PRO gets `wm_` key automatically).
2. World Monitor desktop → Settings → **World Monitor** tab → copy API key.
3. Save to Hermes:
   ```powershell
   hermes worldmonitor-osint setup-auth --mode key --api-key wm_YOUR_40_HEX_CHARS
   ```

Key format: `wm_` + 40 lowercase hex characters. **Do not** send it as `Authorization: Bearer` — use `X-WorldMonitor-Key` only (the plugin handles this).

### OAuth MCP (no key paste)

```powershell
hermes mcp install worldmonitor
hermes mcp test worldmonitor
```

Browser opens **Sign in with World Monitor Pro**. Tokens are stored by Hermes MCP OAuth — not your LLM provider.

### Local sidecar (desktop app)

1. Install [World Monitor desktop](https://github.com/koala73/worldmonitor/releases) (Windows/macOS/Linux).
2. Launch the app — sidecar starts on **port 46123** automatically.
3. Configure Hermes:
   ```powershell
   hermes worldmonitor-osint setup-auth --mode sidecar
   ```

Verify: `hermes worldmonitor-osint status` → `sidecar.running: true`.

### Free web crawl (no Pro / no key)

Collects public JSON from `https://worldmonitor.app` (news digest, GPS jamming, alerts) with browser-like HTTP — no OAuth or `wm_` key.

```powershell
hermes worldmonitor-osint free-crawl --news-limit 20
hermes worldmonitor-osint snapshot --tier free
```

Country intel briefs and risk scores remain **Pro-only**. `snapshot --tier auto` uses Free crawl when no sidecar/key is configured.

### Auto-detect

```powershell
hermes worldmonitor-osint setup-auth
```

Tries sidecar → saved wm_ key → registers OAuth MCP.

Enable toolsets in `hermes tools`: `worldmonitor_osint`, `shinka_osint`, `web`, `search`.

## Tools

| Tool | Purpose |
|------|---------|
| `worldmonitor_status` | API connectivity, Shinka readiness, egov-law MCP hint |
| `worldmonitor_snapshot` | JP-focused real-time snapshot (risk, brief, news) |
| `worldmonitor_country_brief` | Single-country strategic brief |
| `worldmonitor_fusion_report` | WM snapshot + Shinka briefing + egov citation guidance |

## CLI

```powershell
hermes worldmonitor-osint status
hermes worldmonitor-osint snapshot
hermes worldmonitor-osint fusion 台湾有事 --domain taiwan --source-mode real --save
hermes worldmonitor-osint setup-stack
```

## Fusion workflow

1. `worldmonitor_snapshot` — live risk/news from World Monitor
2. `shinka_osint_briefing` with `source_mode=real` — MILSPEC scoring
3. egov-law MCP — `search_laws` / `get_law_article` for 憲法・安保関連法制

Reports saved under `~/.hermes/worldmonitor-osint/reports/` when `save_report=true`.
