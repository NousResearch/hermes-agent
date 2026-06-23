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

### Local dev server (npm run dev)

Clone upstream, install deps, and run the Vite dev stack Hermes can call on **port 3000**:

```powershell
git clone https://github.com/koala73/worldmonitor.git
cd worldmonitor
npm install
hermes plugins enable worldmonitor-osint
hermes worldmonitor-osint dev setup --repo "C:\path\to\worldmonitor"
# or step-by-step:
hermes worldmonitor-osint dev install --repo .
hermes worldmonitor-osint dev start --repo .
```

Dashboard: `http://127.0.0.1:3000` — API base auto-saved to `WORLDMONITOR_API_BASE`.

#### Tailscale (phone / remote dev on tailnet)

Hermes can bind Vite to all interfaces and reach it over your tailnet:

```powershell
# Option A — direct Tailscale IP (recommended for dev)
hermes worldmonitor-osint dev start --repo "C:\path\to\worldmonitor" --tailscale
# → http://100.x.x.x:3000  (this machine's tailscale ip -4)

# Option B — tailscale serve HTTPS proxy (tailnet-only, no Windows firewall fuss)
hermes worldmonitor-osint dev start --repo . --tailscale --tailscale-serve
# → https://<machine>.<tailnet>.ts.net/worldmonitor

# Manual Vite bind only
hermes worldmonitor-osint dev start --bind 0.0.0.0 --host 100.91.183.75
```

Check status (includes `tailscale.ipv4`, `dev_server.tailscale_url`):

```powershell
tailscale status
tailscale ip -4
hermes worldmonitor-osint dev status
```

Env overrides: `WORLDMONITOR_DEV_BIND=0.0.0.0`, `WORLDMONITOR_DEV_HOST=<tailscale-ip>`.

```powershell
hermes worldmonitor-osint dev status
hermes worldmonitor-osint setup-auth --mode dev
hermes worldmonitor-osint dev stop
```

Agent tools: `worldmonitor_dev_status`, `worldmonitor_dev_start`, `worldmonitor_dev_stop`.

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

## PDB-style situation report (08:00 / 18:00 cron)

Twice-daily **President's Daily Brief**–style open-source national-security digest:

- World Monitor HIGH headlines + elevated CII (past 24h)
- Shinka MILSPEC scenario scores
- Japan implications + 24h watchlist

```powershell
# One-shot (mock WM for reliability; use --source-mode real when WM auth is ready)
hermes worldmonitor-osint situation-report --slot morning
hermes worldmonitor-osint situation-report --slot evening --cron-stdout

# Install cron (local wall time — JST if Windows is JST)
hermes worldmonitor-osint cron install
hermes worldmonitor-osint cron install --source-mode real --llm-summary
hermes worldmonitor-osint cron install --deliver telegram,discord --llm-summary --source-mode real
```

| Job | Schedule | Script |
|-----|----------|--------|
| `wm-osint-pdb-morning` | `0 8 * * *` | `~/.hermes/scripts/wm-osint-pdb-morning.py` |
| `wm-osint-pdb-evening` | `0 18 * * *` | `~/.hermes/scripts/wm-osint-pdb-evening.py` |

Saved reports: `~/.hermes/worldmonitor-osint/situation_reports/`. Cron uses `no_agent=True` (script-only); `cron.script_timeout_seconds` is bumped to ≥900 on install.

### MILSPEC / 一次資料規律

PDB レポートは **事実記述に信頼できる一次資料を優先** する:

- **e-Gov Law API v2** — 憲法9条・自衛隊法・サイバー基本法等を自動取得（`egov_primary.py`）
- **PRIMARY backfill** — WM 二次見出しを `site:go.jp OR site:gov …` で公式ドメインへ裏取り（`ddgs`）
- **GitHub provenance** — worldmonitor / egov-law-mcp / hermes-agent の REST メタデータ
- 見出し各行に `[PRIMARY|SECONDARY|UNVERIFIED]` と `[出典: URL]`
- `--no-primary-backfill` / `--skip-egov` / `--skip-github` で段階的に無効化可能

> 防御的 OSINT 方針: 公式 API・サイト制約検索のみ。**ボット回避・ステルスクロールは実装しない**（MILSPEC / Deep Research 防御基準）。

日本法の一次資料: `hermes mcp install egov-law` または `py -3 -m pip install "egov-law-mcp>=0.1.0,<1"`

