# ShinkaEvolve-OSINT Hermes Plugin

Bridge [ShinkaEvolve-OSINT](https://github.com/zapabob/ShinkaEvolve-OSINT) into Hermes as a MILSPEC-grade OSINT agent for daily world-affairs and security briefings.

## What it does

- Runs the evolved **milspec_security_jp** workflow (11 security domains, 42 scenarios)
- Exposes six plugin tools on the `shinka_osint` toolset
- Provides `/shinka-osint` slash commands and `hermes shinka-osint` CLI
- Can schedule daily briefings via `hermes shinka-osint cron install`

ShinkaEvolve-OSINT stays **outside** the Hermes tree. This plugin imports its MCP tool handlers in-process from your checkout.

## Prerequisites

1. ShinkaEvolve-OSINT checkout (default: `C:\Users\downl\Desktop\ShinkaEvolve-OSINT-main\ShinkaEvolve-OSINT-main`)
2. Installed package in that checkout:

```powershell
cd "C:\Users\downl\Desktop\ShinkaEvolve-OSINT-main\ShinkaEvolve-OSINT-main"
py -3 -m pip install -e .
```

3. Enable the plugin in Hermes:

```powershell
hermes plugins enable shinka-osint
hermes tools   # enable toolset: shinka_osint
```

## Setup

```powershell
hermes shinka-osint setup --root "C:\Users\downl\Desktop\ShinkaEvolve-OSINT-main\ShinkaEvolve-OSINT-main"
hermes shinka-osint status
```

## Daily briefing (CLI)

```powershell
# Topic-matched briefing (mock corpus — fast, offline)
hermes shinka-osint briefing 中東情勢 --max-scenarios 3 --save

# Domain shortcut
hermes shinka-osint briefing --domain middle_east --save

# Single scenario deep-dive
hermes shinka-osint scenarios --domain taiwan
hermes shinka-osint analyze taiwan_crisis_overview
```

## Agent tools

| Tool | Purpose |
|------|---------|
| `shinka_osint_status` | Checkout path, import health, examples |
| `shinka_osint_list_scenarios` | List MILSPEC scenarios (optional domain filter) |
| `shinka_osint_analyze` | Run one scenario → score + evidence summary |
| `shinka_osint_briefing` | Multi-scenario daily briefing |
| `shinka_osint_verify` | Corpus allowlist + audit-chain integrity |
| `shinka_osint_audit` | Recent MILSPEC audit log entries |

### Example agent prompt

> 今日の世界情勢・安全保障について `shinka_osint_briefing` を topic=中東情勢 で実行し、各シナリオのスコアと根拠ブロック数を日本語で要約して。

## Scheduled briefing

```powershell
hermes shinka-osint cron install --schedule "every 9am" --topic "世界情勢 安全保障" --save
hermes shinka-osint cron install --dry-run   # preview prompt only
hermes cron list
```

Ensure `cron.script_timeout_seconds` is at least **900** in `config.yaml` for multi-scenario runs.

## Security domains

`middle_east`, `taiwan`, `cyber_defense`, `national_security`, `north_korea`, `us_japan_alliance`, `space_security`, `cognitive_warfare`, `ai_defense`, `japan_russia`, `constitution_defense`

## Source modes

- **mock** (default): bundled government corpus — reproducible, no live network
- **real**: live retrieval when ShinkaEvolve-OSINT real-source adapters are configured

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Tools not visible | `hermes plugins enable shinka-osint` + enable `shinka_osint` toolset |
| `root_exists: false` | `hermes shinka-osint setup --root <path>` |
| Import errors | `pip install -e .` inside the Shinka checkout |
| Low scores / empty evidence | Use `mock` first; verify with `hermes shinka-osint verify` |
