# NOC Alert Context Pipeline

On-demand alert briefing and deep troubleshooting for lab router incidents.

## Architecture

```
[Alert received]
    → webhook_receiver.py
         → enrich_alert_from_dict(alert_dict, source=source, noc_notify=True)
              → NetBox lookup + MiniMax LLM briefing → Telegram (sync, immediate)
              → IF lab device (mgmt_ip in LAB_MGMT_IPS):
                   → run_router_diagnostics() — SSH → read-only IOS commands
                   → send_diagnostic_telegram() — Telegram (sync, immediate)
              → _trigger_noc_md_pipeline() — FIRE AND FORGET
                   → Write ~/noc/inbox/{alert_id}.json
                   → process_alert_from_inbox() — build .md context file

[Later: on-demand]
    /noc <alert_id>        → read ~/noc/context/{alert_id}.md → MiniMax → Telegram
    /noc-troubleshoot <alert_id> [question]
                           → read .md + raw SSH .json → MiniMax → Telegram
```

## Directory Structure

```
~/noc/
  inbox/                    Raw alert JSON (written by webhook_receiver.py)
  context/                 Enriched .md files (consumed by /noc, /noc-troubleshoot)
  diagnostics_raw/          Raw SSH outputs .json (consumed by /noc-troubleshoot)
  logs/                    Processing logs
```

## Key Files

| File | Purpose |
|------|---------|
| `alert_processor.py` | Main enrichment pipeline — NetBox + MiniMax → Telegram |
| `webhook_receiver.py` | Flask webhook receiver — parses format, calls pipeline |
| `noc_md_processor.py` | Builds `~/noc/context/{alert_id}.md` from inbox JSON |
| `router_diagnostics.py` | AI agent — SSHes to lab routers, runs read-only IOS commands |

## Two LLM Calls Per Alert

| Call | When | Purpose | File consumed |
|------|------|---------|---------------|
| Enrichment | Immediate (sync) | Interpret alert + NetBox context → Telegram briefing | NetBox API |
| Diagnostics | Immediate (sync, lab only) | SSH router → read-only commands → Telegram report | Live SSH |
| Context build | Async (fire-and-forget) | Build .md for on-demand access | `~/noc/inbox/*.json` |
| On-demand | Later (on request) | Synthesize .md → Telegram briefing | `~/noc/context/*.md` |

## LAB_MGMT_IPS — Safety Allow-List

Only devices with management IPs in `LAB_MGMT_IPS` get SSH diagnostics.
This is an **explicit allow-list** — no production hardware is ever touched.

```
192.168.1.201  R1-lab
192.168.1.202  R2-lab
192.168.1.203  R3-lab
192.168.1.204  R1-lab-switch
192.168.1.210  R-HQ-LAB
```

Populated by `webhook_receiver.py` at startup via NetBox query.

## Router Diagnostics Command Whitelist

**READ-ONLY — no configure, write, copy, or delete commands.**

| Category | Commands |
|----------|---------|
| Interface | `show ip interface brief`, `show interfaces`, `show interfaces trunk` |
| Routing | `show ip route`, `show ip ospf neighbor`, `show ip bgp summary` |
| Layer 2 | `show cdp neighbors detail`, `show lldp neighbors detail` |
| Health | `show processes cpu`, `show memory summary`, `show platform` |
| Logs | `show log`, `show logging` |
| Reachability | `ping <target>`, `traceroute <target>` (target arg required) |

## Skills

| Skill | Command | Purpose |
|-------|---------|---------|
| `noc-md-storage` | `/noc <alert_id>` | On-demand briefing from saved .md |
| `noc-troubleshoot` | `/noc-troubleshoot <alert_id> [question]` | Deep analysis from .md + raw SSH .json |
| `cisco-lab-access` | (reference) | SSH parameters, PTY script path, known device IPs |
| `router-diagnostic-agent` | (reference) | Full diagnostic agent architecture and command sets |

## MiniMax API Usage

Both enrichment and diagnostics use MiniMax directly (no OpenAI compat layer):

```python
url = "https://api.minimax.io/anthropic/v1/messages"
headers = {
    "Authorization": f"Bearer {MINIMAX_API_SECRET}",
    "x-api-key": MINIMAX_API_KEY,
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}
# DO NOT use openai-compatible base URL with /anthropic suffix — causes 529/503
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `.md` not found on `/noc` | `noc_md_processor.py` hasn't run yet | Wait or run manually: `python3 noc_md_processor.py <alert_id>` |
| No diagnostics section in `.md` | Device not in `LAB_MGMT_IPS` | Production device — SSH diagnostics not available |
| SSH `AuthenticationException` | RSA-SHA2 algorithm mismatch | Ensure `disabled_algorithms` is set in PTY script |
| MiniMax 529/503 on enrichment | Wrong URL construction | Use full URL `https://api.minimax.io/anthropic/v1/messages` directly |
| `LAB_MGMT_IPS` empty | NetBox query failed at startup | Check `mcporter netbox devices --query role:router` manually |
