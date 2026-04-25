# Alert Enrichment Pipeline — Project State

**Last updated:** 2026-04-18
**Project dir:** `/home/jourdan/projects/alert-enrichment/`

---

## Architecture Overview

```
Mock Lab (192.168.1.9:5000)
    │ POST /simulate/power-alert, /simulate/bgp-down etc.
    ▼
webhook_receiver.py (Flask)
    │ AlertRecord dataclass: device, severity, alert_type, affected_component, impact, message, timestamp, raw
    ▼
processor.py — AlertBuffer (deque)
    │ cluster_by_device() → enrich_cluster() → build_cluster_briefing_prompt()
    ▼
enrichment_prompts.py
    │ complete() → LLM call → format_telegram_message()
    ▼
telegram_dispatcher.py
    └── Telegram (hermes-alerts channel — TBD)
```

---

## Files Written

| File | Purpose |
|------|---------|
| `webhook_receiver.py` | Flask app, `AlertRecord` dataclass, `normalize_alalog()` |
| `netbox_lookup.py` | `NbDevice`, `netbox_lookup()`, `get_cables()`, `get_connected_devices()`, `detect_alert_types()`, `compute_severity()`, `compute_delivery()` |
| `processor.py` | `AlertBuffer`, `cluster_by_device()`, `enrich_cluster()`, `process_alerts()` |
| `enrichment_prompts.py` | `EnrichmentContext`, `build_cluster_briefing_prompt()`, `build_site_briefing_prompt()`, `complete()`, `_fallback_summary()` |
| `telegram_dispatcher.py` | `tg_api()`, `send_text()`, `send_html()`, `send_alert()`, `list_updates()`, `resolve_channel_id()` |
| `cron_job.py` | 5-minute batch processor |
| `config.yaml` | All config: mock lab URL, NetBox MCP URL, Telegram bot token, batch interval, thresholds |
| `requirements.txt` | flask>=3.0.0, pyyaml>=6.0, urllib3>=2.0.0 |
| `README.md` | Architecture, setup, API reference |

---

## Mock Lab Devices (8 registered)

```
DC1-CORE-RTR-01   (Juniper MX204, DC1-SG Jurong, BGP/Transit)
DC1-BORDER-01     (DC1-SG Jurong, BGP peering)
DC1-CORE-01       (DC1-SG Jurong, tagged core_device in NetBox)
DC1-LB-01         (DC1-SG Jurong)
DC1-TRANSIT-RING-01-DEV01 (DC1-SG Jurong)
DC1-VMHOST-01     (DC1-SG Jurong, VMs)
DC1-SPINE-01      (DC1-SG Jurong)
DC1-SPINE-02      (DC1-SG Jurong)
```

---

## NetBox Topology (known good)

| Device | ID | Role | Site | Tags |
|--------|----|------|------|------|
| DC1-CORE-RTR-01 | 4 | Core Router | DC1-SG Jurong | — |
| DC1-CORE-01 | 2 | Core Switch | DC1-SG Jurong | `core_device` |
| DC1-BORDER-01 | 3 | Border Router | DC1-SG Jurong | — |
| DC1-TRANSIT-RING-01-DEV01 | 6 | Transit Ring | DC1-SG Jurong | — |
| DC1-VMHOST-01 | 7 | VM Host | DC1-SG Jurong | — |
| DC1-LB-01 | 5 | Load Balancer | DC1-SG Jurong | — |
| DC1-SPINE-01 | 8 | Spine Switch | DC1-SG Jurong | — |
| DC1-SPINE-02 | 9 | Spine Switch | DC1-SG Jurong | — |

### Cable Traces (DC1-CORE-01)
```
DC1-CORE-01 et-1/1  →  DC1-CORE-RTR-01 et-0/0/0  (Cable 3, Cat6A)
DC1-CORE-01 et-1/2  →  DC1-BORDER-01 et-0/0/1     (Cable 18, Cat6A)
DC1-CORE-01 et-1/3  →  DC1-TRANSIT-RING-01-DEV01 et-1/5 (Cat6A)
DC1-CORE-01 et-1/5  →  DC1-VMHOST-01 eth0         (Cat6A)
```

---

## Known Bugs Fixed

1. ✅ mcporter path: `/home/jourdan/.npm-global/bin/mcporter` (NOT `.local/bin`)
2. ✅ mcporter call format: `key=value` flag syntax (NOT `--args '{"key":"val"}'`)
3. ✅ `mcporter_call` unwraps `["results"]` from NetBox paginated responses
4. ✅ `_resolve_cable_peer` handles both nested dict `cable` and bare int `cable`
5. ✅ Removed `fields` param from cable trace path in `get_connected_devices`
6. ✅ `_extract_id()` helper handles both `{"id": N}` dicts and bare int/str IDs
7. ✅ `detect_alert_types` `alert.description` → `getattr(a, "message", "") or getattr(a, "description", "")`

---

## Pending Items

| Item | Status |
|------|--------|
| `detect_alert_types` fix | ✅ Done |
| `AlertBuffer` class | Missing — use raw `deque` for now |
| Telegram channel ID | `hermes-alerts` invite `+AMeiXRAPGrNiYzFl` — needs @username or bot as admin |
| Mock lab push mode | Not yet implemented — webhook from mock lab to Hermes |
| Google Drive access | MCP server forge:8082 unreachable — saving to local file instead |
| End-to-end pipeline test | Not run yet |

---

## mcporter Config

**Location:** `/home/jourdan/.mcporter/mcporter.json`

```json
{
  "servers": {
    "netbox-mcp": {
      "baseUrl": "http://192.168.1.9:8000/mcp",
      "transport": "streamable-http",
      "timeout": 30
    },
    "google-calendar": {
      "baseUrl": "http://forge:8082",
      "transport": "streamable-http"
    }
  }
}
```

---

## Telegram

- Bot token: `8187349242:AAEQ4d5usVCe-GPVG0TbwN7tQ2HC3AI0E-8` (from config.yaml)
- Target channel: `hermes-alerts` (unresolved)
- Invite link: `+AMeiXRAPGrNiYzFl` (private channel, needs @username or bot as admin)
- DM to "stout" works: chat ID `8500351481`

---

## Mock Lab Push Mode (TODO)

Add to mock lab: outbound webhook caller that POSTs to Hermes when alerts fire.
Target: `http://<hermes-host>:5001/alerts` (or whatever Hermes webhook receiver listens on)

Format: same as existing mock lab alert payload format.
