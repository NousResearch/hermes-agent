# Alert Enrichment Pipeline — Documentation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ALERT SOURCE                                                               │
│  Opsgenie webhook          │  Mock lab /test/alerts API                    │
│  (production)              │  (development/testing)                       │
└──────────────┬────────────┴────────────────────────────────────────────────┘
               │  POST /alerts (Flask, webhook_receiver.py)
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  WEBHOOK RECEIVER  (webhook_receiver.py)                                    │
│                                                                              │
│  Responsibilities:                                                           │
│  - Parse the incoming JSON payload                                          │
│  - Detect source format: Opsgenie vs mock_lab vs generic                    │
│  - Route to the correct AlertRecord parser                                  │
│  - Call enrich_alert_from_dict()  ← synchronous, blocking                   │
│  - Return HTTP 200 if Telegram dispatch succeeds, 500 otherwise              │
│                                                                              │
│  Three source formats handled:                                               │
│                                                                              │
│  1. MOCK LAB (single):  {"alert_id": "...", "device": "...", ...}         │
│  2. MOCK LAB (batch):   {"alerts": [...]}  → processes all, returns count  │
│  3. OPSGENIE:           {"action": "Create", "alert": {...}}              │
│                                                                              │
│  Notes:                                                                      │
│  - Processing is SYNCHRONOUS — Flask waits for Telegram before returning.   │
│  - A threading.Lock exists but is not actively used (threaded=True on Flask)│
│  - Non-Create actions (Ack, Resolve) are acknowledged with 200 but ignored. │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │  enrich_alert_from_dict(alert_dict, source)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ALERT PROCESSOR  (alert_processor.py)                                      │
│                                                                              │
│  Entry point:   enrich_alert_from_dict(alert_dict, source) → dict           │
│  Sub-entry:     enrich_alert(alert: AlertRecord) → dict  (used directly)    │
│                                                                              │
│  STEP 1 ─── NETBOX LOOKUP                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  netbox_lookup_device(hostname)  →  raw NetBox device dict            │    │
│  │                                                                       │    │
│  │  mcporter call: netbox_search_objects → netbox_get_object_by_id        │    │
│  │  mcporter binary:  /home/jourdan/.npm-global/bin/mcporter            │    │
│  │  NetBox server:    netbox-mcp  (MCP server name in config)            │    │
│  │                                                                       │    │
│  │  Falls back to empty device dict if not found — proceeds anyway.      │    │
│  └──────────────────────────┬────────────────────────────────────────────┘    │
│                             │  device_id (int)                              │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Three parallel/fast-follow lookups (only if device_id exists):      │    │
│  │                                                                       │    │
│  │  netbox_get_connected_devices(device_id)                             │    │
│  │    → via link_peers (NetBox 3.5+) or cable_trace fallback           │    │
│  │    → returns list of {device_id, device_name, interface, ...}        │    │
│  │                                                                       │    │
│  │  netbox_get_cables(device_id)                                        │    │
│  │    → iterates all interfaces → resolves cable → both terminations    │    │
│  │    → returns list of {cable_id, label, type, status, peer_device}    │    │
│  │                                                                       │    │
│  │  netbox_get_vm_hosts_at_risk(device_id)                              │    │
│  │    → finds connected devices with role slug "vm-host"                │    │
│  │    → queries virtualization.virtual-machine by cluster_id              │    │
│  │    → returns list of {host, site, [vm_names]}                        │    │
│  └──────────────────────────┬────────────────────────────────────────────┘    │
│                               │                                              │
│  STEP 2 ─── BUILD PROMPT                                                     │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  build_enrichment_prompt(alert, nb_device, connected, cables, vms)  │    │
│  │                                                                       │    │
│  │  Raw NetBox output is passed DIRECTLY to the LLM — no pre-digestion. │    │
│  │  The LLM is instructed to interpret the nested dicts itself.         │    │
│  │                                                                       │    │
│  │  Prompt sections:                                                     │    │
│  │  ## RAW ALERT          — alert_id, device, type, severity, message   │    │
│  │  ## NETBOX DEVICE      — role, site, type, status, IP, tags, serial │    │
│  │  ## SOURCE PROVIDED    — netbox_context, netbox_impact, cables, metrics│   │
│  │  ## CONNECTED DEVICES — device_name, interface, role (max 8)        │    │
│  │  ## CABLING            — label, type, status, ISP flag (max 6)       │    │
│  │  ## VM HOSTS AT RISK   — host, site, VM list (max 5 per host)       │    │
│  │  ## YOUR TASK          — 6-section briefing template for LLM         │    │
│  │                                                                       │    │
│  │  Output constraint: max 400 words, must fit in single Telegram msg   │    │
│  │  Not-found device: adds ⚠️ warning, proceeds with alert-only data     │    │
│  └──────────────────────────┬────────────────────────────────────────────┘    │
│                               │                                              │
│  STEP 3 ─── LLM CALL                                                     │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  call_minimax(prompt, max_tokens=600)                               │    │
│  │                                                                       │    │
│  │  URL:     https://api.minimax.io/anthropic/v1/messages              │    │
│  │  Model:   MiniMax-M2.7                                              │    │
│  │  Auth:    Bearer token from MINIMAX_API_KEY env var                  │    │
│  │                                                                       │    │
│  │  .env loading:  Lines 22-30 in alert_processor.py                   │    │
│  │    → Parses ~/.hermes/.env at import time if file exists            │    │
│  │    → os.environ.setdefault(key, value) for each non-comment line     │    │
│  │    → This fixes the bug where Python subprocess didn't inherit shell │    │
│  │      env vars and MiniMax API key came back empty.                   │    │
│  │                                                                       │    │
│  │  Response parsing:                                                   │    │
│  │    → MiniMax returns: [{"type":"thinking",...}, {"type":"text",...}] │    │
│  │    → Only block.type == "text" blocks are joined into the response  │    │
│  │    → thinking blocks are silently discarded                           │    │
│  │                                                                       │    │
│  │  Error handling:                                                     │    │
│  │    → If no API key or call fails: returns _fallback_briefing()      │    │
│  │    → Fallback extracts DEVICE and ALERT lines from prompt manually  │    │
│  │    → Returns: "⚠️ Manual Review Required" message                    │    │
│  │                                                                       │    │
│  │  Token limit: 600 tokens output (sufficient for ~400-word briefing) │    │
│  └──────────────────────────┬────────────────────────────────────────────┘    │
│                               │                                              │
│  STEP 4 ─── SEND TO TELEGRAM                                                 │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  send_telegram(text)                                                 │    │
│  │                                                                       │    │
│  │  Channel:  -1003506715170  (Hermes Alerts, hardcoded + env override)  │    │
│  │  Bot:      8764046749:AAHOX8PsdHiAFiUrzSD8LgDUFDd44zRBbCA            │    │
│  │  Parse:    Markdown (Telegram parse_mode)                            │    │
│  │  URL:      https://api.telegram.org/bot{TOKEN}/sendMessage           │    │
│  │                                                                       │    │
│  │  Message structure:                                                  │    │
│  │    {SEVERITY_HEADER}                                                  │    │
│  │    ─────────────────────────────────────                             │    │
│  │    {LLM_BRIEFING}  (max 4096 chars per Telegram limit)                │    │
│  │                                                                       │    │
│  │  Returns: True (sent ok) / False (failed)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Raw NetBox Dicts → LLM (No Pre-digestion)

NetBox returns deeply nested dicts with nested objects like:
```python
{"role": {"value": "spine-switch", "slug": "spine-switch"}, "site": {...}, ...}
```

Rather than flatten/extract these in Python before sending to the LLM, we pass the **raw nested dict** and let the LLM interpret it. This:
- Keeps the code simple
- Preserves all context (including fields we haven't explicitly extracted)
- Reduces Python-side processing and the risk of data loss

### 2. Synchronous Webhook Processing

`webhook_receiver.py` processes each alert **immediately** — no buffering, no queue. This means:
- `POST /alerts` returns only after Telegram confirms delivery
- Opsgenie sees either HTTP 200 (delivered) or 500 (failed) — useful for their retry logic
- A failed enrichment (NetBox down, LLM timeout) returns 500, triggering Opsgenie retry

### 3. device_found == False: Proceed Anyway

If NetBox doesn't know about a device, the prompt is built with empty context plus a warning:
```
⚠️ DEVICE NOT FOUND IN NETBOX — proceeding with alert data only.
```
This means a **valid alert** (even for an unknown device) still reaches Telegram. The NOC engineer can act on the raw alert; enrichment is bonus context, not a gate.

### 4. mcporter: MCP over stdio, not HTTP

The NetBox MCP server is accessed via `mcporter call netbox-mcp.<tool>` — a CLI subprocess. This is:
- The stable way to use MCP (not the native MCP client which had config issues)
- `NETBOX_SERVER = "netbox-mcp"` is the logical server name, not a hostname
- mcporter lives at `/home/jourdan/.npm-global/bin/mcporter`

---

## AlertRecord Data Model

Normalizes alerts from any source into a common format:

| Field | Type | Source |
|-------|------|--------|
| `alert_id` | str | Unique identifier from source |
| `device` | str | Hostname — **key for NetBox lookup** |
| `alert_type` | str | Freeform: `power-alert`, `bgp-alert`, etc. |
| `severity` | str | `critical` / `high` / `average` / `warning` / `info` |
| `message` | str | Human-readable alert text |
| `site` | str | Data center/site (from source or NetBox) |
| `timestamp` | str | ISO timestamp from source |
| `raw_payload` | dict | Original JSON (for debugging) |
| `is_mock_lab` | bool | True if from mock lab |
| `netbox_context` | dict | Pre-enriched context (from mock lab) |
| `netbox_impact` | dict | Pre-computed impact (from mock lab) |
| `netbox_cables` | list | Pre-computed cable info (from mock lab) |
| `metrics` | dict | Live metrics from mock lab Zabbix API |

---

## Redundancy Assessment

Built-in logic computes redundancy status from NetBox data:

```
HAS_REDUNDANCY                  → 2+ uplinks on a device with redundant role
PARTIAL_REDUNDANCY              → 1 uplink on a partially-redundant device
SINGLE_POINTS_OF_FAILURE        → Critical device + 0-1 uplinks OR 0-1 PSUs
UNKNOWN                         → Device role not in ROLE_REDUNDANCY map
```

This is computed in `_compute_redundancy_status()` and injected into the LLM prompt so the LLM can accurately describe blast radius.

---

## Known Limitations

1. **NetBox device coverage is partial.** Many test alert devices (`DC1-SPINE-01`, `DC1-ACCESS-03`, etc.) are not yet in NetBox. The pipeline falls back gracefully but enrichment is thin.

2. **processor.py (buffered path) is legacy.** It clusters alerts by device and uses a separate prompt builder. It is not the primary path — `webhook_receiver.py` → `alert_processor.py` is the current active pipeline.

3. **MiniMax response can be empty.** Occasional `LLM response (0 chars)` was observed — likely a timeout or empty `text` blocks in MiniMax's response. The Telegram message still sends with an empty briefing body. The LLM fallback does not trigger in this case.

4. **No retry logic for Telegram failures.** If `send_telegram` returns False, the webhook returns 500 and Opsgenie retries — but the LLM was already called (idempotent for the same prompt). No double-billing.

5. **No rate limiting.** If multiple alerts fire simultaneously, Flask threaded mode handles concurrency, but there is no guard against LLM rate limits.

---

## Bug History

### Bug: MiniMax API Key Not Loading
**Symptom:** `call_minimax` returned `⚠️ Manual Review Required` even with correct key in `.env`.
**Root cause:** `alert_processor.py` loaded env vars from its own process environment. The shell (`source ~/.hermes/.env`) sets vars in the shell, but Python subprocesses spawned by Flask don't inherit them unless explicitly sourced.
**Fix:** Added `.env` file parser at lines 22-30 of `alert_processor.py`:
```python
_env_path = os.path.expanduser("~/.hermes/.env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)
```
**Lesson:** When running Python as a subprocess (Flask, cron, systemd), `.env` files are never auto-loaded. Always load explicitly or set env vars in the service runner.

### Bug: Checkmk GUI Crash — `notify_plugin` String vs Tuple
**Symptom:** Checkmk GUI (Notifications page) crashes with `ValueError: too many values to unpack (expected 2)`.
**Root cause:** `notifications.mk` used a bare string `'checkmk-webhook'` for `notify_plugin` on the Hermes webhook rule. The email rule correctly used a 2-tuple `('mail', 'uuid')`. The GUI's `_render_notification_rules()` iterates ALL rules and does `notify_plugin_name, notify_method = rule["notify_plugin"]` unconditionally — one malformed rule crashes the entire page.
**Fix:** Change to tuple format:
```python
'notify_plugin': ('checkmk-webhook', '/omd/sites/cmk/local/bin/checkmk-webhook.sh'),
```
Then `sudo omd reload cmk`.
**Lesson:** Checkmk `notify_plugin` is **always a 2-tuple** for script-based plugins. Never use a bare string. The GUI renders all rules together — one bad tuple crashes the whole view.

### Bug: MiniMax API URL Routing Bug (auxiliary_client.py)
**Symptom:** LLM calls hang and eventually timeout with 529/503 errors.
**Root cause:** `_to_openai_base_url()` strips `/anthropic` from the path, then appends it back. If the input URL already contains `/anthropic`, it gets doubled, producing a non-existent endpoint.
**Fix:** Use the full URL directly:
```python
url = "https://api.minimax.io/anthropic/v1/messages"
```
Do NOT construct from base URL + `/anthropic` path.

### Bug: Checkmk Device Name Empty in Alert Enrichment
**Symptom:** Checkmk alerts reach the pipeline but `device` is empty, NetBox lookup returns nothing.
**Root cause:** Checkmk's form-encoded payload has a nested JSON string in the `context` field. `webhook_receiver.py` was not parsing this nested context before passing to `AlertRecord.from_checkmk()`.
**Fix:** `AlertRecord.from_checkmk()` handles the nested `context` JSON string — but only if the form-encoded field is correctly merged. Flask's `request.form` works for form-encoded Checkmk payloads.

---

---

## Test Alerts (test_alerts_stress.py)

22 alerts covering:

| Category | Count | Examples |
|----------|-------|---------|
| BGP / routing | 5 | Single ISP peer down, multi-BGP, flap discrimination, ISP node failure |
| Power | 2 | PSU failure, UPS low-runtime (critical) |
| VM / compute | 2 | VM host blast (5 VMs), double VM host failure |
| Physical | 3 | Spine fail (cascading spines), DC isolation, cable degradation |
| Cooling/thermal | 1 | Cooling failure → thermal shutdown |
| Security/policy | 1 | ACL asymmetric routing (no device failure) |
| WAN/overlay | 1 | SD-WAN tunnel degradation |
| Storage | 1 | RAID degraded → bulk VM evacuation |
| Certificate | 1 | 48h proactive expiry |
| Unknown device | 4 | Tests graceful degradation when NetBox lookup fails |

All 22 alerts compile and generate prompts (2,543–3,195 chars each). 17 were live-tested and all 17 reached Telegram successfully.
