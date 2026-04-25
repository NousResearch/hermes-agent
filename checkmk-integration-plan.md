# Checkmk → Hermes Alert Enrichment Integration Plan

> Created: 2026-04-22
> Status: Step 1 complete — connectivity verified

---

## Architecture

```
Checkmk (192.168.1.230)
  └─→ [Alert Handler Script] ──POST──→ Hermes Flask (:5001)
                                           └─→ alert_processor.py
                                                 ├─→ NetBox MCP (device lookup)
                                                 ├─→ MiniMax LLM (enrichment)
                                                 └─→ Telegram NOC channel
```

## Connectivity

| Route | Status | Notes |
|-------|--------|-------|
| Checkmk → Hermes :5001 | ✅ HTTP 200 | No firewall/NAT changes needed |

- Hermes IP: `192.168.1.8`
- Checkmk Webhook URL: `http://192.168.1.8:5001/alerts`
- No outbound port blocking on Checkmk Docker

---

## Integration Steps

### Step 1 ✅ — Network connectivity
- **Verified:** `curl http://192.168.1.8:5001/alerts/status` returns `{"status":"ok"}`
- No changes required to Checkmk or network config

### Step 2 — Save integration plan
- File: `/home/jourdan/projects/alert-enrichment/docs/checkmk-integration-plan.md`

### Step 3 — Add Checkmk parser to alert_processor.py
- New `AlertRecord.from_checkmk()` class method
- New dispatch branch in `enrich_alert_from_dict()` for `source="checkmk"`
- Payload format: `application/x-www-form-urlencoded` with `context=<JSON>` field

### Step 4 — Create alert handler script on Checkmk container
```bash
# Path: /omd/sites/cmk/local/bin/checkmk-webhook.sh
docker exec monitoring bash -c 'cat > /omd/sites/cmk/local/bin/checkmk-webhook.sh << 'EOF''
#!/bin/bash
curl -s -X POST http://192.168.1.8:5001/alerts \
  -H "Content-Type: application/json" \
  -d "$(cat)"
EOF
chmod +x /omd/sites/cmk/local/bin/checkmk-webhook.sh'
```

### Step 5 — Register webhook in Checkmk notification rules (CORRECTED FORMAT)
Add to `/omd/sites/cmk/etc/check_mk/conf.d/wato/notifications.mk`:
```python
notification_rules += [{
    'description': 'Hermes Alert Enrichment Pipeline',
    # ⚠️ MUST be a 2-tuple: ('plugin_name', 'script_path')
    # Bare string 'checkmk-webhook' will crash the GUI with:
    # ValueError: too many values to unpack (expected 2)
    # The Checkmk GUI unconditionally unpacks notify_plugin as a 2-tuple
    'notify_plugin': ('checkmk-webhook', '/omd/sites/cmk/local/bin/checkmk-webhook.sh'),
    'match_host_event': ['?d', '?r', '?f', '?x'],
    'match_service_event': ['?c', '?w', '?r', '?f', '?x'],
}]
```
Reload: `sudo omd reload cmk`

---

## Checkmk Alert Handler Payload Format

Checkmk POSTs `application/x-www-form-urlencoded` to the script:
```
context=<JSON>&host_name=<hostname>&service_description=<svc>&...
```

JSON inside `context` field:
| Field | Description |
|-------|-------------|
| `host_name` | Host that generated the alert |
| `service_description` | Service name (empty for host alerts) |
| `service_output` | Alert message / output |
| `service_state` | 0=OK, 1=WARN, 2=CRIT, 3=UNKNOWN |
| `host_state` | For host-level alerts |
| `event_id` | Unique event ID |
| `check_type` | Check type (e.g. `70`, `71`) |
| `attempt` | Attempt number |
| `time` | Unix timestamp |

---

## Checkmk REST API (Optional Future Use)

| Item | Value |
|------|-------|
| Base URL | `http://localhost:8000/cmk/check_mk/api/1.0/` |
| Auth | Automation user + secret (in `/omd/sites/cmk/etc/automation/automation.secret`) |
| Endpoints | `/domain-types/alert/list`, `/domain-types/service/...`, `/domain-types/host/...` |

---

## Existing Notification Rules

```python
# Current email rule (partial — missing ?f, ?x event types)
notification_rules += [{
    'rule_id': '5a39b41a-d4f3-4e17-9aa1-287db9601209',
    'description': 'HTML email to all contacts about service/host status changes',
    'notify_plugin': ('mail', '4869b78e-bbef-5138-a828-5f290405557c'),
    'match_host_event': ['?d', '?r'],
    'match_service_event': ['?c', '?w', '?r'],
}]
```

Event type codes:
- `?d` = DOWN / device down
- `?r` = RECOVERY
- `?c` = CRITICAL
- `?w` = WARNING
- `?f` = FLAPPING
- `?x` = ACKNOWLEDGED

---

## AlertRecord Parser Plan

```python
@classmethod
def from_checkmk(cls, payload: dict) -> "AlertRecord":
    ctx = payload.get("context", {})
    if isinstance(ctx, str):
        import json
        ctx = json.loads(ctx)

    host = ctx.get("host_name", payload.get("host_name", ""))
    service = ctx.get("service_description", payload.get("service_description", ""))
    state = ctx.get("service_state", ctx.get("host_state", "UNKNOWN"))

    # Map Checkmk state: 0=OK, 1=WARN, 2=CRIT, 3=UNKNOWN
    severity_map = {"0": "info", "1": "warning", "2": "critical", "3": "warning"}

    return cls(
        alert_id=payload.get("event_id", f"checkmk-{host}-{service}"),
        device=host,
        alert_type=_detect_alert_type(service or host),
        severity=severity_map.get(str(state), "warning"),
        message=f"[{service}] {ctx.get('service_output', ctx.get('host_output', ''))}",
        site="",
        timestamp=ctx.get("time", payload.get("time", "")),
        raw_payload=payload,
        is_mock_lab=False,
    )
```

---

## Dispatch Routing in enrich_alert_from_dict()

```python
# Existing:
if source == "mock_lab" or ("alert_id" in alert_dict and "alerts" not in alert_dict):
    record = AlertRecord.from_mock_lab(alert_dict)
elif "action" in alert_dict and "alert" in alert_dict:
    record = AlertRecord.from_opsgenie(alert_dict.get("alert", {}), alert_dict.get("action", ""))

# Add:
elif "context" in alert_dict or "host_name" in alert_dict:
    record = AlertRecord.from_checkmk(alert_dict)
    source = "checkmk"
```

---

## Files

| File | Purpose |
|------|---------|
| `/home/jourdan/projects/alert-enrichment/webhook_receiver.py` | Flask receiver — add Checkmk route |
| `/home/jourdan/projects/alert-enrichment/alert_processor.py` | Add `from_checkmk()` + dispatch |
| `/home/jourdan/projects/alert-enrichment/enrichment_prompts.py` | Prompt templates (no change needed) |
| `/home/jourdan/projects/alert-enrichment/docs/checkmk-integration-plan.md` | This file |

---

## Checklist

- [x] Step 1: Network connectivity verified
- [ ] Step 2: Save integration plan
- [ ] Step 3: Add `AlertRecord.from_checkmk()` to `alert_processor.py`
- [ ] Step 4: Add Checkmk dispatch branch in `enrich_alert_from_dict()`
- [ ] Step 5: Create alert handler script on Checkmk container
- [ ] Step 6: Add Checkmk notification rule (webhook) in WATO/conf
- [ ] Step 7: Test end-to-end — trigger test alert from Checkmk
- [ ] Step 8: Verify Telegram briefing received in NOC channel
