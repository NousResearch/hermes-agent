# Customer Intent Escalation Supervisor — QA Report

## Summary

Status: PASS

Implemented the Hybrid Factory plan for Sophie WhatsApp/customer-service escalation:

- CRM-backed `crm.customer_intents` queue.
- Sophie-safe `customer_intent_raise` tool.
- Zeus/supervisor `customer_intent_list` and `customer_intent_update` tools.
- Restricted `customer_service` toolset with no sales/calendar/terminal/cron/delegation or quote/invoice/raw CRM adapter tools.
- Read-only deterministic supervisor script.
- Hermes cron job `customer-intent-supervisor` scheduled every 5 minutes.

## Verification Evidence

### Factory

- Project: `customer-intent-escalation-supervisor`
- Lane: `customer-intent-hybrid`
- Gates recorded as passed: intake, functional, architecture, planning, implementation.

### Migration

Command:

```bash
python scripts/agent_core_db.py migrate
```

Result:

```text
crm:000005 applying db/modules/crm/000005_customer_intents.sql -> zeus_agent
crm:000005 applied
```

### Tests

Command:

```bash
python -m pytest tests/test_customer_intent_tool.py tests/test_customer_service_routing.py tests/gateway/test_whatsapp_dynamic_auth.py tests/gateway/test_whatsapp_reactions.py -q
```

Result:

```text
15 passed in 1.78s
```

### Lint / Compile

Commands:

```bash
python -m py_compile tools/customer_intent_tool.py scripts/customer_intent_supervisor.py
python -m ruff check tools/customer_intent_tool.py scripts/customer_intent_supervisor.py tests/test_customer_intent_tool.py tests/test_customer_service_routing.py
```

Results:

```text
All checks passed!
```

### Customer-Service Boundary Check

Observed `resolve_toolset('customer_service')` behavior:

```text
customer_intent_raise: True
crm_contact_upsert: True
crm_interaction_record: True
crm_quote_create: False
crm_invoice_create: False
crm_product_upsert: False
crm_twenty_raw_request: False
sales_quote_create: False
calendar_create_event: False
terminal: False
cronjob: False
delegate_task: False
```

### Live Tool Smoke

A synthetic intent was raised, listed, and cancelled through the handlers against the live Agent Core DB:

- intent_id: `intent-qa-whatsapp-supervisor`
- status after raise: `pending`
- list returned 1 pending record
- final status: `cancelled`
- result_summary: `QA synthetic intent cancelled after verification`

### Supervisor Script

Command:

```bash
python scripts/customer_intent_supervisor.py --limit 5 --pretty
```

Result with no pending intents:

```text
(no output)
```

This confirms the script stays silent unless there is pending work.

### Cron

Cron job created:

- job_id: `d40d2ba4c6e3`
- name: `customer-intent-supervisor`
- schedule: `every 5m`
- script: `customer_intent_supervisor.py`
- workdir: `/home/jean/Projects/hermes-agent-original`
- last_status after manual run/list: `ok`

### Independent Review

First independent reviewer found a real boundary issue: customer_service included full `crm`, exposing `crm_quote_create`, `crm_invoice_create`, `crm_product_upsert`, `crm_twenty_raw_request`, and `crm_twenty_sync`.

Fix applied:

- Removed `includes: ["crm"]` from customer_service.
- Replaced it with an explicit CRM-safe subset.
- Added negative tests for privileged CRM tools.

Second independent review verdict: PASS.

Codex CLI review was attempted but blocked by standalone Codex CLI 401 auth. Review was completed via independent Hermes subagent instead.

## Final Gate Result

- Spec gate: PASS
- Quality gate: PASS
- Test gate: PASS
- Security gate: PASS
- Delivery gate: pending commit/push
