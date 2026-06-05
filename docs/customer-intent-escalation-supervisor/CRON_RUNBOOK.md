# Customer Intent Supervisor Cron Runbook

## Purpose

Sophie/customer-service conversations are constrained. When a customer asks for an action Sophie cannot execute directly, Sophie raises a CRM-backed customer intent. The supervisor cron scans pending intents and lets Zeus/supervised execution decide and complete the real work.

## Script

```bash
python scripts/customer_intent_supervisor.py --limit 10
```

Behavior:

- Prints nothing when there are no pending intents.
- Prints compact JSON when pending intents exist.
- Does not send messages, create quotes, mutate agenda, or mark work complete by itself.

## Hermes Cron Shape

Name: `customer-intent-supervisor`

Schedule: `every 5m`

Script: `customer_intent_supervisor.py`

Mode: LLM-driven cron with script context (`no_agent=false`). The script gathers pending intents; the prompt instructs Zeus to analyze and execute.

Recommended enabled toolsets:

```yaml
- crm
- customer_intents
- calendar
- notifications
- sales
- signature
- accounting
- messaging
- terminal
- file
```

The broad toolset is for Zeus/supervisor only, not Sophie.

## Supervisor Prompt Contract

When script output contains `pending_customer_intents`:

1. Load the intent and CRM/customer context.
2. Decide whether the requested action is safe and clear.
3. Execute only the concrete requested action.
4. Verify provider acknowledgement (SendGrid 202, WhatsApp bridge success, workspace URL 200, calendar event readback, etc.).
5. Record CRM interaction with provider evidence and artifact links.
6. Update the intent with `customer_intent_update`:
   - `completed` when executed and verified.
   - `blocked` when ambiguous, unsafe, missing required information, or needs Jean.
7. If blocked or strategically important, notify Jean/operator.

## Sophie Prompt Contract

Sophie must not claim execution. She says:

> “Perfecto, ya tomé nota de tu solicitud. La voy a escalar con el equipo de SitioUno para que la revisen y te demos respuesta lo antes posible.”

## QA Checklist

- Customer-service toolset includes `customer_intent_raise`.
- Customer-service toolset excludes `sales_*`, `calendar_*`, terminal, file, cron, delegation, and memory.
- `customer_intent_raise` creates `status='pending'` rows.
- Supervisor script is silent with no pending rows.
- Supervisor cron exists and runs every 5 minutes.
- Intent is not marked `completed` without verified delivery/action evidence.
