# Customer Intent Escalation Supervisor — Hybrid Factory Work Plan

> **For Hermes/Factory:** Execute as a Hybrid lane: BMAD-style planning artifacts + SitioUno Factory DB/gates + Claude/Codex implementation/review + Zeus final verification.

**Goal:** Add a canonical WhatsApp/customer-service escalation path where Sophie captures customer requests she cannot execute, and Zeus/supervisor cron processes them asynchronously with full context and CRM evidence.

**Architecture:** Customer-facing Sophie remains restricted. She can record CRM context and raise a structured `customer_intent`; she cannot directly execute privileged actions such as quotes, email, documents, calendar mutations, file/system work, cron, memory, delegation, or terminal. A deterministic cron/script scans pending intents every 5 minutes and triggers a Zeus/supervised review loop using the full conversation and CRM context.

**Repo:** `/home/jean/Projects/hermes-agent-original`

**Factory Project:** `customer-intent-escalation-supervisor`

**Hybrid Lane:** `customer-intent-hybrid`

---

## Product Contract

### Problem

Voice calls already work because Vapi post-call summaries notify Zeus, who analyzes the transcript and executes follow-up actions. WhatsApp customer-service conversations lack the same handoff: Sophie may understand a request, but must not have privileged tools to execute it directly.

### Product Rule

Sophie captures and escalates. Zeus supervises and executes.

### Customer-Facing Acknowledgement

Sophie should answer naturally, for example:

> “Perfecto, ya tomé nota de tu solicitud. La voy a escalar con el equipo de SitioUno para que la revisen y te demos respuesta lo antes posible.”

She must not say the action was completed unless a verified tool result proves it.

### Security Boundary

The customer-service route must not expose tools that can perform privileged owner/operator actions. The allowed customer-facing tool surface is intentionally narrow: CRM context/notes/follow-ups plus `customer_intent_raise`.

---

## Increment 1 — Agent Core schema + customer intent tools

**Objective:** Persist structured customer intents in the shared Agent Core CRM schema and expose safe tools.

**Files:**
- Create: `db/modules/crm/000005_customer_intents.sql`
- Create: `tools/customer_intent_tool.py`
- Modify: `toolsets.py`
- Create: `tests/test_customer_intent_tool.py`

**Acceptance Criteria:**
- `crm.customer_intents` table exists with status lifecycle fields.
- Runtime role `crm_runtime` can insert/select/update safe fields.
- Sophie-safe tool `customer_intent_raise` creates pending intents only.
- Supervisor tools can list/update intents without exposing secrets.
- Tests cover schema-free SQL generation/handler behavior with monkeypatched SQL helpers.

**Implementation Steps:**
1. Write migration for `crm.customer_intents` with references to CRM contact/org/opportunity when available.
2. Add statuses: `pending`, `processing`, `completed`, `blocked`, `cancelled`.
3. Add intent types: free text field plus recommended values (`send_email`, `formal_quote`, `document`, `calendar_request`, `follow_up`, `escalation`, `other`).
4. Register tools:
   - `customer_intent_raise` — customer_service-safe creation.
   - `customer_intent_list` — supervisor/admin listing.
   - `customer_intent_update` — supervisor/admin status/action result update.
5. Add `customer_intents` toolset for supervisor usage and add only `customer_intent_raise` to the restricted customer-service tool list.

---

## Increment 2 — Restrict Sophie/customer_service and prompt escalation behavior

**Objective:** Ensure Sophie is architecturally prevented from executing privileged work and prompted to escalate.

**Files:**
- Modify: `toolsets.py`
- Modify: `gateway/run.py`
- Modify: `tests/test_customer_service_routing.py`

**Acceptance Criteria:**
- `customer_service` no longer includes `sales` or `calendar` toolsets.
- `customer_service` includes CRM and `customer_intent_raise` only for escalations.
- Customer-service prompt explicitly says to raise an intent and acknowledge escalation for non-executable actions.
- Tests assert the restricted toolset shape and prompt content.

**Implementation Steps:**
1. Update `_CUSTOMER_SERVICE_DEFAULT_PROMPT` with explicit escalation instruction.
2. Update `TOOLSETS["customer_service"]` to remove privileged `sales` and `calendar` includes.
3. Add test assertions that `customer_intent_raise` is available and sales/calendar are absent.
4. Keep owner/operator WhatsApp route unchanged.

---

## Increment 3 — Supervisor scan script + cron registration

**Objective:** Add a deterministic script and scheduled job that surfaces pending intents to Zeus/supervisor every 5 minutes.

**Files:**
- Create: `scripts/customer_intent_supervisor.py`
- Create/modify docs: `docs/customer-intent-escalation-supervisor/CRON_RUNBOOK.md`
- Register Hermes cron job: `customer-intent-supervisor` every 5 minutes, script-driven with LLM supervisor prompt.

**Acceptance Criteria:**
- Script prints no output when there are no pending intents.
- Script prints compact JSON/Markdown context when pending intents exist.
- Script does not execute privileged actions itself; it only collects and presents pending work.
- Cron job exists or is updated with self-contained prompt and correct toolsets.
- Cron prompt instructs Zeus to process pending intents, verify delivery, log CRM, then update intent status.

**Implementation Steps:**
1. Implement `scripts/customer_intent_supervisor.py --limit N` using Agent Core SQL helpers.
2. Include recent CRM timeline pointers and intent metadata, not secrets.
3. Create/update cron job with schedule `every 5m` and deliver `origin`.
4. Keep the script quiet when no pending rows are found.

---

## Increment 4 — QA, docs, final verification, push

**Objective:** Prove the flow works and push the changes.

**Files:**
- Create: `docs/customer-intent-escalation-supervisor/QA_REPORT.md`
- Update skills if lessons changed.

**Acceptance Criteria:**
- Unit tests pass for customer intent tools/routing.
- Migration applies against local Agent Core DB or SQL syntax is validated if DB unavailable.
- A synthetic pending intent can be raised/listed/updated through the tool handlers or script.
- Cron job is listed and has the intended schedule.
- Diff is independently reviewed by Codex or a review subagent.
- Changes are committed and pushed to `origin`.

**Verification Commands:**

```bash
python -m pytest tests/test_customer_intent_tool.py tests/test_customer_service_routing.py tests/gateway/test_whatsapp_dynamic_auth.py -q
python -m py_compile tools/customer_intent_tool.py scripts/customer_intent_supervisor.py
python scripts/agent_core_db.py migrate
python scripts/customer_intent_supervisor.py --limit 5
hermes cron list | grep -i customer-intent || true
```

---

## Factory Gates

- Intake: User request clear — customer-service escalation for WhatsApp.
- Functional: Sophie captures, Zeus executes.
- Architecture: CRM-backed queue + restricted tool surface + supervisor cron.
- Planning: Four increments above.
- Implementation: Code/docs/tests complete.
- Spec: Behaves as requested; no privileged Sophie tools.
- Quality: Tests/lint compile pass.
- Security: No secrets; customer route restricted.
- Delivery: Commit pushed to SiteOneTech repo.
