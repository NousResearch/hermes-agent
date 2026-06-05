# vapi-post-call-sales-supervisor Implementation Plan

> For Hermes: Use subagent-driven-development or a dedicated coding engine to implement task-by-task. Keep the deterministic worker as collector first; move customer-facing decisions into the supervisor service.

Goal: Build the canonical post-call sales supervisor service for SitioUno voice sales, with one quality path for all customer-facing sales calls.

Architecture: Vapi callbacks append audit events. A deterministic collector creates idempotent post-call tasks. An intelligent supervisor analyzes transcript + CRM context, executes approved actions, runs QA, sends material, logs CRM, and schedules follow-up. Formal quote requests route through Sales/Signature Core acceptance flow.

Tech Stack: Python 3.11, Agent Core DB modules, Hermes cron/jobs, CRM/Sales/Signature/notification/WhatsApp/TTS tools, Vapi REST, SendGrid, ElevenLabs, document-worker runtime.

---

## Sprint 0 — Methodology and Safety Lock

### Task 0.1: Rename conceptual service and freeze old worker semantics

Objective: Make `vapi-post-call-sales-supervisor` the canonical name and demote `vapi_postcall_worker.py` to collector/legacy bridge.

Files:
- Modify: `scripts/vapi_postcall_worker.py`
- Create: `scripts/vapi_post_call_sales_supervisor.py`
- Test: `tests/test_vapi_post_call_sales_supervisor.py`

Steps:
1. Add module docstring explaining collector vs supervisor.
2. Add deprecation note: deterministic classifier is not final authority.
3. Tests assert collector creates tasks but does not send customer-facing material without supervisor approval.

Verification:
`python3 -m pytest -q tests/test_vapi_post_call_sales_supervisor.py tests/test_vapi_postcall_worker.py`

### Task 0.2: Add one-quality-path policy test

Objective: Prevent semantic branching into `demo` vs `real` quality paths.

Test behavior:
- `classify_call_context(...)` may return action types like `capability_material`, `formal_quote`, `followup`, `escalation`, but never quality modes `demo` or `real`.
- all outbound customer material must pass the same QA gate list.

Verification:
`python3 -m pytest -q tests/test_vapi_post_call_sales_supervisor.py::test_no_demo_real_quality_split`

---

## Sprint 1 — Post-call task model

### Task 1.1: Add Agent Core schema for post-call tasks

Objective: Persist call tasks, analysis, QA and delivery evidence.

Files:
- Create: `db/modules/voice/000003_post_call_sales_supervisor.sql`
- Test: migration smoke under existing DB migration test pattern if available.

Schema:
- `voice.post_call_tasks`
- `voice.post_call_events`
- `voice.post_call_artifacts`
- `voice.post_call_qa_results`

Required fields:
- `task_id`, `provider`, `call_id`, `contact_id`, `opportunity_id`, `status`, `transcript_ref`, `tool_calls_json`, `analysis_json`, `action_plan_json`, `qa_json`, `delivery_json`, `created_at`, `updated_at`.

Verification:
- migration applies cleanly,
- inserting duplicate `call_id` is idempotent/upserted.

### Task 1.2: Implement collector-to-task conversion

Objective: Convert Vapi events into `post_call_tasks` without deciding final material.

Files:
- Modify: `scripts/vapi_postcall_worker.py`
- Create: helper module if needed: `tools/voice_post_call_task_tool.py`
- Test: `tests/test_vapi_post_call_sales_supervisor.py`

Acceptance:
- groups by call_id,
- waits until final call event,
- stores transcript/tool calls,
- marks task `pending_supervisor`,
- no SendGrid/WhatsApp/document generation in collector.

---

## Sprint 2 — Supervisor analysis engine

### Task 2.1: Build call context loader

Objective: Load transcript, Vapi artifact, tool calls and CRM timeline.

Files:
- Create: `scripts/vapi_post_call_sales_supervisor.py`
- Test: `tests/test_vapi_post_call_sales_supervisor.py`

Inputs:
- `call_id` or `task_id`.

Output:
- normalized `CallBrief` dataclass with customer, company, requested actions, promised commitments, transcript excerpt, CRM context.

### Task 2.2: Build action planner

Objective: Decide action types from context using reasoning prompt/tool contract.

Action types:
- `capability_material`
- `formal_quote_with_acceptance`
- `followup_only`
- `schedule_meeting`
- `escalate_to_zeus`
- `no_action_needed`

Rules:
- If customer asks prices/packages/monthly fee/cotización/proposal terms → `formal_quote_with_acceptance`.
- If customer asks to see examples/capability/formats/demo material → `capability_material`.
- If customer already received material and needs nudging → `followup_only` or `schedule_meeting`.
- If ambiguous/high-risk → `escalate_to_zeus`.

Test cases:
- admin/accounting formats → capability material.
- “cuánto cuesta / mándame precio” → formal quote.
- “llámame la semana que viene” → follow-up/call scheduled.
- strategic custom pricing → escalation.

---

## Sprint 3 — QA gates

### Task 3.1: Implement QA gate runner

Objective: Block wrong material before send.

Files:
- Create/modify: `scripts/vapi_post_call_sales_supervisor.py`
- Test: `tests/test_vapi_post_call_sales_supervisor.py`

Gates:
- transcript fit,
- customer specificity,
- wrong vertical blocker,
- professional copy,
- no placeholders,
- document parse/readback,
- visual QA marker for PDFs when possible,
- provider delivery gate after send,
- CRM evidence gate.

Acceptance:
- wrong-vertical material fails before send,
- missing company/name does not fail if unavailable but creates a warning,
- formal quote without price source escalates.

### Task 3.2: Add customer-facing copy guard

Objective: Prevent internal process language in customer material.

Block terms:
- TODO, lorem, blocker, strategy notes, approval placeholder, internal rationale, “cliente-facing”, “copy alignment”, raw agent reasoning.

Verification:
- tests fail if these terms appear in outbound material.

---

## Sprint 4 — Formal quote with acceptance path

### Task 4.1: Integrate formal quote generation

Objective: When customer asks prices, create a formal quote using existing Sales/Signature pattern.

Files:
- Modify: `scripts/vapi_post_call_sales_supervisor.py`
- Reuse: CRM/Sales/Signature tools and workspace `/w/<token>/` pattern.
- Test: `tests/test_vapi_formal_quote_path.py`

Required output:
- quote record,
- PDF or workspace quote link,
- acceptance/action link,
- CRM interaction,
- follow-up scheduled.

Policy:
- Do not invent prices. Use configured catalog or escalate to Zeus.
- If quote amount/terms are available, generate and send.
- If not available, draft quote and ask Zeus for price approval.

### Task 4.2: Acceptance template QA

Objective: Ensure the formal quote includes acceptance affordance.

Verification:
- document/link includes approval/acceptance action,
- Signature Core event/audit path exists,
- quote status transitions are tested.

---

## Sprint 5 — Delivery orchestration

### Task 5.1: Email/WhatsApp send with evidence

Objective: Send approved artifacts through selected channels and record provider ACKs.

Rules:
- Email for attachments/formal quote by default.
- WhatsApp for short confirmations, links and voice notes when valid.
- Do not duplicate the same material without idempotency key.

Evidence:
- SendGrid `X-Message-Id`,
- WhatsApp message id,
- artifact path or workspace URL,
- CRM interaction id.

### Task 5.2: Sophie voice-note follow-up

Objective: After material/quote sent and WhatsApp exists, send short Sophie voice note when commercially appropriate.

Steps:
1. Fetch Sophie voice config from Vapi assistant.
2. Generate TTS with provider/voice id.
3. Send audio as native WhatsApp media.
4. Log CRM interaction.

Tests:
- voice note is not attempted without WhatsApp,
- uses configured voice id,
- logs message id.

---

## Sprint 6 — Funnel follow-up engine

### Task 6.1: Add follow-up sequence model

Objective: Track open opportunities and next actions.

Files:
- Create: `db/modules/sales/00000X_followup_sequences.sql` or equivalent module migration.
- Test: `tests/test_voice_sales_followup_funnel.py`

Fields:
- `sequence_id`, `opportunity_id`, `contact_id`, `status`, `stage`, `cadence_json`, `next_action_at`, `last_action_at`, `stop_reason`.

### Task 6.2: Implement cadence policy

Objective: Follow up like a good salesperson without overwhelming.

Default cadence:
- T+0 material/quote + optional voice note.
- T+1 business day WhatsApp confirmation.
- T+3 business days Sophie call if no response.
- T+5 business days value-add message.
- T+8-10 business days second call or close-soft message.
- T+14 days nurture/pause if silent.

Stop conditions:
- customer rejects,
- customer asks not to be contacted,
- opportunity closed-won/lost,
- Zeus pauses sequence,
- invalid contact channel.

### Task 6.3: Lead response integration

Objective: If the customer replies via WhatsApp/email, update funnel and avoid blind scheduled nudges.

Acceptance:
- inbound reply pauses next automated follow-up until agent handles it,
- positive reply creates next-step task,
- rejection stops sequence.

---

## Sprint 7 — Observability and learning loop

### Task 7.1: Supervisor report artifact

Objective: Every processed call gets a structured report.

Fields:
- call_id,
- customer/company,
- action plan,
- artifacts,
- QA results,
- delivery evidence,
- CRM IDs,
- follow-up sequence,
- learning notes.

### Task 7.2: Learning loop into skills/playbooks

Objective: Reusable process corrections become skill updates, not one-off memory bloat.

Rules:
- customer-specific data → CRM only,
- stable architecture preference → memory if needed,
- reusable workflow fix → skill/reference patch,
- code regression → test.

---

## Sprint 8 — Migration and cleanup

### Task 8.1: Disable deterministic customer-facing sends

Objective: Ensure old worker cannot silently send wrong material.

Acceptance:
- old worker only creates supervisor tasks,
- any legacy direct-send path requires explicit env flag and logs warning,
- cron points to new supervisor pipeline.

### Task 8.2: Port to `sitiouno-agent-runtime`

Objective: Propagate canonical implementation to derived-agent runtime.

Commands:
```bash
cp scripts/vapi_post_call_sales_supervisor.py /home/jean/Projects/sitiouno-agent-runtime/scripts/
cp tests/test_vapi_post_call_sales_supervisor.py /home/jean/Projects/sitiouno-agent-runtime/tests/
python3 -m pytest -q tests/test_vapi_post_call_sales_supervisor.py
```

### Task 8.3: Final verification

Run:
```bash
python3 -m pytest -q tests/test_vapi_post_call_sales_supervisor.py tests/test_vapi_postcall_worker.py tests/tools/test_notification_tool.py tests/test_customer_service_routing.py
```

Manual smoke:
- create synthetic Vapi call asking for capability material,
- create synthetic call asking for formal quote,
- create synthetic follow-up sequence,
- verify no customer-facing send occurs before QA.

---

## Delivery criteria

- PRD/ADR/prompt/plan committed.
- Tests prove no demo/real split.
- Tests prove formal quote path is selected on pricing request.
- Tests prove follow-up funnel schedules next action.
- Tests prove wrong-vertical material is blocked.
- Collector no longer sends material by itself.
- Supervisor produces structured report and CRM evidence.
- Changes propagated to `sitiouno-agent-runtime`.
