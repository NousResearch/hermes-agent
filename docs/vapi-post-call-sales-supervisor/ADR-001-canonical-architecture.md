# ADR-001 — Canonical architecture for vapi-post-call-sales-supervisor

Status: Accepted for implementation planning

Date: 2026-06-03

## Decision

Replace the current deterministic post-call worker as the commercial decision-maker with a canonical service named `vapi-post-call-sales-supervisor`.

The deterministic worker remains as an ingestion/idempotency dispatcher only. Customer-facing post-call decisions are made by an intelligent supervisor agent, normally Zeus or an explicitly delegated post-call sales supervisor with tool access and QA obligations.

There is no semantic split between “demo” and “real”. Every sales call is handled as a real opportunity and every outbound artifact must meet commercial closing quality.

## Context

Sophie/Vapi successfully captured a customer commitment, but the deterministic worker initially sent wrong-vertical material. This was a product/process failure, not only a classifier bug.

A rule-only worker can be safe for plumbing, but sales post-call actions require:

- transcript interpretation,
- CRM context,
- intent and funnel stage analysis,
- creative document generation,
- QA against what the customer actually requested,
- channel strategy,
- follow-up strategy,
- and learning from mistakes.

## Architecture

```text
Vapi / Sophie
    ↓ public callback
Delivery sandbox event log
    ↓ deterministic collector
Post-call task queue / Agent Core DB
    ↓
vapi-post-call-sales-supervisor
    ↓
CRM Core / Sales Core / Signature Core / Calendar / Notification / WhatsApp / TTS / Document worker
```

## Component responsibilities

### 1. Public callback service

- Receives Vapi tool calls and status events.
- Authenticates Vapi.
- Appends audit events.
- Does not hold broad CRM/document/email secrets.

### 2. Deterministic collector

- Reads audit events.
- Normalizes call id and payloads.
- Fetches final call artifact if needed.
- Creates or updates `post_call_tasks`.
- Maintains idempotency.
- Emits work for supervisor.
- Does not generate or send final customer-facing material by itself.

### 3. Sales supervisor agent

- Loads call transcript, summary and tool calls.
- Loads CRM/customer timeline.
- Creates an action plan.
- Applies QA gates.
- Executes actions if approved by policy.
- Escalates to Zeus/human when required.
- Updates CRM and schedules follow-up.
- Records learning/procedure improvements.

### 4. Sales/Signature path

When the customer asks for prices, package, terms, monthly fee, proposal or “cotización”, the supervisor must route to formal quote flow:

- create formal quote in Sales Core,
- produce the formal quote document using the existing SitioUno quote/acceptance template,
- create workspace link under `https://zeus-sandbox.kidu.app/w/<token>/` when applicable,
- attach Signature Core approval/acceptance flow,
- send via email/WhatsApp,
- log CRM interaction,
- schedule follow-up.

This is distinct from “show me capabilities” material, but it is not a lower/higher quality path. Both are sales-critical.

### 5. Follow-up funnel

Every open opportunity enters a scheduled follow-up cycle. The supervisor chooses channel/cadence based on:

- last response,
- customer preference,
- urgency,
- value of opportunity,
- whether quote was sent,
- whether material was opened/acknowledged if tracking is available.

## State machine

```text
received_event
→ call_ready
→ supervisor_assigned
→ context_loaded
→ analyzed
→ plan_ready
→ execution_started
→ artifact_ready
→ qa_passed
→ sent
→ crm_logged
→ followup_scheduled
→ learning_recorded
→ closed
```

Blocked branches:

```text
qa_failed → supervisor_revision
missing_context → supervisor_question_or_research
high_risk → zeus_review
delivery_failed → retry_or_alternative_channel
customer_replied → update_funnel_and_replan
```

## Autonomy policy

Single policy, not demo/real split:

- Auto-execute low-risk follow-up messages if they pass QA and are based on existing CRM context.
- Auto-generate but require Zeus review for first-time customer-facing documents in a new vertical or unknown request type.
- Require Zeus review for formal quotes above configured threshold, unusual terms, legal/compliance claims, or strategic accounts.
- Always block if artifact does not match transcript.

## Data model sketch

Tables/schema may live under Agent Core `voice` and `sales` schemas.

```text
voice.post_call_tasks
- task_id
- provider
- call_id
- contact_id
- opportunity_id
- status
- supervisor_profile
- transcript_ref
- summary_json
- tool_calls_json
- requested_actions_json
- action_plan_json
- qa_results_json
- delivery_results_json
- learning_notes_json
- created_at
- updated_at

sales.followup_sequences
- sequence_id
- opportunity_id
- contact_id
- status
- cadence_json
- next_action_at
- last_action_at
- channel_policy_json
- stop_reason

sales.followup_events
- event_id
- sequence_id
- opportunity_id
- contact_id
- channel
- action_type
- scheduled_at
- executed_at
- result
- provider_message_id
- notes
```

## Why not deterministic-only?

Rejected because:

- customer asks are open-ended,
- wrong material can lose the sale,
- closing requires persuasion and judgment,
- rigid template selection creates false confidence,
- learning must update playbooks, not just keyword lists.

## Why not make Sophie handle everything during the call?

Rejected because:

- low-latency voice runtime should not perform heavy document generation and QA,
- Sophie should not claim completed work before it is executed,
- post-call actions need deeper context and tools,
- separation keeps call experience fast and safe.

## Quality invariant

A provider ACK proves delivery, not correctness. Success requires:

1. artifact matches transcript,
2. artifact passes professional QA,
3. delivery provider acknowledges send,
4. CRM records evidence,
5. follow-up is scheduled if opportunity remains open.

## Migration from current worker

1. Keep current worker alive temporarily as collector.
2. Stop using its template classifier as final authority.
3. Add post-call task records.
4. Add supervisor execution mode.
5. Add quote formal path.
6. Add follow-up sequence engine.
7. Deprecate deterministic customer-facing send except where supervisor explicitly approves.

## Consequences

Positive:

- higher sales quality,
- fewer wrong-material errors,
- better CRM/funnel discipline,
- reusable architecture for derived agents,
- aligns with SitioUno value proposition.

Trade-offs:

- more moving parts,
- requires supervisor prompt/tool contract,
- more QA gates,
- higher token/cost usage on meaningful sales calls.

This trade-off is accepted because losing sales opportunities is more expensive than running a supervised post-call agent.
