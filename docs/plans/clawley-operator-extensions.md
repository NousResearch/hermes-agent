# Clawley Operator Extensions Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Give Clawley/Hermes read-only operator primitives for daily briefs, maintainer-sweep-to-kanban proposals, and a later dashboard command center.

**Architecture:** Scripts accept already-redacted local JSON snapshots and produce proposal/brief artifacts. They do not mutate GitHub, Kanban, Home Assistant, broker accounts, or QuantOS state. Later dashboard/cron wiring can consume these artifacts behind explicit gates.

**Tech Stack:** Python scripts, pytest, Hermes cron, dashboard plugin system, maintainer-sweep output, Kanban proposal import.

---

## Task 1: Daily brief payload builder

**Objective:** Build a read-only JSON/Markdown daily brief from redacted status sections.

**Files:**
- Create: `scripts/clawley_daily_brief.py`
- Test: `tests/test_clawley_operator_extensions.py`

**Acceptance:**
- Emits `schema=clawley_daily_brief.v1`.
- Emits `write_performed=false` and `read_only=true`.
- Recommends only terminal next actions like reviewing blockers or failed cron jobs.
- Does not call messaging APIs itself.

## Task 2: Maintainer sweep → Kanban proposals

**Objective:** Convert read-only maintainer-sweep `summary.json` payloads into Kanban proposal rows.

**Files:**
- Create: `scripts/maintainer_sweep_to_kanban_proposals.py`
- Test: `tests/test_clawley_operator_extensions.py`

**Acceptance:**
- Emits `mutation_allowed=false` and `write_performed=false`.
- Human-gates security/privacy/auth/infra/broker/live/trading labels.
- Does not call Kanban or GitHub APIs.

## Task 3: Clawley Command Center dashboard plugin

**Objective:** Add a read-only dashboard tab for gateway, cron, kanban, maintainer-sweep, QuantOS and Home Assistant health.

**Files:**
- Future create: `plugins/clawley-cockpit/dashboard/manifest.json`
- Future create: `plugins/clawley-cockpit/dashboard/plugin_api.py`
- Future tests: dashboard/plugin API tests

**Acceptance:**
- Read-only endpoints only.
- No raw secrets/prompts/private Telegram text.
- Works on Tailscale-only dashboard deployment.

## Task 4: Redacted observability event model

**Objective:** Define local event records suitable for later OpenTelemetry/Langfuse/Phoenix export.

**Files:**
- Future create: `scripts/clawley_observability_event.py`
- Future tests: redaction and schema tests

**Acceptance:**
- No prompt bodies, API keys, raw private IDs, or raw Telegram content.
- Records run ID, tool name, duration/status, artifact refs, and safety flags.

## Global gates

- LLM output is proposal-only.
- GitHub/Kanban/Home Assistant writes need separate deterministic applicators or explicit operator commands.
- Cron jobs should deliver concise terminal-state briefs, not noisy continuous chatter.
