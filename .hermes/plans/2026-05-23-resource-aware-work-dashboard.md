# Hermes Resource-Aware Work Dashboard Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a quota-aware Hermes/MITC work dashboard that displays Codex quota, protects interactive work from background quota drain, and lets Ed prepare prompts in an Outbox for delayed sending when quota recovers.

**Architecture:** Implement one shared resource/policy layer for Codex quota, then have Web UI, TUI, cron policy, n8n alerts, and the Prompt Outbox consume that same state. Hermes remains source of truth for prompts, history, attachments, retention, and quota policy; n8n observes transitions and triggers alerts/actions through Hermes APIs.

**Tech Stack:** Hermes Agent Python backend, SQLite state store, Hermes API/Web UI, Hermes cron scheduler, OpenAI Codex account usage, n8n on srv-n8n, Telegram notifications, TUI Node.js v20.19.2 for ui-tui build validation.

---

## Product Decisions Validated by Ed

- Treat quota monitoring and delayed prompts as one product epic, not isolated features.
- Interactive user requests take priority over background crons.
- Non-critical project crons such as Graph'it can be paused below quota thresholds.
- n8n should send transition-only alerts at 10%, 5%, 0%, and recovery.
- Web UI should expose Codex quota directly.
- Web UI should add an Outbox / "À envoyer" tab for prompt drafts queued for later.
- Sending at 0% quota requires explicit confirmation.
- Prompt-response history retention must be configurable for enterprise/legal policy.
- Prompt drafts can later support attachments, merge/fusion, priority, and project tags.

## Shared Concepts

### Resource Status

Normalize Codex quota into a reusable resource object:

```json
{
  "provider": "openai-codex",
  "ok": true,
  "status": "ok|economy|warning|critical|exhausted|degraded",
  "checked_at": "2026-05-23T12:00:00Z",
  "stale": false,
  "session": {
    "remaining_percent": 72,
    "reset_at": "2026-05-23T13:00:00Z"
  },
  "weekly": {
    "remaining_percent": 41,
    "reset_at": "2026-05-28T08:00:00Z"
  }
}
```

Status policy:

- `ok`: >= 20%
- `economy`: < 20%
- `warning`: < 10%
- `critical`: < 5%
- `exhausted`: 0%
- `degraded`: quota fetch failed repeatedly; never treat this as 0%

### Prompt Outbox Entity

```json
{
  "id": "uuid",
  "title": "Short title",
  "content": "Prompt text",
  "status": "draft|queued|scheduled|waiting_quota|blocked|sent|answered|archived|failed",
  "priority": 50,
  "project": "Graph'it|MITC|infra|personal|null",
  "tags": ["graphit", "codex-heavy"],
  "provider": "openai-codex",
  "send_condition": {
    "mode": "manual|quota_positive|quota_above_threshold|quota_full|scheduled_time",
    "threshold_percent": 10,
    "scheduled_at": null,
    "require_confirmation": true
  },
  "attachments": [],
  "merged_from": [],
  "retention_policy": "default|short|project|legal_hold|custom",
  "created_at": "iso8601",
  "updated_at": "iso8601",
  "queued_at": null,
  "sent_at": null
}
```

### Prompt Response History Entity

```json
{
  "id": "uuid",
  "prompt_id": "uuid",
  "session_id": "Hermes session id",
  "request_snapshot": "exact prompt sent",
  "response_snapshot": "assistant answer/final result",
  "provider": "openai-codex",
  "model": "gpt-5.5",
  "quota_snapshot": {},
  "attachments_snapshot": [],
  "project": "MITC",
  "tags": [],
  "retention_policy": "default",
  "created_at": "iso8601",
  "expires_at": null,
  "archived": false
}
```

---

## Phase 1: Codex quota resource and visibility

### Task 1: Locate existing account usage implementation

**Objective:** Find the current OpenAI Codex account usage/quota code path and avoid duplicating logic.

**Files:**
- Inspect: `agent/`, `hermes_cli/`, `gateway/`, `tools/`, `tests/`

**Steps:**
1. Search for `account_usage`, `quota`, `openai-codex`, `/quota`, and `gquota`.
2. Identify the canonical function or command that returns Codex quota.
3. Document source file and existing JSON shape in a short note inside this plan or an implementation issue.
4. If no stable JSON exists, plan a small provider-neutral adapter.

**Verification:**
- There is a named Python function/module selected as the single source for quota fetch.
- The implementation does not use `hermes insights` as remaining provider quota.

### Task 2: Add quota resource service with TTL cache

**Objective:** Expose a reusable Python service for cached Codex quota status.

**Files:**
- Create or modify: likely `agent/account_usage.py`, `agent/resource_policy.py`, or a new `agent/resources.py`
- Test: `tests/agent/test_resource_policy.py` or similar

**Implementation guidance:**
- Cache result for 60-120 seconds.
- Return stale/degraded state on fetch failures.
- Do not block UI rendering on quota API latency.
- Do not encode API failure as 0%.

**Verification:**
- Unit tests cover ok/warning/critical/exhausted/degraded.
- Unit tests cover TTL cache hit and refresh.

### Task 3: Expose quota through Hermes API

**Objective:** Make quota available to Web UI/n8n via an internal API endpoint.

**Files:**
- Locate API server adapter under `gateway/` or Web UI backend.
- Add endpoint: `GET /api/resources/codex-quota` or existing API naming equivalent.
- Test route behavior.

**Verification:**
- Curling endpoint returns stable JSON.
- Failed quota fetch returns `ok: false` / `status: degraded`, not 0%.

### Task 4: Display quota in Web UI

**Objective:** Show Codex quota badge/status in the Web UI chat/dashboard without blocking the UI.

**Files:**
- Locate Web UI frontend sources.
- Add resource polling hook with TTL/backoff.
- Add badge text such as `Codex S72% W41% reset 3h`.

**Verification:**
- Normal quota visible.
- Warning/critical/exhausted style visible.
- Degraded state visible.
- UI remains responsive if endpoint is slow/unavailable.

### Task 5: Keep TUI statusbar compatible

**Objective:** Ensure TUI quota/statusbar work remains buildable after quota service changes.

**Files:**
- `ui-tui/packages/hermes-ink/...`
- `ui-tui/dist/entry.js` if generated artifact is intentionally tracked in this branch.

**Commands:**
```bash
cd /home/edou/src/hermes-agent/ui-tui
export PATH=/home/edou/.local/bin:$PATH
npm run type-check
npm run build
```

**Verification:**
- Type-check passes.
- Build passes with Node.js v20.19.2.

---

## Phase 2: n8n transition alerts and cron protection

### Task 6: Implement n8n quota monitor workflow

**Objective:** Create workflow on srv-n8n that monitors quota and sends transition-only Telegram alerts.

**Files/Systems:**
- srv-n8n n8n instance.
- Hermes quota API from Task 3.
- Telegram notification credential/channel.

**Workflow nodes:**
1. Schedule Trigger every 5-10 minutes.
2. HTTP Request to Hermes quota endpoint.
3. Code node computes status + transitions using `$getWorkflowStaticData('global')`.
4. Telegram notification node.
5. Optional HTTP calls to pause/resume quota-managed crons.

**Verification:**
- Simulated 10%, 5%, 0%, and recovery transitions each send exactly one alert.
- Consecutive unchanged states send no spam.
- Fetch failure after N attempts sends degraded alert but does not pause jobs.

### Task 7: Add quota-managed cron policy surface

**Objective:** Let Hermes/n8n identify which crons may be paused by quota policy.

**Files:**
- Inspect `cron/` job model and CLI/API.
- Add metadata/tags if supported; otherwise use explicit allowlist config.

**Rules:**
- Never pause critical jobs.
- Pause only explicitly `quota-managed`/allowlisted jobs.
- Track IDs paused by policy.
- Resume only jobs paused by policy, not manually paused jobs.

**Verification:**
- Test job marked critical is not paused.
- Test manually paused job is not resumed.
- Test policy-paused job is resumed when quota >= 10%.

---

## Phase 3: Outbox MVP

### Task 8: Add outbox persistence model

**Objective:** Store prompt drafts/queued prompts in Hermes state DB or a profile-aware DB table.

**Files:**
- Inspect `hermes_state.py` and migration patterns.
- Add tables for `prompt_outbox` and optionally `prompt_response_history`.
- Test migrations on fresh and existing DB.

**Minimum fields:**
- id, title, content, status, priority, provider, project, tags JSON, send_condition JSON, created_at, updated_at, queued_at, sent_at, retention_policy.

**Verification:**
- CRUD operations work in tests.
- Existing state DB migrates safely.

### Task 9: Add Outbox API CRUD

**Objective:** Expose prompt outbox operations to Web UI.

**Endpoints:**
- `GET /api/outbox/prompts`
- `POST /api/outbox/prompts`
- `PATCH /api/outbox/prompts/{id}`
- `DELETE /api/outbox/prompts/{id}`
- `POST /api/outbox/prompts/{id}/queue`
- `POST /api/outbox/prompts/{id}/send`

**Verification:**
- API tests cover create, edit, delete, queue, manual send validation.
- Deleting a sent prompt either archives or requires explicit destructive action.

### Task 10: Add Web UI Outbox tab

**Objective:** Add second tab to chat page for "À envoyer".

**UI requirements:**
- list prompt drafts;
- create/edit/delete;
- save draft;
- queue;
- priority selector;
- send condition selector;
- send now button;
- blocked reason display.

**Verification:**
- User can create a prompt and refresh page without losing it.
- User can queue prompt with quota threshold.
- User sees why a prompt is blocked.

### Task 11: Implement quota-aware send guard

**Objective:** Enforce confirmation and policy before sending queued prompts.

**Rules:**
- If quota = 0 and manual send: return confirmation-required response.
- If quota fetch degraded: do not auto-send.
- If prompt older than configured threshold: require review before auto-send.
- If prompt has attachments in future: default to confirmation-required.

**Verification:**
- Unit tests for all guard decisions.
- UI shows confirmation flow at quota 0%.

### Task 12: Add prompt-response history MVP

**Objective:** Record sent prompt and final response snapshot linked to Hermes session ID.

**Files:**
- State DB model.
- API endpoint `GET /api/outbox/history`.
- Web UI history tab or section.

**Verification:**
- Sent prompt creates history record.
- History record includes session id, provider/model, prompt snapshot, response snapshot, timestamps.

---

## Phase 4: Advanced dashboard features

### Task 13: Attachments model

**Objective:** Track attachments separately from prompt text.

**Fields:** path/object key, original name, MIME type, size, checksum, retention policy, included_at_send.

**Verification:**
- Attachment metadata persists.
- Missing attachment blocks send with clear reason.
- Retention can purge attachment independently from prompt text.

### Task 14: Merge/fusion action

**Objective:** Let selected prompts merge into a single new prompt without losing provenance.

**Modes:**
- simple concat;
- structured requirements doc;
- deduplicate;
- implementation plan.

**Verification:**
- New prompt contains `merged_from` IDs.
- Source prompts are not destroyed by default.

### Task 15: Retention policy

**Objective:** Add configurable retention for drafts, history, and attachments.

**Configuration:**
- default days for archived drafts;
- default days for response history;
- shorter default for attachments;
- project override;
- legal hold/pinned exemption.

**Verification:**
- Dry-run purge lists records.
- Purge skips pinned/legal-hold records.

### Task 16: Batch send / process-ready endpoint

**Objective:** Let n8n or scheduler process eligible prompts when quota recovers.

**Endpoint:**
- `POST /api/outbox/process-ready`

**Rules:**
- Order by priority and scheduled time.
- Stop if quota drops below threshold.
- Respect confirmation-required prompts.
- Return summary of sent/skipped/blocked.

**Verification:**
- Recovery from 0 triggers process-ready but sends only eligible prompts.
- Summary can be sent via Telegram.

---

## Quality Gates

Run before merging implementation branches:

```bash
cd /home/edou/src/hermes-agent
python -m pytest tests/ -o 'addopts=' -q
cd ui-tui
export PATH=/home/edou/.local/bin:$PATH
npm run type-check
npm run build
```

If Web UI has separate frontend commands, add them after locating the package.

## Open Questions to Resolve During Implementation

- Exact Web UI source location and framework/package commands.
- Existing Hermes API route naming and authentication model.
- Whether cron jobs already support tags/metadata; if not, define allowlist config first.
- Whether prompt outbox belongs in `state.db` or profile-specific side DB.
- Exact provider quota fetch function for OpenAI Codex OAuth.
- Telegram destination/credential mechanism for srv-n8n.

## Related Skills

- `codex-quota-n8n-monitoring`
- `hermes-prompt-outbox-dashboard`
- `hermes-agent`
- `n8n-operations`
- `writing-plans`
