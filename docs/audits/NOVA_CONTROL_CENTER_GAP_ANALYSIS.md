# NOVA Control Center — gap analysis

Companion to `NOVA_HERMES_CAPABILITY_AUDIT.md`. One section per recommended capability,
in the order they should be built. Each states the gap precisely enough to implement
against, and what would have to be true before the result could be called validated.

Machine-readable source of truth: `nova/capabilities/catalog.yaml`.

---

## 1. Task detail — run history, comments, artifacts

**Current state.** The Work screen shows state, agent, title and relative age. Opening a
task is not possible; there is no `/tasks/{id}`.

**Hermes capability.** `list_runs` (status, outcome, error, started/ended),
`list_comments` (author, body, time), `list_attachments` (filename, content type, size,
`stored_path`), `list_events`. All durable, all in `hermes_cli/kanban_db.py`.

**NOVA capability.** `nova/runtime/hermes/work.py` reads the `tasks` table only.

**Gap.** No adapter method and no endpoint for a task's history.

**Required adapter.** `AgentRuntime.task_detail(task_id) -> TaskDetail` in
`nova/runtime/base.py`, implemented in `nova/runtime/hermes/work.py`, read-only URI mode
as the rest of that module already is.

**Required API.** `GET /platform/v1/tasks/{id}` — viewer role, tenant resolved from the
bundle, added to `ROUTE_ROLES`.

**Required UI.** A task drawer or route: timeline of runs, comment thread, artifact list.

**Security.** `stored_path` is an absolute host path and must **never** leave the API —
return `{filename, content_type, size, uploaded_by, created_at}` only. The tenant scope
fixed in §7 of the audit is the precondition: without it this endpoint would have
served another tenant's comments to anyone who guessed an id. Run `error` text can carry
provider messages — pass it through the same redaction the review path uses.

**Testing.** Adapter unit tests; a tenant-isolation test asserting a foreign task id
returns 404 and not 403; an assertion that `stored_path` appears in no response body.

**Live validation.** Create a task on a real board, claim it, complete it with a comment
and an attachment, and read it back through the running control API.

---

## 2. Automations (scheduled work)

**Current state.** Nothing. The Control Center has no scheduling surface at all.

**Hermes capability.** `cron/jobs.py` — `list_jobs(include_disabled)`, `get_job(id)`,
jobs carrying `schedule` (`kind`: cron/interval, `expr`), `prompt`, `skills`, `enabled`,
`state`, `next_run_at`, and a `latest_execution` already joined in by `list_jobs`.
`cron/executions.py` holds the ledger; `cron/incidents.py` holds failures. The store is
`CRON_DIR = HERMES_DIR/"cron"`, i.e. per profile — **proven** in this audit by resolving
it under a temporary `HERMES_HOME`.

**NOVA capability.** None. `RuntimeCapabilities` has no scheduling flag.

**Gap.** Everything: capability flag, adapter, endpoint, screen.

**Required adapter.** `scheduling` capability flag; `list_automations()` returning a
NOVA-shaped view (id, title, schedule in human words, owner agent, enabled, last run,
next run). Import `cron.jobs` lazily inside `nova/runtime/hermes/` so the boundary test
in `tests/platform/test_boundaries.py` keeps holding.

**Required API.** `GET /platform/v1/automations` — viewer.

**Required UI.** An Automations screen: what runs, when, who owns it, did it last
succeed.

**Security.** Cron jobs carry a **prompt**, which is instruction text executed by an
agent. Displaying it is fine; allowing a viewer to edit it is equivalent to granting
arbitrary agent instruction and must be admin-gated and audited if ever added. Writes
are deliberately out of scope for the first pass — a schedule that NOVA did not compile
is a governance hole, and the design for that does not exist yet.

**Testing.** Adapter tests against a seeded `jobs.json`; a test proving one profile's
jobs are invisible from another profile's home.

**Live validation.** Would require running the cron scheduler loop and observing a job
fire. **Not achievable in this sandbox** (provider egress blocked), so the first release
must describe automations as "read from the runtime's schedule", not "verified to fire".

---

## 3. Usage depth — by model, by provider, estimated vs actual

**Current state.** Three aggregate tiles (tokens, API calls, recorded cost) and an
honest "observed, never enforced" banner.

**Hermes capability.** `session_model_usage` carries `model`, `billing_provider`,
`api_call_count`, five token counters, `estimated_cost_usd`, `actual_cost_usd`,
`cost_status`, `cost_source`.

**NOVA capability.** `nova/runtime/hermes/usage.py` **already selects every one of those
fields** and aggregates them. The UI shows three.

**Gap.** Presentation only. No adapter or API work.

**Required UI.** Breakdown by model and provider; cache and reasoning tokens separated
from input/output; estimated and actual cost shown as distinct figures with
`cost_status` explaining which is which.

**Security.** None beyond the existing admin gate on `/budget`.

**Testing.** A test asserting the UI never renders `actual_cost_usd` as authoritative
when `cost_status` says it is an estimate.

**Live validation.** Requires a real model call to populate the table. **Not achievable
here** — the sandbox blocks provider egress. Until then this screen must keep the
existing caveat.

---

## 4. Workforce health

**Current state.** The Overview shows counts derived from the task list.

**Hermes capability.** `DispatchResult` carries `reclaimed`, `promoted`, `crashed`,
`timed_out`, `auto_blocked`, `stale`, `skipped_unassigned`, `skipped_nonspawnable`,
`skipped_per_profile_capped`, `skipped_per_tenant_capped`, `respawn_guarded`,
`reconciled_orphans`, `memory_pressure`, `skipped_locked`. Plus
`kanban_db.release_stale_claims` and the `consecutive_failures` breaker.

**NOVA capability.** `health()` reports runtime reachability and agent count.

**Gap.** The dispatcher's own telemetry is never read.

**Required adapter.** A health surface that reads the **board**, not a live tick —
`DispatchResult` exists only inside the dispatcher process. Counts derivable from SQL:
tasks running, blocked, with `consecutive_failures > 0`, with expired claims.

**Security.** Board-wide dispatcher counts are **not tenant-scoped** — the catalog marks
`tenant_safe: unknown` for that reason. Either derive per-tenant counts from the tasks
table (safe) or keep this admin/operator-only. Do not show one tenant another's load.

**Testing.** Per-tenant derivation tests; a test that a tenant's health view excludes
other tenants' tasks.

**Live validation.** Achievable without a model: seed a board, kill a worker, observe the
health view report it.

---

## 5. Artifacts / deliverables

**Current state.** Nothing.

**Hermes capability.** `add_attachment`, `list_attachments`, blobs under
`attachments_root(board)/<task_id>/<stored_name>`.

**Gap.** No listing, no download path.

**Required API.** Listing is part of §1. Download needs a NOVA-mediated route that maps
`(tenant, task, attachment_id)` → bytes, so the client never learns a filesystem path.

**Security.** The highest-risk item in this document. `stored_path` is absolute and
attacker-useful; the file is customer content; `get_attachment` is addressed by
attachment id, which is a small integer and therefore trivially enumerable. That reader
is now tenant-scoped (audit §7) — a download route must re-check on every request rather
than trusting a list obtained earlier.

**Testing.** Enumeration test: a tenant iterating attachment ids 1..N receives only its
own. Path-disclosure test across every response shape.

**Live validation.** Achievable: attach a file on a real board, download it through the
API as the owning tenant, and fail to as another.

---

## 6. Task graph / objective structure

**Current state.** Objectives list their steps flat, with a `blocking` line.

**Hermes capability.** `task_links` is durable and enforced (`_parents_satisfied` gates
`ready → running`).

**Gap.** The edges are never sent to the client.

**Required UI.** A dependency view of an objective's real parent/child edges.

**Security.** Low — same data as the step list, differently shaped.

**Important limit.** This can only ever show the **task** graph. In-process subagent
trees are not persisted (audit §6.1), so a visualisation labelled "agent team" would be
inventing structure the runtime does not record. Either build on `task_links` and call
it what it is, or add a subagent ledger to Hermes core first.

---

## 7. Agent memory (read-only)

**Current state.** Nothing.

**Hermes capability.** `MemoryStore` over `MEMORY.md` / `USER.md` under
`get_memory_dir() = get_hermes_home()/"memories"`, with char budgets and an injection
scan on write. Per-profile, proven.

**Gap.** No adapter, no endpoint.

**Security.** Memory is free text an agent wrote about the customer's business, and it
enters the system prompt. Reading it is a legitimate transparency feature ("what does
this agent believe?"). **Editing** it is a prompt-injection vector by definition and
needs the same scan the write path applies — do not add an edit path that bypasses
`_scan_memory_content`.

**Testing.** Cross-profile read test; a test that the injection scanner is applied on any
write path NOVA adds.

**Live validation.** Achievable without a model: write memory through the store, read it
through the API.

---

## Not recommended for the Control Center

| Capability | Why not |
|---|---|
| MCP server management | Arbitrary tool surface; adding a server routes around NOVA's tool policy entirely |
| Plugin enable/disable | A plugin can register `pre_tool_call` and therefore outrank NOVA's own policy hook |
| External memory providers | Sends customer content to a third party |
| Hermes web dashboard | 0/23 routers are tenant-aware; operator-only, permanently |
| Cron job *writes* (first pass) | A schedule NOVA did not compile is an ungoverned agent instruction |
| Raw dispatcher tick counts to tenants | Board-wide, would leak other tenants' load |
