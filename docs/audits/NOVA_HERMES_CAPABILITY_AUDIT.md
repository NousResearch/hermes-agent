# NOVA ← Hermes capability audit

Evidence-based inventory of what the Hermes runtime actually does, and what the NOVA
Control Center can honestly surface on top of it.

Every row cites a file and symbol. Nothing is marked supported because a config key, a
schema, a CLI verb or a docstring exists — the question asked throughout is *what
happens at runtime*.

**Scale, for context.** Hermes is ~600k lines across `agent/`, `gateway/`, `tools/`,
`hermes_cli/`, `plugins/` and `cron/`. NOVA is ~15k. NOVA's job is not to reimplement
any of it; it is to govern it and present it.

## Ladder

| Level | Meaning |
|---|---|
| `declared` | Config/schema/type exists. No runtime consumer proven. |
| `wired` | A runtime call site reads it. |
| `enforced` | The runtime refuses/acts on it; bypass is not trivially available. |
| `tested` | Covered by a test in this repo. |
| `live-proven` | Observed in this audit against a real runtime, with the output recorded. |

---

## 1. Executive summary

Hermes already provides nearly all of the *mechanism* an enterprise AI-workforce
product needs: a durable task board with claims, run history, retries, circuit
breakers and orphan recovery; a full cron/automation engine with its own execution
ledger and incident tracking; per-profile memory; per-task artifacts; token/cost
accounting with an estimated-vs-actual distinction; 22 bundled channel platforms; a
~40-hook plugin system; and MCP.

NOVA surfaces a minority of it. The largest honest finding of this audit is not a
missing feature but a **category error waiting to happen**: the Hermes web dashboard
(23 routers under `hermes_cli/web_routers/`) contains **zero** references to `tenant`.
It is an operator console and can never become the customer surface. NOVA Control
Center has to be that surface, and everything it shows must come through NOVA's own
tenant-scoped control API.

This audit also found and fixed a **live cross-tenant read**: hiding a task did not
hide its children. See §7.

## 2. Capability matrix

`NOVA` column = what the Control Center exposes today.

### Workforce / task system

| Capability | Hermes evidence | Enforcement | NOVA | Customer safe? | Core change? | Priority |
|---|---|---|---|---|---|---|
| Durable task state | `hermes_cli/kanban_db.py:SCHEMA_SQL` (`tasks`, `task_runs`) | live-proven | surfaced (Work) | yes | no | — |
| Atomic claim / CAS | `kanban_db.claim_task`, `_claim_and_open_run` | live-proven | implicit | yes | no | — |
| Run history per task | `task_runs` (status, outcome, error, timings) | live-proven | **not surfaced** | yes | no | **P1** |
| Task comments | `kanban_db.list_comments`, `add_comment` | live-proven | **not surfaced** | yes | no | **P1** |
| Attachments / artifacts | `kanban_db.add_attachment`, `list_attachments`, `attachments_root` | live-proven | **not surfaced** | yes, path must be withheld | no | **P1** |
| Task events timeline | `task_events` + `_append_event` | live-proven | partial (policy decisions only) | yes | no | P2 |
| Task graph / dependencies | `task_links`, `create_task(parents=…)`, `kanban_db_graph` | enforced | partial (objective steps, not the graph) | yes | no | P2 |
| Priorities | `tasks.priority`, dispatcher `ORDER BY` | enforced | not surfaced | yes | no | P3 |
| Per-profile concurrency | `max_in_progress_per_profile` | enforced | compiled from `limits` | yes | no | — |
| Per-tenant concurrency + fair share | `kanban_db_dispatch._tenant_fair_order`, `max_in_progress_per_tenant` | tested | not surfaced | yes | no | P2 |
| Retry backoff | `kanban_db_dispatch.retry_backoff_seconds` | tested | not surfaced | yes | no | P3 |
| Circuit breaker | `consecutive_failures`, `_record_task_failure` | enforced | not surfaced | yes | no | P2 |
| Orphan / crash recovery | `detect_crashed_workers`, `_reclaim_dead_workers` | tested | not surfaced | yes | no | P2 |
| Stale-claim reclaim | `kanban_db.release_stale_claims` | tested | not surfaced | yes | no | P2 |
| Dispatcher tick telemetry | `DispatchResult` (12+ buckets incl. `skipped_per_tenant_capped`) | live-proven | not surfaced | yes | no | **P1** |
| Graceful shutdown | `kanban_db_dispatch.run_daemon` (SIGINT/SIGTERM) | wired | n/a | operator-only | no | — |

### Scheduling / automations

| Capability | Hermes evidence | Enforcement | NOVA | Customer safe? | Core change? | Priority |
|---|---|---|---|---|---|---|
| Cron/interval jobs | `cron/jobs.py` (`JOBS_FILE = CRON_DIR/jobs.json`, `list_jobs`, `get_job`) | live-proven | **surfaced** (Automations, Phase 11) | yes | no | done |
| Per-profile cron store | `cron/jobs.py:70 CRON_DIR = HERMES_DIR/"cron"` | live-proven | relied on for tenancy | yes | no | done |
| `use_cron_store` contextvar seam | `cron/jobs.py:136` | live-proven | the read/write path | yes | no | done |
| Pause / resume | `pause_job`, `resume_job` | live-proven | **surfaced** (admin, audited) | yes | no | done |
| Create / delete / run-now | `create_job`, `remove_job`, `trigger_job` | live-proven | **withheld** — an ungoverned agent instruction; PHASE_11 §3 | not yet | no | P2 |
| Ticker liveness | `record_ticker_heartbeat`, `get_ticker_heartbeat_age`, `get_ticker_success_age` | live-proven | **surfaced** — the banner | yes | no | done |
| Execution ledger | `cron/executions.py` (`executions` table: status, pid, timings, error) | wired | surfaced (proven on seeded data only) | yes | no | done |
| Job incidents | `cron/incidents.py` (`cron_incidents`) | wired | nothing | yes | no | P2 |
| Delivery queue + tombstones | `cron/delivery_queue.py` | wired | nothing | operator-only | no | P3 |
| One-shot / delayed | `cron/jobs.py` fire-claim fence | wired | carried in `schedule_kind`, not distinguished in UI | yes | no | P2 |
| **Scheduled execution actually firing** | needs a running gateway (`hermes_cli/cron.py:70`) | **wired, NOT live-proven** | reported honestly as "nothing is running these" | — | no | — |

### Memory

| Capability | Hermes evidence | Enforcement | NOVA | Customer safe? | Core change? | Priority |
|---|---|---|---|---|---|---|
| Curated file memory | `tools/memory_tool_store.py` (MEMORY.md/USER.md, char budgets) | wired | nothing | yes (read) | no | P2 |
| Per-profile scoping | `tools/memory_tool.py:38 get_memory_dir() = get_hermes_home()/"memories"` + dispatcher sets `HERMES_HOME` per profile (`kanban_db_dispatch.py:2352`) | **live-proven** | nothing | yes | no | P2 |
| Injection scan on write | `memory_tool_store._scan_memory_content` (strict scope) | enforced | nothing | yes | no | P3 |
| External providers | `agent/memory_manager.py`, `plugins/memory/{byterover,supermemory}` | wired | nothing | **no** — third-party egress | no | — |

### Delegation / agent teams

| Capability | Hermes evidence | Enforcement | NOVA | Customer safe? | Core change? | Priority |
|---|---|---|---|---|---|---|
| Kanban parent/child graph | `task_links`, `kanban_db_graph.initial_task_state` | enforced | partial | yes | no | P2 |
| `delegate_task` child agents | `tools/delegate_tool*.py`, `agent/subagent_lifecycle.py` | wired | nothing | — | — | — |
| **Durable record of in-process children** | **none found** — no sqlite/kanban writes in `delegate_tool_child_run.py`, `delegate_tool_results.py`; `SubagentHandle` is in-memory | **absent** | n/a | n/a | **yes, to persist** | see §8 |
| Depth limit | `tools/delegate_tool_config.py:21 MAX_DEPTH`, `_get_max_spawn_depth` | enforced | compiled (`delegation.max_depth`) | yes | no | — |
| Concurrent children | `_get_max_concurrent_children` | enforced | compiled | yes | no | — |

### Observability / usage

| Capability | Hermes evidence | Enforcement | NOVA | Customer safe? | Core change? | Priority |
|---|---|---|---|---|---|---|
| Token counters | `hermes_state_usage.py:17 _TOKEN_COUNTERS` (input, output, cache_read, cache_write, reasoning) | wired | tokens only, aggregate | yes | no | **P1** |
| Per-model / per-provider | `session_model_usage` (`model`, `billing_provider`) | wired | read by `nova/runtime/hermes/usage.py` but **not shown** | yes | no | **P1** |
| Estimated vs actual cost | `estimated_cost_usd`, `actual_cost_usd`, `cost_status`, `cost_source` | wired | adapter reads all four; UI shows one | yes, with the distinction kept | no | **P1** |
| API call count | `api_call_count` | wired | shown | yes | no | — |
| Policy decisions | `nova/runtime/hermes/enforcement.py` → `policy.decision` audit rows | tested | surfaced (Activity) | yes | no | — |

### Channels

Already audited in depth — see `docs/NOVA_CHANNEL_AUDIT.md` (22 bundled platforms,
`gateway/platform_registry.py`, `ctx.register_platform`, profile routes, multiplex
allowlist, `agent/secret_scope.py`, `gateway/delivery_ledger.py`, webhook HMAC + replay
window). NOVA surfaces declaration, routing, readiness and per-channel approval.
**Not surfaced:** `gateway/delivery_ledger.py` outbound delivery state and failures
(P2), channel connection lifecycle/health (P2).

### Plugins / extensibility

| Capability | Hermes evidence | Enforcement | NOVA | Customer safe? | Core change? | Priority |
|---|---|---|---|---|---|---|
| Plugin discovery/lifecycle | `hermes_cli/plugins.py` (`register(ctx)`, 57 found / 51 enabled observed) | live-proven | nothing | **operator-only** | no | P3 |
| Hook surface | `plugins.py:107 VALID_HOOKS` — ~40 hooks incl. `pre_tool_call`, `post_tool_call`, `pre/post_approval_*`, `on_kanban_*`, `subagent_start/stop` | wired | NOVA installs one (`pre_tool_call`) | no | no | — |
| `pre_tool_call` fail-closed | `nova/runtime/hermes/enforcement.py` | **enforced + tested** | governs every agent | yes | no | — |
| MCP servers | `tools/mcp_tool.py`, `hermes_cli/config.py:1029 "mcp_servers"`, `_disable_suspicious_mcp_servers` | wired | nothing | **no** — arbitrary tool surface | no | — |

### Security boundaries

Classified per the brief: REAL / PARTIAL / DECLARED / UNSUPPORTED.

| Boundary | Class | Evidence |
|---|---|---|
| Process isolation | **REAL** | worker is a separate OS process (`_default_spawn`) |
| Profile/config isolation | **REAL** | `env["HERMES_HOME"] = resolve_profile_env(profile)` (`kanban_db_dispatch.py:2352`) |
| Memory isolation | **REAL** | follows `HERMES_HOME`; per-profile dir, live-proven |
| Cron store isolation | **REAL** | `CRON_DIR = HERMES_DIR/"cron"` |
| Tool *denial* | **REAL** | `pre_tool_call` returns `block`; fail-closed on every error path |
| Tool *allowlist* (positive scoping) | **DECLARED** | NOVA's own apply warns: "positive tool scoping (toolsets/allow) is recorded but not yet enforced by the hermes adapter" |
| Approval gate | **REAL** | `pre_tool_call` → `{"action":"approve"}`; fails closed with no human present |
| Credential isolation | **PARTIAL** | per-agent `.env` resolves per agent, but a worker inherits the host environment and the multiplex guard is inactive in dispatcher-spawned workers — documented verbatim in `nova/runtime/base.py:48-61` |
| Tenant isolation (tasks) | **REAL** | `kanban_tenant` + guards; 45 tests |
| Tenant isolation (child rows) | **REAL as of this audit** | was a live leak — see §7 |
| Hermes web dashboard | **UNSUPPORTED** | 0/23 routers reference `tenant` |
| Webhook verification | **REAL** | HMAC-SHA256 + Svix + replay window (see channel audit) |

## 3. Hidden capabilities — real, unsurfaced, safe

Ranked by customer value × existing runtime support ÷ effort.

1. ~~**Automations (cron).**~~ **Done — Phase 11**, see `PHASE_11_AUTOMATIONS.md`. The
   audit's central caveat survived into the product: Hermes runs its ticker inside the
   gateway, so the screen leads with whether anything is actually running the schedules.
2. **Task detail: run history, comments, artifacts.** All three durable, all three
   already in SQLite, none surfaced. This is what makes a task page feel like a record
   rather than a status light.
3. **Usage breakdown by model and provider, estimated vs actual.** The adapter already
   reads every field; only the UI is missing.
4. **Dispatcher health.** `DispatchResult` carries per-tick reclaim/crash/timeout/
   capped/guarded counts — a real Workforce Health view with no new telemetry.
5. **Agent memory (read-only).** Per-profile, isolated, injection-scanned.

## 4. Recommended Control Center architecture

Derived from what is actually surfaceable, not from a generic list. `*` = new.

```
Overview        Agents        Objectives     Work
Approvals       Activity      Knowledge      Channels
Automations     Artifacts*    Usage          Policies
Health*         Settings
```

Dropped from the brief's candidate list, with reasons: **Integrations** (MCP/plugins are
operator-only), **Deployment** (belongs to the operator CLI, not the customer),
**Tasks** separate from Work (duplicate), **Agent Teams** as its own area (the durable
graph is task-level; see §8).

## 5. Runtime boundary

| NOVA owns | Hermes owns |
|---|---|
| Tenant identity, RBAC, principals | Task execution, worker lifecycle |
| Policy compilation and the audit log | `pre_tool_call` invocation |
| Product vocabulary (agent, objective, channel) | Profiles, kanban, cron, memory |
| The customer-facing control plane | The operator console and CLI |
| Governance decisions | Transport and delivery |

Direction of travel is one-way: **Hermes runtime → NOVA adapter → NOVA control API →
Control Center.** The frontend is never authoritative and never touches the runtime
database.

## 6. Missing *runtime* capabilities (genuine Hermes gaps)

Distinguished from NOVA UI gaps:

1. **In-process subagent runs are not persisted.** `delegate_tool_child_run.py` and
   `delegate_tool_results.py` contain no persistence; `SubagentHandle` is in-memory.
   A live agent-team tree is therefore **not** reconstructable after the fact. Only the
   kanban task graph survives. Any "Agent Teams" visualisation must be built on
   `task_links`, or Hermes core must gain a subagent ledger.
2. **Positive tool scoping is not enforced** (see matrix).
3. **No per-tenant spend/queue quota primitive.** Concurrency is capped; spend is not.

## 7. Security finding, found and fixed during this audit

**Hiding a task did not hide its children.** `get_task` was tenant-scoped, but
`list_comments`, `list_comments_after`, `list_events`, `list_runs`, `list_attachments`,
`get_attachment`, `get_run` and `latest_run` were not. Observed against a real board:

```
As tenant-b, holding tenant-a's task id:
  get_task      : None
  list_comments  : 1 rows LEAKED
      -> the merger closes Friday
  list_events    : 4 rows LEAKED
  list_runs      : 1 rows LEAKED
```

A task id is guessable, `stored_path` on an attachment is an absolute filesystem path,
and comment bodies are free text written by the business. Fixed by scoping every child
read on the child row's own `tenant` column — which exists precisely so this costs one
predicate rather than a join. 11 tests added; all readers verified closed and
own-tenant access verified intact.

## 8. Recommended implementation phases

* **Phase A — Task detail.** `/tasks/{id}` returning runs, comments, artifact metadata
  (never `stored_path`). Highest value, zero new runtime concepts, and the security fix
  in §7 is its precondition.
* ~~**Phase B — Automations.**~~ **Shipped (Phase 11)** with pause/resume as the write
  surface. Create/delete stay withheld pending the governance design in
  `PHASE_11_AUTOMATIONS.md` §3 — a NOVA-created automation should be a compiled bundle
  artifact, not free-text prompt on a schedule.
* **Phase C — Usage depth.** Per-model/provider breakdown, estimated vs actual, using
  fields the adapter already reads.
* **Phase D — Workforce Health.** `DispatchResult` + `release_stale_claims` +
  circuit-breaker counts.
* **Phase E — Artifacts.** Download flow; requires a NOVA-mediated file route so
  `stored_path` never reaches the client.

## 9. Not claimed

* No live end-to-end run of a real LLM worker was performed in this audit; the sandbox
  blocks provider egress (`inference-api.nousresearch.com`, `openrouter.ai` refused by
  the proxy). Everything marked `live-proven` was proven against the real runtime
  *libraries and databases*, not against a model call.
* Cron **execution** was not observed end to end, before or after Phase 11. Reads,
  writes, isolation and liveness reporting are live-proven; a job firing is not, and no
  test asserts that it does. The scheduler loop remains `wired`.
* MCP, plugin isolation and the external memory providers were mapped, not exercised.
* `gateway/` channel internals are cited from the earlier Phase 9 audit, not re-proven.
