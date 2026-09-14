# Multi-tenant execution hardening — verification report

Scope: turn the Hermes execution kernel into something that can run several tenants
on one host without them reaching each other, and without one of them monopolising
the machine.

The brief said to inspect before changing. That mattered: **most of what was asked
for already existed**, and the honest result of the audit was a short list of real
defects rather than a rewrite.

---

## 1. What Hermes already had (left alone)

Found by reading `hermes_cli/kanban_db.py` (4.1k lines), `kanban_db_dispatch.py`
(2.3k) and `kanban_db_connect.py` (1.2k) before writing any code.

| Requirement | Already present |
|---|---|
| Durable task state | `tasks` + `task_runs` in SQLite; every attempt is a row, not memory |
| Queue + atomic claim | `claim_lock` / `claim_expires`, compare-and-swap in `_claim_and_open_run` |
| Survives restart | Claims and runs are on disk; a fresh process reclaims by PID liveness |
| Heartbeats | `last_heartbeat_at`, `heartbeat_claim`, `release_stale_claims` |
| Timeouts | `max_runtime_seconds` → SIGTERM → grace → SIGKILL, `outcome='timed_out'` |
| Stuck-worker recovery | `detect_crashed_workers`, `detect_stale_running`, orphan reconcile |
| Retry limit | `consecutive_failures` + circuit breaker (`max_retries`, `failure_limit`) |
| Graceful shutdown | `run_daemon` installs SIGINT/SIGTERM handlers, drains on `stop_event` |
| Global + per-profile concurrency | `max_in_progress`, `max_in_progress_per_profile` |
| Backpressure | memory-pressure guard degrades to 1 spawn, then 0 |
| Multi-writer safety | single-writer `write_txn`, cross-process dispatch lock |
| Tenant plumbed to workers | `HERMES_TENANT` already exported in `_default_spawn` |

None of this was rebuilt.

## 2. What was actually broken

`tasks.tenant` existed as a **column**, not a boundary. `kanban_db_graph` called it
"this soft namespace" in its own docstring, and nothing read it back.

Reproduced on a real board before any fix:

```
A created: t_38e28f0a
B created: t_38e28f0a  <-- SAME ID
   B's 'own' task actually belongs to tenant: tenant-a | title: Tenant A payroll export
get_task(A) from any caller: Tenant A payroll export
assign_task(A) -> True     A.assignee is now: attacker-profile
archive_task(A) -> True    A.status is now: archived
```

| # | Defect | Consequence |
|---|---|---|
| 1 | Idempotency keyed on the key alone | Tenant B's submission returned **tenant A's task id**: B's job never ran, and B held a readable, mutable handle to A's card |
| 2 | Idempotency lookup outside the write txn | Admitted in the code's own comment: "a concurrent-create race may insert twice" |
| 3 | `get_task`/`list_tasks` unscoped | Any caller with an id read any tenant's task |
| 4 | `assign`/`archive`/`delete`/`claim`/`complete`/`block` unscoped | Any caller with an id **mutated or destroyed** any tenant's task |
| 5 | `task_runs`/`task_events`/`task_comments`/`task_attachments` had no tenant column | Executions, events and artifacts did not carry `tenant_id` |
| 6 | Dispatcher ordered globally by `priority DESC, created_at ASC` | One tenant enqueueing 500 cards owns the head of the queue forever — the noisy-neighbour case, verbatim |
| 7 | No per-tenant concurrency cap | Nothing bounded one tenant's share of the host |
| 8 | No backoff on crash/timeout retries | A card that crashes on startup re-spawned **every tick** until the breaker tripped, spending a shared slot each time |
| 9 | NOVA `work.get_task`: `view.tenant_id and view.tenant_id != tenant_id` | Short-circuit on an unstamped row handed it to **every** tenant that asked |

## 3. Changes made

New: `hermes_cli/kanban_tenant.py` (193 lines) — an ambient tenant (contextvar),
`scope_clause()`, normalisation, and `HERMES_TENANT_STRICT`.

**Default is unscoped, deliberately.** A single-tenant workstation binds nothing and
behaves exactly as before; the boundary engages when a tenant is bound, which is what
a multi-tenant deployment does at its edges.

**Cross-tenant access reads as absent, not forbidden.** Returning "forbidden" for a
row that exists and "not found" for one that does not is an existence oracle for other
tenants' task ids. So the accessors return their ordinary not-found value and log — which
also means no caller's contract changed.

| Area | Change |
|---|---|
| Schema | `tenant` added to the four child tables, **back-filled from the parent task**; unique partial index `(COALESCE(tenant,''), idempotency_key) WHERE key IS NOT NULL AND status != 'archived'`; covering index for the fair-dispatch scan |
| `create_task` | Idempotency lookup moved **inside** the write txn and scoped to the resolved tenant; `IntegrityError` on the new index resolves to the race winner's id instead of retrying with a fresh id |
| Accessors | `_guard()` on `get_task`, `assign`, `archive`, `delete`, `claim`, `complete`, `block`; `list_tasks` inherits the bound tenant |
| Child rows | Tenant stamped by SQL subquery in the same statement, so a child row cannot disagree with its task and no call site has to remember |
| Scheduler | `_tenant_fair_order()` — group by tenant, preserve each tenant's own priority order, deal round-robin, least-loaded tenant first |
| Scheduler | `max_in_progress_per_tenant` + `skipped_per_tenant_capped`, with within-tick accounting |
| Retries | `retry_backoff_seconds()` — exponential, capped, deterministic; new `failure_backoff` respawn-guard reason |
| Worker | `HERMES_TENANT` bound on first kanban `connect()`, so a spawned worker inherits its boundary with no new plumbing |
| NOVA | `_existing_id` probe scoped by tenant; `work.get_task`/`list_tasks` refuse unowned rows under strict tenancy |

Config: `kanban.max_in_progress_per_tenant` (CLI + gateway),
`HERMES_KANBAN_RETRY_BACKOFF_SECONDS`, `HERMES_KANBAN_RETRY_BACKOFF_MAX_SECONDS`,
`HERMES_TENANT_STRICT`.

## 4. Verification

### Tests

`tests/hermes_cli/test_multitenant_execution.py` — **36 tests, all passing.**

Concurrency is real: threads with their own connections and a `threading.Barrier` to
force overlap, plus **eight separate OS processes** for the idempotency guarantee —
threads share a GIL and a process cache; processes share only the database, which is
where the guarantee has to live.

| Requirement | Tests |
|---|---|
| Concurrent A/B execution | `test_two_tenants_execute_concurrently` (8 threads, 4 per tenant), `test_a_claim_is_won_by_exactly_one_worker` (12 contenders, 1 winner) |
| Isolation | `test_reads_do_not_cross_tenants`, `test_mutations_do_not_cross_tenants` (6 parametrised ops, each of which *succeeded* before), `test_worker_subprocess_inherits_its_boundary` |
| Duplicate execution | `test_concurrent_submits_create_exactly_one_task` (16 threads), `test_separate_processes_cannot_duplicate_an_idempotent_job` (8 processes), `test_same_key_in_two_tenants_is_two_tasks` |
| Restart recovery | `test_claim_survives_process_restart_and_is_recovered`, `test_recovery_preserves_run_history` |
| Long-running jobs | `test_a_long_running_job_holds_its_claim_via_heartbeats`, `test_stale_claim_is_reclaimed_when_heartbeats_stop`, `test_a_heartbeat_from_another_claimer_is_refused` |
| Failure recovery | `test_a_failing_task_is_held_off_instead_of_hot_looping`, `test_one_tenants_crash_does_not_disturb_another` |
| Noisy neighbour | `test_flooding_tenant_does_not_starve_a_quiet_one`, `test_fair_ordering_alone_interleaves_tenants`, `test_per_tenant_cap_is_not_exceeded_within_one_tick`, `test_least_loaded_tenant_is_served_first` |
| No single-tenant regression | `test_unscoped_callers_still_see_everything`, `test_untenanted_board_keeps_global_dedupe`, `test_single_tenant_board_keeps_its_priority_order` |

Plus 3 in `tests/platform/test_tenant_isolation.py` for the NOVA unowned-row leak.

### Measured

Noisy neighbour, with the flood holding every advantage (enqueued first, priority 9):

```
spawn budget 6, per-tenant cap 3
  spawned: {'loud': 3, 'quiet': 2}
  quiet tenant got a slot: True
  loud tenant held to cap: True
  per-tenant-capped defers: 47
```

16 concurrent threads, same tenant, same key: 1 task in the DB, 1 distinct id returned,
0 errors. 8 separate processes: same.

### Regression

| Suite | Before | After |
|---|---|---|
| kanban + platform, per file | 986 passing, 1 file failing | **989 passing, 1 file failing** |
| NOVA platform suite | 680 passing | **680 passing** |
| New multi-tenant suite | — | **36 passing** |

Then, separately: **every one of the 85 test files that imports any module changed
here**, run individually. Four files had failures, and all four fail identically on
upstream `origin/main` in a clean worktree:

| File | Failures | Cause |
|---|---|---|
| `tests/hermes_cli/test_kanban_notify.py` | 13 | `pytest-asyncio` not installed |
| `tests/gateway/test_busy_wake_admission.py` | 3 | pre-existing on upstream |
| `tests/gateway/test_kanban_wake_acceptance.py` | 3 | pre-existing on upstream |
| `tests/run_agent/test_run_agent.py` | 1 | Anthropic interrupt handler; pre-existing on upstream |

Run **per file**: the kanban suite has pre-existing cross-test pollution when run as one
batch (28 failures on a clean tree, every file green in isolation). That is the
pre-existing state, not something this change introduced.

The whole `tests/hermes_cli` directory in one batch could **not** be completed: it hangs
at ~84% on tests that reach for network endpoints the sandbox proxy refuses
(`inference-api.nousresearch.com`, `openrouter.ai` and others), and
`tests/gateway/relay` aborts collection outright on the missing `pytest-asyncio`. So no
whole-repo pass/fail number is claimed here — only the per-file results above.

## 5. Two failures that were mine, not the product's

Recorded because both looked like bugs and neither was:

* `max_spawn` is a **live concurrency cap** (running + spawned this tick), not a
  per-tick budget — so `max_spawn=1` with one task already running correctly spawns
  nothing. My test's premise was wrong.
* Dead-worker reclaim is **host-scoped on purpose**: only claims carrying this host's
  prefix are eligible, because another host's PID is meaningless. My test minted a
  non-host-prefixed claim, so the reclaim correctly ignored it.

## 6. What is **not** claimed

* **The race fix is the in-transaction lookup, not the unique index.** A control run
  with the index dropped did not reproduce a duplicate, because SQLite serialises
  writers and the lookup now happens inside the write transaction. The unique index is
  a backstop — it makes the guarantee structural rather than a consequence of one
  engine's locking. Both are tested; the distinction is stated rather than blurred.
* **"Credential" and "memory" isolation are not covered here.** The brief lists them.
  Credentials already have their own boundary (`agent/secret_scope.py`, per-profile
  `.env`, fails closed under multiplexing) and were not changed; agent memory
  (`hermes_state_*`) was **not** audited in this pass and no claim is made about it.
* **Quotas beyond concurrency are not implemented.** Per-tenant *concurrency* and fair
  ordering are. A queued-item or spend quota is not, and would need a place to hold the
  counter across ticks.
* **No load testing.** Fairness is demonstrated on tens of tasks, not thousands, and
  not under sustained multi-hour pressure.
* **The Hermes dashboard was not audited.** NOVA's control API resolves through its
  bundle's `tenant_id` and is covered; the separate Hermes web dashboard was out of
  scope for this pass and is not claimed to be tenant-safe.
* **`HERMES_TENANT_STRICT` is opt-in.** Left off, an unbound caller still runs
  unscoped — the single-tenant behaviour every existing install depends on. A
  multi-tenant deployment must set it, and should.
