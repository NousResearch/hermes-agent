# Phase 11 — NOVA Automations

Surfacing the scheduling engine Hermes already runs, and governing it, without building
a second scheduler.

---

## 1. Cron audit

Traced end to end before any code. Ladder: `declared → wired → enforced → tested →
live-proven`.

| Capability | Evidence | Class |
|---|---|---|
| Job storage | `cron/jobs.py:70` `CRON_DIR = HERMES_DIR/"cron"`, `JOBS_FILE = CRON_DIR/"jobs.json"` | **live-proven** — created jobs and read them back |
| Per-store targeting | `cron/jobs.py:136` `use_cron_store(home)` — a **contextvar**, "without mutating process globals" | **live-proven** |
| Job record | `id, name, schedule{kind,expr,display}, enabled, state, next_run_at, last_run_at, last_status, last_error, failure_streak, paused_at, paused_reason, created_at, repeat, deliver` (34 fields) | **live-proven** |
| `next_run_at` computation | `compute_next_run` via `create_job` | **live-proven** — `every day at 07:00` → `2026-09-15T07:00:00+00:00` |
| Human schedule text | `schedule_display` / `_schedule_display_for_job` | **live-proven** |
| Read API | `list_jobs(include_disabled)`, `get_job`, `resolve_job_ref` | **live-proven** |
| Write API | `create_job`, `update_job`, `pause_job`, `resume_job`, `remove_job`, `trigger_job`, `rearm_oneshot` | **live-proven** for pause/resume/remove |
| On-disk locking | `_jobs_lock()` flock, `_fire_job_lock`, `claim_dispatch`, `heartbeat_run_claim` | wired |
| Execution ledger | `cron/executions.py` — `executions` table (status, pid, claimed/started/finished, error), `list_executions`, `latest_executions` | wired |
| Incidents | `cron/incidents.py` — `cron_incidents` | wired |
| Ticker liveness markers | `record_ticker_heartbeat`, `get_ticker_heartbeat_age`, `get_ticker_success_age`, `get_ticker_last_error` — per store via `_current_cron_store()` | **live-proven** |
| Due-job scan & dispatch | `get_due_jobs`, `cron/scheduler.py:3666 tick()` | wired |
| **Actual scheduled execution** | requires a running gateway — see §2 | **NOT live-proven** |

### Two findings that shaped the implementation

**a) There is no profile field on a job.** `create_job` takes prompt, schedule, model,
skills, workdir — and no agent or profile. A job belongs to *whatever store it is in*.
Since a NOVA agent **is** a Hermes profile with its own home, automations are per-agent
by construction, and tenant isolation is the directory rather than a check. Proven: from
agent A's store, `get_job(B_id)` → `None`, `pause_job(B_id)` → `None`,
`remove_job(B_id)` → `False`, and B's job is untouched.

**b) `use_cron_store` does not retarget the execution ledger.** It moves `jobs.json` and
the ticker markers, but `cron/executions.py:37` resolves its path from
`get_hermes_home()`, which a contextvar does not move. Proven:

```
get_hermes_home() inside use_cron_store: /root/.hermes
 -> executions.db would resolve to:      /root/.hermes/cron/executions.db
agentB's own executions.db would be:     .../cr/agentB/cron/executions.db
```

`list_jobs` internally calls `latest_executions`, so the `latest_execution` it attaches
is **read from the wrong store**. NOVA ignores that field and opens
`<profile>/cron/executions.db` itself, read-only — the same pattern
`nova/runtime/hermes/work.py` already uses. The alternative, setting the module-global
`EXECUTIONS_FILE`, is not thread-safe in a server.

## 2. The thing this screen exists to say

Hermes' own CLI documents it (`hermes_cli/cron.py:70`):

> The cron ticker only runs inside the gateway (`_start_cron_ticker` in gateway/run.py);
> there is no standalone cron daemon. Without a running gateway, `next_run_at` passes but
> jobs never fire and `last_run_at` stays null — the most common cron support report.

So a deployment can hold a perfectly correct schedule that nothing executes, and a naive
Automations screen would present that as a working automation suite. The screen therefore
leads with scheduler liveness, read from this agent's own markers:

* `ticker_heartbeat` — the loop iterated (`running`)
* `ticker_last_success` — it iterated *without raising* (`healthy`)

A ticker stuck failing every tick keeps the first fresh and the second stale; that is
reported as "running but recent ticks have failed", not as healthy.

## 3. What was implemented

```
Hermes cron store  →  nova/runtime/hermes/automations.py  →  ControlAPI  →  Control Center
```

| Layer | Change |
|---|---|
| Contract | `AutomationView`, `AutomationRunView`, `SchedulerHealth`; `scheduling` capability; default `list_automations` / `scheduler_health` / `set_automation_enabled` |
| Adapter | `nova/runtime/hermes/automations.py` — reads via `use_cron_store`, history via a direct read-only ledger open, liveness from the markers |
| Runtime | `HermesRuntime.list_automations/scheduler_health/set_automation_enabled`, `scheduling=True` |
| API | `GET /platform/v1/automations` (viewer); `POST /platform/v1/automations/{id}/decide` (admin) |
| UI | Automations screen with the liveness banner, per-agent cards, pause/resume |

**Reads** return what runs, for whom, when, whether it is paused, the next and last run,
recent attempts, and failure streak. **The prompt is never returned** — instruction text
is not needed to answer "what runs, when, did it work", and every returned field can leak.

**Writes are pause and resume only.** Both are transitions on a schedule the runtime
already holds, delegated to `pause_job`/`resume_job` so resuming recomputes `next_run_at`
and clears the pause marker — writing `enabled` directly would leave an automation that
looks active and never fires.

### Why create and delete are not offered *(superseded by Phase 12)*

> **Update.** Phase 12 implements create and delete, through the design this section
> asked for: automations are now a first-class spec compiled against the tenant before
> the runtime sees them. See `PHASE_12_GOVERNED_AUTOMATIONS.md`. The reasoning below is
> kept because it is why the compiler exists.

Creating an automation means handing an agent a **prompt that NOVA never compiled and no
policy reviewed**, on a recurring schedule. That is an ungoverned agent instruction: it
bypasses the spec → policy → `pre_tool_call` chain that every other agent behaviour goes
through. Deleting removes an operator's schedule with no NOVA-side record of what it was.

Neither is hard to build; both need a design first — automations declared in the tenant
bundle and compiled like agents and channels, so a NOVA-created automation is a governed
artifact rather than free text. That is a Phase 12 proposal, not an oversight. Pause and
resume need no such design because they act on what the operator already declared.

## 4. Security

| Requirement | How it holds |
|---|---|
| Tenant isolation | Structural: the store is a profile directory. Cross-agent reads and writes return `None`/`False` from the runtime's own API. Tested four ways. |
| RBAC | `/automations` viewer, `/automations/decide` admin, declared in `ROUTE_ROLES` / `WRITE_ROUTES`. `WRITE_ROUTES` refuses undeclared routes outright. |
| Owning agent | Resolved **server-side** by searching this tenant's agents. The client never supplies the pair, so its claim is not part of the lookup. |
| Audit | `automation.decision` intent → committed/failed, actor = the human principal, with the reason. |
| Credential isolation | Untouched; no credential is read or returned. |
| Runtime-owned transitions | NOVA asks; `cron.jobs` decides what pausing means. |
| Unknown id | 404, identical to another tenant's — distinguishing them confirms ids. |
| Frontend authority | None. The UI has no role knowledge; it posts and renders whatever the server says, including "this endpoint requires the admin role". |

### A note on testing RBAC

`nova serve` treats **any loopback caller as a local admin**, by design and documented in
`nova/control/server.py:113`: someone on the host can already read the principals file,
the bundle, the audit log and every profile directory straight off disk, so demanding a
bearer token from them protects nothing — principals gate *remote* access.

A `curl` from `127.0.0.1` therefore succeeds regardless of the token it carries, and an
attempt to demonstrate the role gate that way proves nothing. (This was confirmed the
hard way during this phase: a "viewer" token appeared to be allowed to pause an
automation, which turned out to be the loopback rule, not a permissions failure.) The
role gate is asserted where it is enforced — `ControlAPI.write` with a viewer principal —
in `test_a_viewer_is_refused_by_the_api_not_merely_by_the_ui`, which also checks that
nothing was recorded and nothing changed.
| CSP | Unchanged; verified on the live response. |

## 5. Validation

| Check | Result |
|---|---|
| `tests/platform/test_automations.py` | **23 passed** |
| Full platform suite | **715 passed** |
| kanban + platform, per file | **1022 passing** (was 999) |
| Live read | `GET /automations` against a real `nova serve` returned both schedules with computed `next_run_at` and honest `scheduler_health` |
| Live write | `POST .../decide` paused and resumed; audit shows `intent → committed` pairs with the acting principal |
| Live RBAC | Not demonstrable over loopback (see above); asserted at `ControlAPI.write` |
| Live UI | Browser drove pause → Paused pill → Resume → Scheduled, no console errors |

### Live-validation status

**Reads, writes, isolation and liveness reporting are live-proven** against the real
runtime.

**Scheduled execution is NOT live-proven.** No job was observed firing. Firing requires a
running Hermes gateway, and this environment's proxy refuses provider egress
(`inference-api.nousresearch.com`, `openrouter.ai`), so an agent-backed job could not
complete even if the ticker ran. Accordingly:

* no test asserts that a schedule fires;
* the execution-history path is exercised against a **seeded** ledger, which proves NOVA
  reads the right file — not that Hermes writes it;
* the UI says "never run" and "nothing is running these schedules" because that is the
  true state of this deployment, not a placeholder.

## 6. Remaining limitations

1. **Execution history is proven only on seeded data.** The schema was read from
   `cron/executions.py`; no observed run produced a row here.
2. **Incidents (`cron/incidents.py`) are not surfaced.** Mapped, not integrated.
3. **One-shot vs recurring is not distinguished in the UI.** `schedule.kind` is carried
   through the API (`once`/`cron`/`interval`) but the cards do not yet treat a one-shot
   differently.
4. ~~**No create/delete**~~ — **added in Phase 12**, via the compiler described in §3.
5. **`trigger_job` (run now) is not exposed**, in Phase 11 or 12. It is a real seam and would be a genuinely
   useful admin action, but running an automation on demand executes its prompt
   immediately; it belongs with the create/delete governance design.
6. **Liveness is inferred from markers, not from the gateway.** If a gateway writes
   markers and then dies mid-tick, NOVA reports the staleness after
   `HEARTBEAT_STALE_SECONDS` (180s), not instantly.
