# Cost budget + alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An opt-in, per-profile `cost_budget:` block that skips discretionary cron jobs when the profile is over budget, alerts once per threshold crossing per period through the owning profile's adapter, alerts on per-run overspend and unpriced tokens, and gives the agent a read-only view of its own spend.

**Architecture:** One new module, `agent/cost_budget.py`, owns config parsing, spend (via `agent.cost_attribution.attribute_sessions` on `get_hermes_home()/state.db`, opened read-only per call, cached per home), alert composition, dedupe state, and sending (through `cron.scheduler._deliver_result` with the adapters the caller passes). The scheduler gets two one-line hooks: a pre-run check in `_run_one_job_body` (skip + ledger note, or loud overrun alert) and a throttled watcher call in `tick()`. A read-only tool `cost_budget_status` in its own `cost_budget` toolset reads the same snapshot.

**Tech Stack:** Python, sqlite (`SessionDB(read_only=True)`), the existing cron scheduler and delivery path.

**Spec:** `docs/superpowers/specs/2026-11-16-product-education-teammate-design.md`, "Dependencies → 3. Cost budget + alerts" (approved 2026-11-16).

## Global Constraints

- A profile with no `cost_budget:` block is byte-for-byte unchanged: no store open, no state file, no new log/audit writes, no ledger difference.
- Spend is resolved against the **owning profile's** store via `get_hermes_home()` per call — never a module-level path capture (multiplex: one process ticks several homes).
- Unpriced tokens are never counted as $0 silently: they raise an alert and every surface says spend is a lower bound.
- Scheduler hooks stay small and self-contained (a parallel branch edits `cron/scheduler.py`).
- Every budget path fails open (a budget bug must never stop cron or the gateway) and logs.
- Tests only via `scripts/run_tests.sh <path>`; never without a path; one package at a time.

## Config shape (decided)

```yaml
cost_budget:
  daily_usd: 5          # optional
  monthly_usd: 100      # optional
  alert_at: [0.5, 0.8, 1.0]   # default; fractions of each budget
  per_run_usd: 1.5      # optional; cron runs only (a run = one cron root session + its lineage)
  alert_channel: slack:C0123ABCD   # deliver-target syntax; bare "C…" means slack:C…
  alert_mention: "<!subteam^S123>" # optional; prefixed to LOUD alerts only
  discretionary_jobs: [explore, 3f9a1c2d4e5b]  # job ids or job names
  cache_seconds: 300    # spend snapshot TTL (default 300, min 30)
```

The block is "active" when it is a dict with at least one of `daily_usd`, `monthly_usd`, `per_run_usd` > 0. A job is discretionary when `job.get("discretionary") is True` or its id or name is in `discretionary_jobs`. Days and months use `hermes_time.now()` (configured timezone). Over budget = daily spend ≥ `daily_usd` or monthly ≥ `monthly_usd`.

Alert keys (each sent at most once, recorded in `${HERMES_HOME}/cost_budget/alerts.json` only after a successful send; entries older than 40 days pruned):

| Key | When | Loud |
|---|---|---|
| `daily:<YYYY-MM-DD>:<t>` | daily fraction ≥ t (only the highest newly crossed t is sent; lower ones are marked) | t ≥ 1.0 |
| `monthly:<YYYY-MM>:<t>` | same, monthly | t ≥ 1.0 |
| `run:<root_session_id>` | a cron run's cost ≥ `per_run_usd` | no |
| `unpriced:<YYYY-MM-DD>` | today has unpriced tokens | no |
| `overrun:<period-key>:<job_id>` | a non-discretionary job starts while over budget | yes |

No `alert_channel` → alerts are logged at WARNING and not recorded (so they send once a channel is configured).

## Review Focus

- Spend jumps past several thresholds between two checks (e.g. 40% → 110%): exactly one alert (the 1.0 one), all lower thresholds marked sent.
- Budget block present but `state.db` missing (fresh profile): zero spend, no crash, jobs run.
- Store or config raises mid-check: discretionary job still runs (fail open), warning logged.
- A secondary profile's tick must read its own store and send through its own adapters: the hook takes `adapters`/`loop` from the caller and the home from `get_hermes_home()` at call time.
- Delivery fails (adapter not yet connected): the alert is not marked sent, retried on the next watch.

---

### Task 1: Budget core — config, spend snapshot, status

**Files:**
- Create: `agent/cost_budget.py`
- Test: `tests/agent/test_cost_budget.py`

**Interfaces:**
- Produces:
  - `BudgetConfig` (frozen dataclass): `daily_usd: Optional[float]`, `monthly_usd: Optional[float]`, `alert_at: Tuple[float, ...]`, `per_run_usd: Optional[float]`, `alert_channel: Optional[str]`, `alert_mention: Optional[str]`, `discretionary_jobs: FrozenSet[str]`, `cache_seconds: int`.
  - `load_budget_config(raw: Optional[dict] = None) -> Optional[BudgetConfig]` — `raw` is the whole config dict; `None` reads `hermes_cli.config.read_raw_config_readonly()`. Bad values are dropped with a warning.
  - `RunSpend(root_id: str, job_id: str, started_at: float, cost_usd: float, unpriced_tokens: int)`.
  - `SpendSnapshot`: `day_key: str`, `month_key: str`, `day_usd`, `month_usd`, `day_unpriced_tokens`, `month_unpriced_tokens`, `unpriced_models: Tuple[str, ...]`, `runs: Tuple[RunSpend, ...]`, `as_of: float`.
  - `compute_spend(home: Path, now: datetime) -> SpendSnapshot` — `attribute_sessions` over [month start, now] with no-op job/target resolvers; a missing `state.db` yields a zero snapshot.
  - `get_spend(cfg: BudgetConfig, *, now: Optional[datetime] = None) -> SpendSnapshot` — cache keyed on `str(get_hermes_home().resolve())`, invalidated by TTL or day change.
  - `BudgetStatus`: `cfg`, `snap`, `day_fraction: Optional[float]`, `month_fraction: Optional[float]`, `over_daily: bool`, `over_monthly: bool`, property `over_budget`.
  - `budget_status(cfg, snap) -> BudgetStatus`.
  - `is_discretionary(job: dict, cfg: BudgetConfig) -> bool`.

- [ ] Write failing tests: `test_absent_block_is_none`, `test_block_without_any_limit_is_none`, `test_parses_full_block` (all keys above; bare `C123` → `slack:C123`), `test_bad_values_dropped` (`daily_usd: "x"`, `alert_at: [2, -1, 0.5, "q"]` → `(0.5, 2.0)`: sorted, non-positive and non-numeric dropped), `test_compute_spend_day_and_month` (seed sessions via the `_seed` pattern from `tests/agent/test_cost_attribution.py`: one priced $2 today, one $3 earlier this month, one $10 last month → day 2.0, month 5.0), `test_unpriced_tokens_counted_not_zeroed` (unknown-status session → `day_unpriced_tokens > 0`, model listed), `test_cron_runs_grouped_by_root` (root `cron_job1_20261116_080000` + child → one `RunSpend` summing both), `test_missing_store_zero_snapshot`, `test_get_spend_caches_per_home` (two homes via `HERMES_HOME` monkeypatch return their own numbers; second call within TTL does not call `compute_spend`), `test_status_over_daily` / `test_status_over_monthly`, `test_is_discretionary_by_field_id_and_name`.
- [ ] Run `scripts/run_tests.sh tests/agent/test_cost_budget.py` → fail (module missing).
- [ ] Implement in `agent/cost_budget.py`.
- [ ] Run → pass. Commit `feat(cost_budget): config, spend snapshot and status`.

### Task 2: Alerts — composition, dedupe, sending, watcher

**Files:**
- Modify: `agent/cost_budget.py`
- Test: `tests/agent/test_cost_budget.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `Alert(key: str, text: str, loud: bool)`.
  - `pending_alerts(status: BudgetStatus) -> List[Alert]` — pure; covers daily/monthly/run/unpriced keys; texts state dollars, the limit, "(lower bound: N unpriced tokens)" when relevant, and for the 1.0 crossing: "Discretionary jobs are skipped until <period resets>; release (non-discretionary) work keeps running."
  - `send_once(alerts: List[Alert], cfg, *, adapters=None, loop=None) -> int` — filters by `alerts.json`, marks lower thresholds of the same period, delivers through `cron.scheduler._deliver_result({"id": "cost_budget", "name": "Cost budget", "deliver": cfg.alert_channel}, text, adapters=adapters, loop=loop)`, prefixing `cfg.alert_mention` on loud alerts; records a key only when `_deliver_result` returns `None`; returns the number sent.
  - `watch(*, adapters=None, loop=None) -> None` — no block → return before any IO; throttled per home to `cfg.cache_seconds`; never raises.

- [ ] Failing tests: `test_single_alert_for_multiple_crossings`, `test_alert_not_resent_same_period`, `test_new_period_alerts_again`, `test_run_overspend_alert_once_per_root`, `test_unpriced_alert`, `test_delivery_failure_not_recorded` (stub `_deliver_result` returning `"boom"` → state empty, next call retries), `test_loud_alert_gets_mention`, `test_no_alert_channel_logs_only`, `test_watch_without_block_touches_nothing` (patch `compute_spend` to raise; no `cost_budget/` dir created), `test_watch_throttled`, `test_watch_never_raises`.
- [ ] Run → fail. Implement. Run → pass. Commit `feat(cost_budget): threshold alerts with once-per-period dedupe`.

### Task 3: Scheduler hooks — pre-run skip + tick watcher

**Files:**
- Modify: `agent/cost_budget.py` (add `preflight_cron_job`)
- Modify: `cron/scheduler.py` (`_run_one_job_body`, `tick`)
- Modify: `cron/executions.py` (`finish_execution(..., note: Optional[str] = None)` — stored in `error` on a completed row)
- Test: `tests/cron/test_cost_budget_scheduler.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `preflight_cron_job(job: dict, *, adapters=None, loop=None) -> Optional[str]` — returns `"skipped: over budget (<daily|monthly> $X of $Y)"` for a discretionary job when over budget; for a non-discretionary job over budget, sends the loud `overrun:` alert via `send_once` and returns `None`; `None` when no block, under budget, or on any error.

Hook in `_run_one_job_body`, right after the secret scope is set and before `run_job`: on a reason, `mark_job_run(job_id, True, None, status="skipped_budget", [expected_fire_owner])`, `finish_execution(execution_id, success=True, note=reason, delivery_outcome="suppressed")`, `return True`. Hook in `tick()` after the `can_dispatch` gate and before `get_due_jobs()`: `watch(adapters=adapters, loop=loop)` wrapped in `try/except Exception`.

- [ ] Failing tests (real `run_one_job` against a temp cron store, `run_job` patched to a sentinel): `test_discretionary_job_skipped_when_over_budget` (run_job not called; latest execution `status == "completed"`, `error == "skipped: over budget …"`; job `last_status == "skipped_budget"`), `test_named_discretionary_job_skipped`, `test_non_discretionary_job_runs_and_alerts_loudly` (run_job called; one `overrun:` alert delivered; second run → no second alert), `test_under_budget_runs`, `test_no_block_unchanged` (run_job called, `compute_spend` never called), `test_preflight_error_fails_open`, `test_tick_calls_watch_with_adapters`.
- [ ] Run `scripts/run_tests.sh tests/cron/test_cost_budget_scheduler.py` → fail. Implement. Run → pass.
- [ ] Run `scripts/run_tests.sh tests/cron/` → no new failures. Commit `feat(cron): skip discretionary jobs over budget; watch budget each tick`.

### Task 4: `cost_budget_status` tool

**Files:**
- Create: `tools/cost_budget_tool.py` (`registry.register(name="cost_budget_status", toolset="cost_budget", …)`, sync handler, `check_fn` always true)
- Modify: `toolsets.py` (add `"cost_budget"`), `hermes_cli/tools_config.py` (label row)
- Test: `tests/tools/test_cost_budget_tool.py`

**Interfaces:**
- Consumes: `load_budget_config`, `get_spend`, `budget_status`.
- Produces: JSON `{configured, as_of, today: {spent_usd, budget_usd, remaining_usd, fraction}, month: {…}, over_budget, discretionary_jobs_skipped, per_run_usd, unpriced_tokens_today, unpriced_tokens_month, unpriced_models, spend_is_lower_bound, costliest_runs_this_month: [{job_id, cost_usd, started_at}] (top 5)}`. No parameters. Without a block: `configured: false`, budgets null, spend still reported.

- [ ] Failing tests: `test_reports_spend_and_remaining`, `test_unconfigured_still_reports_spend`, `test_lower_bound_flag_with_unpriced`, `test_registered_in_own_toolset`.
- [ ] Run → fail. Implement. Run → pass. Commit `feat(tools): cost_budget_status read-only tool`.

### Task 5: Docs + inventory guard

**Files:**
- Modify: `CLAUDE.md` (new "Cost budget + alerts" subsection after "Cost attribution"; deployment notes: grant `cost_budget` to the role, list it in `platform_toolsets.slack`, needs a `pricing.overrides` entry for the model, `alert_channel` must be reachable by the profile's own bot)
- Modify: `tests/test_fork_feature_inventory.py` (entries for `_cost_budget.preflight_cron_job(` and `_cost_budget.watch(` in `cron/scheduler.py`, `toolset="cost_budget"`, `note:` in `finish_execution`)

- [ ] Add entries; run `scripts/run_tests.sh tests/test_fork_feature_inventory.py` → pass. Commit `docs: cost budget + alerts`.
- [ ] Push branch, open PR against main.

---

## Review fixes (2026-11-17, PR #5 review)

Supersedes the matching parts above; CLAUDE.md "Cost budget + alerts" is the current description.

1. Spend is booked by delta of each usage row's cumulative cost, dated by `last_seen` (`cost_budget/spend.json`); cold start baselines pre-month rows and flags whole-session bookings as upper bound.
2. One over-budget alert per period (`…:1` threshold key), shared by `watch` and every over-budget preflight, naming the triggering job.
3. Per-run alerts only since tracking began, batched per check; keys claimed under the lock, sent on a context-carrying daemon thread, claim released on failure.
4. `1.0` always a threshold; `alert_at` values > 5 rejected with a "did you mean" warning; skips always raise the deduped alert.
5–7. Tool `remaining_usd: null` under a lower bound; failed scans cached 60s; `local`/`origin` alert channels rejected, unresolved targets not marked sent.
8. `cron.jobs.mark_job_skipped`: no streak/repeat change, one-shot re-armed at budget reset, fire-claim loss honoured.
9. `cronjob(discretionary=true)` (set-only); `discretionary_jobs` matches ids only.
10. Two-profile test through `set_hermes_home_override`; stubs keyed on `now` so the cache path runs.
