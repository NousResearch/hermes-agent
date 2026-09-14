# Phase 3 — Budget controls

Only the controls the audit proved the runtime can enforce, each labelled with what it
actually does. **Core patches: still 1** (the `AGENTS.md` routing row).

Everything here follows from one finding in
[`BUDGET_ENFORCEMENT_AUDIT.md`](BUDGET_ENFORCEMENT_AUDIT.md):

> **NOVA cannot stop a model call.** The runtime's LLM-boundary hooks discard their return
> values (`agent/turn_api_request.py:44`), and `pre_tool_call` is the only entry in
> `_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS`.

So a token or cost ceiling is not implementable, and NOVA does not pretend otherwise.

---

## The five enforcement classes

Every control carries its class **as data**, not as prose, so nothing downstream can
present a measurement as a ceiling. The vocabulary is in `nova/policy/limits.py`; the
*claims* live with each runtime adapter, because enforcement is a property of the
*(limit, runtime)* pair rather than of the limit alone. A future runtime that cannot veto
a tool call classifies the same limit differently, and the contract lets it say so through
`AgentRuntime.limit_facts()`. An adapter that declares nothing is taken to enforce nothing.

| Class | Meaning |
|---|---|
| `hard_preemptive` | The runtime refuses before the work happens |
| `hard_boundary` | NOVA refuses at the tool-call boundary |
| `soft_advisory` | Asks the agent to wrap up. **Not a limit** |
| `recorded_only` | Carried through; nothing NOVA controls enforces it |
| `observed_only` | Measured after the fact. **Never a ceiling** |

Only the first two are in `ENFORCING_CLASSES`. A control may not claim an enforcing class
without a `verified_at` call site — a test asserts it.

---

## What is enforced

### Hard, pre-emptive — compiled to verified runtime keys

| NOVA limit | Runtime key | Verified at |
|---|---|---|
| `max_turns` | `agent.max_turns` | `cli.py::_init_turn_limits` → `conversation_loop.py:1514` |
| `max_concurrent_tasks` | `kanban.max_in_progress_per_profile` | `gateway/kanban_watchers_dispatcher.py:116` |
| `delegation.max_concurrent_children` | `delegation.max_concurrent_children` | `tools/delegate_tool.py:469` |
| `delegation.max_depth` | `delegation.max_spawn_depth` | `tools/delegate_tool.py:188` |
| `delegation.max_child_turns` | `delegation.max_iterations` | `tools/delegate_tool.py:451` |
| `delegation.child_timeout_seconds` | `delegation.child_timeout_seconds` | `delegate_tool_config.py::_get_child_timeout` |
| `delegation.orchestrator_enabled` | `delegation.orchestrator_enabled` | `tools/delegate_tool.py:189` |

The delegation block reaches the runtime because `_load_config()` reads it through
`load_config_readonly()`, which the runtime's own docstring says *"follows HERMES_HOME/
profile"* — so a block written into an agent's profile is the block that agent's
delegation runs under. Verified before any of this was written.

A test asserts **every emitted runtime key traces back to a registered, verified control**,
which is the standing regression guard against the `enabled_toolsets` class of bug: a key
that looks applied and is silently ignored.

### Hard at the tool boundary — the per-run tool-call ceiling

`max_tool_calls_per_run` is enforced by the Phase 2 policy plugin. Runaway spend is, in
practice, a runaway tool loop, and that hook fires on every one and fails closed.

Three ordering decisions, each load-bearing:

- **Baseline tools are exempt.** An agent out of budget can still call `kanban_complete`.
  A ceiling that silenced task reporting would produce work that runs and never closes —
  the failure that kept positive tool scoping out of Phase 1.
- **Explicit denials still win.** The ceiling never promotes a denied tool.
- **The ceiling beats approval.** An exhausted agent should not queue work for a human it
  can no longer perform.

Refused calls do not consume budget: a denied call costs nothing. The counter is
per-process, which is per-task for a dispatched worker — hence the name, so it is never
mistaken for a per-day or per-tenant allowance the runtime cannot provide.

---

## What is not enforced, and is labelled so

### `soft_wrapup_after_seconds` — advisory

Compiles to `agent.run_budget_seconds`, which at `conversation_loop.py:119` **injects a
wrap-up message**. The agent may ignore it; nothing terminates. It appears under
`advisory`, never under `controls`, and its own summary says "NOT a limit".

### Token counts and estimated cost — observed only

Read from the runtime's `session_model_usage` (input, output, cache, reasoning tokens,
estimated and actual cost) and reported read-only. Three caveats travel **inside the
payload** rather than in documentation, so a consumer cannot render them as a budget
without contradicting the data it was handed:

1. Not a spending limit — no plugin can veto a model call.
2. Lagging — usage is written by a coalescing background thread.
3. Estimated — reconcile against the provider's billing before charging anyone.

### `daily_token_budget` — removed

It promised a ceiling the runtime cannot enforce. A bundle declaring it now **fails to
load** with an explanation pointing at `max_turns` and `max_tool_calls_per_run`. Failing
loudly is better than carrying a field that reads like a spend cap and is not one.

### `max_task_runtime_seconds`, `max_retries` — recorded only

The dispatcher genuinely kills workers past a per-task runtime cap
(`kanban_db_dispatch.py:418`), but that cap is a *task column* NOVA cannot set while it
does not create tasks. Recorded under the `nova:` key, reported as `recorded_only`.

---

## Surfaces

`GET /platform/v1/budget` returns `controls`, `advisory`, `recorded` and `observed` as
**separate keys** rather than one list with a flag. A flag is easy to drop in a UI; a
missing key is not. Each control names the runtime key that enforces it, so a reviewer can
check the claim rather than trust it.

The dashboard renders enforced controls with an affirmative pill and everything else
neutrally, and states the usage caveat **above** the numbers — it is the thing most likely
to be misread, and a footnote under a total is read last if at all.

---

## Known limitations

- **No spend ceiling exists, by construction.** The nearest honest control is
  `max_tool_calls_per_run`, which bounds tool loops, plus `max_turns`, which bounds
  iterations. Neither is denominated in money.
- **Budgets are per agent, not per tenant.** A tenant-wide rollup needs aggregation across
  profile databases, which nothing in the runtime does. Per-agent is the smaller and more
  honest first step.
- **The tool-call counter is per process.** For a dispatched worker that is per task; for a
  long-lived interactive session it is per session.
- **Reported usage lags.** Nothing here calls `flush_token_counts()`; the figures are a
  recent reading, not a live one.
- **No hard per-task wall clock.** It needs a task column NOVA cannot set, or an upstream
  config-level default that does not exist.

---

## Extension points

**Add a control.** Register it in the adapter's `LIMIT_FACTS` with a `verified_at` call
site you have actually read, then compile it in `build_config`. The test suite rejects an
enforcing claim with no citation, and rejects any emitted runtime key not in the register.

**Add a runtime.** Declare your own register from `limit_facts()`. If your runtime cannot
enforce something, classify it honestly — the platform reads your claim rather than
assuming the Hermes answer.

**Never** move a control into an enforcing class without a call site, and never label a
measurement as a limit. The whole design is arranged to make that a test failure rather
than a judgement call.

---

## Tests

`tests/platform/test_budget.py` — 31 tests, within 259 total. The load-bearing ones assert
the *absence* of misrepresentation: no token or cost control claims enforcement, every
enforcing control cites a call site, unknown keys default to the weakest class, the budget
route never lists usage or the advisory wrap-up among its controls, and every emitted
runtime key traces to a verified entry.
