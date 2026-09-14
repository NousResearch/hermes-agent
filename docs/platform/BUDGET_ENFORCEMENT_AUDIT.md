# Budget Enforcement Audit

**Where the runtime exposes model usage, token counts, iteration limits, delegation and
termination — and which of those NOVA can actually enforce against.**

Written before implementing budget controls, because the same discipline already caught
two expensive mistakes: a config key the runtime silently ignored, and a package name that
would have broken OS detection everywhere.

Every claim below is a file and line read at HEAD `2d84f38`.

---

## The finding that shapes everything

> **NOVA cannot stop a model call.** The runtime's LLM-boundary hooks are observers, not
> gates. Only `pre_tool_call` can veto, and it fires on *tool* calls.

`agent/turn_api_request.py:44` — `_fire_pre_api_request_hook(...) -> None`. The hook is
invoked and its return value is discarded; the surrounding comment describes the payload as
"raw langfuse passthroughs", i.e. tracing. `pre_llm_call` (`agent/turn_context.py:658`)
collects *context to inject into the user message*, not a verdict.

`hermes_cli/plugins_dispatch.py:48` settles it:

```python
# Policy hooks: timeout / still-running must fail closed (block the tool).
_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS: Set[str] = {"pre_tool_call"}
```

One policy hook exists. Everything else on the LLM path is an observer or a transformer.

**Consequence for the design:** a token or cost ceiling cannot be enforced *before* the
call that would breach it. Any NOVA spend budget is necessarily **trailing** — it stops the
agent shortly after a threshold, not before. A control that claims otherwise would be
lying, and the lie would surface on a customer's invoice.

---

## 1. Model usage and token counts

### What is measured

`session_model_usage` (`hermes_state_schema.py:68`), keyed by
`(session_id, model, billing_provider, billing_base_url, billing_mode, task)`:

| Column | |
|---|---|
| `api_call_count` | calls |
| `input_tokens`, `output_tokens` | the basics |
| `cache_read_tokens`, `cache_write_tokens` | cache accounting |
| `reasoning_tokens` | thinking tokens |
| `estimated_cost_usd`, `actual_cost_usd` | money |
| `cost_status`, `cost_source` | how the money was derived |

This is a genuinely good measurement layer — better than most platforms start with.

### How it is written

`hermes_state_usage.py:1` — "the coalescing background token writer". Deltas queue and a
background thread applies them, coalescing where safe. `flush_token_counts(timeout=5.0)`
(line 149) forces a flush.

**So usage is eventually consistent.** A NOVA budget check reads a figure that lags the
true spend by the queue depth unless it flushes first. That lag is a second source of
overshoot on top of the in-flight call.

### Per-agent scoping comes free

Each NOVA agent is a profile, each profile is a runtime home, each home has its own
`state.db`. Usage is therefore already partitioned per agent with no work — but it is
*also* partitioned per agent, so a **tenant-wide** daily budget needs aggregation across
profiles, which nothing in the runtime does.

### What is NOT here

**No token ceiling and no cost ceiling exist anywhere in the runtime.** Cost is measured
and never enforced against. The one nearby thing, `agent/credits_tracker.py`, parses
`x-nous-credits-*` response headers into a notice policy — provider-specific, informational,
and useless against Bedrock or an enterprise's own endpoint.

---

## 2. Iteration limits

### `IterationBudget` — the strongest primitive already present

`agent/iteration_budget.py:25`: a thread-safe counter with `consume()`, `refund()`, `used`
and `remaining`. `agent_init.py:2235` notes **"the budget is shared with subagents"**, and
`execute_code` iterations are refunded so programmatic tool calling does not eat the budget.

The loop bound, `agent/conversation_loop.py:1514`:

```python
while (s.api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) \
        or agent._budget_grace_call:
```

Two independent bounds plus a grace call. `turn_finalizer.py:126` reports exhaustion as a
distinct terminal reason, and delegation surfaces `exit_reason == "max_iterations"` "only
for genuine budget exhaustion" (`delegate_tool.py:305`).

### `agent.max_turns` is real, and NOVA already sets it

`cli.py:2728` — *"max_turns: CLI arg > config > env var > default"*, resolving
`CLI_CONFIG["agent"]["max_turns"]` before `HERMES_MAX_ITERATIONS`.
`gateway/run_startup.py:759` logs the same chain.

**This is the one hard, pre-emptive, exact bound NOVA can already set per agent**, and
Phase 1 writes it. Verified read — unlike `enabled_toolsets`, which was not.

### `agent.run_budget_seconds` is soft

`conversation_loop.py:119` `_maybe_inject_run_budget_wrapup` **injects a "wrap up" message**
when the wall-clock budget elapses. It asks the agent to finish; it does not terminate it.
Useful, but it must never be described to a customer as a limit.

---

## 3. Delegation

Enforced for real in `tools/delegate_tool.py`, reading the `delegation` config block
(`delegate_tool_config.py:29`):

| Key | Effect | Site |
|---|---|---|
| `max_concurrent_children` | Parallel child cap | `delegate_tool.py:469`, `:563` |
| `max_spawn_depth` | Delegation tree depth | `:188` |
| `orchestrator_enabled` | Kill switch for nested orchestration | `:189`, `:505` |
| `max_iterations` | Per-child iteration budget | `:160`, `:235` |
| `child_timeout` | Per-child wall clock | `delegate_tool_config.py:131` |

These are hard caps in code paths NOVA can reach by writing the `delegation:` block into a
profile's `config.yaml` — the same mechanism already proven for `agent` and `kanban`.

**NOVA does not write this block today.** It is the cheapest real win available: the
runtime's own config comments warn that concurrency above 10 multiplies cost linearly.

---

## 4. Termination

The strongest enforcement in the system, and the only place a process is actually killed.

| Control | Where | Nature |
|---|---|---|
| `tasks.max_runtime_seconds` | `kanban_db_dispatch.py:418` | **Kills the worker process tree** |
| `kanban.dispatch_stale_timeout_seconds` | config default 14400 | Reclaims heartbeat-less workers |
| `consecutive_failures` breaker | `kanban_db.py` | Parks a task after repeated failure |
| `agent/deadline.py` | unified | Bounded execution driven by a daemon timer, so a blocked event loop cannot disable it |
| `agent.gateway_timeout` | config | Inactivity timeout for gateway runs |

`max_runtime_seconds` is a **per-task column**, not a config key. NOVA cannot set it today
because NOVA does not create tasks — the Control API is read-only. There is no config-level
default for it either; `dispatch_stale_timeout_seconds` only catches workers whose heartbeat
stopped, so a healthy runaway loop never trips it.

---

## 5. What NOVA can enforce, ranked

| # | Control | Mechanism | Nature | Cost |
|---|---|---|---|---|
| 1 | Iteration cap | `agent.max_turns` in profile config | **Hard, pre-emptive, exact** | Already written |
| 2 | Delegation caps | `delegation:` block in profile config | **Hard, pre-emptive** | A few lines |
| 3 | Tool-call ceiling | `pre_tool_call` plugin | **Hard at the tool boundary** | Reuses Phase 2 |
| 4 | Concurrency cap | `kanban.max_in_progress_per_profile` | Hard | Already written |
| 5 | Token / cost ceiling | Count in the plugin, refuse tools past it | **Trailing, approximate** | Moderate |
| 6 | Wall-clock wrap-up | `agent.run_budget_seconds` | **Soft — a request, not a limit** | One line |
| 7 | Hard task timeout | `tasks.max_runtime_seconds` | Hard kill, but needs task creation | Blocked |

### Why the tool boundary is the right place for spend control

A runaway agent burning money is, in practice, a *tool-call loop* — searching, reading,
retrying. `pre_tool_call` fires on every one of those, can block, and fails closed. It is
the same hook Phase 2 already enforces policy through, so the plumbing exists.

A pure-reasoning loop with no tool calls is bounded by `max_turns` instead. Between the two,
every realistic runaway is covered — one exactly, one approximately.

### The overshoot, stated honestly

A NOVA token budget can exceed its threshold by:

1. **The in-flight call.** The call that crosses the line completes; nothing can veto it.
2. **Writer lag.** Usage is coalesced on a background thread unless flushed.
3. **The grace call.** `_budget_grace_call` permits one extra iteration past exhaustion.

So the honest contract is *"stops within roughly one model call of the ceiling"*, not
*"never exceeds it"*. That number should appear in customer-facing documentation rather than
being discovered.

---

## 6. Recommendation

**Build budget controls in three layers, and describe each one truthfully.**

1. **Hard bounds, compiled to config** — `agent.max_turns` plus the `delegation:` block.
   Pre-emptive and exact, free to add, and they bound the worst case absolutely. Do this
   first; most of the value is here.
2. **A tool-call ceiling in the existing policy plugin** — count calls against a per-task
   limit and refuse past it, reusing the Phase 2 enforcement point and its fail-closed
   behaviour.
3. **A trailing token/cost ceiling** — read `session_model_usage`, flush first, refuse tools
   past a threshold. Label it as trailing everywhere it is surfaced.

**Do not build:** a pre-call token gate (impossible), a spend cap presented as hard
(dishonest), or a second usage accounting system (the runtime's is good; use it).

**Needs a decision:** a *tenant-wide* daily budget requires aggregating usage across
profiles, which nothing in the runtime does. Either NOVA aggregates by reading each
profile's `state.db`, or budgets stay per-agent in this phase. Per-agent is the smaller,
more honest first step.

**One core-patch candidate, not yet requested:** a config-level default for
`max_runtime_seconds` would give a hard per-task kill without NOVA creating tasks. That is
an upstream feature request rather than a patch we should carry.
