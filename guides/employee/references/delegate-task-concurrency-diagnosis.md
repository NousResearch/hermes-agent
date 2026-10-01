# delegate_task: diagnosing "my batch was capped"

When a user reports `delegate_task` ran fewer subagents than requested, inspect
the actual tool result, session type and effective configuration before blaming
model self-limiting. Missing log lines alone do not establish that the model
chose the limit.

## Runtime limits and warnings

Batch concurrency is resolved by
`tools.delegate_tool_config._get_max_concurrent_children()` from
`delegation.max_concurrent_children` (or `DELEGATION_MAX_CONCURRENT_CHILDREN`,
then the native default). It has a floor of 1 and no hard ceiling, but separate
session budgets, depth limits and spawn controls still apply.

1. **Per-call hard reject** — `tools/delegate_tool_tasks.py::_normalize_task_list`.
   If `len(tasks) > max_children`, the call returns a `tool_error` with the
   exact message: `"Too many tasks: {N} provided, but
   max_concurrent_children is {M}. ..."` The model sees this as a failed
   tool call and usually retries with fewer tasks.

2. **Per-turn truncator** — `run_agent.py::AIAgent._cap_delegate_task_calls`. If the model emits *multiple separate* `delegate_task`
   tool_calls in a single assistant turn, the count of those calls is
   truncated to `max_children`. Logs as
   `Truncated N excess delegate_task call(s) to enforce
   max_concurrent_children=M limit` at WARNING.

3. **Cost-warning** — same `_get_max_concurrent_children()`. When the
   resolved value is `> 10`, logs once at WARNING:
   `delegation.max_concurrent_children=N: each child consumes API tokens
   independently. High values multiply cost linearly.` This is **just a
   log line** — it does not cap anything. Easy to mis-read as "Hermes is
   refusing my value."

4. **One-shot total budget** — `tools/delegate_tool.py::_oneshot_spawn_budget`
   limits total spawned children across a finite `hermes chat -q` run through
   `delegation.oneshot_max_children` (default 2; 0 disables this budget).
   A rejected request returns `Delegation budget for this one-shot run is exhausted`
   and names the setting. Neither batch-concurrency log pattern detects this.
   Interactive and gateway sessions are not charged against this budget.
   When exhausted, do the remaining work inline; splitting or retrying the batch
   does not restore the budget.

Also inspect returned errors for paused spawning and the configured depth/role
limit. A full background pool can force synchronous execution; that is different
from silently dropping requested children.

## Diagnostic recipe

When a user says "delegate is capped at N":

```bash
# 1. What does the loaded config actually say?
hermes config get delegation.max_concurrent_children
hermes config get delegation.oneshot_max_children
hermes config get delegation.max_spawn_depth

# 2. Did Hermes' truncator or rejector actually fire?
grep -E "Truncated.*delegate_task|Too many tasks" ~/.hermes/logs/agent.log | tail
# Also inspect the actual delegate_task result for one-shot budget, depth or pause errors.

# 3. Confirm the resolver returns what config says (in venv with hermes on path)
python -c "from tools.delegate_tool_config import _get_max_concurrent_children; \
           print(_get_max_concurrent_children())"
```

Compare requested tasks with the actual tool call and returned result. Only
consider model self-limiting after checking runtime errors, the effective
one-shot budget (including children already spawned), depth and concurrency.

## Why models self-limit batches

Reasoning models (Claude Opus/Sonnet, GPT-5, Grok-4) routinely trim a
13- or 15-task batch to a "rounder" number (5, 8, 9, 10) when their
internal reasoning says the coordination cost outweighs parallelism. The
cost-warning log line printed at startup *reinforces* this — the model
reads its own reasoning trace and sees "each child consumes API tokens
independently" and concludes a smaller batch is "more responsible."

A model that chooses a smaller batch may narrate the choice as "the runtime caps at 9" or
"despite the config saying 15, max parallel is 9," which is **not true**
— it's post-hoc rationalisation. Calling this out to the user is fine;
it is a real, well-known reasoning-model failure mode (face-saving
attribution to the system rather than admitting a self-imposed limit).

## Requesting N parallel children

Only when the session budget, role/depth and concurrency allow N, tell the
model explicitly in the prompt:

> "Send all 13 tasks in **one** `delegate_task` call with a `tasks` array
> of 13 items. Do not split into multiple calls. The runtime supports
> this; `delegation.max_concurrent_children` is set to 15."

Submit the complete task list through a direct `delegate_task` call. Python can
help construct the list, but `execute_code` cannot submit delegation: its tool
allowlist excludes `delegate_task`. Inspect the direct call and its result;
do not retry a rejected runtime budget through a different tool.

## Pitfalls / gotchas

- **`max_concurrent_children` is a per-parent cap, not a global cap.**
  Confirmed in `ui-tui/src/components/appChrome.tsx`. Two different
  parents can each spawn `max_children` workers concurrently.
- **`subagent_auto_approve: false` does not cap concurrency.** It only
  controls whether children inherit yolo / approval bypass. Don't mistake
  it for a throttle.
- **The cost-warning is emitted once for configured high concurrency.**
  It does not indicate a cap. Inspect actual tool errors and session budgets.
- **Don't suggest reverting `max_concurrent_children` to fix this.** The
  user set it deliberately; diagnose the real constraint first. Do not alter
  the user's configuration to work around a runtime budget.
