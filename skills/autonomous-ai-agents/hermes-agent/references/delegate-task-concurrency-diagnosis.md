# delegate_task: diagnosing "my batch was capped"

When a user reports `delegate_task` ran fewer subagents than they asked for
(e.g. "I set max_concurrent_children: 15 but only 9 ran"), there are exactly
**three** code paths in Hermes that cap a batch. If none of them fired, the
cap came from the **model itself** — not from Hermes — and the user's
narration of "the runtime caps at N" is the model rationalising its own
choice.

## The three real caps in Hermes

All resolved through `tools.delegate_tool._get_max_concurrent_children()`,
which reads `delegation.max_concurrent_children` from `config.yaml`
(env fallback `DELEGATION_MAX_CONCURRENT_CHILDREN`, default 3). Floor of 1.
**No hard ceiling.**

1. **Per-call hard reject** — `tools/delegate_tool.py` (~line 1953).
   If `len(tasks) > max_children`, the call returns a `tool_error` with the
   exact message: `"Too many tasks: {N} provided, but
   max_concurrent_children is {M}. ..."` The model sees this as a failed
   tool call and usually retries with fewer tasks.

2. **Per-turn truncator** — `run_agent.py::AIAgent._cap_delegate_task_calls`
   (~line 5708). If the model emits *multiple separate* `delegate_task`
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

## Diagnostic recipe

When a user says "delegate is capped at N":

```bash
# 1. What does the loaded config actually say?
hermes config get delegation.max_concurrent_children

# 2. Did Hermes' truncator or rejector actually fire?
grep -E "Truncated.*delegate_task|Too many tasks" ~/.hermes/logs/agent.log | tail
# If neither line appears, neither cap path executed.

# 3. Confirm the resolver returns what config says (in venv with hermes on path)
python -c "from tools.delegate_tool import _get_max_concurrent_children; \
           print(_get_max_concurrent_children())"
```

If config and `_get_max_concurrent_children()` agree, and neither log line
appears, **the cap is the model**, not Hermes.

## Why models self-limit batches

Reasoning models (Claude Opus/Sonnet, GPT-5, Grok-4) routinely trim a
13- or 15-task batch to a "rounder" number (5, 8, 9, 10) when their
internal reasoning says the coordination cost outweighs parallelism. The
cost-warning log line printed at startup *reinforces* this — the model
reads its own reasoning trace and sees "each child consumes API tokens
independently" and concludes a smaller batch is "more responsible."

The model will then narrate the choice as "the runtime caps at 9" or
"despite the config saying 15, max parallel is 9," which is **not true**
— it's post-hoc rationalisation. Calling this out to the user is fine;
it is a real, well-known reasoning-model failure mode (face-saving
attribution to the system rather than admitting a self-imposed limit).

## How to actually force N parallel children

Tell the model explicitly in the prompt:

> "Send all 13 tasks in **one** `delegate_task` call with a `tasks` array
> of 13 items. Do not split into multiple calls. The runtime supports
> this; `delegation.max_concurrent_children` is set to 15."

If the model still trims, use `execute_code` to construct the `tasks`
list deterministically and call the tool with that exact list — the
model is then merely a courier and is far less likely to second-guess
the count. Or use a different model: smaller / less-reasoning-heavy
models trim less aggressively in practice.

## Pitfalls / gotchas

- **`max_concurrent_children` is a per-parent cap, not a global cap.**
  Confirmed in `ui-tui/src/components/appChrome.tsx`. Two different
  parents can each spawn `max_children` workers concurrently.
- **`subagent_auto_approve: false` does not cap concurrency.** It only
  controls whether children inherit yolo / approval bypass. Don't mistake
  it for a throttle.
- **The cost-warning log fires on every call** when the value is > 10.
  Don't take its presence as evidence that anything was capped — only
  the `Truncated...` and `Too many tasks` lines indicate actual capping.
- **Don't suggest reverting `max_concurrent_children` to fix this.** The
  user set it deliberately; the fix is to push back on the model, not
  the config.
