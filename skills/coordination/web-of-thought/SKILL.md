---
name: web-of-thought
description: When to invoke wot_chat (multi-agent reasoning) and how to design the inner agents — when this beats answering directly.
metadata:
  hermes:
    tags: [debate, perspectives, critique, red-team, multi-agent, synthesize, wot, web-of-thought]
---

# Web-of-Thought (`wot_chat`)

**Opt-in only.** The tool is unavailable unless `HERMES_ENABLE_WOT=1` (or
`true`/`yes`/`on`) is set in the environment. Even then, include the `wot`
toolset deliberately — do not assume every session has it.

Inner agents call **`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_DEFAULT_MODEL`**
(or `OLLAMA_URL`), **not** the parent Hermes session provider. Configure those
env vars before relying on `wot_chat`.

You have access to a multi-agent reasoning tool. Use it when a task genuinely benefits from multiple independent perspectives feeding into one synthesis. Don't use it for everything.

## When to use it

Reach for `wot_chat` when **at least two of** the following apply:

- The question is **inherently multi-perspective**: design tradeoffs, policy/strategy decisions, code review across orthogonal dimensions (correctness × perf × security), risk assessment.
- A single answer is likely **brittle or biased**: the model's first instinct is probably partial, and parallel takes catch what one pass misses.
- The question carries **real downside cost** if you're wrong — recommendation that drives a purchase, a hire, a deploy, a clinical or regulatory decision.
- You'd otherwise be tempted to **simulate "let me think from another angle"** in your own response — let actual independent agents do that instead.

## When NOT to use it

- Simple lookups, single-fact questions, definitions, casual chat.
- Tasks the user already framed precisely enough that one good answer suffices.
- Anything time-sensitive (each `wot_chat` round costs seconds-to-minutes; don't make a user wait for `2+2`).
- Token-budget-constrained sessions — `wot_chat` multiplies token usage by `agents × rounds`.

## Designing the inner agents — the only rule that matters

**Don't pre-assign specialist roles.** Give each agent a **minimal** differentiating system prompt (1–2 sentences) and let the agent develop its own viewpoint from the conversation. Examples:

✅ **Good** (low specification, high autonomy):
```json
{"name": "alpha", "system_prompt": "Argue the case for the change. Be direct, brief."}
{"name": "beta",  "system_prompt": "Argue the case against. Be direct, brief."}
{"name": "gamma", "system_prompt": "Synthesize alpha and beta into a single recommendation."}
```

❌ **Avoid** (over-scripted, role-cargo):
```json
{"name": "Senior_Fullstack_Architect_with_15_Years_Experience", "system_prompt": "You are a senior fullstack architect with 15 years of React experience who has built scalable production dashboards for Fortune 500 companies and you believe React is the most proven choice and you must defend it with examples from your career..."}
```

Long over-scripted prompts make agents perform a role rather than think. The point of WoT is *emergent disagreement and synthesis*, not theatre.

**Naming**: alphanumeric / underscore / dash. Spaces in names are auto-sanitized to underscores.

## Picking the mode

| Mode | When |
|---|---|
| `parallel` (default) | All agents react to the task simultaneously. Best for independent perspectives that meet at the synthesis. |
| `streaming` | Parallel rounds that surface peer *partials after/as each peer stream finishes in the round* — not true mid-token injection into another agent's still-open request (OpenAI-compat cannot do that client-side). Prefer `parallel` unless you specifically want chunked peer visibility. |
| `sequential` | Round-robin; each agent sees the full prior transcript. Use for refinement chains where agent N improves on agent N-1's output. |
| `queue` | Reserved for tag-driven pull. **Current engine does not yet route ordinary messages by `@name` / `interests`** — only `DONE` early-stop is implemented. Prefer `parallel` until routing lands. |

If unsure, use `parallel` with 3 agents. It's the safest default.

## Cost discipline

- Default `max_rounds` is 5. For most tasks 2–3 is plenty. **Pass `max_rounds: 2` explicitly** if you want fast cheap output.
- Default `token_budget: 0` means unbounded. **Set `token_budget: 8000` for hard caps** (cumulative chars × ~4 = tokens) — lets you bound per-call spend.
- 3 agents × 3 rounds covers most cases. 7 agents × 5 rounds is rarely justified.

## Reading the result

`wot_chat` returns JSON with these fields you should look at, in order:

1. **`errors`** — non-empty means at least one inner agent failed. Surface the count to the user; don't silently smooth over.
2. **`agents_done`** — agents that emitted `DONE` on their own line (signaled they had nothing more to add). Empty list is fine.
3. **`rounds_run`** vs **`stop_reason`** — `"all_done"` (clean), `"max_rounds"` (capped), `"budget"` (token cap hit).
4. **`transcript`** — the actual messages. Each entry has `from` (agent name), `content`, and optionally `reasoning` (the agent's chain-of-thought, if `propagate_reasoning != "strip"`).

Synthesize the user's answer **from the transcript**, not from the schema metadata. Cite which agent said what when relevant.

## Example invocation

```json
{
  "agents": [
    {"name": "alpha", "system_prompt": "Argue the case for. Brief."},
    {"name": "beta",  "system_prompt": "Argue the case against. Brief."},
    {"name": "gamma", "system_prompt": "Synthesize alpha and beta into one recommendation."}
  ],
  "task": "Should the team migrate from REST to gRPC for the internal services this quarter?",
  "mode": "parallel",
  "max_rounds": 3,
  "token_budget": 12000
}
```

The tool returns a JSON transcript. Read it, then write the user a clean synthesis that names the tradeoffs and gives a clear recommendation with caveats.

## Anti-patterns to avoid

1. **Calling `wot_chat` for everything.** If you can answer in one paragraph confidently, just answer.
2. **Specifying `model` per agent.** The engine ignores caller-supplied `model` and uses the configured backend default — don't waste tokens guessing model names.
3. **Long ceremonial system prompts.** See "Designing the inner agents" above.
4. **Ignoring the `errors` field.** A 3-agent run with 9 errors is a failed run; surface that to the user.
5. **Using WoT to look authoritative on something that's just a guess.** Multi-agent doesn't manufacture truth; it surfaces tradeoffs.
