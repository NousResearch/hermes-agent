---
name: subagent-model-routing
description: Model selection guide for delegate_task and cron jobs. Tiers are LLM decision guides (not API constraints).
version: 3
tags: [delegation, cost, models, routing, auto-router]
---

# Subagent Model Routing

> **Last updated:** 100526 | **Sources:** `~/.hermes/scripts/refresh_openrouter_models.py → WHITELISTS` + OpenRouter Pareto Code Router / Artificial Analysis coding percentiles
> If today is >10 days past the date above, verify pricing before high-stakes decisions.

Load this skill every time you use `delegate_task` without an explicit model.

## Tiers

Budget/standard/premium are mutually exclusive. Coding overlaps all.

**PREMIUM** — synthesis, judgment, reviews
`x-ai/grok-4.20` | `x-ai/grok-4.20-multi-agent` | `x-ai/grok-4.3`
`anthropic/claude-opus-4.7` | `anthropic/claude-sonnet-4.6`
`google/gemini-2.5-pro` | `google/gemini-3.1-pro-preview` | `openai/gpt-5.5`

**STANDARD** — ops, analysis, general delegation
`anthropic/claude-haiku-4.5` | `openai/gpt-5.4-mini` | `deepseek/deepseek-v4-pro`
`moonshotai/kimi-k2.6` | `minimax/minimax-m2.7` | `z-ai/glm-5.1`

**BUDGET** — cron jobs, extraction, parsing (account default)
`google/gemini-2.5-flash` | `google/gemini-2.5-flash-lite` | `x-ai/grok-4.1-fast`
`openai/gpt-5-nano` | `deepseek/deepseek-v4-flash`

**CODING** — anything touching code (overlaps permitted)
Use `openrouter/pareto-code` by default for coding delegation. It routes across OpenRouter's Pareto-efficient coding frontier; set the global quality floor with `openrouter.min_coding_score`.

*HIGH tier* (`min_coding_score >= 0.66`) — strongest coders, highest cost:
`openai/gpt-5.5` | `google/gemini-3.1-pro-preview` | `anthropic/claude-opus-4.7` | `deepseek/deepseek-v4-pro`

*MEDIUM tier* (`0.33 <= score < 0.66`) — solid coders, mid cost:
`openai/gpt-5.4-mini` | `anthropic/claude-sonnet-4.6` | `moonshotai/kimi-k2.6` | `x-ai/grok-4.3`

*LOW tier* (`score < 0.33`) — lighter coders, lowest cost:
`xiaomi/mimo-v2.5-pro` | `qwen/qwen3.6-max-preview` | `z-ai/glm-5.1` | `deepseek/deepseek-v4-flash` | `anthropic/claude-haiku-4.5`

> **Slug verified 100526:** MiMo-V2.5-Pro is Xiaomi, NOT Minimax. Display name in OR UI was misleading. Good: `xiaomi/mimo-v2.5-pro`; bad: `minimax/mimo-v2.5-pro`. Always verify OR slugs via live API when vendor name isn't obvious from the model card.

## Pareto Router (`openrouter/pareto-code`)

OpenRouter's curated coding router selects a model from the Pareto-efficient quality/cost frontier. No router fee; you pay only for the underlying model.

**Status (100526): ✅ NATIVELY SUPPORTED** — Hermes wires `min_coding_score` automatically via the OpenRouter plugin when model is `openrouter/pareto-code`. PR #22838 shipped in the +80 commit sync.

**Active config (100526):**
```yaml
delegation:
  model: openrouter/pareto-code   # default for all delegate_task calls

openrouter:
  min_coding_score: 0.33          # Medium tier floor — Sonnet 4.6 / Grok 4.3 / Kimi K2.6 range
```

**Score is a floor, not a target** — OpenRouter picks the cheapest available coding model that meets or exceeds the quality bar.
- `0.66+` → HIGH only (Opus 4.7, GPT-5.5, Gemini 3.1 Pro)
- `0.33–0.65` → MEDIUM or better (Sonnet 4.6, Grok 4.3, Kimi K2.6)
- `< 0.33` → any tier

**Per-task score override is NOT supported.** One global `min_coding_score` applies to all delegation batches. Use explicit model pinning (`model: "anthropic/claude-opus-4.7"`) for tasks needing a different tier in the same batch.

**Aux tasks are independent** — `openrouter.min_coding_score` does NOT propagate to auxiliary tasks (compression, vision, session_search, etc.). Wire separately via `extra_body` if needed:
```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openrouter/pareto-code
    extra_body:
      plugins:
        - id: pareto-router
          min_coding_score: 0.5
```

**Usage:**
```python
# delegate_task — inherits min_coding_score from config automatically
delegate_task(goal="...", model="openrouter/pareto-code", provider="openrouter")
```

Response `model` shows which concrete model was used. `min_coding_score` is silently dropped for non-`openrouter/pareto-code` models, so it is safe to leave configured globally.

## Routing Matrix

| Task | Model | Tier |
|------|-------|------|
| Extraction / parsing / summarization | gemini-2.5-flash | budget |
| Web research | gemini-2.5-flash | budget |
| Simple file ops / renames | gpt-5-nano | budget |
| New script / feature | openrouter/pareto-code (MEDIUM) | coding |
| Refactor / complex integration | explicit premium pin, or temporarily raise Pareto floor to HIGH | coding |
| Code review | explicit premium coding pin + grok-4.3 second opinion | coding + premium |
| Business ops / analysis | claude-haiku-4.5 | standard |
| Reasoning / math | deepseek-v4-pro or gpt-5.4-mini | standard |
| Intelligence synthesis | grok-4.20 | premium |
| Architecture judgment | gpt-5.5 or grok-4.3 | premium |
| Legal research (Mexican/labor law) | grok-4.3 | premium ← confirmed 050526: LFT PTU, STPS, SAT |
| Domain law research (any jurisdiction) | grok-4.3 | premium — Grok outperforms on law questions |

## Per-Call Model Pinning (030526 — verified working)

Per-task model pinning works on the feat branch (PR #12794). Pass `model` at the task level:

```python
delegate_task(tasks=[
    {"goal": "...", "model": "anthropic/claude-haiku-4.5", "toolsets": [...]},
    {"goal": "...", "model": "anthropic/claude-opus-4.7",  "toolsets": [...]},
])
```

**Verified live (030526):** Three tasks with distinct pins ran on their correct models. The `model_observability` plugin confirmed `match: true` for each pinned task via JSONL evidence.

**Note:** This requires the gateway to run from `hermes-agent-feat/` (feat branch). Upstream main does NOT have `model` in the `delegate_task` signature — pins are silently discarded there. Until PR #12794 merges, `delegation.model` in `config.yaml` is the only routing lever on upstream main.

## Hard Rules

0. **NEVER second-guess a model slug the user provides.** If Jordan says "use grok-4.3" or any other slug, do NOT say the model doesn't exist. Load this skill and `openrouter-expert`, verify the whitelist, then attempt execution. He almost certainly knows it exists. Only flag if the API call actually fails. (Canonical failure 050526: told Jordan grok-4.3 didn't exist — it was in the premium tier the entire time.)
1. **Grok (ALL variants):** reasoning, intelligence synthesis, and **legal/domain research**. Never code, never general tasks. Confirmed strong: LFT labor law, STPS/SAT compliance, jurisdiction-specific statutes.
2. **Coding tasks:** coding-tier models ONLY. General models invent abstractions instead of following integration requirements.
3. **Adversarial code review:** use a different coding model (different training DNA), not a reasoning model.
4. **Mixed providers in one batch:** always `provider="openrouter"`. Native providers only serve their own models.
5. **Never self-select Premium autonomously** without exhausting cheaper options. Cost difference is 10–40×.
6. **Escalating past Standard when user is present:** offer the tradeoff before acting.
7. **No `openrouter/auto` for cron jobs** where tool execution is mandatory — it can route to a model that fabricates output instead of calling tools. Pin a specific model.
8. **Rule 0 is not aspirational — it has fired twice.** If you are reading this and Jordan just named a model slug you don't recognize, stop reading, open a terminal, run `curl -s 'https://openrouter.ai/api/v1/models' | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]" | grep <slug>`, and confirm before saying anything.

## When Not to Delegate

- Task is 3 tool calls or fewer
- Task needs 3+ existing codebase functions/patterns already in your context — write it directly
- Two prior delegations already failed on the same task
- Task requires your session state or ongoing conversation context

## Ephemeral Subagents vs. Persistent Profiles — Do NOT Conflate

`delegate_task` spawns **ephemeral anonymous subagents** — they live for one task, have no memory, no SOUL.md, no accumulated context. They are cheap throwaway workers.

**Persistent profiles** are a fundamentally different concept: a named profile with its own `config.yaml`, `SOUL.md`, memory store, cron schedule, and identity that accumulates context across sessions.

**Common mistake (caught 040526):** Connecting `delegate_task`'s per-task model routing to the design of persistent multi-agent profiles (e.g., a Librarian profile or Ops profile). PR #12794 adds model overrides to ephemeral subagents only. A persistent profile running in the gateway as its own named agent is a completely separate architectural decision that has nothing to do with delegate_task's model parameter.

**Rule:** When discussing multi-agent architecture — which profiles to create, what each specialist accumulates over time — do not reference `delegate_task` or PR #12794 as the mechanism. Those are for ephemeral fan-out only.

## Syntax

```python
# Single task
delegate_task(goal="...", model="anthropic/claude-haiku-4.5", provider="openrouter")

# Batch — mixed models
delegate_task(
    provider="openrouter",
    tasks=[
        {"goal": "implement feature", "model": "openai/gpt-5.1-codex-mini"},
        {"goal": "review output",     "model": "x-ai/grok-4.20"},
    ]
)

# Cron job model pin
cronjob(action="create", model={"model": "google/gemini-2.5-flash", "provider": "openrouter"}, ...)
```

## Provider Rules

- Omit `provider` only when all tasks share the same provider as the parent session
- `provider="openrouter"` for mixed-provider batches or any OpenRouter model
- `provider="anthropic"` for Anthropic-only batches (better latency, prompt caching)
- Per-task `provider` override does NOT exist — top-level `provider` applies to all tasks in a batch

## Escalation Ladder

```
1. Handle yourself       free — no context transfer cost
2. Budget                default starting point
3. Standard              when budget underperforms or task warrants reliability
4. Coding                mandatory for any code writing/modification
5. Premium               review, synthesis, architecture — after cheaper options exhausted
```

## Observability

Every `delegate_task` result includes an `observability` field:
```json
{
  "models_used": {"google/gemini-2.5-flash": 8},
  "models_requested": {"openrouter/auto": 8},
  "api_calls": 8,
  "tokens": {"input": 24312, "output": 1847},
  "auto_router_resolutions": {"openrouter/auto": {"google/gemini-2.5-flash": 8}},
  "override_mismatches": []
}
```

If `override_mismatches` is non-empty → alert Jordan immediately. A model override was silently dropped and the task ran on the wrong model.

## Aux Slot Routing (v0.13.0+)

These are gateway-internal slots — not `delegate_task` params — but they follow the same routing logic and slug conventions.

| Slot | Current model | Rationale |
|---|---|---|
| `triage_specifier` | `anthropic/claude-haiku-4.5` | Kanban task dispatch; judgment-lite routing decisions |
| `curator` | `anthropic/claude-haiku-4.5` | Skill archive/prune; monitor for over-aggressive pruning — bump to `sonnet-4.6` if needed |
| `vision` | `google/gemini-2.5-flash` | Powers `video_analyze` + screenshot analysis |

**`triage_specifier` watch note (070526):** This is the model that reads an incoming Kanban task and decides which worker profile picks it up, what toolsets it gets, and how to scope it. Haiku is appropriate for structured routing decisions. If the first real Kanban board shows mis-dispatch (wrong worker, wrong toolset, wrong scope), bump to `anthropic/claude-sonnet-4.6` immediately — don't wait for a second failure.

## Known Limitations

- No per-task `provider` in batches. Workaround: use `provider="openrouter"` for mixed batches — it serves all providers.
- OpenRouter uses dot notation for Anthropic slugs: `claude-haiku-4.5` not `claude-haiku-4-5`. Hyphens will silently fail or fall back to the main model. Always verify a new slug via live API before writing it into config, code, or cron — canonical verification command:
  ```bash
  curl -s 'https://openrouter.ai/api/v1/models' | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]" | grep <slug>
  ```
  Canonical failure (080526): attempted `anthropic/claude-haiku-4-5` (hyphen) — corrected to `anthropic/claude-haiku-4.5` (dot) after live check.

## Whitelist Maintenance

**Canonical source:** `~/.hermes/scripts/refresh_openrouter_models.py → WHITELISTS`

The refresh cron runs every Sunday 11 AM, delivers a read-only digest, and requires operator approval before any changes are applied. When approving changes, patch **both** the script's `WHITELISTS` dict and this skill's tier tables atomically in the same session. Also copy the updated script to the feat branch — see the MAINTENANCE section inside the script for the exact commands.

**Tier exclusivity is enforced by the script itself.** The `WHITELISTS` dict has a validator that raises at import time if any model appears in more than one of budget/standard/premium. Coding is explicitly exempt. If you add a model, ensure it belongs to exactly one non-coding tier.

**Price delta tracking uses a local cache.** The script writes `~/.hermes/caches/openrouter_prices_last.json` after each run and diffs against it the following week. Section 2b of the report shows changes ≥20% with direction arrows. No external persistence needed — the cache is the baseline.

**OpenRouter response caching (030526):** Default TTL is 300s and is on by default — no config needed. Decision: leave at 300s. Dynamic cron prompts (briefings, intel) never hit the cache because they include fresh dates/content; 300s captures legitimate repeat aux calls within a session. Longer TTLs offer marginal savings but more exposure to the beta feature's edge cases. Do not increase unless cost audits show a clear win.

**Do not embed a `CRON_PROMPT_TEMPLATE` in the refresh script.** The live prompt lives in `jobs.json` (job `6d0271a4d5cb`). A template in the script is a diverging copy with no enforcement mechanism — it goes stale silently and misleads future agents reading the script. The script's job is data collection; the prompt belongs in the scheduler.

> See `references/routing-rationale.md` for case studies, QA history, whitelist change log, and slug verification protocol.
