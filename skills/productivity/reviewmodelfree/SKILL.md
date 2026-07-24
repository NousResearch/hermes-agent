---
name: reviewmodelfree
description: Use when a user types /reviewmodelfree to review currently-free cloud LLM models and update the local free-worker router if a better zero-price replacement exists, using verified-free workers/search only.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [free-models, openrouter, worker-router, cost-control, model-review]
    related_skills: [hermes-agent]
---

# Review Free Cloud Models

## Overview

`/reviewmodelfree` is a cost-control workflow for keeping a Hermes free-worker router fresh. It checks current cloud LLM catalogs, filters for models that are actually free right now, compares them against the active worker set, and updates the router when a better free replacement exists.

The goal is not to chase every new model. The goal is to preserve a safe, useful, zero-cost worker pool for delegation, summaries, coding review, long-context reading, and structured classification.

## When to Use

Use this skill when a user asks for any of these:

- `/reviewmodelfree`
- `review model free`
- “review free cloud models”
- “find a better free model for the worker router”
- “replace old free workers if a better one exists”

Do **not** use this skill for:

- choosing the primary paid assistant model
- benchmarking paid models unless the user explicitly approves cost
- changing API keys or printing secrets
- blindly switching to a model that is only rumored to be free

## Hard Cost Rules

- Prefer verified-free workers/search for the review itself whenever possible.
- Never use a paid model or paid worker without explicit approval.
- Never silently fall back to paid if a free check fails.
- Verify a model is free before using it or writing it into router config.
- For OpenRouter, check both model catalog pricing and endpoint availability.
- If a real free-worker run fails because of auth, quota, rate limit, or API error, report that no free worker was used and continue only with safe main-agent inspection.

## Primary Files

Typical user-local router files:

```text
~/.hermes/hermes-agent/scripts/worker_models.free.json
~/.hermes/hermes-agent/scripts/worker_router.py
~/.hermes/scripts/worker_router
```

Do not print API keys. For OpenRouter, the expected env var is usually in `~/.hermes/.env` as `OPENROUTER_API_KEY`, but only check presence unless the user explicitly asks to edit it.

## Preferred Workflow

### 1. Inspect current router state

```bash
~/.hermes/scripts/worker_router --check-free --no-cache
```

Read current router config:

```text
~/.hermes/hermes-agent/scripts/worker_models.free.json
```

Use token-efficient file tools or `rtk read` when available.

### 2. Discover current free cloud candidates

For OpenRouter, fetch the live catalog:

```text
https://openrouter.ai/api/v1/models
```

For every candidate that looks free, also check endpoints:

```text
https://openrouter.ai/api/v1/models/<model_id>/endpoints
```

Filter to candidates where:

- catalog `pricing.prompt == 0`
- catalog `pricing.completion == 0`
- at least one provider endpoint has `pricing.prompt == 0`
- at least one provider endpoint has `pricing.completion == 0`
- the model is usable for chat/text work

### 3. Compare against worker routes

Evaluate candidates by route, not by hype:

- default / general / reasoning
- Thai / Asian-language explanation
- coding / repo / review / debugging
- long context / logs / large notes
- structured output / classify / fallback
- writing / content / summary

Prefer stability and fit over headline parameter count.

### 4. Replace only when clearly better

A replacement is allowed when most of these are true:

- verified zero-price now
- usable zero-price endpoint exists
- not removed from catalog
- not obviously deprecated
- context window fits the route
- output style fits the route
- safer or more useful than the current worker
- no auth/quota blocker was seen during checks

Remove or de-prioritize models when:

- catalog lookup fails
- pricing is nonzero or unclear
- zero-price endpoints disappear
- the model is deprecated or unstable
- the endpoint exists but consistently fails for the user’s account

### 5. Update router config

Edit only the router config needed for the new worker set, typically:

```text
~/.hermes/hermes-agent/scripts/worker_models.free.json
```

Keep policy strict:

```json
{
  "allow_paid": false,
  "require_zero_pricing_before_run": true
}
```

Do not add paid fallbacks.

### 6. Verify after edits

Run:

```bash
~/.hermes/scripts/worker_router --check-free --no-cache
~/.hermes/scripts/worker_router --dry-run --prompt 'Reply OK'
~/.hermes/scripts/worker_router --task code --dry-run --prompt 'Reply OK'
~/.hermes/scripts/worker_router --task content --dry-run --prompt 'Reply OK'
```

If a real worker was used, include only the worker name on the final line if that is the user’s reporting preference. Remember: `--check-free` and `--dry-run` do **not** count as real worker usage.

## Reporting Format

Keep the final answer short:

- whether a real free worker was used
- what was added / removed / replaced
- new default worker, if changed
- blocked candidates and why
- verification result

Example:

```text
อัปเดตแล้วค่ะ
- default: deepseek
- code: laguna
- removed: old-model เพราะ endpoint หาย
- verify: check-free + dry-run ผ่าน
```

## Common Pitfalls

1. **Counting dry-run as worker usage.** `--check-free` and `--dry-run` only verify routing/catalog; they do not mean a free worker did the review.
2. **Trusting catalog only.** OpenRouter can list a model while no usable free endpoint exists. Always check `/endpoints`.
3. **Silent paid fallback.** If free fails, block and report. Do not switch to paid automatically.
4. **Leaking secrets.** Never print API keys, bearer tokens, auth files, or `.env` contents.
5. **Replacing too aggressively.** A new free model is not automatically better. Compare by route and stability.
6. **Forgetting gateway/session cache.** New or changed skills may require a new session or gateway restart before a messaging platform sees the command.

## Verification Checklist

- [ ] Existing router state inspected
- [ ] Free catalog checked live
- [ ] Endpoint availability checked for OpenRouter candidates
- [ ] Only zero-price candidates considered
- [ ] Router config updated only if a better candidate exists
- [ ] No paid fallback added
- [ ] `worker_router --check-free --no-cache` passed
- [ ] Route dry-runs passed
- [ ] Final answer states whether a real free worker was used
