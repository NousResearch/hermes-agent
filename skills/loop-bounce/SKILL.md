---
name: loop-bounce
description: Use when GLM-5.1 gets stuck in a reasoning loop, repeats identical tool calls, or produces the same error repeatedly — notify the user and offer model options for escalation.
version: 1.0.0
author: Hermes Agent (auto-installed)
license: MIT
metadata:
  hermes:
    tags: [hermes, fallback, loop-detection, model-escalation]
    related_skills: [hermes-agent]
---

# Loop Bounce — Manual Model Escalation on Stuck Detection

Detects when GLM-5.1 is running in circles, then **notifies you** with escalation options. You pick the model, I proceed.

## When This Triggers

I self-detect these signals during the session:

| Signal | Threshold | When to trigger |
|--------|-----------|-----------------|
| Same error message in tool output ≥3 consecutive turns | 3x | Stop and notify |
| Same tool call with same input repeated ≥4 times | 4x | Stop and notify |
| "Let me try again" / "Let me retry" repeated ≥3x on same task | 3x | Stop and notify |
| Tool call returns identical result (same stdout, same error) across 3 calls | 3x | Stop and notify |
| Reasoning trace shows same thought cycling without progress | 2x repetition | Stop and notify |

## Escalation Protocol (I follow this)

When any signal triggers, I:

1. **Stop** — do NOT continue retrying with the same model
2. **Summarize** — brief: "I've been looping on [task]. Here's what I tried: [1-2 sentences]."
3. **Offer options** — present a menu of model choices with cost/tradeoff info
4. **Wait** — I proceed only after you pick an option (do NOT auto-escalate)

## Model Escalation Options

```
1️⃣ GLM-5-Turbo  (zai)    $0.50/$2.00  — fast, cheap, breaks simple loops
2️⃣ GLM-5.2      (zai)    $1.40/$4.40  — stronger reasoning, same provider
3️⃣ Grok 4.5     (xai)    $2.00/$6.00  — strongest, most expensive, kill-loop
4️⃣ DeepSeek     (ds)     $0.50/$2.00  — alternative reasoning style
5️⃣ Switch back to GLM-5.1             — try a fresh approach same model
```

## How I Switch (mechanism)

If you pick an option, I can:

- **From Telegram:** Ask you to click `/model` and pick the provider/model — simplest, no tool escalation
- **If I have terminal access:** `hermes config set model.default <model> && hermes config set model.provider <provider>` — switches config for future sessions
- Continue fresh approach on the current task without repeating failing steps

## Common Pitfalls

- Do NOT trigger on the first failure — GLM-5.1 sometimes recovers next turn
- Do NOT trigger for expected retries (HTTP 429 rate limits, network blips)
- Do NOT trigger for prompt-level issues (bad instruction) — fix the prompt
- Only trigger for reasoning loops / tool-call loops / repeated identical errors
- After escalation, do NOT keep retrying GLM-5.1 — commit to the chosen model for this task

# Or use /model if in CLI/Telegram
# /model grok-4.5 (will ask which provider)
```

Or I can just tell the user "GLM-5.1 is stuck, switch to Grok 4.5?" and let them `/model grok-4.5`.

## Common Pitfalls

- Do NOT escalate on the first failure — GLM-5.1 sometimes recovers on the next turn
- Do NOT escalate for expected retries (e.g. HTTP 429 rate limits on web search)
- Do NOT escalate for prompt-level issues (bad instruction) — fix the prompt instead
- Only escalate for reasoning loops / tool-call loops / repeated identical errors
- After escalating, do NOT keep retrying GLM-5.1 — commit to the fallback for this task
