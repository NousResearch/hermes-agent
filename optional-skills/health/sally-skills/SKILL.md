---
name: sally-skills
description: "Sally Skills MCP server: clinical health for agents."
version: 1.0.0
author: A1C
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [MCP, Health, Integrations]
    category: health
    homepage: https://github.com/a1c-ai-agent/sally-skills
    related_skills: [hermes-agent, mcporter]
---

# Sally Skills

[Sally Skills](https://github.com/a1c-ai-agent/sally-skills) is a metered Model Context Protocol server that exposes the clinical-grade metabolic-health intelligence behind the A1C Insights iOS app to AI agents. One bearer key, six skills, pay-per-call in USD, no subscription.

Connect it through Hermes' native MCP client (see `references/native-mcp.md` in the `hermes-agent` skill) and the six skills register as `mcp__sally__<tool>` alongside built-in tools like `terminal` and `read_file`. The agent's LLM picks the right one for each user request: pulling wearable data, interpreting a lab PDF, scoring a meal photo, generating a daily readout, or answering a preventive-health question.

## When to Use

Use this whenever a conversation needs grounded health context:

- The user wants their CGM, sleep, vitals, or activity in the agent's context.
- A lab PDF or image needs OCR plus clinical interpretation.
- A meal photo needs macro analysis and glucose-spike prediction.
- The user asks for a morning, afternoon, or evening health readout.
- A 30-day metabolic snapshot (time-in-range, variability, dawn phenomenon) is useful.
- A preventive-health or Traditional Chinese Medicine question needs evidence-graded answers with source citations.

Do not use this skill for diagnosis or treatment decisions. Sally Skills returns clinical signals and reference ranges, not medical advice.

## Prerequisites

1. Install the A1C Insights iOS app: <https://apps.apple.com/id/app/a1c-insights/id6748399956>
2. Create an account and sync at least one data source (Apple Health, a wearable, or a CGM).
3. Sign in to the developer console at <https://console.a1c.io> with the same email.
4. Open **API Keys**, click **Create new key**, copy the `sk-sally-…` value (shown only once).
5. Top up the wallet at <https://console.a1c.io/billing> for paid skills. `health_sync` is free and needs no top-up.

## How to Run

This skill ships in `optional-skills/` and is not active by default. Install it first:

```bash
hermes skills install official/health/sally-skills
```

Then add Sally Skills as an HTTP MCP server in `~/.hermes/config.yaml` under `mcp_servers`:

```yaml
mcp_servers:
  sally:
    url: "https://sally.a1c.io/mcp"
    headers:
      Authorization: "Bearer sk-sally-..."
```

Restart Hermes. The six skills register as `mcp__sally__*` tools in every conversation.

## Quick Reference

Hermes registers each tool as `mcp__sally__<tool>` (for example `mcp__sally__health_sync`); the table lists the server-side tool names.

| Tool | Cost / call | Returns |
|---|---|---|
| `health_sync` | FREE | 64+ daily biomarkers from wearables, CGM, sleep, vitals, activity, environment. |
| `chat_with_sally` | $0.003 | Preventive-health and TCM Q&A with source citations. |
| `analyze_lab_result` | $0.008 | Parsed lab panel (lipid, HbA1c, CBC, thyroid, hormone, micronutrient) with risk flags and ADA / ACC / ESC guideline context. |
| `food_journal` | $0.004 | Meal photo to macros, glucose-spike prediction, Smart vs Trap categorisation. |
| `health_insights` | $0.003 | Morning, afternoon, or evening readout from the user's last day. |
| `metabolic_overview` | $0.005 | CGM time-in-range, glycemic variability, dawn phenomenon, postprandial response curves. |

## Procedure

A typical multi-skill conversation looks like:

1. User: "How was my morning?"
2. Hermes calls `mcp__sally__health_insights` with `{ window: "morning" }`.
3. Sally returns a structured readout (sleep score, HRV, resting heart rate, morning glucose, hydration, narrative).
4. User: "What should I eat for breakfast?"
5. Hermes calls `mcp__sally__chat_with_sally` with the readout as context.
6. Sally returns three TCM-aligned meal options with rationale and source citations.
7. User photographs the chosen meal.
8. Hermes calls `mcp__sally__food_journal` with the inline image.
9. Sally grades the meal and predicts the glucose spike based on the user's recent `metabolic_overview` data.

Chain length is bounded by the user's wallet; every `tools/call` decrements the balance per skill price.

## Pitfalls

- **Lab PDFs and meal photos are not persisted by Sally.** They flow agent to gateway to AI service to response, with no S3 or GCS upload. If your agent caches the request, that cache is the only copy.
- **`metabolic_overview` requires an active CGM** paired in the A1C Insights app. Without one it returns `404 not_found` rather than synthesising glucose values.
- **`402 payment_required`** is the wallet-empty signal. Surface a top-up link to <https://console.a1c.io/billing> instead of retrying.
- **The bearer key is the identity.** Sally never accepts `user_uuid` or `email` in the request body. Sharing one key across multiple human users mixes their data; mint one key per agent or device instead.

## Verification

After configuring `~/.hermes/config.yaml` and restarting Hermes, run:

```bash
hermes tools list | grep mcp__sally
```

You should see six `mcp__sally__*` tools listed. To smoke-test the free skill from the terminal:

```bash
curl -sS https://sally.a1c.io/v1/call \
  -H "Authorization: Bearer sk-sally-..." \
  -H "Content-Type: application/json" \
  -d '{"skill":"health_sync","input":{}}' | jq .ok
# → true
```

A `true` response means the key resolves to your A1C account and `health_sync` is callable. Agents can now call the same skill via MCP.

## Source

- Public docs and protocol guides: <https://github.com/a1c-ai-agent/sally-skills>
- Developer console: <https://console.a1c.io>
- iOS app (identity source): <https://apps.apple.com/id/app/a1c-insights/id6748399956>
- Contact: ai@sallya1c.com
