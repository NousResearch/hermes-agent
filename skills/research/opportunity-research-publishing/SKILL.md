---
name: opportunity-research-publishing
description: Run comparative product/vendor research or business-opportunity analysis, including "is X better than the competition" evaluations, and produce publish-ready business/opportunity reports.
version: 1.0.0
author: Hermes Agent
metadata:
  hermes:
    tags: [research, business-ideas, opportunity-analysis, scheduled-research]
---

# Opportunity Research Publishing

Use this skill when a task asks for Lurkbot-style scheduled research, business idea discovery, opportunity analysis, comparative product/vendor evaluation for a specific ICP, or publish-ready business/opportunity reports.

Do not use it for general non-business research, future-of-AI forecasting, health explainers, science summaries, or broad expert-consensus memos.

## Desired behavior

- Treat prompts like "is X better than the competition," "compare A vs B," or "best software/POS for this buyer" as deep comparative research, not a digest.
- For Brian/Telegram, competitive product research should use the background `research` profile path rather than inline browsing in the active chat.
- Focus on business ideas/opportunities, not generic news summaries or non-business explainers.
- If the user asks what researchers/experts believe about a non-business topic or what is likely to happen in a field over time, that should route to `evidence-report-publishing`, not this skill.

## Output shape

Produce a compact business/opportunity report with:
1. Title
2. One-line thesis
3. ICP
4. Pain evidence
5. Why now
6. MVP
7. Distribution wedge
8. Competition / substitutes
9. Risks
10. Scorecard
11. Sources

## Background spawning model

Brian-facing research should not run inline in the active Telegram/chat session.

Preferred execution shape:
1. Enqueue/spawn a temporary run under the single `research` profile.
2. Use `hermes -p research chat -q ...` (or `hermes chat -q ...` when already inside the research profile).
3. Do not use cron/cronjob for manual Brian/Telegram research unless the user explicitly asks to schedule it.
4. The spawned session gathers sources, synthesizes, self-critiques, publishes, verifies the public page, and returns only the public URL.

See `references/research-background-profile.md` for the preferred prompt contract.

## Scheduled Lurkbot behavior

- run in standard mode without prompting the user
- generate only business-idea research
- keep Telegram output concise
- use the existing `research.briankeefe.dev` renderer/pipeline

## Publishing behavior

- Publish through the established `/home/brian/research-output` pipeline using the existing modern renderer.
- For Brian-facing manual Telegram research, prefer returning a public `https://research.briankeefe.dev/...` URL.
- If publishing fails, return `PUBLISH_FAILED: <brief reason>`.
