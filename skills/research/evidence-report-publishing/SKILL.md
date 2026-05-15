---
name: evidence-report-publishing
description: Run non-business deep research or expert-consensus reports, including future outlooks like what researchers believe will happen over time, and publish them to research.briankeefe.dev using the established renderer.
version: 1.0.0
author: Hermes Agent
metadata:
  hermes:
    tags: [research, evidence, synthesis, forecasting, expert-consensus, publishing, scheduled-research]
---

# Evidence Report Publishing

Use this skill for non-business research that should end as a published report on `research.briankeefe.dev` rather than an inline chat answer.

Typical triggers:
- what most researchers believe will happen in 10 years with AI
- what do experts think about X
- future outlook for Y
- deep research memo on a non-business topic
- summarize the evidence on X and publish it

Do not use this skill for:
- current-events/news digests (`executive-news-digest`)
- business-idea ranking / microsaas opportunity hunts (`opportunity-research-publishing`)
- comparative vendor/product evaluations for a buyer (`opportunity-research-publishing`)

## Desired behavior

- For Brian/Telegram, non-business research should not run inline in the active chat. Spawn the research under the durable `research` profile first, then return a short status.
- Produce an evidence-backed synthesis, not a business memo.
- Be explicit about uncertainty, disagreement, and time horizons.
- Prefer representative expert/primary-source evidence over SEO summaries.
- Publish via the existing `research.briankeefe.dev` pipeline using the same modern renderer as other research pages.

## Output shape

The published report should include:
1. Title
2. Short thesis
3. What researchers/experts broadly believe
4. Main reasons/evidence behind that view
5. Major disagreements or uncertainty bands
6. What could change the outlook
7. Practical implications / watch items
8. Sources

## Background spawning model

Brian-facing manual research should not run inline in the active Telegram/chat session.

Preferred execution shape:
1. Spawn a temporary run under the `research` profile.
2. Use the exact Hermes CLI form `hermes -p research chat -q ...` (or `hermes chat -q ...` when already inside the research profile).
3. Do not use bare `hermes -q ...` without the `chat` subcommand.
4. Do not use cron/cronjob for manual Brian/Telegram research unless the user explicitly asks to schedule it.
5. The spawned session does the research, publishes the page, verifies it, and returns only the public URL or `PUBLISH_FAILED: <brief reason>`.

## Research method

1. Frame the question precisely.
2. Gather varied evidence:
   - expert surveys
   - primary lab/company statements when relevant
   - high-quality syntheses
   - skeptical / cautionary views
   - recent analysis that updates older consensus
3. Distinguish consensus, minority views, and speculation.
4. Do not collapse uncertainty into false certainty.
5. If extraction tooling is weak, use terminal fallbacks to retrieve source text.

## Publishing behavior

- Publish through the same established `/home/brian/research-output` pipeline and renderer used by other research pages.
- Use `research_type: ad_hoc` and `mode: research`.
- Verify the public page returns HTTP 200.
- Verify the content contains the expected thesis/evidence sections.
- Prefer returning a public `https://research.briankeefe.dev/...` URL, not local paths.
- If publishing fails, return exactly `PUBLISH_FAILED: <brief reason>`.
