---
name: executive-news-digest
description: Produce concise current-events digests only; not for product/vendor comparisons, competitive evaluations, or deep research.
version: 1.0.0
author: Hermes Agent
metadata:
  hermes:
    tags: [research, news, briefing, digest, scheduled-jobs, executive-summary]
---

# Executive News Digest

Use this skill when the task is a current-events digest rather than a deep research memo, comparative product evaluation, software/vendor comparison, or business-opportunity report.

Typical triggers:
- morning digest
- weekday briefing
- send Brian's digest
- concise AI/tech/business updates
- summarize what matters today

Do not use this skill for prompts like:
- is X better than the competition
- compare vendor A vs vendor B
- evaluate the best software/POS for a specific buyer
- coffee-shop / restaurant / vertical SaaS competitive analysis
- what experts or researchers believe will happen over the next 5-10 years

Those should route to a deeper research workflow, not an inline digest.

This skill is for high-signal, time-sensitive briefings delivered in chat/Telegram/email-sized format.

## Output contract

Unless the user asks otherwise:
- keep it to one Telegram screen, usually 4-6 bullets max
- each bullet should say what happened, why it matters, and optionally what to watch next
- if there is nothing worth sending and silence is allowed, return exactly `[SILENT]`

## Source strategy

1. Use current web search for discovery.
2. Prefer reputable current sources and first-party announcements.
3. Cross-check important claims when practical.
4. Add your own framing instead of copying vendor language.
