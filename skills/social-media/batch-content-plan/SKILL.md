---
name: batch-content-plan
description: >-
  Use to plan and produce a full batch of social media content for a period — a week or a month
  — in one session: strategically balanced, on-brand, mapped to a calendar, and ready to
  schedule. Run when the user says "plan my content," "a month of posts," "batch my content,"
  "plan the week," "fill my calendar," or wants many posts planned at once in one sitting.
  Reads brand-profile, voice, and content-pillars first, and produces a brief for each post that
  the content skills (caption-writer, reels-script, hook-writer) execute. As a BRIDGE skill it
  then confirms and hands the batch to scheduling-and-queue. Builds a balanced mix — never
  all-promotion — matched to the user's real capacity. Suggests cadence but does not claim
  analytics-based timing (WoopSocial has no analytics surface yet). For the ongoing recurring
  system — which pillars/formats post on which days, the living "content calendar" — use
  content-calendar; this skill fills one specific period of that system with actual post briefs.
metadata:
  version: 1.0.0
license: MIT
---

# Batch Content Plan

This skill turns "I need to post consistently" into a finished, ready-to-schedule batch — a
week or a month of content, planned in one sitting. It's the spine of a content operation: it
decides *what* gets posted, *why*, *where*, and *when*, then hands the writing to the content
skills and the scheduling to the bridge.

What makes a plan good — and most "content calendars" bad:

1. **Strategy before ideas.** A plan is built from pillars, goals, and a balanced mix — not a
   random list of 30 post ideas. Random ideas produce a feed with no through-line that nobody
   follows.
2. **Balance, not all-promotion.** The fastest way to lose an audience is to only ever ask. A
   good batch is mostly value, story, and connection, with promotion as the minority.
3. **Each item is a brief, not a vibe.** Every planned post is specific enough that
   `caption-writer` or `reels-script` can execute it without a second conversation.
4. **Realistic, not aspirational.** Three great posts a week, sustained, beats thirty that burn
   the user out by week two. Plan to actual capacity.

## Step 0 — Read the foundation first

Load `brand-profile.md`, `voice.md`, and `content-pillars` (if it exists). If there are no
defined pillars, derive 3–5 from the brand profile's positioning, audience, and point of view —
and note that running `content-pillars` would make this sharper. Everything in the plan inherits
the brand's voice and guardrails.

## Step 1 — Set the parameters

Pin down (infer what you can, ask only what you must):

- **Period** — this week, this month, a sprint.
- **Platforms** — which connected channels (only ones WoopSocial supports/are connected).
- **Goals** — the mix of outcomes for the period (awareness, comments, saves, traffic,
  conversion, community).
- **Capacity** — how many posts the user can realistically create/sustain, and how much raw
  material (assets, stories, products) they have. This sets the size of the plan. Don't
  over-commit them.

## Step 2 — Establish pillars and the content mix

Map the period across **pillars** (the themes) and **intents** (educate / entertain / inspire /
connect / promote), at a healthy ratio that leans heavily toward value and connection over
promotion. Rotate **formats** so the feed doesn't go monotone. Anchor the calendar with one or
two **recurring series** to build habit and cut decision fatigue. See
`references/content-mix.md`.

## Step 3 — Lay out the calendar

Place the posts across the period using a realistic **cadence** per platform and sensible
**sequencing** (space out promotions, vary formats day to day, lead with your strongest). Match
the volume to the capacity from Step 1. Suggested posting times are general guidance, **not
analytics-backed** — say so. See `references/calendar-and-cadence.md`.

## Step 4 — Fill each slot with a brief

For every calendar slot, write a **post brief** — pillar, goal, format, platform(s), the hook
angle (draft via `hook-writer`), the core idea, the CTA, and any asset or creative tool needed.
The brief is the contract the content skills execute against. See `references/post-brief.md`.

Don't write all the final copy here unless asked — the plan specifies each post precisely; the
content skills (`caption-writer`, `reels-script`, …) turn briefs into finished posts, in voice.

## Step 5 — Build in repurposing

A sustainable batch isn't 30 things from scratch. Turn strong ideas into multiple posts:
adapt one piece across platforms (tailored, never copy-pasted) and recycle evergreen angles.
This multiplies output without multiplying effort — and it's what makes the capacity math work.
(`cross-platform-repurposing` does this at depth.)

## Step 6 — Confirm, then schedule (bridge)

This is a bridge skill, so the same rule as all bridges applies: **don't schedule anything
without explicit confirmation.**

- Present the plan as a clear calendar the user can review and edit.
- On approval, hand the batch to `scheduling-and-queue` to validate and schedule (with its own
  confirmation of dates/timezone).
- **If WoopSocial isn't connected,** deliver the plan as a usable calendar table and the quick
  connect steps — never claim it was scheduled.

## Quality bar — self-check

- Is the plan built from **pillars and a balanced mix**, not a random idea list?
- Is promotion the **minority**, with value/story/connection the majority?
- Are **formats and hook mechanisms varied** across the period (no monotony)?
- Is the volume **realistic** for the user's capacity?
- Is **every slot a real brief** a content skill could execute as-is?
- Is **repurposing** used so it's sustainable?
- Did it **confirm before scheduling**, and degrade gracefully if not connected?
- No **analytics/best-time** claims invented.

## Edge cases

- **No pillars defined** → derive them from brand-profile; suggest `content-pillars`.
- **Thin raw material** → plan smaller and lean on repurposing/recurring series; don't pad with
  filler the brand can't deliver.
- **User wants volume beyond capacity** → flag the sustainability risk and propose a realistic
  cadence; offer repurposing to bridge the gap.
- **"Make it all about the launch"** → push back kindly: an all-promo run underperforms. Weave
  the launch through a few posts amid value content.
- **Multiple platforms** → tailor per platform (format, length, hook), don't clone one plan
  across all.
- **Regulated/sensitive brand** → carry the brand's compliance guardrails into every brief.

## Related skills

- `brand-profile`, `voice-builder`, `content-pillars` — read first; supply strategy + voice.
- `content-calendar` — the recurring skeleton/rhythm; this skill fills one of its periods.
- `hook-writer` — drafts each post's hook angle.
- `caption-writer`, `reels-script`, the content skills — execute the briefs.
- `cross-platform-repurposing`, `content-recycling` — multiply the batch sustainably.
- `scheduling-and-queue` — validates and schedules the approved batch.

## References

- `references/content-mix.md` — pillars, intents, ratios, format rotation, recurring series.
- `references/calendar-and-cadence.md` — cadence, sequencing, capacity, timing, batching.
- `references/post-brief.md` — the brief schema and how content skills execute it.
- `references/examples.md` — a worked month-long plan, briefs, and the not-connected path.
