---
name: scribe-social-post-writer
description: >-
  Draft platform-native social copy (hook/body/CTA) for a Mission Control
  contentItem while it is still a `candidate`. Use when a contentItem needs
  fresh draft copy for its `drafts` field — from a Drive-imported asset, an
  idea, or an AI playground item — before it can go through Editor and Critic,
  Platform Adapter, and brand-voice review on the way to approval.
---

# Scribe: Social Post Writer

Write draft social copy for one `contentItem` at a time, grounded only in
that item's own material and the brand rules that already exist in code.
Never invent a hook, feature, or outcome the source material doesn't support.

## Canonical source of truth

Do not improvise voice, pillars, or claims. Read these fresh for every draft:

- `MARKETING_MODULES` in [server.js](../../../server.js) — resolve the item's
  module via `brandModule` (falling back to `projectId`) and pull `voice`,
  `pillars`, `defaultPlatforms`, and `hardRules` from it.
- The `scribe-brand-voice-guardian` skill — run its enforcement checklist
  against your draft before treating it as done. If Guardian flags voice
  drift, a pillar mismatch, or a hard-rule violation, rewrite, don't argue.
- The `personal-voice` skill — use this instead of `MARKETING_MODULES` when
  the content is Matt's own first-person voice (`brandModule` is `matt`,
  absent, or unmappable to a product module). Never blend the two.
- The source `contentItem` itself — `title`, `notes`, `assetType`,
  `sourceFolder`, `channel`, `contentPillar`, `campaignPhase`, `campaign`.
  These are the only facts you're allowed to write from.

If the item's module can't be resolved from either source, stop and say so —
do not default it into whichever module seems closest.

## Source grounding

Every claim in the draft must trace back to one of:

1. The contentItem's own fields (what the asset actually is/shows/says).
2. The resolved module's `product`, `pillars`, or `hardRules`.

Do not add capabilities, metrics, pricing, release dates, outcomes, or
integrations that aren't already present in those sources. If the item's
`notes` are thin, write a thinner post — don't pad with invented specifics.

## What to produce

For each requested platform, write a distinct entry with three visible parts:

- **Hook** — the first line, written to survive a feed scroll on that specific
  platform (a scroll-stopping visual claim reads differently on LinkedIn than
  on X).
- **Body** — the substantiation: the workflow, the proof point, the process —
  drawn from the item's own material.
- **CTA** — what the reader should do next, proportionate to `campaignPhase`
  (validation/authority phases invite conversation, not a sale; conversion
  phase can point at the product).

Do not just restate the hook and call it a body.

## Hard ban: one post resized across platforms

Never write one post and reflow it into other platforms' character limits.
Each platform's draft must differ in structure, not just length — the hook
mechanic, the pacing, and the CTA phrasing all change per platform. If two
platform drafts in the same batch would read identically with the hashtags
stripped, rewrite one of them. (Structural per-platform shape is owned by the
`scribe-platform-adapter` skill — invoke it for the platform-specific frame,
then fill it with grounded copy from this skill.)

## Output

Write drafts into the shape the pipeline already expects — one entry per
variant with a `copy` map keyed by platform (see `buildDrafts`/`copyForPlatform`
in server.js for the existing shape) — or, if asked for prose review copy,
present each platform's Hook/Body/CTA clearly labeled. Always state which
module/voice source and which `contentPillar` you drafted against.

## Scope boundary

This skill only writes candidate draft copy. It does not:

- Judge whether the draft is well-written or overclaims — that's
  `scribe-editor-critic`.
- Decide platform-specific structural sections beyond hook/body/CTA — that's
  `scribe-platform-adapter`.
- Approve, schedule, publish, or otherwise change `status` on the contentItem.
  Every contentItem this skill touches stays a `candidate`; approval and
  publishing only happen through the app's gated endpoints
  (`/api/content/:id/approve`, `/api/content/:id/publish`), never as a side
  effect of drafting copy.
