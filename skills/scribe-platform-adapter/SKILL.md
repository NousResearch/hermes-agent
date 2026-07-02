---
name: scribe-platform-adapter
description: >-
  Shape a Mission Control contentItem's draft copy into the correct
  structural frame for a specific platform — LinkedIn, Instagram, X, YouTube,
  or Substack. Use after `scribe-social-post-writer` has grounded copy ready,
  to adapt it into that platform's native structure rather than reflowing one
  generic post into every character limit.
---

# Scribe: Platform Adapter

Each platform gets its own structural skeleton below. Fill the skeleton with
copy that's already grounded in the source `contentItem` and the resolved
brand voice — this skill decides *shape*, not *substance* or *voice*.

## Canonical source of truth

- The source `contentItem` and the module resolved from its `brandModule`
  (`MARKETING_MODULES` in [server.js](../../../server.js)) or the
  `personal-voice` skill for Matt's first-person content — same resolution
  rule as `scribe-social-post-writer` and `scribe-brand-voice-guardian`.
- `item.platforms` — only adapt for platforms already on the item (or
  explicitly requested); don't invent a platform target that isn't part of
  the item's distribution intent.

## Platform structures

Each platform below is structurally distinct — do not reuse sections between
them, and do not produce one post and re-cut it into these five shapes.

### LinkedIn
`Insight → Process → Evidence → Takeaway`
Open on the insight (the non-obvious point), walk through the process that
produced it, ground it in evidence from the contentItem (the actual
workflow/asset/result), close with a takeaway the reader can act on. Written
for people reading at a desk, not scrolling.

### Instagram
`Visual context → Narrative → Caption → CTA`
Lead with what the viewer is looking at (the asset itself carries the hook;
the caption supports it). Narrative is short and personal/process-driven.
Caption stays conversational, not a LinkedIn post pasted in. CTA is
proportionate to the item's `campaignPhase`.

### X
`Compact argument → Observation → Thread (optional)`
One sharp claim stated in the first post, no throat-clearing. If the point
needs more than one post, structure it as an explicit thread: each post
extends the argument with a new observation, not a restatement. Never pad a
single-post idea into a thread just to look substantial.

### YouTube
`Title → Hook (first line of description) → Description → Chapters → Thumbnail copy`
Title states the concrete subject (not a vague teaser). Hook is the first
line of the description, written to work whether or not the viewer expands
it. Description gives real context. Chapters only if the underlying asset
has distinct segments — don't fabricate chapter markers for a single-shot
clip. Thumbnail copy is a short, distinct phrase, not a duplicate of the
title.

### Substack
`Narrative → Argument → Evidence → Reflection`
Opens with a narrative frame (a moment, a problem encountered), states the
argument it's building toward, backs it with evidence from the contentItem,
and closes with reflection rather than a hard CTA — Substack readers expect
essay pacing, not a pitch.

## No shared boilerplate

If a sentence would fit unchanged into two different platform structures
above, it's doing generic work, not platform-native work — rewrite it inside
each structure separately. Hashtags, emoji conventions, and line-break
rhythm should also differ per platform; don't apply Instagram's caption
style to a LinkedIn post or vice versa.

## No unsupported publishing claims

This skill produces draft structure and copy only. Never state or imply that
a post has been published, scheduled, or sent — those are facts owned by
`item.status`, `item.distributionPlan`, and `item.deliveryLog`, set only by
the app's gated endpoints (`/api/content/:id/publish` via `publishContentItem`
in server.js, which itself requires `status === 'approved'`). Do not write
"posted," "live," or a URL into draft copy — those fields don't exist yet at
draft time.

## Scope boundary

This skill only adapts structure per platform. It does not:

- Decide the underlying voice or pillar — that's `MARKETING_MODULES` /
  `personal-voice`, enforced by `scribe-brand-voice-guardian`.
- Judge whether the copy is well-written or substantive — that's
  `scribe-editor-critic`.
- Change a contentItem's `status`, `distributionPlan`, or `deliveryLog`.
  Adapting a draft's structure never approves, schedules, or publishes it.
