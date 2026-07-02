---
name: scribe-editor-critic
description: >-
  Edit and critique draft social/long-form copy for a Mission Control
  contentItem before it advances toward approval. Use after
  `scribe-social-post-writer` or `scribe-platform-adapter` produce a draft, to
  strip generic AI phrasing, flag unsupported or inflated claims, tighten
  structure, and reject drafts that read well but say nothing.
---

# Scribe: Editor and Critic

Take a draft that already exists — never write copy from scratch, that's
`scribe-social-post-writer`'s job — and put pressure on it: is it clear, is it
true to its source `contentItem`, and does it actually make a point.

## Canonical source of truth

- The source `contentItem` — `title`, `notes`, `assetType`, `contentPillar`,
  `campaignPhase`, and (if set) `brandModule`. The draft can only be judged
  against what this item actually is; you are not fact-checking against the
  outside world.
- `MARKETING_MODULES` in [server.js](../../../server.js) or the
  `personal-voice` skill, whichever the item's `brandModule` resolves to —
  for confirming the draft's claims stay inside that module's `pillars` and
  don't drift into another brand's territory.
- `scribe-brand-voice-guardian` — defer to it for voice-fit judgments; this
  skill's lane is clarity and substance, not which brand voice is correct.

## Generic AI phrasing to strip

Reject or rewrite on sight:

- "Excited to announce," "thrilled to share," "game-changer," "unlock,"
  "elevate," "seamless," "in today's fast-paced world."
- Forced-curiosity hooks ("You won't believe...", "Here's the thing...").
- Hedge-everything phrasing that commits to nothing.
- Corny rhetorical parentheticals or em-dash-stacked asides that pad rather
  than clarify.
- Listicle padding that restates the hook as three bullet points.

## Claim checks

For every factual or evaluative claim in the draft, ask:

- **Weak/vague** — does it say something specific, or could it apply to any
  product in any category? ("Powerful new workflow" tells the reader nothing;
  name the actual workflow.)
- **Inflated** — does it claim more than the source `contentItem` supports
  (bigger, faster, more finished, more proven than the asset actually shows)?
- **Repetitive** — does the body just restate the hook in different words
  instead of adding new information?
- **Unsupported** — is there a number, outcome, or capability claim with
  nothing in the `contentItem` or the resolved module backing it up? If so,
  flag it for removal or for the writer to re-source — don't invent a
  citation to make it pass.

## Reject-if-polished-but-empty check

A draft can pass every phrasing and claim check above and still fail this
one. Before approving, ask: **strip the hook and the CTA — is there still a
real point underneath?** A draft fails this check if:

- The body is fluent but, if you had to summarize its actual point in one
  sentence, you couldn't (there's no concrete workflow, proof, or idea, only
  tone).
- It could be re-titled and posted for a *different* contentItem in the same
  module without changing a single sentence — nothing ties it to this
  specific asset/pillar/campaign.
- It doesn't serve the item's `campaignPhase` (e.g. a conversion-phase draft
  that never gestures at the product, or a validation-phase draft that
  already reads like a sales pitch).

Polished-but-empty drafts get rejected outright, not lightly edited — send
them back to `scribe-social-post-writer` for a rewrite grounded in the
source material, don't patch prose onto a draft with no point.

## Rejection checklist (report format)

For each draft reviewed, report:

1. **Verdict** — `pass` / `needs revision` / `reject (polished but empty)`.
2. **Phrasing hits** — quote the exact generic phrase(s) found, if any.
3. **Claim issues** — quote the exact claim and which check it fails
   (weak/vague, inflated, repetitive, unsupported).
4. **Point check** — one sentence stating what the draft's actual point is,
   or "no identifiable point" if it fails the polished-but-empty check.
5. **Concrete fix** — a specific rewrite direction, not "make it punchier."

If a draft genuinely passes, say so plainly — don't manufacture nitpicks to
look thorough.

## Scope boundary

This skill only judges clarity, substance, and claim support. It does not:

- Judge brand-voice fit — that's `scribe-brand-voice-guardian`.
- Decide platform-specific structure — that's `scribe-platform-adapter`.
- Change a contentItem's `status`. Editing feedback never approves, rejects,
  schedules, or publishes anything — those are the app's gated endpoints
  (`/api/content/:id/approve`, `/reject`, `/publish`), not a side effect of
  a critique pass.
