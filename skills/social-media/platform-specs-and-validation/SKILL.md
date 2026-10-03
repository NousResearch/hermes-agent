---
name: platform-specs-and-validation
description: >-
  Platform specs and publish validation — per-platform requirements and the validate-before-publish
  step for multi-platform fan-out. Use when fanning a post out to multiple platforms, when a post is
  getting rejected, when mapping per-platform postType/required fields, or to run the
  validate-before-publish step. Encodes the per-platform postType enums, required-field matrices,
  and media rules from the OpenAPI spec, and runs POST /posts/validate so a post lands correctly on
  every target. Uses the CHECK framework. The keystone the per-platform format skills point to.
  WoopSocial validates + publishes atomically and has no update endpoint; the agent maps fields and
  fixes spec violations, but human-judgment calls (disclosure truthfulness, privacy intent) stay
  with the person; metrics never fabricated. Distinct from the per-platform format skills (one
  platform's content) and scheduling-and-queue (timing).
version: 1.0.0
---

# platform-specs-and-validation

The **cross-cutting publish-readiness skill** — make a post correct on **every** target (required fields,
media rules, limits) and run the **validate-before-publish** loop. It's the **keystone the format skills
point to.** The agent **maps fields + fixes spec violations + validates**; **WoopSocial validates and
publishes**; **content craft and human-judgment calls stay elsewhere.**

## The POV: publish-ready, atomically, the first time
WoopSocial **validates and publishes atomically** — `POST /posts` checks every target together, and **if
any one fails, nothing is created.** There's **no update endpoint**, so a post that publishes wrong can't
be patched (edit = **delete + recreate**). That makes **getting every target's fields right *before*
publish** the whole game: the right `postType`, the required per-platform fields, the media rules, and
account-specific values (Pinterest board, TikTok privacy) resolved from `platform-inputs` — then
`POST /posts/validate` to catch what's left. This skill is where the format skills come to **make sure it
will actually go through.**

## The framework: CHECK
(Depth: `references/the-check-framework.md`.)
- **C — Collect the targets + project:** list the social accounts; confirm they're in the **same project**;
  note each platform's field set.
- **H — Hydrate per-platform required fields:** fill each target's required fields (FB/IG `postType`;
  TikTok's eight; YouTube `title`+`privacy`; Pinterest `pinterestBoardId`); **resolve account values** from
  `GET /social-accounts/{id}/platform-inputs`.
- **E — Enforce media + content rules:** FB no image+video mix; IG Stories need media; **Pinterest single
  media**; TikTok PHOTO/VIDEO field applicability; use `contentOverride` or separate posts where fan-out
  conflicts; respect text limits.
- **C — Confirm truthful declarations:** `isBrandedContent`/`isYourBrand`/`madeForKids` are **truthful
  declarations, not validation knobs** — never flip them to clear an error; these stay with the person.
- **K — Knock it through validation:** `POST /posts/validate` → fix all errors, review warnings → create
  (atomic). **No update** → delete + recreate (only while `NOT_STARTED`).

## The spec (ground truth — verify-quarterly)
Per-platform postType enums + required-field matrices + media rules + the platform-inputs discovery
endpoint + lifecycle facts (atomic create, no update, delivery status, schedule types):
`references/woopsocial-publish-spec.md`. The validate-then-create loop, the full `ValidationField` code
table + `CreatePostErrorCode` list + worked examples: `references/validation-reference-and-recipes.md`.
(The live OpenAPI spec / `tools/integrations/woopsocial.md` are the source of truth; enums and limits move.)

## Honest scope (never violate)
- **This skill ensures publish-READINESS** (fields/media/spec correct + validated) — it **doesn't write or
  improve the creative** (the format/content skills do) and **can't guarantee performance.**
- **WoopSocial validates + publishes atomically; one failing target fails the whole post.** **No update** →
  delete + recreate (only while `NOT_STARTED`).
- **Disclosure/compliance declarations stay truthful** (`isBrandedContent`/`isYourBrand`/`madeForKids`) —
  **never flipped to pass validation**; privacy intent + disclosure truthfulness stay with the **person**.
- **Resolve account values** (board, privacy) from `platform-inputs` — don't guess. **Never fabricate
  metrics** — native analytics only. (Scope, distinctions + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**platform-specs-and-validation (this)** = the cross-platform field/media/validation layer that makes a post
publish-ready on every target · **the per-platform format skills** (`tiktok-photo-mode`,
`tiktok-video-publishing`, `threads-post`, `youtube-publishing-and-metadata`, `carousel-writer`, …) =
**craft one platform's content** and point here for validation · **scheduling-and-queue** = timing/cadence,
not correctness · `tools/integrations/woopsocial.md` = the shared integration guide this applies post-level.

## Where this connects
Receives from: every **format skill** (crafted content + chosen postType) and **content-calendar** (the
plan). Resolves with: `GET /social-accounts/{id}/platform-inputs` (boards, privacy options) + the
**integration guide**. Hands to: **scheduling-and-queue → WoopSocial** (`POST /posts` after `isValid`),
then **webhooks/delivery status** for confirmation. Supports: **cross-platform-repurposing** (per-target
`contentOverride`), **analytics-and-reporting** (native delivery readout), **experimentation** (valid
variants). Content craft + truthful declarations stay with the format skills + the person.

## Definition of done
Every target collected and confirmed in one project; each target's required fields hydrated (postType /
privacy / board / disclosure / title as applicable) with account-specific values resolved from
platform-inputs; media + content rules enforced (no FB mix, IG-Story media, single Pinterest media, TikTok
field applicability) with contentOverride or separate posts where fan-out conflicts; disclosure/madeForKids
declarations left truthful (never flipped to pass); `POST /posts/validate` run with all errors resolved and
warnings reviewed before an atomic create; edits handled via delete+recreate, nothing fabricated and no
promised performance; correctly distinguished from the per-platform format skills and scheduling-and-queue.
