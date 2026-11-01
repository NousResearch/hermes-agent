# Scope, distinctions + connections

platform-specs-and-validation is the **cross-cutting publish-readiness skill** — it makes a post correct on
**every** target (fields, media rules, limits) and runs the **validate-before-publish** loop. It's the
**keystone the format skills point to** for the validation step. The agent **maps fields + fixes spec
violations + validates**; **WoopSocial validates and publishes**; **content craft and human-judgment calls
stay elsewhere.**

## Honest scope (never violate)
- **This skill ensures publish-READINESS** (required fields hydrated, media rules satisfied, validated) —
  it **doesn't write or improve the creative** (that's the per-platform format/content skills) and **can't
  guarantee performance.**
- **WoopSocial validates + publishes atomically** (`POST /posts/validate` → `POST /posts`); **one failing
  target fails the whole post.** **No update endpoint** → edit = **delete + recreate** (only while
  `NOT_STARTED`).
- **Disclosure/compliance declarations are truthful, not validation knobs** — TikTok `isBrandedContent`/
  `isYourBrand`, YouTube `madeForKids`. **Never flip a declaration to clear an error**; fix the real field.
  Privacy intent + disclosure truthfulness stay with the **person**.
- **Resolve account-specific values** (Pinterest board, TikTok `privacyLevel`) from **platform-inputs** —
  don't guess. **Never fabricate metrics** — native analytics only. **Re-verify the spec quarterly**
  (`references/woopsocial-publish-spec.md`).

## Distinct from its siblings
- **platform-specs-and-validation (this)** — the **cross-platform field/media/validation layer** that makes
  a post publish-ready on every target.
- **the per-platform format skills** (`tiktok-photo-mode`, `tiktok-video-publishing`, `threads-post`,
  `youtube-publishing-and-metadata`, `carousel-writer`, `reels-script`, …) — **craft one platform's
  content**; they **point here** for the validate step.
- **scheduling-and-queue** — **timing/cadence** (when), not correctness (whether it'll publish).
- `tools/integrations/woopsocial.md` — the **shared integration guide** (endpoints/media/auth); this skill
  applies its rules at the post level.

## Where this connects
- **Receives from:** every **format skill** (the crafted content + chosen postType) and `content-calendar`
  (the plan).
- **Resolves with:** `GET /social-accounts/{id}/platform-inputs` (boards, privacy options), the
  **integration guide** (`tools/integrations/woopsocial.md`).
- **Hands to:** `scheduling-and-queue → WoopSocial` (`POST /posts` after `isValid`), then **webhooks/
  delivery status** for confirmation.
- **Supports:** `cross-platform-repurposing` (per-target `contentOverride`), `analytics-and-reporting`
  (native delivery/performance readout), `experimentation-and-ab-testing` (valid variants). **Content craft + truthful
  declarations stay with the format skills + the person.**
