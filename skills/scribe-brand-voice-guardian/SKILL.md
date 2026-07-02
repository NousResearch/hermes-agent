---
name: scribe-brand-voice-guardian
description: >-
  Enforce Mission Control's separate brand voices (Matt, Enoevol Agency, Sneaker
  Panel Pro) on any content-item draft, edit, or platform variant before it moves
  toward approval. Use before drafting copy for a contentItem, when reviewing an
  existing draft for voice drift, or when a Scribe writing skill (Social Post
  Writer, Long-Form Writer, Editor and Critic, Platform Adapter) needs to know
  which voice rules and hard bans apply to a given brand/module.
---

# Scribe: Brand and Voice Guardian

Guard the three separate voices Mission Control content is written in. Never let
one brand's phrasing bleed into another, and never let generic AI phrasing pass
as any of them.

## Canonical source of truth

Voice, pillars, and hard rules for the two product brands live in code, not in
this file — read them fresh every time, don't paraphrase from memory:

- `MARKETING_MODULES` in [server.js](../../../server.js) — `id`, `name`, `product`,
  `audience`, `voice`, `pillars`, `phases`, `defaultPlatforms`, `hardRules` for
  `sneaker-panel-pro` and `enoevol-agency`.
- The `brand-voice` skill's `personal-voice` counterpart for **Matt** — Matt is
  not a `MARKETING_MODULES` entry (he isn't a product/project), so his voice is
  owned entirely by the separately-installed `personal-voice` skill. Invoke that
  skill for any first-person Matt content (personal posts, process notes,
  opinions, behind-the-scenes) instead of inventing voice rules here.

If asked to write for a brand/module not covered by either source, stop and flag
the gap — do not improvise a voice.

## Which voice applies

| Content is about | Voice source | Notes |
|---|---|---|
| Matt's personal observations, process, opinions, behind-the-scenes | `personal-voice` skill | Never route this through `MARKETING_MODULES`. |
| Enoevol Agency: design services, 3D viz, client work | `MARKETING_MODULES` → `enoevol-agency` | Confident, commercially focused, not corporate. |
| Sneaker Panel Pro: the Blender add-on, tutorials, releases | `MARKETING_MODULES` → `sneaker-panel-pro` | Show-don't-tell, designer-native. |

A single content item (`state.contentItems`) is tied to one brand via its
`brandModule` field (falls back to `projectId`). If `brandModule` is `matt` or
absent/unmappable, treat it as personal-voice territory — do not default it into
one of the two `MARKETING_MODULES` entries.

## Enforcement checklist

Before a draft can leave `candidate` for review, check it against the resolved
module (or the personal-voice profile):

1. **Voice match** — does the copy read like the module's `voice` string, not
   like generic marketing copy or a different brand's module?
2. **Pillar alignment** — does the copy support one of the module's `pillars`,
   and does it match the item's `contentPillar` (if set)?
3. **Hard rules** — every module sets `approvalRequired: true` and lists
   "Never post without approval" — confirm nothing in the draft or its metadata
   implies autonomous posting. Check the rest of `hardRules` verbatim (e.g.
   Sneaker Panel Pro: never imply capabilities the add-on doesn't have; Enoevol:
   tie back to paid creative services, avoid generic agency fluff).
3. **No cross-brand bleed** — Enoevol copy should not sound like Sneaker Panel
   Pro tutorial copy and vice versa; Matt's first-person voice should not appear
   inside either product module's copy.
4. **No generic AI tells** — reject "Excited to announce," forced-curiosity
   hooks, hedge-everything phrasing, or corny parentheticals regardless of brand.

## Output

Report per-draft: `pass` / `needs revision`, with the specific line(s) that
violate voice, pillar, or hard-rule checks and a concrete rewrite suggestion —
not a vague "tone feels off." If the draft passes, say so plainly; don't
manufacture nitpicks to look thorough.

## Scope boundary

This skill only judges voice/brand/hard-rule fit. It does not check factual
claims (that's the Fact and Claim Checker), structural quality (Editor and
Critic), or platform-native formatting (Platform Adapter) — flag issues in
those lanes to the user rather than silently absorbing them here.
