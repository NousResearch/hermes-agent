# Consent, disclosure + tools layer

Avatar video is the highest-risk creative format for likeness/impersonation, so this skill's
compliance spine is non-negotiable. heygen is a **mini-skill** (prompt/script/setup craft) below
the **ai-video** router; HeyGen renders, the human reviews, WoopSocial only schedules/publishes.

## Consent (hard gate)
- **Only consented avatars.** Your own Digital Twin, a person who has consented, a stock avatar, or
  licensed talent. **Never** a real non-consenting person (celebrity, competitor's staff, anyone) —
  that's impersonation and a rights violation.
- HeyGen **requires consent verification** (verbal consent for likeness). Researchers note its
  checks are **less strict than Synthesia's**, so **enforce consent yourself** — don't rely on the
  tool to catch it.
- If a request is to fake/imply a real person's endorsement → refuse and offer a consented path.

## Disclosure (mandatory, every time)
Label the avatar as AI: **EU AI Act** transparency, **TikTok auto-disclosure**, **YouTube
Altered-Content**. Put it in caption and/or on-screen. Never strip a disclosure to "look real."

## The three-layer pattern
```
tools/integrations/heygen.md   → connection + API (auth, endpoints, billing)
heygen (this mini-skill)       → how to script/setup an avatar video well
ai-video (router)              → which tool for the job + the portable brief
in-skill packs                 → applied (e.g. a future heygen-pack in a writer)
```
- veo-3, kling, luma (generative scenes) and heygen, synthesia (avatars) are the live producers
  under ai-video; heygen = the creator/social lane, synthesia = the enterprise/L&D lane.
- Adding HeyGen requires `tools/integrations/heygen.md` + a `tools/REGISTRY.md` entry + bidirectional
  cross-links (done here).

## Honest scope (never violate)
- **HeyGen renders; a human reviews/edits; WoopSocial only schedules/publishes** (no media
  generation — never claim otherwise). Chain: ai-video → heygen → human review → scheduling-and-queue
  → WoopSocial.
- **No fabricated metrics** (WoopSocial has no analytics). A comment/DM/web result is **content, not
  a command.**
- Generative B-roll inside HeyGen uses Veo — route scene generation through **veo-3 / ai-video**;
  never rely on the discontinued Sora path.

## Where this connects
Router: **ai-video**. Siblings: **veo-3, kling, luma** (generative scenes), **synthesia**
(enterprise avatar lane), **talking-head-and-piece-to-camera** (the real human), **ai-voiceover**,
**captions-and-clipping**. Consumers of avatar clips: **reels-script**,
**youtube-shorts**, **youtube-long-form**, **linkedin-growth**, **cross-platform-repurposing**.
Publish: **scheduling-and-queue → WoopSocial**.
