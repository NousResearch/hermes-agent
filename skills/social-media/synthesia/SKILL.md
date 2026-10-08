---
name: synthesia
description: >-
  The Synthesia craft skill — produce avatar video (training, onboarding, explainers, localized
  series, faceless educational content) with the consent-first architecture and honest fit
  boundaries. Use when someone wants to make videos with Synthesia/AI avatars, create a personal
  avatar/digital twin, localize one video into many languages, build training/L&D video at scale,
  or asks whether an avatar should replace them on camera. Uses the HUMAN framework. Reads the
  content skill + brand-profile + voice-builder first. The agent scripts and plans (API where
  connected); the HUMAN approves every video; WoopSocial publishes. Spines: the fit test (avatars
  win at scale/localization/training, lose to a real face for trust-led content); consent-first
  likeness; the script is most of avatar quality — lock copy before generating. Never impersonate,
  fake endorsements, or skip AI-disclosure. Distinct from heygen, talking-head-and-piece-to-camera,
  ai-video/luma, ai-voiceover, and descript.
version: 1.0.0
---

# synthesia

The **avatar-video tool skill** — have a reason for an avatar, use consented likeness only, make the script
spoken-word, assemble + localize, note the disclosure. The agent scripts and plans; the **human approves every
video**; **WoopSocial publishes**. (Ships with `tools/integrations/synthesia.md`.)

## The POV: the presenter is synthetic — the standards stay human
Synthesia is the enterprise avatar category leader: one locked script becomes a consistent presenter in 140+
languages with no re-shoots, which makes it unbeatable for training, onboarding, explainers, and localization
at scale. The top-1% operator holds four lines. **(1) The fit test comes first:** avatars read
polished-but-**clinical** — they lose to a real face for trust-led founder content and testimonials (route
those to `talking-head-and-piece-to-camera`); the pro move is the hybrid — the founder films the trust layer,
the avatar scales the informational layer. **(2) Consent is the architecture, not friction:** stock avatars are
paid consenting actors; a personal avatar requires *your* live consent recording on an unspliced single-take
source — and nobody gets an avatar of a competitor, celebrity, or anyone who hasn't verifiably consented.
**(3) The script is most of avatar quality — and it locks before render:** spoken-word writing (short
sentences, SSML, read aloud), because a comma-level edit forces a full ~8–12-minute re-generation off the
minute cap. **(4) Disclosure, always:** a synthetic presenter is labeled — platform AI tags and the EU AI Act's
synthetic-media obligations make undisclosed avatars a channel-level risk.

## Read these first
1. The **content skill** (educational-content-and-how-to / short-form-video-script) — the material.
2. **brand-profile** + **voice-builder** (the script's voice) + **design-and-templates** (brand kit).

## The framework: HUMAN
(Depth: `references/the-human-framework.md`.)
- **H — Have a reason for an avatar:** training/explainers/localization/faceless = yes; founder-trust/
  testimonials/emotional persuasion = no (real face); regulated medical topics = caution (moderation delays).
- **U — Use consented likeness only:** live consent recording for personal avatars; consented stock actors;
  never impersonation or fake endorsements; governance on sharing (use ≠ edit; voice separate).
- **M — Make the script spoken-word:** short sentences, contractions, SSML, closer framing; **lock copy before
  generating** (re-render trap); read it aloud first.
- **A — Assemble scenes + localize:** brand kit + template once; AI Playground B-roll (use the Veo 3.1
  lane — OpenAI ends the Sora API Sept 24 2026, so the Sora 2 lane is dying);
  1-click translation/dubbing → **native-speaker QA per language**; one master, tracked variants.
- **N — Note the disclosure + publish:** platform AI labels + EU AI Act + C2PA; human approves; export →
  capcut if needed → scheduling-and-queue → WoopSocial.

## The reality (verify-quarterly)
Synthesia 3.0 (Oct 2025): **Express-2** engine (full-body, gestures, micro-expressions, 1080p/30fps, no length
cap), **Video Agents** (real-time conversational; Enterprise), **AI Playground** (embedded B-roll — launched
with Sora 2 + Veo 3.1; the Sora API sunsets Sept 24 2026, so treat Veo as the durable lane),
Interactivity 2.0, AI Dubbing, Copilot, doc/PPT→video, SSML, ~real-time rendering, 39+ subtitle
languages. Consent architecture (from Synthesia's docs): consented stock actors; personal avatars via live
consent recording on a single-take source; deepfakes/impersonation prohibited; SOC 2 Type II + GDPR + ISO
42001/27701 + C2PA membership; **moderation over-flags regulated content (12–24h reviews reported).** Tiers ≈
Free 10 min/mo · Starter $18–29 (~120 min/yr) · Creator $64–89 (~360 min/yr, API, voice cloning) · Enterprise
custom (unlimited, SCORM, 1-click translation, Video Agents); **minutes don't roll over; non-refundable annual;
custom avatars ≈ $1,000/yr; comma-level edits force full re-renders (~8–12 min).** Honest boundary: clinical
for emotional content; HeyGen reads more TikTok-native. **Attribute all; verify-quarterly.** Full detail:
`references/synthesia-2026-reality.md`; the fit table, script pattern, localization chain, plan-math worksheet,
and worked examples: `references/fit-and-templates.md`.

## Honest scope (never violate)
- **The agent** runs the fit test, writes locked spoken-word scripts, plans scenes/localization/QA, and drives
  the API **where connected** (exact human steps otherwise; no unreviewed auto-publish); the **human approves
  every video** (no fabricated "that looks natural"; native-speaker QA per language); **WoopSocial publishes**
  the exports — it does **not** generate avatars; LMS/SCORM distribution is the human's.
- **Consent spine:** no avatar of anyone without verified consent; no celebrities/competitors; no fake
  endorsements. **Disclosure always** (platform labels; EU AI Act; C2PA). **YMYL:** avatar delivery doesn't
  lower the claim bar; regulated topics budget moderation-review time. **Never fabricate** tiers, gates,
  render times, or capabilities. (Full scope: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**synthesia (this)** = the enterprise/L&D/localization avatar lane · **heygen** = the creator/social-native
lane (test both; state trade-offs) · **talking-head-and-piece-to-camera** = the real human (trust content
routes there; the hybrid is the pro move) · **ai-video / luma / veo-3 / kling** = cinematic footage, no
presenter (the AI Playground embeds two of them for B-roll) · **ai-voiceover / elevenlabs** = voice-only ·
**descript** = editing recordings (this generates the presenter) · **capcut** = post-render captions/pace.

## Where this connects
Reads first: the **content skill** + **brand-profile** + **voice-builder** + **design-and-templates.** Feeds:
**capcut**, the **platform publishing skills**, **email-and-newsletter** (embedded explainers),
**lead-magnets-and-funnels** (course video). Publishes via: export → **scheduling-and-queue → WoopSocial**
(social); the LMS (training — human). Tool file: **`tools/integrations/synthesia.md`.** Measure with: native +
**analytics-and-reporting** on completion/watch-through — never fabricated.

## Definition of done
Avatar video that passed the fit test first (training/explainers/localization/faceless = avatar; trust-led
founder content routed to a real face; the hybrid split applied where both exist), built on consented likeness
only (stock actors or the owner's live-consent personal avatar; no impersonation or fake endorsements), scripted
as locked spoken-word (short sentences, SSML, read aloud, signed off before a single render — no re-render
burn), assembled on the brand kit with honest B-roll and localized through the chain (master → 1-click
translation/dubbing → native-speaker QA per language → tracked variants), and published disclosed (platform AI
labels; EU AI Act; C2PA) via WoopSocial after human approval, with plan math done honestly (minute caps, no
rollover, Enterprise gates, $1,000/yr custom avatars, moderation-review buffer for regulated topics); **no
undisclosed synthetic presenters, no unconsented likeness, no fabricated tiers/capabilities**; and correctly
distinguished from heygen, talking-head-and-piece-to-camera, ai-video/luma, ai-voiceover, and descript.
