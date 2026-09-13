# Scope, distinctions & connections

## Honest scope (never violate)
- **The agent** writes the shot briefs, reference plans, keyframe/consistency strategy, the draft-first credit
  plan, and — **where an API key/connection exists** — can drive the official SDK (lumaai-python: create a
  generation, poll status, download). Where it doesn't, exact steps for the human in the web/iOS app. No
  pretended automation, and **no unreviewed auto-publish** — the human judges every clip.
- **The human judges every render** — the agent can't see video and never fabricates "that shot works";
  multi-shot sets get a side-by-side drift review.
- **WoopSocial publishes** the finished exports (measurement: the platforms' native analytics); it does **not** generate or edit
  video.
- **The rights spine:** free tier = watermarked, non-commercial; the uncontested commercial floor is the
  Plus-level plan (tier structures conflict across sources and two generations of plans coexist — **verify
  in-app**); **Luma retains a marketing/service license on your generations** (Enterprise for NDA/data-privacy);
  high-stakes → counsel (not legal advice).
- **Likeness + misinformation:** no real-person likeness (real or AI lookalike) without consent; **no photoreal
  fake-event footage targeting real people/companies** (misinformation + defamation — "satire" doesn't launder
  it); **AI-disclosure** where platform/region requires (EU AI Act; C2PA). **Sound:** no native audio — music/VO
  added in post with proper licenses. **Never fabricate** credit rates, tier terms, benchmarks, or capabilities
  — verify-quarterly.

## Distinct from its siblings (route correctly)
- **ai-video** = the model-agnostic video-generation router/craft (read it first; this is the Luma-specific
  lane).
- **veo-3 / kling / runway** = sibling video skills (runway = the control-focused/edit-grade lane) —
  route sound-native generation to Veo/Kling (Luma has no native audio); strengths shift per
  release, test on your brief.
- **flux / image-prompt** = the brand-true stills the still-first workflow starts from (Photon is Luma's
  in-house image model).
- **capcut** = where shots become films (assembly, pace, captions) + **ai-music-and-sound** = the audio Luma
  doesn't generate.
- **heygen / synthesia** = avatar/talking-head generation (a different job: presenters, not cinematic shots).
- **short-form-video-script / scripting skills** = the shot list a sequence starts from.

## Where this connects
- **Reads first:** ai-video (router) + brand-profile + design-and-templates (the brand feel) +
  short-form-video-script (the beats a sequence serves).
- **Pulls stills from:** flux / image-prompt (canonical, brand-true reference frames).
- **Feeds:** capcut (assembly + captions), ai-music-and-sound (the audio), tiktok-video-publishing /
  instagram-reels-publishing / youtube-shorts (the finished clip), before-after-and-transformation (honest
  visuals only).
- **Publishes via:** export → scheduling-and-queue → **WoopSocial.**
- **Tool file:** `tools/integrations/luma.md` (models, tiers, API wallet, the rights spine).
- **Measure with:** native + analytics-and-reporting on the published posts — never fabricated.
