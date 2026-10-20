# When & How to Prompt Veo

When to reach for Veo over another tool, and the prompt anatomy that gets cinematic results. (The
API/connection layer — model IDs, async generate-and-poll — is in `tools/integrations/veo.md`; this is
prompt craft.)

> Fast-moving model area — re-verify model names/specs quarterly. As of 2026: **Veo 3.1** is the
> current model (`veo-3.1-generate-preview`), with **Fast** and **Lite** tiers for cheaper/faster
> iteration. Use Fast/Lite to iterate, Quality for the final.

## Reach for Veo when the job is…

Its genuine strengths:

- **Native, synchronized AUDIO** — the standout. Veo generates **dialogue (with lip-sync), sound
  effects, ambient sound, and music in the same pass as the video.** Every other model bolts audio on
  afterward; Veo doesn't. If sound matters, this is the reason to use it. (See `audio-and-camera.md`.)
- **Cinematic realism + physics + prompt adherence** — believable motion, lighting, and camera work.
- **Image-to-video** — animate a still (e.g. a `nano-banana` image) into motion (see
  `inputs-length-and-recipes.md`).
- **Reference-image consistency ("ingredients")** + **first/last-frame** control — keep a
  character/product consistent and direct the shot's start/end.
- **Native 9:16 vertical + up to 4K** — true vertical composition for Reels/TikTok/Shorts (not cropped).

## Reach for something else when…

- **A pure talking-head / spokesperson explainer with lots of scripted dialogue** → an **avatar tool**
  (e.g. `heygen`) is usually better and cheaper for sustained presenter video.
- **Long-form continuous video** → Veo clips are short (≤8s); plan to **stitch/extend**, or use a tool
  built for length.
- **Highly stylized/anime or very high-volume cheap ad output** → other video models may fit cost/style
  better. Match the job; don't default.

## The prompt anatomy (cinematic, natural language)

Describe a **shot**, like a director. Include:

- **Subject** — who/what, specifically.
- **Action** — what they do (describe motion with attention to physics).
- **Scene / setting** — where, time of day, context.
- **Camera** — shot type + movement + angle + lens ("medium shot, slow push-in"; "POV"; "180° arc";
  "lens flare").
- **Lighting / mood / atmosphere** — ("warm golden-hour light, hopeful").
- **Style** — ("cinematic," "documentary," "commercial product film").
- **Audio** — dialogue in quotes, plus SFX/ambient/music cues (see `audio-and-camera.md`).
- **Timing / pacing** — what happens across the clip; aspect ratio (**9:16** for social).

Use real cinematography vocabulary; be descriptive and clear (prompts can be long, ~1,024 tokens).
Avoid vague one-liners — Veo rewards specific direction.

## Plan for short clips, iterate cheaply

- **One generation ≈ 8 seconds** (4/6/8 options). Build the idea to fit; put the **hook in the first
  second**. For longer → scene-extension/stitch (`inputs-length-and-recipes.md`).
- **Iterate at low-res / Fast** to lock the prompt, then **finalize the chosen clip at high-res/4K**
  (4K adds ~40–60% time/cost). Off-prompt generations still cost — **prompt skill is the cost lever.**
- **Async** — video isn't instant; the agent submits and polls (per the integration guide).

## Brand-grounded

Read `brand-profile.md` — bake the brand's **visual style, palette, tone, do/don't** into the prompt so
the clip looks like the brand, not generic stock.
