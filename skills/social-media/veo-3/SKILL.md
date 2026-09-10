---
name: veo-3
description: >-
  Use to write great prompts for Google Veo (Veo 3.1) to generate or animate video for social media —
  the video-prompt-craft mini-skill, the video counterpart to nano-banana. Run when the user wants a
  Veo / AI video prompt, a short video clip for a post (b-roll, product-in-motion, hook visual,
  spokesperson/UGC clip, ad), to animate a still image into video, or a vertical Reel/TikTok/Short
  clip. Reads brand-profile for brand style. Veo's standout is native synchronized audio in one
  pass, plus image-to-video and native 9:16 vertical. Teaches the prompt anatomy, audio prompting,
  the image-to-video pipeline, and the 8-second constraint + extension/stitching. Honest: iterate
  cheap then finalize, disclose AI video (SynthID watermark), never generate real identifiable
  people or copyrighted IP. This is the
  prompt-craft layer — the API/connection and the generate -> upload to WoopSocial Media -> attach
  flow live in tools/integrations/veo.md; the consuming pack is reels-script's veo-prompt-pack.
metadata:
  version: 1.0.0
license: MIT
---

# Veo (video prompt craft)

Write prompts that get cinematic clips out of **Veo** — Google's video model (**Veo 3.1**; Fast/Lite
tiers for cheap iteration). This is the **prompt-craft** layer of a three-layer setup, and the **video
counterpart to `nano-banana`**:

- **Connection/API** (model IDs, async generate-and-poll, the *generate → upload to WoopSocial Media →
  attach* flow) → `tools/integrations/veo.md`.
- **Prompt craft** (this skill) → how to direct the right clip well.
- **In-skill application** → `reels-script`'s veo-prompt-pack.

> Fast-moving area — re-verify model names/specs quarterly.

## Reach for Veo when… (match the job)

Its real strengths: **native synchronized audio** (dialogue + lip-sync, SFX, ambient, music — the
differentiator), **cinematic realism/physics**, **image-to-video**, **reference-image consistency**,
**first/last-frame** control, and **native 9:16 vertical / 4K**. Reach **elsewhere** for a **pure
talking-head explainer** (→ avatar tool like `heygen`), **native-4K/multi-shot/motion-transfer at
lower cost** (→ `kling`), **HDR/atmospheric mood shots** (→ `luma`), **long continuous video**
(stitch/extend or a length-built tool), or **highly stylized/very-high-volume** output. Don't
default it to every job. (Details: `references/when-and-how-to-prompt.md`.)

## Step 0 — Read the brand + the job

Load `brand-profile.md` (visual style, palette, tone). Identify the **job** (b-roll / product motion /
hook visual / spokesperson / ad / animate-a-still) and the **aspect ratio** (9:16 for social).

## Step 1 — Direct the shot (natural language, not quality spam)

Describe a shot like a director: **subject · action · scene · camera (type/movement/angle/lens) ·
lighting/mood · style · audio · timing**. One clear motion per clip. Set **9:16**. Ground in the brand.
See `references/when-and-how-to-prompt.md`.

## Step 2 — Use the superpower: audio

Veo generates **native synchronized audio** in one pass — describe it explicitly: **dialogue in
quotes** (+ who/tone), **SFX**, **ambient**, **music** mood. Treat generated audio as a **draft/guide
track** for branded work (record real VO / license music for final) and **verify** lip-sync. Direct
the **camera/motion** for the cinematic feel. See `references/audio-and-camera.md`.

## Step 3 — Inputs, length, recipes

- **Image-to-video:** animate a still (e.g. a `nano-banana` frame) — it becomes the first frame; you
  describe motion + audio. Use **reference images** / **first-and-last frame** for consistency/control.
- **Length:** one generation ≈ **8 seconds** (4/6/8); **hook in the first second**; for longer use
  **scene-extension or stitch** clips. Plan short beats.
- Pick a **recipe** for the job (b-roll, product motion, hook, spokesperson, ad, animate-a-still). See
  `references/inputs-length-and-recipes.md`.

## Step 4 — Iterate cheaply, then verify, disclose, ship

- **Iterate at low-res / Fast** to lock the prompt; **finalize the keeper at high-res/4K** (4K costs
  ~40–60% more time/$). Off-prompt generations still consume credits — **prompt skill is the cost
  lever**; video is **async**.
- **Verify** every clip (artifacts, lip-sync, physics) before publishing.
- **Disclose** AI video per platform/region (EU AI Act; TikTok auto-disclosure); every clip carries a
  **SynthID** watermark — don't pass it off as real footage.
- **Ship:** generate per the integration guide → **upload to WoopSocial Media → attach** via
  `scheduling-and-queue`. WoopSocial doesn't generate video.

## Quality bar — self-check

- Did I **match the tool to the job** (and route talking-heads/long-form elsewhere)?
- Is the prompt a **directed shot** (subject/action/camera/lighting/style), **brand-grounded**, **9:16**,
  with an **explicit audio cue**?
- Did I plan for **8-second** clips (hook first) and **iterate cheap → finalize**?
- For stills, did I use **image-to-video** (and reference/first-last frame where useful)?
- Did I handle **SynthID + disclosure**, **audio-as-draft**, **verify the output**, and **refuse real
  people / IP**?
- Did I point to **`tools/integrations/veo.md`** for the API + WoopSocial flow (no claim WoopSocial
  generates video)?

## Edge cases & pushback

- **Talking-head explainer / lots of dialogue** → suggest an avatar tool (`heygen`); don't force Veo.
- **"Make a 40-second video"** → ~8s per gen; scene-extend/stitch; plan short beats.
- **"Generate 10 final 4K-with-audio variations now"** → iterate cheap first; 4K/audio is costly + async.
- **"a person, cinematic, 4k, amazing"** → rewrite into a directed shot (subject/camera/lighting/audio).
- **Real person / copyrighted IP / "post as real footage"** → refuse; SynthID + disclosure; offer an
  original alternative.
- **"Generate it in WoopSocial"** → WoopSocial doesn't generate; this prompts Veo, then the clip is
  uploaded to Media and attached.

## Related

- `tools/integrations/veo.md` — API/model IDs, async generate-and-poll, pricing, the upload-to-WoopSocial flow.
- `reels-script` (veo-prompt-pack) — the consuming skill; `nano-banana` — the image sibling + image-to-video source.
- `brand-profile` — the visual brand; `hook-writer` — the in-clip hook/line; `heygen` — avatar/talking-head alternative.
- `ai-video` — the router above this skill; `kling` (4K/multi-shot/motion) and `luma` (HDR/mood, silent) — generative siblings.
- `scheduling-and-queue` — attach the video to a post and publish.

## References

- `references/when-and-how-to-prompt.md` — when to reach for Veo vs other tools, and the shot/prompt anatomy.
- `references/audio-and-camera.md` — the native-audio superpower (dialogue/SFX/ambient) and camera/motion direction.
- `references/inputs-length-and-recipes.md` — image-to-video, reference/first-last frame, the 8s limit + extension, social recipes.
- `references/examples.md` — weak→strong prompts, an audio-rich clip, image-to-video, a vertical hook, and honest scope.
