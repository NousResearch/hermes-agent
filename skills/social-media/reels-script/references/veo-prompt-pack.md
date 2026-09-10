# Veo Prompt Pack — AI video for Reels

When a Reel is **faceless** (voiceover or text-overlay + visuals) or needs b-roll/establishing
shots, this pack maps the script's beats to AI-video prompts. It's the Reels counterpart to the pack
in `tiktok-script`. The **connection/API layer** (auth, model IDs, limits, disclosure) lives in
`tools/integrations/veo.md` — read it before generating. Cinematic prompt craft lives in the
`veo-3` mini-skill; this pack is how `reels-script` applies it with Reels-native judgment.

> Tools and model versions move fast — re-verify quarterly against `tools/integrations/veo.md` and
> the official docs.

## Honest first: film faces, generate b-roll

Instagram rewards original, person-first content, and a real face converts profile visits to
follows in a way b-roll can't. So:

- **Default to filming** for talking-head, story, POV — anything with a face or a personal voice.
- **Use AI video for:** faceless niches, b-roll/cutaways, establishing shots, concepts you can't
  film (impossible scenes, product-in-context), or scaled faceless output you've validated.
- **Never generate fake people delivering testimonials** or results — a compliance line, not a
  style choice.

Reels tolerates a slightly more polished look than TikTok, but "aesthetic" still beats "ad": aim
for the look of a well-shot phone clip, not a commercial.

## Veo for Reels — what fits

- **Vertical 9:16** native — the correct Reels frame (1080×1920 at publish; specs →
  `instagram-reels-publishing`).
- **~8-second clips** per generation → one prompt per beat; stitch/extend in an editor.
- **Native audio** (ambience/SFX) — but the Reel's voiceover/music is usually added in edit, and
  IG-library audio is added natively in-app (`formats-and-audio.md`).
- **Reference images + character consistency** — keep a subject/style stable across beats.

## The beat → prompt skeleton

For each script beat, write an 8-second prompt:

```
8-second vertical 9:16 video for Instagram Reels, well-shot phone-clip feel (aesthetic, not
glossy-commercial).
Scene: {what happens}
Subject & action: {subject}, {motion}
Camera: {handheld / slow push-in / static phone shot}
Lighting/mood: {natural, real} ; Style: {clean, authentic — NOT ad-like}
On-screen text (added in edit): "{hook/label}" — keep clear space top and bottom for UI
Audio: {ambience} (voiceover/music added in edit)
```

Keep key action **centered** — the frame gets cropped in the feed and on the profile grid, and
on-screen text must land in the safe zone (see `instagram-reels-publishing`).

## Workflow

1. Generate per beat (each ~8s; expect ~60–180s/clip; retries are normal).
2. Assemble in an editor: order beats, add the **voiceover**, **on-screen text/keywords**, and
   burned-in captions (→ `captions-and-clipping`); export clean — **no other-platform watermarks**.
3. Keep the **3-second hook** on the strongest beat first, and pick the **cover frame**.
4. Upload the finished MP4 to WoopSocial's **Media** domain, publish via
   `instagram-reels-publishing` / `scheduling-and-queue`; add any IG-library audio natively.

## Compliance — AI-disclosure

Veo output carries a **SynthID watermark** — it's AI-generated video. On Instagram, disclosure is
**native/manual**: follow Meta's AI-labeling rules and the brand's policy (WoopSocial auto-discloses
on TikTok only — do not assume the same here). Note EU AI Act synthetic-media obligations where
relevant. Never fake testimonials or put fabricated claims in generated video.

## Relationship to other pieces

- `tools/integrations/veo.md` = the API/connection layer (shared).
- `veo-3` mini-skill = cinematic prompt craft.
- This pack = how `reels-script` applies Veo, with Reels-native judgment (crop-safe, aesthetic-not-ad,
  disclosure-manual).
