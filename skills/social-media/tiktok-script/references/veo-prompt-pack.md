# Veo Prompt Pack — AI video for TikTok

When a TikTok is **faceless** (voiceover + visuals) or needs b-roll/establishing shots, this pack
maps the script's beats to AI-video prompts. It's the TikTok counterpart to the Veo pack in
`reels-script`. The **connection/API layer** (auth, model IDs, limits, disclosure) lives in
`tools/integrations/veo.md` — read it before generating.

> Tools and model versions move fast — re-verify quarterly against `tools/integrations/veo.md` and
> the official docs.

## Honest first: native filming usually beats AI on TikTok

TikTok rewards real, native, slightly-raw content (see `tiktok-mechanics.md` — "make TikToks, not
ads"). A founder talking to a phone camera typically **out-performs** a polished AI clip, because it
reads as authentic. So:

- **Default to filming** for talking-head, storytime, POV, reactions — anything with a face or a
  personal voice.
- **Use AI video for:** faceless niches, b-roll/cutaways, establishing shots, concepts you can't
  film (impossible scenes, product-in-context), or scaled faceless output where you've validated it
  works.
- Don't pitch AI as the better TikTok option when it isn't. Recommend it where it genuinely helps.

## Veo for TikTok — what fits

- **Vertical 9:16** native — correct TikTok frame.
- **~8-second clips** per generation → one prompt per beat; stitch/extend in CapCut.
- **Native audio** (Veo 3.1) — ambience/SFX; but TikTok voiceover/sound is usually added in edit.
- **Reference images + character consistency** — keep a subject/style stable across beats.

## The beat → prompt skeleton

For each script beat, write an 8-second prompt:

```
8-second vertical 9:16 video for TikTok, native/handheld feel (not glossy/ad-like).
Scene: {what happens}
Subject & action: {subject}, {motion}
Camera: {handheld / quick push-in / static phone shot}
Lighting/mood: {natural, real} ; Style: {authentic, slightly raw — NOT polished commercial}
On-screen text (added in edit): "{hook/label}"
Audio: {ambience} (voiceover added in edit)
```

Note "native/handheld, not glossy" deliberately — default AI output skews over-polished, which is
exactly what underperforms on TikTok.

## Workflow

1. Generate per beat (each ~8s; expect ~60–180s/clip; retries are normal).
2. Assemble in CapCut: order beats, add the **voiceover**, **on-screen text/keywords**, captions,
   and the **sound** (trending/original — see `formats-and-sound.md`).
3. Keep the **1-second hook** on the strongest beat first.
4. Upload the finished MP4 to WoopSocial's **Media** domain (raw-bytes) and schedule via
   `scheduling-and-queue`.

## Compliance — AI-disclosure

All Veo output carries a **SynthID watermark** — it's AI-generated video. WoopSocial
**auto-discloses on TikTok**; follow TikTok's synthetic-media rules and the brand's policy (and EU
AI Act synthetic-media disclosure where relevant). Never fake testimonials or put fabricated claims
in generated video.

## Relationship to other pieces

- `tools/integrations/veo.md` = the API/connection layer (shared).
- `veo-3` mini-skill (Tier 2) = cinematic prompt craft.
- This pack = how `tiktok-script` applies Veo, with TikTok-native judgment.
