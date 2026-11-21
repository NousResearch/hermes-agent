# Inputs, Length & Recipes

How to drive Veo with images, control the shot, work within the 8-second clip, and apply it to common
social jobs.

> Re-verify feature/limit specifics quarterly.

## Image-to-video (animate a still)

Feed Veo an **input image** and it animates it — the image becomes the **first frame**, and your prompt
describes the **motion + audio**. Great for:

- **Animating a `nano-banana` still** — the cross-tool pipeline: design the exact frame as an image
  (precise text/composition/brand), then bring it to life in Veo. Image gen gives control; Veo gives
  motion + sound.
- **Product / character motion** — animate a clean product shot or a brand character.

## Reference images & frame control (consistency + direction)

- **Reference images ("ingredients")** — supply up to ~3 images of a character/product/scene to
  **maintain identity/style** across the clip (and across clips). Use 2–3 angles for best consistency.
- **First & last frame** — provide a start and end image and Veo generates the **transition** between
  them (with audio) — precise control over a shot's composition (e.g. front view → POV behind).

## The 8-second reality + going longer

- **One generation ≈ 8 seconds** (4/6/8). Design the beat to fit; **hook in the first second** (most
  social viewers decide instantly → ties to `reels-script`).
- **Longer videos:** use **scene extension** (each new clip continues from the last second, keeping
  continuity/audio) or **generate several clips and stitch** them (in Google Flow or an editor). Plan a
  shot list of short beats rather than expecting one long take.

## Social recipes (adapt the brackets)

Always set **9:16** for vertical social and ground in the brand. Verify + disclose before publishing.

- **B-roll / establishing:** "[Scene], [time of day], [lighting]. [Slow camera move]. [Ambient sound].
  Cinematic, 9:16."
- **Product in motion:** "[Product] [action — rotating / steam rising / liquid pouring], [surface/
  setting], [lighting]. Macro to medium, shallow depth of field. [SFX]. 9:16." (Or animate a product
  still via image-to-video.)
- **Hook visual (for a Reel):** a striking 1–2s opener that earns the watch — "[surprising visual],
  [fast camera move], [punchy SFX]. 9:16." (→ `reels-script` owns the full script; this is the visual.)
- **Spokesperson / UGC-style clip:** "[Person] in [setting] looks to camera and says, "[short line]",
  [tone]. [Natural ambient]. Handheld, 9:16." (For sustained dialogue, prefer an avatar tool.)
- **Ad / scene:** "[Hero/scene], [mood], [camera], with "[short on-audio line or VO]". [Music mood].
  9:16." (Verify any claim/offer; disclose AI.)
- **Animate a still:** input the image → "Bring this to life: [describe the motion + ambient audio],
  subtle and natural. 9:16."

## Ship it

Generate per `tools/integrations/veo.md` (async generate-and-poll) → **upload the video bytes to
WoopSocial Media → attach to the post** via `scheduling-and-queue`. WoopSocial publishes; it does not
generate video.
