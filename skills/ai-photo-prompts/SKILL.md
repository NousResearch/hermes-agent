---
name: ai-photo-prompts
description: "Authoring high-quality AI photo-editing prompts for GPT-5.5, Gemini, and similar models — restoration, upscaling, face preservation, background editing."
version: 1.0.0
author: Session learned
license: MIT
metadata:
  hermes:
    tags: [AI, photo, GPT, editing, restoration, prompt-engineering]
    related_skills: [claude-design, popular-web-designs]
---

# AI Photo Prompts

Engineering prompts for generative AI photo editing (GPT-5.5, Gemini, etc.) — restoration, upscaling, background editing, and composite creation.

## Core Principles

1. **Faces are sacred.** AI models default to "beautifying" and "smoothing" faces. You must actively prevent this.
2. **Backgrounds must be explicitly protected.** Without explicit KEEP instructions, the model will replace, idealize, or invent background elements.
3. **One-photo vs. multi-photo workflows differ.** Do not assume the same prompt works for both.
4. **Specific removal > general cleanup.** "Remove distractions" leads to the model inventing a cleaner scene. "Remove ONLY the woman in the pink jacket" keeps everything else authentic.

## Prompt Structure Template

Every prompt should contain these sections in this order:

```
SUBJECTS (MUST remain completely unchanged)
- Name/role, position, clothing, expression

ABSOLUTE RULES — DO NOT ALTER THE SUBJECTS
- Faces must remain pixel-perfect identical to originals. Only edit background, lighting, and color. Do not touch faces.
- Do NOT make younger, smoother, slimmer, different in any way.
- Clothing, poses, body proportions must remain identical.

SCENE / BACKGROUND — MUST REMAIN THE SAME LOCATION
- Describe real location and architecture.

KEEP ALL EXISTING BACKGROUND ELEMENTS AS THEY ARE:
- Keep vehicles, street signs, pedestrians, buildings exactly as in original.
- Do NOT replace, redraw, or invent new architecture.
- Do NOT add extra domes, towers, or elements not in original.
- Do NOT change background to different location.

ONLY REMOVE THESE SPECIFIC DISTRACTIONS:
- Remove ONLY [named person/object].
- Do NOT remove distant pedestrians, vehicles, or infrastructure.

ROTATION / ORIENTATION (if needed)
- Photo is rotated 90° counter-clockwise. Rotate first.

COMPOSITION / LIGHTING / COLOR / QUALITY
- Specific improvements, no vague "make it better."

OUTPUT:
- One photorealistic photo. Instantly recognizable subjects. No AI-smoothness.
```

## Pitfalls and Corrections

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| "Improve the photo" without face protection | Faces smoothed, skin plastic, features altered | Add "Faces must remain pixel-perfect identical. Do not touch faces." |
| "Remove distractions" without specifics | Model removes bus, replaces with building, invents architecture | Replace with "Remove ONLY the woman in pink jacket. Keep bus, signs, pedestrians." |
| "Clean up the background" | Entire background redrawn as idealized version | Use "KEEP ALL EXISTING BACKGROUND ELEMENTS" section |
| Two-photo merge request | GPT fails to composite, or creates Frankenstein image | Give ONE photo as base, second as optional reference. Do not expect true multi-image compositing. |
| "Enhance lighting" without constraints | Studio flash look, unnatural on outdoor photo | Specify "subtle warm key light from front-left, photorealistic, not studio flash" |
| Vague color instructions | Oversaturated Instagram filter | Specify "vibrant but natural" and name exact targets (domes, brick, skin) |

## One-Photo vs. Multi-Photo Workflows

**One photo (base image):**
- GPT handles well: rotation, color correction, face sharpening, specific object removal, lighting.
- Use when there is one clearly best shot.

**Two photos (merge request):**
- GPT handles poorly: true compositing of two different camera angles.
- Better approach: Pick the best single photo. Mention the second only as "reference for scene understanding" — do not ask GPT to merge them.
- If merge is truly needed, use Photoshop or inpainting tool instead.

## Model-Specific Notes

**GPT-5.5:**
- Best skin texture preservation among commercial models.
- Still requires explicit "pixel-perfect" face protection.
- Respects explicit KEEP/DO NOT REMOVE sections better than vague instructions.

**Gemini:**
- Tends to "beautify" faces more aggressively. Extra emphasis on "Do NOT smooth skin" needed.
- Good at architectural detail but may invent elements.

**restorephotos.io:**
- Excellent for technical restoration (scratches, tears, fade) without altering composition.
- Use BEFORE generative editing for old/damaged photos.

## References

- `references/prompt-examples.md` — Real session examples: aquarium portrait, Red Square travel photo, old photo restoration.
- `templates/photo-prompt.md` — Starter template for quick authoring.
