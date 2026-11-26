# Cross-Tool Pipelines

Guidance for chaining multiple AI generation tools together for consistent, production-ready output.

## Nano Banana → Kling (Character Video)

The most common pipeline: create a character in Nano Banana, animate them in Kling.

### Step 1: Character Sheet in Nano Banana

Generate 2-3 angles in a single frame:

```
Character reference sheet for a young woman with sharp cheekbones,
dark wavy hair, amber eyes, wearing a black leather jacket.
Three panels: front-facing portrait, left profile, right profile.
Neutral expression, soft studio lighting, white background.
Shot on medium format, 80mm lens, shallow depth of field.
```

**Critical**: Use identical identity language in every subsequent tool. Copy-paste the character description verbatim.

### Step 2: Export at Highest Resolution

- Kling's I2V works best with 1080p+ input
- Ensure no text overlays, watermarks, or compression artifacts
- Good lighting and contrast in the source image dramatically improves video output

### Step 3: Kling Prompt Focuses on Motion Only

**Never redescribe the subject.** Describe only movement and camera:

```
Camera slowly tracks right while maintaining focus on the central figure,
subtle wind animation affecting the subject's hair and clothing,
leaves in background sway gently, warm lighting gradually intensifies.
```

Bad (redescribes subject):
```
A young woman with dark wavy hair and amber eyes walks forward...
```

Good (motion only):
```
Subject walks forward with confident stride, camera tracks backward
maintaining medium shot, subtle head turn to the left, natural hand swing.
```

## Image → Video Consistency Rules

1. **Lock identity language** — Copy-paste character descriptors across tools
2. **Lock color palette** — Specify exact hex codes or Pantone names when possible
3. **Lock lighting** — "Soft Rembrandt key from camera-left" travels across tools
4. **Motion only in video** — I2V models hallucinate subject changes if you redescribe

## Nano Banana → Veo (Cinematic Sequence)

Veo 3.1 accepts up to 3 reference images per shot.

**Workflow**:
1. Generate establishing shot in Nano Banana (wide environmental frame)
2. Generate character close-up in Nano Banana
3. Use both as references in Veo with frame conditioning
4. Prompt Veo with cinematic shot list language

## Krea → Nano Banana (Refinement)

Use Krea's real-time canvas to explore directions, then lock the best output into Nano Banana for final generation.

1. Sketch rough composition in Krea
2. Iterate live until direction is clear
3. Take the final Krea output as a reference image for Nano Banana
4. Write a precise Nano Banana prompt that preserves the Krea composition but adds detail

## Elevenlabs → Veo / Kling (Synced Audio-Visual)

For scenes with dialogue or narration:

1. Generate audio in Elevenlabs with proper expression brackets
2. Use the audio track as a timing reference
3. In Kling 2.6, write beat-matched prompts aligned to speech cadence
4. In Veo 3.1, include dialogue in the audio section of the prompt

## Common Pitfalls

1. **Identity drift** — Using slightly different descriptions across tools breaks consistency. Copy-paste exact phrases
2. **Resolution mismatch** — 512px source → 1080p video = blurry output. Always use highest resolution
3. **Lighting discontinuity** — "golden hour" in image + "studio lighting" in video = jarring cut
4. **Over-prompting I2V** — Describing the subject again in the video prompt causes morphing and artifacts
5. **Ignoring audio sync** — In Kling 2.6, lip sync requires explicit beat markers. Don't guess timing
6. **Style contamination** — Mixing "anime" in image + "photorealistic" in video = uncanny results. Lock one aesthetic
