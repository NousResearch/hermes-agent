# Kling Video Models

## Model Selection

| Model | Best For | Key Feature |
|-------|----------|-------------|
| Kling 3.0 | Character-driven narratives | 6-shot storyboards, native audio, 10s durations |
| Kling 2.6 | Music videos, performance | Audio-visual choreographer, beat-matched, lip sync |
| Kling O1 | UGC, product demos | Grounded realism, hands-on demonstrations |
| Kling 2.5 Turbo | Quick iterations | Speed-optimized, 3-4 elements max |

## When to Use

- Character consistency across multiple shots (Kling 3.0)
- Beat-synchronized motion (Kling 2.6)
- Product demonstrations or UGC-style content (Kling O1)
- Image-to-video with high fidelity to source image
- Complex camera movements (tracking, whip-pans, crash zooms)

## Universal Formula

Four parts: **Subject** (specific details) + **Action** (precise movement) + **Context** (3-5 elements) + **Style** (camera, lighting, mood)

## Kling 3.0 Specifics

**Multi-shot generation**: Up to 6 shots in single output. Label shots explicitly:

```
Shot 1: [framing], [subject], [motion]
Shot 2: [framing], [subject], [motion]
```

Understands: profile shots, macro close-ups, tracking shots, POV, shot-reverse-shot

**Native audio**: Explicitly indicate who speaks when in multi-character scenes

**Motion instructions**: Describe camera behavior over time (tracking, following, freezing, panning)

**Long takes**: Define camera's relationship to subject ("staying in medium shot", "freezing when subject pauses")

## Kling 2.6 Specifics

**Beat-matched prompting** for audio sync:

```
Beat 0-4s: Slow motion, character walks toward camera
Beat 4s: EXPLOSION of color, rapid zoom (synced to bass drop)
Beat 8s: Character lip syncs vocal line
```

**Audio-visual workflow**:
1. Generate/upload audio track
2. Identify beat timestamps
3. Write timeline script matching markers
4. Upload audio with character reference for lip sync

**Capabilities**: 5-7 distinct environmental elements, complex lighting setups, nuanced micro-expressions

## Kling O1 Specifics

Best for: Hands-on demos, product-focused clips, UGC aesthetic

## Image-to-Video (All Versions)

**Critical**: Don't redescribe what's in the image. Focus only on motion:

```
Camera slowly tracks right while maintaining focus on the central figure,
subtle wind animation affecting the subject's hair and clothing,
leaves in background sway gently, warm lighting gradually intensifies
```

**Image prep**: High resolution (1080p+), clear composition, no text overlays/watermarks, good lighting/contrast

## Camera Language

Motion verbs matter. Use cinematic phrasing:

- dolly push, whip-pan, shoulder-cam drift, crash zoom, snap focus
- Avoid generic: "moves", "goes"

**Narrative-motivated movement**: Connect camera to story purpose
- Pan-to-reveal: slow pan reveals information
- Push-in: heightens pressure/tension
- Pull-out: creates isolation from intimacy

## Common Failures & Fixes

| Problem | Fix |
|---------|-----|
| Unwanted distortion | Reduce complexity, specify "stable camera movement" |
| 99% hangs | Include end-state descriptions |
| Physics issues | Use spatial relationship language |
| Objects changing mid-video | Add continuity notes: "preserve shape", "keep colors consistent" |

## Pro Tips

- Specify camera distance and framing to prevent style confusion
- Don't mix lighting terms ("golden hour" + "studio lighting")
- Use film stock references: "VHS camcorder aesthetic", "shot on 35mm"
- Color language should be literal but emotive: "cool blue haze", "amber nightclub strobe"
- For I2V, the prompt should read like a cinematographer's shot list, not a description
