# Veo 3.1

Google's video generation model. Native audio, multi-shot continuity, up to 3 reference images.

## When to Use

- Cinematic video with professional camera movement
- Scenes requiring native generated audio (dialogue, ambience, SFX)
- Multi-shot sequences with continuity
- Need to extend existing clips
- Frame-conditioned transitions between shots

## Capabilities

- **Resolution**: 720p or 1080p
- **Duration**: 4/6/8 seconds per shot
- **Aspect ratio**: 16:9 or 9:16
- **Audio**: Native generation (dialogue, ambient, sound effects)
- **Reference images**: Up to 3 per shot for consistency
- **Extension**: Extend clips, frame conditioning for transitions

## Prompt Structure

Six elements for consistent results:

1. **Scene** — One clear sentence describing action and vibe
2. **Visual style** — Cinematic, realistic, animated, surreal
3. **Camera movement** — Static, slow pan, tracking, aerial
4. **Main subject** — Who/what camera focuses on
5. **Background** — Setting and era
6. **Lighting and mood** — Emotional tone through lighting

## Cinematography Language

The model responds strongly to film vocabulary.

**Camera movement**: Dolly shot, tracking shot, crane shot, aerial view, slow pan, POV shot

**Composition**: Wide shot, close-up, extreme close-up, low angle, two-shot

**Lens/focus**: Shallow depth of field, wide-angle, soft focus, macro, deep focus

## Audio Prompting

Use separate sentences for audio. Categories:

- **Sound effects**: "the sound of a phone ringing", "water splashing"
- **Ambient noise**: "city traffic and distant sirens", "waves crashing"
- **Dialogue**: "the man says: [dialogue]", "voiceover with British accent"

## Multi-Shot Continuity

- Use frame conditioning: generate shot N to clean last frame, use that frame to start N+1
- Stable camera (tripod/dolly) for transitions reduces model freedom, improves cuts
- Repeat palette descriptors and time-of-day across connected shots
- Prepare 6-8 expressions/poses as references for character consistency

## Example Template

```
[Shot intent]: A [subject] in [location], [time of day].
Camera: [camera move], [lens/focal length], [framing].
Lighting: [key light quality], [practicals/accents].
Motion cues: [subject action], [environmental movement].
Aesthetic: [color palette], [film stock], [era reference].
Audio: [ambient], [dialogue if any], [sound effects].
Aspect/Duration: [ratio], [seconds].
```

## Style References

- **Photorealistic**: "ultra-realistic rendering", "shot on 8K camera"
- **Cinematic**: "cinematic film look", "shot on 35mm film", "anamorphic widescreen"
- **Animation**: "Japanese anime style", "Pixar-like 3D", "claymation"
- **Vintage**: "sepia tone", "1980s vaporwave", "Kodak 2383 print"

## Common Pitfalls

1. **Overstuffed prompts** — Naming 7+ locations creates chaos, not direction
2. **Vibes only, no physics** — Add friction (wind, gravity, fabric) for realism
3. **Missing negatives** — Explicitly ban artifacts: "no double faces, no rubber hands"
4. **Ignoring audio** — Veo 3.1's native audio is a superpower. Use it
5. **Static descriptions** — "a person standing" produces boring video. Always include motion
