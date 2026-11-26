# Elevenlabs Text-to-Speech

Eleven v3 is the most expressive TTS model. Uses audio tags in square brackets for delivery control.

## When to Use

- Expressive narration or character voices
- Need emotional variation within a single sentence
- Sound effects or ambient audio mixed with speech
- Accent switching mid-sentence
- Audiobooks, video voiceover, game dialogue

## Audio Tags (Expression Brackets)

Tags are inline, lowercase, wrapped in square brackets. Can be combined.

### Emotion Tags

Set emotional tone: `[sad]`, `[angry]`, `[happily]`, `[sorrowful]`, `[excited]`, `[nervous]`

### Delivery Tags

Control how speech is delivered:
- `[whispers]` - quiet, intimate
- `[shouts]` - loud, intense
- `[instructional]` - teaching tone
- `[sarcastic]` - ironic delivery

### Reaction Tags

Natural, unscripted moments:
- `[laughs]`, `[sighs]`, `[clears throat]`
- `[gasps]`, `[groans]`

### Sound Events

Non-speech audio in scene:
- `[gunshot]`, `[explosion]`, `[applause]`
- `[leaves rustling]`, `[footsteps]`

### Accent Tags

Switch regions: `[American accent]`, `[British accent]`, `[Southern US accent]`

## Example Usage

```
[whispers] Something's coming... [sighs] I can feel it.
```

```
[excited] Oh my god, this is amazing! [laughs] I can't believe it worked!
```

```
[instructional] First, you'll want to gather your materials. [pause] Then, carefully measure out two cups of flour.
```

## Combining Tags

Layer multiple effects:

```
[nervously] I... I don't think we should do this. [whispers] They might hear us.
```

## Pause Control

- Use `<break time="x.xs" />` for precise pauses up to 3 seconds (not supported in v3)
- Alternatives: dashes (- or —) for short pauses, ellipses (...) for hesitant tone
- Natural punctuation affects pacing

## Pronunciation Control

SSML phoneme tags for specific words (Flash v2, Turbo v2, English v1 only):
- CMU Arpabet (recommended)
- IPA

## Text Normalization

Expand for clarity:
- Numbers: "123" → "one hundred twenty-three"
- Dates: "01/02/2023" → "January second, twenty twenty-three"
- Abbreviations: write out fully

## Voice Settings

- **Stability**: Higher = more consistent, lower = more expressive variation
- **Similarity boost**: How closely to match voice characteristics
- **Speed**: 0.7 (slow) to 1.2 (fast), default 1.0

## Model Selection

- **Eleven v3**: Most expressive, requires more prompt engineering, best for video/audiobooks
- **Flash v2.5**: Ultra-low 75ms latency, real-time applications
- **Multilingual v2**: Highest quality, nuanced expression, 32 languages

## Tips

- Emotional context from text itself: "she said excitedly" influences delivery
- Exclamation marks affect speech emotion
- Write in natural, narrative style for better pacing
- Test different voices; each handles tags differently

## Common Pitfalls

1. **Over-tagging** — Too many tags in one sentence sound robotic. Space them out
2. **Wrong model for tags** — v3 handles tags best; Flash v2.5 ignores many
3. **Inconsistent accent** — Switching accents mid-sentence only works with compatible voices
4. **Literal punctuation** — Write "dot" or "period" if you want the word spoken, not a pause
5. **Ignoring stability** — Low stability + long text = voice drift. Lock stability for narration
