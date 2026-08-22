# SUNO Music Generation

## When to Use

- Full songs from text descriptions
- Need structured lyrics with song architecture
- Style-specific music (genre, era, mood)
- Outcome-focused generation rather than pure genre labels

## Output Format

When generating SUNO prompts, always provide:

1. **Lyrics** in a code block (with metatags for structure)
2. **Style prompt** in JSON format (1000 character max)

### Example Output

**Lyrics:**
```
[Intro]
(Atmospheric synths fade in)

[Verse 1]
Walking through the neon rain
City lights calling my name
Every shadow tells a story
Of the dreams that went astray

[Pre-Chorus]
And I know, I know

[Chorus]
We're running out of time
Chasing what we left behind
In the glow of midnight signs
We're running out of time

[Verse 2]
Strangers pass like ghosts in frames
Nothing here will stay the same

[Bridge]
(Instrumental break)

[Outro]
Running out of time...
(Fade)
```

**Style Prompt (JSON):**
```json
{
  "genre": "synthwave, electronic pop",
  "mood": "melancholic, nostalgic, driving",
  "tempo": "120 BPM, medium-fast",
  "vocals": "ethereal female vocals, layered harmonies",
  "instruments": "analog synths, arpeggiated bass, gated reverb drums, pad textures",
  "production": "80s-influenced, polished, spacious mix",
  "references": "Chvrches meets Kavinsky, midnight drive energy"
}
```

## Metatags for Structure

Use in lyrics to guide song architecture:

- `[Intro]`, `[Outro]`
- `[Verse 1]`, `[Verse 2]`
- `[Pre-Chorus]`, `[Chorus]`
- `[Bridge]`, `[Break]`
- `[Instrumental]`, `[Solo]`
- `[Hook]`, `[Drop]`

Parentheticals for production notes: `(builds intensity)`, `(stripped back)`, `(fade out)`

## Style Prompt Components (4-7 descriptors optimal)

1. **Genre/subgenre**: Be specific (chill trap, synthwave, neo-soul)
2. **Mood**: Emotional tone (melancholic, euphoric, aggressive)
3. **Tempo**: BPM or descriptor (fast-paced, slow burn)
4. **Vocals**: Type and style (raspy male, ethereal female, spoken word)
5. **Instruments**: Key sounds (acoustic guitar, 808 bass, orchestral strings)
6. **Production**: Mix approach (lo-fi, polished, raw)
7. **References**: Style descriptors, not artist names

## Artist-Style Workarounds

Can't use artist names directly. Describe characteristics instead:

| Instead of | Use |
|------------|-----|
| Taylor Swift | Confessional pop storytelling, catchy melodic hooks, emotional narratives, polished studio sound |
| The Killers | Indie rock, synth textures, anthemic choruses, 2000s retro tone |
| Nirvana | 90s grunge, raw distorted guitars, dynamic quiet-loud shifts, angst-filled male vocals |

## Voice Tags (in style prompt)

Specify vocal characteristics:
- Gender: male vocals, female vocals
- Quality: raspy, smooth, breathy, powerful
- Style: soulful, punk, operatic, whispered

## Common Mistakes

- **Too vague**: "sad song about love" → generic output
- **Too many elements**: Overloaded prompts confuse the model
- **Artist names**: Will be filtered; use characteristics
- **Ignoring structure**: No metatags = unpredictable arrangement

## Pro Tips

- Keep style prompt under 1000 characters
- 4-7 descriptors is the sweet spot
- Use decades to refine sound ("80s synth-pop" vs "modern synth-pop")
- Outcome-focused prompting works better than pure genre labels
- Iterate: adjust one element at a time

## Common Pitfalls

1. **Artist name drops** — "like Radiohead" gets filtered. Describe the sound instead
2. **No metatags** — Without `[Verse]` / `[Chorus]` labels, structure is random
3. **Overstuffed style JSON** — 1000 char max; prioritize the 4 most important descriptors
4. **Mismatched lyrics and style** — Dark lyrics + upbeat style = confused output. Align them
5. **Ignoring production notes** — `(builds intensity)` and `(stripped back)` guide dynamics powerfully
