# Flux Image Generation

Black Forest Labs model. Dual text encoders (T5 + CLIP). Excellent text rendering.

## When to Use

- Text needs to appear inside the image (logos, signs, posters)
- Complex scenes with clear object relationships
- Need structured/JSON prompting for precision
- Transparent materials (glass, ice, plastic)
- Art movement or film stock emulation

## Prompt Structure

Use natural language for relationships, direct specifications for technical elements.

**Subject** + **Action** + **Style** + **Context**

Front-load important elements. Flux weights early content more heavily.

## Example

```
Red fox sitting in tall grass, wildlife documentary photography, misty dawn
```

## Layered Approach

Build complexity in stages:

1. **Foundation**: Subject + Action + Style + Context
2. **Visual layer**: Lighting, color palette, composition details
3. **Technical layer**: Camera settings, lens specs, quality markers
4. **Atmospheric layer**: Mood, emotional tone, narrative elements

## Text in Images

Flux handles text well. Best practices:

- Use quotation marks: `"The text 'OPEN' appears in red neon letters above the door"`
- Specify placement relative to other elements
- Describe typography style: "elegant serif", "bold industrial lettering"

## Content-Type Strategies

**Portraits/characters**: Start with detailed character description, then add rest
```
Elderly wizard with long white beard and piercing blue eyes, casting a spell,
fantasy art style, dramatic lighting
```

**Landscapes/scenes**: Start with environment, add subjects
```
Misty coastal cliff at blue hour, waves breathing, lone hiker in distance,
cinematic photography, 50mm lens
```

## Style References

Works well with:
- Art movements: "in the style of Van Gogh", "Impressionistic", "Art Deco"
- Film stocks: "shot on 35mm film", "Kodak Portra aesthetic"
- Specific looks: "gritty graphic novel", "watercolor painting"

## Dual Encoder Tips (T5 + CLIP)

- **T5**: Understands context, relationships, detailed descriptions
- **CLIP**: Identifies visual elements, reinforces key concepts

Use positive phrasing. Tell Flux what to include, not what to avoid.

## Common Issues

- **White background bug** (dev only): Avoid "white background" phrase, causes fuzzy output
- **No prompt weights**: Unlike Stable Diffusion, `(word)++` syntax doesn't work
- Alternative: Use "with emphasis on" or "with focus on"

## Transparent Materials

Flux handles glass/ice/plastic well:
- Explicitly state object layering: "object A in front, object B behind"
- Describe what's visible through material

## FLUX.2 JSON Prompting

Supports structured JSON for precise control:

```json
{
  "subject": "a woman in her 30s with auburn hair",
  "setting": "modern Tokyo coffee shop",
  "lighting": "soft morning light through large windows",
  "camera": "85mm lens, shallow depth of field",
  "mood": "contemplative, warm"
}
```

## Does Not Support

- Negative prompts (describe what you want, not what to avoid)
- Prompt weights/emphasis syntax

## Common Pitfalls

1. **Negative phrasing** — "no blur" confuses the model. Say "sharp focus" instead
2. **Prompt weights** — `(word)++` and `[word:0.5]` are ignored. Rephrase naturally
3. **Over-describing relationships** — T5 understands context; trust it
4. **Generic style terms** — "beautiful" is empty. "Art Deco" is specific
