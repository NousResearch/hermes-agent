# Nano Banana Pro 2

Built on Gemini 3 Pro. Understands intent, physics, and composition. Stop using tag soup.

## When to Use

- Best overall image quality needed
- Character consistency across multiple generations
- Editing existing images (color, outfit, environment)
- Text rendering under 25 characters
- Complex scenes with multiple objects
- Physics-aware edits (filling a glass, draping fabric)

## Prompt Structure

Six factors (not all required):

1. **Subject** — Who/what (a bartender, a runner)
2. **Composition** — Camera angle, framing (low angle, close-up, dutch tilt)
3. **Action** — What's happening (holding coffee, looking away)
4. **Environment** — Setting (urban street, studio backdrop)
5. **Lighting** — Light quality (golden hour, three-point, rim light)
6. **Style** — Render approach (photorealistic, Kodak Portra 400)

## Character Consistency

Lock identity first:

```
Keep the facial features of the person in the uploaded image exactly consistent.
```

For edits preserving identity:

```
Keep the facial features of the person in the uploaded image exactly consistent,
including all tattoos, piercings, glasses, and skin texture. Change only [ELEMENT].
Preserve the original lighting, background, clothing, and composition.
```

**Character sheets**: Generate 2-3 angles (front, left profile, right profile) in a single frame for multi-scene projects. This is critical for the Nano Banana → Kling pipeline.

## Strengths

- Color blocking, graphic compositions
- Architectural elements, environments
- Pose refinement from reference
- Text rendering under 25 characters
- Physics-aware edits ("fill this glass with liquid")
- Semantic in-painting (describe changes, no manual masking)
- Up to 14 reference images simultaneously

## Limitations

Pivot to post-production after 3 failures:

- Exact logo reproduction → Photoshop overlay
- Precise text placement → add in post
- Scan lines, glitch textures → layer effects
- Consistent helmets/visors in groups → composite separately

## Edit Templates

**Hair color:**
```
Change only the hair color to [COLOR] with [UNDERTONE] undertones, maintaining
the exact same hairstyle, length, wave pattern, and parting.
```

**Outfit swap:**
```
Keep the person's face and pose identical. Change their clothing to [DESCRIPTION].
Maintain the original background and lighting.
```

**Environment change:**
```
Place this exact person in [NEW ENVIRONMENT]. Maintain identical facial features
and expression. Adjust lighting to match the new scene naturally.
```

## Photography Grounding

End prompts with technical terms for photorealism:

- **Lens**: 85mm f/1.4 (portrait), 35mm (environmental), 24mm (wide)
- **Camera**: Sony A7III, Hasselblad (medium format look)
- **Film**: Kodak Portra 400 (warm), Fuji Pro 400H (neutral)
- **Lighting**: three-point, Rembrandt, butterfly, rim light separation

## Example Prompts

**Portrait:**
```
Editorial portrait of a woman in her 30s with freckles and auburn hair,
wearing an oversized camel cashmere coat, walking through a foggy London
street at dawn. Shot on 85mm f/1.4, Kodak Portra 400, soft natural light
from the left, shallow depth of field.
```

**Product:**
```
Minimalist product shot of a matte black perfume bottle on a travertine
pedestal, soft studio lighting with a single warm key light, subtle shadow
falling to the right, cream backdrop, shot on medium format, 120mm lens.
```

## Common Pitfalls

1. **Tag soup** — "8k, highly detailed, masterpiece" adds nothing. Describe the scene instead
2. **Over-describing** — Nano Banana reasons about intent. A sentence beats a list
3. **Ignoring physics** — "liquid in glass" works better than "glass with water"
4. **Forgetting reference limits** — 14 refs max; prioritize the most important
