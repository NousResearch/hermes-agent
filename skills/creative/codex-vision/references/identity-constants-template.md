# Identity Constants Reference Template

Use this template to document entity identity constants for image generation prompts. Copy and customize it for each entity you generate images for.

## Physical Identity

```
- Gender/Age: [description]
- Build: [height, frame, posture]
- Hair: [color, length, style, texture]
- Face: [shape, distinguishing features, expression tendency]
- Skin tone: [description]
- Signature details: [glasses, jewelry, tattoos, scars, etc.]
```

## Reference Image Guidelines

Store canonical reference images in a stable directory:

```
/path/to/references/<entity>-primary.jpg     # Primary: well-lit, identity-forward
/path/to/references/<entity>-alt-1.jpg       # Alternate angle or context
```

Choose reference images that are:
- **Local, not iCloud placeholders** — verify with `file <path>` (look for `dataless`)
- **Well-lit and clear** — blurry/dark references produce poor outputs
- **Identity-forward** — clearly shows the face and build to preserve
- **Single-entity** — avoid group photos as identity references

## Prompt Patterns

### Brief (Fast, Iterative)

```
Generate a [scene]: [identity constants], [scene context with time of day]. Reference: ref.jpg. Save to output.jpg
```
Expected time: ~60s

### Verbose (Quality, Final)

```
Use case: identity-preserve
Asset type: portrait
Primary request: [full scene description with time of day]
Input images: Image 1: reference — preserve exact face, hair, build, skin tone
Style: photorealistic, [lighting description]
Constraints: lock identity from reference. Change only scene and clothing. No text, no watermark.
Output: save to output.jpg
```
Expected time: 2–3 min. May timeout at 300s — if so, condense.

## Time-Aware Scene Library

Document your entity's common scenes matched to time of day:

| Time of Day | Scene | Location | Props | Mood |
|---|---|---|---|---|
| Morning (6–11 AM) | | | | |
| Midday (11 AM–5 PM) | | | | |
| Evening (5–8 PM) | | | | |
| Night (8 PM–6 AM) | | | | |

## Multi-Entity Generation

When generating images with multiple entities in the same scene:

- Describe each entity's **distinctive visual markers** in the prompt
- Specify **relative positions** and **interactions**
- Note shared context (lighting, location, time of day)
- Use brief prompts first; escalate to verbose if quality is insufficient

## Success Tracking

Keep a log of what works:

- Which prompt styles produce the best identity preservation
- Which scenes yield the most naturalistic results
- Which reference images produce the strongest identity match
- Recurring artifacts or failure modes to avoid
