# 2026-05 prompt-correction notes

These are condensed lessons from an image-generation session using the glossy neon sticker-bomb character poster style.

## User corrections

### `NickZag` integration

The user corrected that `NickZag` should not behave like a watermark or corner signature. It should be part of the depicted world:

- graffiti on torn tape
- printed sticker or poster fragment
- clothing patch
- prop label
- barcode / evidence tag
- lens reflection
- object surface inscription

It must share the scene's lighting, perspective, print texture, halftone, and distortion.

### Pose/layout repetition

The user flagged that character pose and poster layout were too consistent. For future batches, vary the whole composition skeleton, not just subject text:

- camera angle
- body orientation
- character placement
- dominant foreground prop
- typography direction
- negative-space distribution
- motion path
- background geometry
- `NickZag` object placement

A standard centered poster with subject filling ~75%, title behind, and labels near an edge should be treated as the failure mode.

### Prompt adherence review

When the user says a generation did not apply the prompt, do not defend from the prompt text alone. Inspect the image and distinguish:

- which requested mechanisms applied
- which mechanisms failed or softened
- what caused the drift
- what explicit negative/positive constraints should be added next

For example, a gothic character prompt can drift into a clean fantasy/gothic advertisement; correct by specifying sticker-bomb chaos, diagonal/floating composition, fashion remix details, and blocking organized text columns / clean key art.

### Mascot drift

A mascot prompt drifted into official cute merchandise / scrapbook style. Corrective language:

- block official-merchandise, theme-park, children’s-book, stationery, scrapbook, pastel-only, and clean mascot render
- keep mascot wholesome and non-humanized
- add deep black contrast, chrome foreground scanner/lens/foil, glitch UI, torn vinyl, barcode fragments, evidence/warning labels, cyan/magenta rim light, and aggressive cropped typography

## Operational note

For batch generation, assign archetypes before generation and run parallel calls only after confirming each prompt has a distinct skeleton. If image quota or token issues interrupt the batch, preserve the remaining prompts and resume only after real smoke-test verification of the active image account.
