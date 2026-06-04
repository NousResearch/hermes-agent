# Pokemon Mew/Mewtwo reroll and empty-response handling — 2026-05-30

## Context
Nick asked for neon sticker-bomb Pokemon generations, then rerolled Mew and Mewtwo. Mew succeeded. Mewtwo repeatedly returned `empty_response` from the image backend under long, medium, and compact prompts before one ultra-compact prompt succeeded. A later fresh Mewtwo reroll again failed after several prompt variants.

## Durable lesson
For repeated reroll requests, treat the request as fresh generation, not a resend. Do not reuse the previous successful Mewtwo path as if it were a new image. If the provider returns no image after multiple prompt reductions, preserve a sidecar manifest with the failed slot and plainly report `empty_response` while optionally including the last successful prior path only as a fallback/reference.

## Prompting pattern that finally worked once
A very compact, generic visual-identity prompt reduced direct-IP dependence and produced an image:

```text
Portrait neon stickerbomb poster of a tall pale gray alien psychic monster with purple belly and huge purple tail, smooth feline head, three fingers, glowing energy orb, off-center dynamic pose, black comic outlines, glossy highlights, cyan magenta lights, graffiti stickers torn tape halftone barcode collage, text PSYCHIC CLONE, small NickZag tape tag. No photorealism, no centered pose, no five fingers.
```

## Retry guidance
1. Preserve any successful sibling image and its manifest entry.
2. Retry only the failed slot.
3. Reduce the prompt in stages: full canon-tight prompt → compact anchor-first prompt → ultra-compact generic visual-identity prompt.
4. If 3–4 attempts still return `empty_response`, stop retrying in the same turn, write/patch the manifest as failed, and report the blocker. Do not duplicate or recycle an older path as a new result.
5. If a previous successful path exists, label it clearly as the prior usable fallback, not the current reroll.
