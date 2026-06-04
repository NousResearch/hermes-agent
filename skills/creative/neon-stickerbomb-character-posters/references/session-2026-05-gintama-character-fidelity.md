# Session note — Gintama neon sticker-bomb character fidelity (2026-05-17)

## Context

Nick generated multiple Gintama characters in the neon sticker-bomb poster style. The style surface was strong, but two repeated corrections exposed a durable lesson for known-IP human/mascot prompts: exact canon costume/hair anchors must be protected before adding fashion/sticker-bomb remix details.

## Corrections that mattered

### Kamui / 神威

Nick corrected Kamui repeatedly. The bad tendency was to drift into a generic white/black martial outfit or red/yellow accented outfit. The corrected anchor is:

- bright orange hair tied into one long braid/queue
- bright blue eyes
- cheerful/predatory Yato smile
- **deep charcoal/dark-gray Chinese-style mandarin-collar robe/jacket/changshan**
- visible frog-button fasteners down the front
- loose dark trousers
- **large clean white cape/cloak draped over both shoulders and flowing behind him**
- red umbrella weapon as a prop, not as a replacement for costume fidelity

Hard negatives that helped:

- no white martial suit as the main outfit
- no red/yellow jacket or modern jacket
- no missing white cape
- no missing braid
- keep body visible from knees/torso up so robe + cape are readable
- put sticker-bomb density in background/borders, not over the corrected costume

### Shimura Tae / 志村妙 / Otae

Nick corrected the hairstyle. The failure mode was model-fashion drift: freeform layered hair, incorrect bangs, or generic anime styling. The corrected anchor is:

- long dark brown/black hair
- original simple straight style
- smooth side-parted front bangs/fringe framing the forehead
- hair gathered neatly back/behind shoulders with a low tied-back feel
- purple kimono / traditional outfit
- gentle refined smile with hidden menace
- optional burnt-food gag / serving tray / wooden sword

Hard negatives that helped:

- no short hair
- no curly hair
- no twin buns
- no high ponytail
- no messy layered fashion hair

## Class-level lesson

For known-IP humans in this neon style, labels and overall poster quality do not compensate for wrong costume/hair. If Nick corrects a specific outfit or hair detail, reroll with a `CANON-FIDELITY PRIORITY` line and move the neon density to background, borders, typography, props, and tape layers. Keep the corrected costume/hair large, clean, and unobscured.

Use direct language in the prompt:

```text
CANON COSTUME/HAIR PRIORITY FIRST: [exact corrected anchors]. Character must be immediately recognizable without text. Keep [costume/hair anchors] visible and unobscured. Neon stickerbomb style lives mostly in background/borders; character fidelity first, stickerbomb second.
```

## Operational note

When a long multi-image batch hangs before stdout, a one-image smoke-test / serial generation path with compact prompts can still work. Preserve the manifest and patch only the rerolled indices, rather than starting a fresh untracked batch.
