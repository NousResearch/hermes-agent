# Session notes: Gintama and Dragon Ball canon rerolls — 2026-05-17

## Why this matters

Nick repeatedly corrected neon sticker-bomb outputs where the surface style was acceptable but canonical identity anchors drifted. Future generations should treat outfit/hair/prop silhouette as hard constraints before applying the neon remix.

## Gintama corrections

### Kamui / 神威

Initial rerolls overfit generic white/black/red/yellow martial clothing. Nick corrected the outfit:

- orange hair tied into one long braid/queue
- bright blue eyes
- deep charcoal / dark-gray Chinese-style mandarin-collar robe or jacket
- frog-button / Chinese fastener front detail
- clearly visible white cape/cloak draped over the shoulders
- red umbrella weapon

Prompt lesson: put a hard `CANON COSTUME PRIORITY FIRST` line before any neon/stickerbomb wording. Use `body visible from knees up so the robe and cape are readable`; keep the umbrella behind or diagonal so it does not replace/obscure the clothes. Negatives should explicitly block `white martial outfit`, `red/yellow coat`, `modern jacket`, and `missing cape`.

### Shimura Tae / 志村妙 / Otae

The main failure mode was hairstyle drift. Use original simple dark straight hair with side-parted bangs/fringe, gathered back/behind shoulders with a neat low tied-back feel. Block bob, twin tails, wild curls, short bangs, high ponytail, and fashion-layered hair. Keep purple kimono and gentle-but-dangerous big-sister aura.

### Elizabeth / 伊丽莎白

Keep as a plainly non-humanized white mascot silhouette first. Do not make a humanized fashion figure.

## Dragon Ball villain batch lessons

For Frieza / Perfect Cell / Majin Buu, the canon silhouette and palette should be protected the same way as mecha/creatures:

- Frieza: compact sleek white alien tyrant, glossy purple head dome/plates, long white tail, red eyes, death-ball finger pose. Block gold form, bulky monster, missing tail, wrong palette.
- Perfect Cell: tall green bio-android, black spotted carapace, black torso panels, pale face, purple cheek marks, crown head crest, black side fins, wing-like back plates, spotted tail. Block generic bug monster, robot armor, missing spots/wing plates.
- Majin Buu: distinguish requested variant. Fat Buu = rounded pink body, head antenna, black-gold vest, white baggy pants, yellow gloves/boots. Super/Slim Buu = tall lean pink body, long antenna, narrow angry eyes, black-gold vest, white baggy pants with M belt, yellow gloves/boots. Block the other Buu forms when one form is requested.

## Operational lesson

Long three-image batches may hang after partial completion. Keep sidecar manifest as source of truth, inspect it after interruption, skip completed items, and retry only pending indices with shorter prompts. If a pending index still hangs in batch mode, generate that single item with a compact one-image script and patch the same manifest index.
