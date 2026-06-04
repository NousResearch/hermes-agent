# Evangelion Angels — canon fidelity + stickerbomb density balance (2026-05-16)

## Trigger
Nick rejected two opposite failure modes in the Evangelion Angels 1–4 neon batch:

1. **Neon/stickerbomb strong, shape wrong** — attractive posters, but Angel silhouettes drifted too far from TV/anime source shapes.
2. **Canon-tight, stickerbomb lost** — silhouettes improved, but the result became sparse border/UI archive art and lost the neon skill’s core stickerbomb language.

The durable lesson: for silhouette-driven known-IP creatures/mecha, **canon fidelity and stickerbomb density are both hard requirements**. Do not solve one by sacrificing the other.

## Correct balance rule
Use this priority frame in prompts:

```text
CANON + STICKERBOMB BALANCE: preserve the original Angel silhouette, face/mask/core, body color, and signature anatomy as the clean readable center. Restore dense neon stickerbomb as the core style through background, borders, huge cropped typography, torn vinyl decals, warning tape, barcode fragments, graffiti tags, hazard/evidence labels, halftone cutouts, chromatic split, and a few edge-overlap decals that do not hide the face/core.
```

Do **not** reduce stickerbomb to a thin UI frame. Do **not** cover or redesign the Angel body with random stickers. The subject should read without labels; the poster should still feel like dense neon stickerbomb if the subject is ignored.

## Working prompt pattern
Shorter prompts were more reliable than long dense prompts on the local CLIProxyAPI `/v1/responses` path. A proven shape:

```text
Vertical 3:4 glossy neon cyber-pop sticker-bomb poster of [ANGEL NAME], [canon anchor sentence]. Dense torn warning labels, barcode tags, cropped [NAME] typography, cyan magenta neon rim light, thick black manga ink, glossy highlights, halftone, graffiti, NERV hazard stickers, [subject-specific negatives], no watermark.
```

Keep `json.dumps(payload)` default JSON formatting if a previous compact script hung unexpectedly; one-image smoke tests can confirm the payload style before batch generation.

## Angel-specific corrections

### Adam
- Failure: embryo/ghost/cocoon, then sparse white giant archive poster.
- Better anchor: gigantic white humanoid giant of light, upright pale glowing body, smooth blank head, long arms, red core/halo, dark containment setting.
- Keep stickerbomb dense with huge cropped `ADAM`, warnings, barcodes, torn labels.
- Avoid embryo, ghost, baby, cocoon, wings, armor, superhero muscles.

### Lilith
- Better anchor: massive white faceless giant, open cruciform arms, multi-eye mask symbol, black cables/restraints, Terminal Dogma, red LCL/containment fluid.
- Dense stickers can live on side walls, lower foreground labels, cable tags, warning tape, and dossier cards.
- Watch for mild feminine body curves; if too strong, reroll toward heavy plain white non-sexual sacred-corpse body.
- Avoid sexy android, goddess pose, human face.

### Sachiel
- Failure: bird beak / plague-doctor mask.
- Better anchor: tall dark purple-black simple humanoid, broad shoulders, long arms, chest red spherical core, flat/simple white mask with exactly two black round eye holes and a short blunt face plate.
- Dense stickerbomb worked well when surrounding the subject as warning labels and huge `SACHIEL` typography.
- Avoid bird beak, plague doctor, horns, wings, demon, extra masks, armor.

### Shamshel
- Failure: dense version became a mechanical worm/centipede; energy whips read as a closed decorative ring.
- Better anchor: short/simple magenta segmented aerial body, only several large smooth rounded pink segments, small blunt front cap with red core, exactly two separate glowing pink energy-whip arms clearly attached left/right and sweeping outward.
- Dense stickerbomb should surround the whip paths and background, not turn the body into a busy mechanical insect.
- Avoid legs, dragon head, centipede, many segments, mechanical bug, skull head.

## QC language
Do not say “passed” unless both are true:

- **Stickerbomb core**: dense torn decals, huge cropped type, barcodes, warning labels, graffiti/halftone/chromatic chaos are visually strong, not merely a neat border.
- **Canon silhouette**: subject reads as the named Angel without labels; face/core/body anchors are not hidden or redesigned.

Use explicit results:

- `Pass: stickerbomb strong + canon anchors usable`.
- `Fail: stickerbomb strong, shape drift significant`.
- `Fail: canon closer, stickerbomb core missing`.

## Manifest/source-of-truth note
For reroll chains, create a fresh manifest for each rejected balance mode rather than overwriting the accepted/failed evidence. Preserve prior paths for comparison, but report only the current usable manifest to Nick.
