# Session note — Evangelion Angels 1–4 canon-tight neon generation

## Trigger

Nick requested neon generation for Evangelion Angels 1–4: Adam, Lilith, Sachiel, and Shamshel.

The important session lesson is that known-IP creature subjects need **canon silhouette first, neon sticker-bomb second**. The prior Evangelion Angels references already warned about this, and this run reinforced the rule with successful prompt/retry patterns.

## Working priority

Use this order for Angel prompts:

1. Canon silhouette / shape fidelity.
2. Key identity anatomy and colors.
3. Clear subject body, not hidden by labels.
4. Put neon sticker-bomb density in the background, borders, tape strips, labels, and warning UI.
5. Optional in-scene `NickZag` tape/patch tag.

Prompt phrase that worked well:

```text
CANON-TIGHT SILHOUETTE PRIORITY: subject fidelity first, neon sticker-bomb treatment second. Keep the Angel body clean, centered enough to read, and not covered by labels. Put dense sticker-bomb chaos in background, borders, warning tape, labels, typography, cables and containment UI.
```

## Subject anchors used

### Adam

- Pale/white primordial seed-life or embryo-source entity.
- Smooth, eerie, monumental, non-human.
- Minimal/featureless head, red core glow, halo/containment ring.
- Avoid handsome angel, armored hero, superhero muscles, wings.

Provider note: an explicit `First Angel Adam from Evangelion` prompt repeatedly returned no image. A safer wording succeeded by describing `a primordial pale white origin entity called FIRST ANGEL ADAM` and avoiding the direct IP phrase near the start.

Successful image:

`/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2_20260516_004629_first_angel_adam_primordial_seed_neon_canon_retry2.png`

QC: passed. Style close high, shape drift low. It reads as a white primordial seed/embryo-source entity rather than a human angel or armored hero.

### Lilith

- Massive white faceless restrained terminal entity.
- Open-arm containment posture, black restraints/cables, red ribbons/core accents.
- Ancient non-human mother-of-life presence.
- Avoid sexy android, goddess pin-up, human eyes/mouth, feminine body emphasis.

Successful image:

`/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2_20260516_003852_second_angel_lilith_restrained_terminal_neon_canon.png`

QC: passed. Style close high, shape drift low-to-mid. It keeps giant faceless restrained-Lilith identity; added bio-organic details are acceptable.

### Sachiel

- Tall dark purple-black humanoid Angel.
- One clean white beak/skull-like mask with two black eye holes.
- Red spherical chest core.
- Broad/simple shoulders, long arms.
- Avoid multiple masks, horns, wings, robot armor, generic demon drift.

Successful image:

`/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2_20260516_004206_third_angel_sachiel_mask_core_neon_canon.png`

QC: passed. Style close high; shape drift light-to-medium because the mask became more bird/plague-doctor-like and shoulders were rounder, but core Sachiel anchors were intact.

### Shamshel

- Long smooth magenta/pink segmented aerial body.
- Small front/head with red core.
- Exactly two flexible pink energy-whip arms connected to front/body and forming a large arc/ring.
- Avoid legs, dragon head, mechanical centipede armor, hard spikes, humanoid torso.

Successful image:

`/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2_20260516_004431_fourth_angel_shamshel_energy_whip_neon_canon_retry.png`

QC: passed. Style close high, shape drift low. The body/whip silhouette stayed readable and stickerbomb elements remained mostly border/background.

## Manifest

`/Users/nick/.hermes/profiles/jea/state/neon_eva_angels_01_04_canon_tight_20260515.json`

## Operational lesson

When one Angel fails or drifts, retry only that index and preserve the batch manifest mapping. For provider no-image failures, compact the prompt and, if needed, move direct IP naming out of the first sentence while preserving visible typography such as `FIRST ANGEL ADAM`.
