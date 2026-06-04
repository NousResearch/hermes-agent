# Session note — Evangelion Angels canon-fidelity correction

## Trigger

Nick corrected the Evangelion Angels 1–4 neon batch: the outputs were in the right neon/sticker-bomb surface style, but the Angels drifted too far from their TV/anime silhouettes.

The correction matters for all known-IP creature/mecha batches: **style success is not enough when the subject identity is shape-driven**. Big labels like `SACHIEL`, `LILITH`, or `SHAMSHEL` cannot compensate for a wrong silhouette.

## What went wrong

Earlier rerolls over-applied the neon poster language:

- Lilith became a white sexy/android/fashion humanoid instead of a massive faceless restrained terminal entity.
- Sachiel became a generic dark demon/monster with too many masks and overlong horror limbs.
- Shamshel became a mechanical dragon/centipede/worm instead of a simple magenta segmented aerial body with two energy-whip arms.
- Adam became a polished heroic/muscular faceless angel instead of a more primordial, eerie, non-human seed/giant-of-light entity.

The poster style was loud and attractive, but it redesigned the subjects.

## Correct prompt priority

For Evangelion Angels, use this order:

1. Canon silhouette / shape fidelity.
2. Key identity anatomy and color.
3. Clear readable subject unobscured by labels.
4. Neon sticker-bomb background, borders, typography, warning tape.
5. Optional in-scene `NickZag` tag.

Write this explicitly near the top of the prompt:

```text
CANON-TIGHT SILHOUETTE PRIORITY: subject fidelity first, neon sticker-bomb treatment second. Keep the Angel body clean and immediately recognizable; put sticker density in the background, borders, tape strips, labels, and typography, not covering or redesigning the Angel.
```

## Angel-specific fidelity anchors

### Adam

- Pale / white primordial giant or embryo-source entity.
- Smooth, eerie, non-human, monumental.
- Featureless or minimal head; sacred-science aura, halo/core allowed.
- Avoid muscular superhero anatomy, ornate angel armor, handsome faceless mannequin, wings/horns/insect parts.

### Lilith

- Massive white faceless restrained giant.
- Cruciform/open-arm terminal containment posture.
- Mask/head should feel strange, ancient, non-human; if possible, emphasize original-like mask/multi-eye symbolism.
- Thick, heavy, sacred corpse / mother-of-life presence.
- Avoid sexy android, goddess pin-up, slim cyber fashion body, human face/eyes/mouth, feminine anatomy emphasis.

### Sachiel

- Tall dark purple-black humanoid Angel.
- One clean white beak/skull-like mask with two black eye holes.
- Visible red spherical core in chest.
- Broad simple shoulders, long arms, non-human but not generic demon.
- Avoid multiple masks/heads, horns, wings, claws dominating, armor plates, muscular demon anatomy, swords/robot parts.

### Shamshel

- Long smooth magenta/pink segmented aerial body.
- Small simple front/head/body with red core area.
- Two flexible pink energy-whip arms clearly connected to the front/body.
- Simple soft serpent/whip Angel silhouette.
- Avoid legs, hard spikes, dragon head, skull/bird mask, mechanical centipede armor, humanoid torso, worm-only body.

## QC rule

After generation, QC must answer shape fidelity directly, not just style:

- Does the silhouette still read as the named Angel without relying on text labels?
- Which canon anchors are present?
- Which overdesign drift occurred?
- Is the sticker-bomb density helping the image or covering the subject?
- Is this `style close, shape drift` or genuinely usable?

Use direct labels in the report, e.g.:

- `Style strong, shape drift significant`.
- `Recognizable Sachiel, medium fidelity; over-humanized proportions`.
- `Shamshel has the color/body/whips but is too worm/centipede-like`.

## Operational lesson

If Nick says an IP creature/mecha image differs too much from the source, do **not** just intensify style adjectives. Reroll with a reduced, canon-tight prompt and move visual chaos to background/borders. If only one or two subjects drift, retry those indices only and preserve the batch manifest mapping.