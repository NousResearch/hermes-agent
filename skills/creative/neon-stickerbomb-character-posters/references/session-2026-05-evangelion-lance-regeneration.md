# Evangelion Unit-02 Lance of Longinus regeneration — 2026-05-12

## Trigger

Nick corrected the generated Unit-02 direction with:

`重新生成二号机，手握朗基努斯之枪`

The previous Unit-02 used dual blades / generic spear-like weapon. The correction required the Lance of Longinus to be the central readable prop.

## Prompting correction

For Unit-02 + Lance of Longinus, do not just say `spear`. Use explicit weapon mechanics:

- `hand-holding the Lance of Longinus / 朗基努斯之枪`
- `both hands grip the shaft with visible mechanical fingers`
- `long crimson organic spear running diagonally`
- `distinctive forked/double-pronged spearhead clearly visible inside the frame`
- `must read as the Lance of Longinus, not a sword or ordinary polearm`
- `the lance crosses in front but does not hide Unit-02's head, green visor, chest, and shoulders`

Negative constraints should explicitly block the common wrong outputs:

```text
no missing Lance of Longinus, no wrong weapon, no ordinary sword, no dual blade, no spear hidden outside frame
```

## Composition that worked

`TWO-HANDED LANCE OF LONGINUS DIAGONAL, CLEAR WEAPON SILHOUETTE`

Unit-02 dominates in a dynamic three-quarter assault stance. The Lance runs from lower-left foreground to upper-right, with the forked head visible. A white/cyan rim outline separates red Unit-02 from the red weapon.

## Output verified

Generated path:

`/Users/nick/.hermes/profiles/jea/cache/images/cliproxyapi_gpt-image-2-medium_20260512_224316_eva_unit_02_lance_of_longinus.png`

Vision verification passed:

- Unit-02 recognizable: red armor, green eye/visor, EVA-02 labels, angular head and shoulder pylons.
- Weapon recognizable: long crimson spear, both hands gripping shaft, clear forked/double-pronged spearhead.
- Neon sticker-bomb poster style preserved.

## Manifest handling

For repeated numbered IP batches, update the existing manifest entry for the corrected index instead of creating a disconnected new batch. In this case, index 6 in:

`/Users/nick/.hermes/profiles/jea/state/neon_evangelion_batch_20260512_195240.json`

was updated to the Lance of Longinus image so future `发布6` maps to the corrected image.
