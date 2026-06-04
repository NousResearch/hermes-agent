# Demon Slayer Mitsuri whip-sword continuity correction — 2026-05

## Context
Nick liked the overall neon sticker-bomb Mitsuri / Love Hashira poster direction, but rejected versions where the flexible sword did not physically connect correctly to the hilt.

## Durable lesson
For Mitsuri Kanroji / Love Hashira, the distinctive weapon is not just a decorative ribbon. The image must show a readable katana-style handle/tsuka and guard, with the thin flexible blade emerging from the **front side of the guard** as one continuous metallic whip-blade.

## Failure modes observed
- The sword hilt and ribbon blade appeared as separate objects.
- The blade connected to the bottom/end of the handle instead of from the guard/front side.
- Hair/ribbons/stickers visually competed with the blade and made the weapon connection ambiguous.
- A generated result could look good overall while still failing because the hilt/blade junction was wrong.
- `vision_analyze` may say “connected” too broadly; ask specifically whether the blade exits from the front side of the guard versus the handle bottom/behind.

## Prompting pattern that should be tried next
Use a composition that makes the weapon junction deliberately readable before maximizing body dynamism:

```text
Mitsuri Kanroji / Love Hashira, glossy neon sticker-bomb poster. Canon anchors: long pink hair with lime-green tips, green eyes, white haori, dark uniform, cheerful brave expression; covered and non-sexualized.

WEAPON-JUNCTION PRIORITY: place the katana-style handle, flower-shaped guard, both hands, and first segment of the blade in a clean readable area near the visual center. The thin flexible metallic whip-blade emerges from the FRONT side of the guard as one continuous attached blade, then curves outward into an S-curve ribbon slash. Keep hair ribbons, stickers, and typography away from the hilt/guard/blade junction. Do not let the blade emerge from the handle bottom, behind the handle, or as a detached floating ribbon.

Use dense sticker-bomb collage around borders/background: torn vinyl decals, barcode strips, graffiti tags, halftone dots, cyan/magenta rim light, huge angled text MITSURI / LOVE HASHIRA / WHIP BLADE. NickZag appears once as a small background sticker, not near the weapon junction.
```

## QC question
Ask vision to judge the exact failure mode:

```text
Check only the Love Hashira weapon junction. Are the handle/guard/both hands visible, and does the flexible blade emerge from the FRONT side of the guard as one continuous blade? Or does it connect from the handle bottom/behind, detach, or get confused with hair/ribbons? Brief verdict.
```

## Workflow note
If the overall poster is good but weapon anatomy is wrong, do **not** update the manifest to that version as final. Keep it as a failed candidate and retry the weapon-junction-first prompt. If repeated generation fails, narrow the composition to half-body/three-quarter with the hilt/guard/blade junction unobscured, and push sticker density to the frame/background.
