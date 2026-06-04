# EVA Units 05–08 glossy dense stickerbomb generation — 2026-05-13

## Context

Nick requested Evangelion Units 05, 06, 07, and 08 after a corrected Unit-01/00/02/03/04 direction: **premium glossy mecha material + dense neon sticker-bomb core**. The previous correction still applies: do not trade sticker density away to get cleaner material; keep the mecha glossy and move dense decals/typography into background, borders, cable arcs, weapon surfaces, and light armor-edge overlaps while protecting head/chest readability.

## Generation targets that worked

Use these identity anchors when generating this batch or similar Rebuild-era EVA mecha:

1. **Unit-05 / 五号机 / Provisional Unit-05**
   - Green/white provisional mecha.
   - Experimental exposed machinery.
   - Wheel/roller leg assemblies must be visible; do not let it become ordinary humanoid legs only.
   - Long spear/drill-lance arm or foreground track-lance diagonal.
   - Palette: toxic green, white armor, black machinery, orange/yellow hazard accents, cyan/magenta sticker lighting.

2. **Mark.06 / 六号机**
   - Deep navy/blue armor with gold/yellow accents.
   - Long horn-like EVA head silhouette, elegant mysterious posture.
   - Moon/lunar halo or sealed moon-test context.
   - Long spear-like weapon can be a left-edge graphic spine; if Nick asks for stronger weapon fidelity, require it to be held by the unit.
   - Palette: midnight blue, gold trim, moon white, cyan/magenta edge split.

3. **Unit-07 / 七号机**
   - No single strong canon public silhouette; treat as a plausible mass-production/test-body concept when unspecified.
   - White/silver skeletal EVA-like production unit.
   - Scanner wall, repeated ghost silhouettes, serial stickers, `MASS PRODUCTION / TEST BODY` language.
   - Make clear that this is concept interpretation, not strict canon replication.

4. **Unit-08 / 八号机**
   - Hot pink/magenta EVA with white/black armor accents and green eye/visor glow.
   - Sniper/marksman role works well: foreground rifle barrel/scope with HUD reticle, stickers on weapon surface.
   - Protect face/chest from the scope; no cute magical-girl redesign.

## Operational lesson

CLIProxyAPI sometimes fails individual items with:

```text
Tool choice 'image_generation' not found in 'tools' parameter.
```

For an interrupted batch, keep the sidecar manifest as source of truth, preserve successful paths, and retry only failed indices. A shorter retry payload without explicit `tool_choice` succeeded for Unit-05 and Mark.06 while preserving already completed Unit-07 and Unit-08.

## QC expectations

A passing EVA neon mecha result should satisfy all three:

- unit identity is readable from palette/silhouette/weapon/context, not only from text labels;
- mecha material remains glossy/lacquered/chrome/highlighted, not muddy industrial grunge;
- dense neon sticker-bomb typography/decals/barcodes/tape/halftone chaos remains visible and central to the style.

Common acceptable caveats: small pseudo-text, heavy information density, and style-driven armor redesign. Common failures: generic Gundam face, foreground prop covering head/chest, clean official key art, or sparse sticker background.
