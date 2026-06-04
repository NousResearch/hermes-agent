# Session: EVA extended mecha generation + numbered publication (2026-05-13)

## What happened

Nick continued an Evangelion neon sticker-bomb batch beyond Unit-00/01/02/03/04 and asked for:

- Unit-05 / 五号机
- Mark.06 / 六号机
- Unit-07 / 七号机
- Unit-08 / 八号机

The earlier correction still applied: premium glossy mecha material is necessary, but the neon skill core is **dense sticker-bomb collage**. A clean glossy mecha render without dense decals/typography/torn labels is a fail.

## Prompting pattern that worked

Use one explicit per-unit identity block plus the fixed balance block:

```text
Core style balance: premium glossy mecha material PLUS dense neon sticker-bomb visual core. Keep stickers dense in background, borders, cable arcs, weapon surfaces and slight armor-edge overlaps, but never hide the head/chest/core silhouette.
```

Unit-specific cues:

1. **Unit-05 / Provisional Unit-05**
   - green/white provisional EVA
   - exposed experimental machinery
   - roller/wheel leg assemblies
   - long spear/drill-lance arm
   - orange/yellow hazard accents
   - composition: low-angle track/lance diagonal
   - negatives: no ordinary humanoid legs only, no missing wheel system, no missing spear/lance arm

2. **Mark.06 / Unit-06**
   - midnight blue/navy armor with gold trim
   - long horn-like head silhouette
   - lunar/moon halo, sealed/test aura
   - long spear-like weapon
   - composition: moon-halo side split / spear descent
   - negatives: no Gundam V-fin face, no generic blue knight, no missing gold accents

3. **Unit-07**
   - no universally stable canon visual; treat as a mass-production/test-body concept
   - white/silver skeletal EVA-like body
   - black joint gaps, exposed torso/rib machinery
   - scanner wall and repeated ghost silhouettes
   - composition: multiple test-silhouette scanner wall
   - caveat in reporting: concept-rationalized EVA production unit rather than strict canon replica

4. **Unit-08**
   - pink/magenta EVA
   - white/black armor accents
   - green eyes/visor
   - marksman/sniper combat role
   - foreground sniper scope/rifle barrel with targeting decals
   - negatives: no generic pink Gundam, no face hidden behind rifle, no missing sniper/scope motif

## Tool/provider quirk

The direct CLIProxyAPI Responses request can intermittently fail on some items with:

```text
HTTP 400: Tool choice 'image_generation' not found in 'tools' parameter
```

Successful recovery: preserve the sidecar manifest, skip existing `status: done` items, and retry only failed indices with a shorter payload that omits `tool_choice` while keeping the same image_generation tool. In this session, indices 3 and 4 succeeded first, indices 1 and 2 failed with this error, then 1/2 succeeded on retry without `tool_choice`.

Do not regenerate successful siblings after this failure.

## Manifest/source of truth

Save and preserve a manifest like:

```text
/Users/nick/.hermes/profiles/jea/state/neon_eva_units_05_06_07_08_dense_glossy_20260513.json
```

Each item should contain index, title, slug, full prompt, status, path, elapsed time, provider/model/size/quality. This keeps later `发布1234` selection mapping exact.

## QC standard

Pass only if both are true:

1. Unit identity is recognizable from color/silhouette/weapon/role cues, not only from big labels.
2. The dense neon sticker-bomb core is visible: torn decals, warning labels, barcodes, graffiti tags, halftone print energy, cropped typography, chromatic split, and zine-poster chaos.

Common acceptable caveats:

- generated microtext may be pseudo-text
- the designs are stylized/tasteful redesigns, not strict production sheets
- Unit-07 should be reported as a concept/rationalized mass-production test body if used
