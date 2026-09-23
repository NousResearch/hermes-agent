# Lunar City review publication

This branch preserves the current Lunar City review runtime, its tests and asset-generation scripts, the exact runtime asset graph, the source GLBs needed by `build-review-pack.mjs`, basic interior exports, source cards and selected finishing studies. It is not final game or asset acceptance.

The runtime graph has 52 files (166.5 MB), including the compile-time interior plan JSON. The generated renderer emits the other 51 assets. `runtime-public-assets.ts` follows the manifest graph; Electron additionally excludes `public/lunar-city/**` because the runtime assets already ship under `dist/**`.

The multi-gigabyte Blender workshop, automatic Blender backups, unused raw meshes and superseded intermediate exports remain local and are not in this Git publication. Scripts that operate on the authoring workshop require that local source file; a clone can run the game and rebuild the review pack from the included source GLBs, but cannot recreate all manual Blender editing history. Earlier asset-preservation drafts remain separate.

## Current review behavior

- Fourteen basic interior/courtyard layouts support reversible cutaways and indoor job-pickup paths.
- Eight leaders begin at their own home anchors. Automatic activity uses scoped source observations; local preview controls can override and reset it.
- The cat has verified home, desk and town walking, pause/remount preservation and corrected foot contact.
- Seven other leaders remain stationary while their rigs await deformation repair. Ground-corrected copies were deliberately not promoted after meaningful clothing/prop stretching was measured.
- The core worker's static v3 study improves the helmet, visor and probe; it is not a finished or rigged production replacement.

See `game/interiors-gameplay-handoff-2026-09-08.md` for behavior, test receipts and remaining shell-fitting limits. A successful renderer build is not packaged desktop or authenticated live-gateway acceptance.
