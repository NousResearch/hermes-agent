# Loaded historical model bounds census — 2026-09-07

Actual localhost WebGL scene measurements, not GLB metadata estimates and not packaged hardware acceptance. [Raw census](evidence/loaded-historical-bounds.json), [comparison data](evidence/loaded-historical-bounds-analysis.json), and [reusable census script](../../../apps/desktop/scripts/lunar-city/census-loaded-bounds.mjs).

The script measures enabled meshes with `isVisible` and positive `visibility`, using their world bounding boxes after world matrices update. Disabled source templates, inactive LODs and zero-vertex nodes are counted separately, never unioned into a visible measurement. It captures overview and actual focus-selected representations for every historical model ID. Current review replacement buildings, terrain and eight new leader meshes are preserved and excluded from historical mismatch conclusions. A second browser-only request mapping loads the unchanged original v2 manifest and asset bytes to measure historical replacements too; nothing on disk is redirected or normalized.

Dimensions are geometric extents in world metres. The manifest comparison is `(bounds.max - bounds.min) * scale`; it is a declared envelope, not necessarily an approved intended physical height. A shorter visible LOD does not itself prove an error. Heights exceeding that envelope, or visible geometry outside its rotated XZ range, indicate camera/collision contract risk. Low geometry may be authored foundations rather than a misplaced model; inspect before changing it. Model rotations here are around Y, so height comparisons do not conflate horizontal rotation with scale.

## Historical pack, most detailed observed representation

| Model | Actual height m | Declared height m | Minimum world Y m | Placement Y m | Actual minus placement m |
| --- | ---: | ---: | ---: | ---: | ---: |
| terrain | 19.890 | 26.000 | -10.120 | 0.000 | -10.120 |
| library | 18.182 | 16.000 | 4.000 | 4.000 | -0.000 |
| research-lab | 23.734 | 17.000 | 4.600 | 5.000 | -0.400 |
| depot | 15.807 | 14.000 | 1.983 | 2.000 | -0.017 |
| review-office | 17.100 | 15.000 | 2.978 | 3.000 | -0.022 |
| triage | 5.280 | 9.000 | 1.000 | 1.000 | +0.000 |
| garden | 4.911 | 5.000 | -0.425 | 0.000 | -0.425 |
| council | 19.222 | 14.000 | -1.880 | 1.000 | -2.880 |
| arts-studio | 17.250 | 15.000 | 0.356 | 2.000 | -1.644 |
| engineering-workshop | 18.150 | 16.000 | 0.192 | 2.000 | -1.808 |
| release-gatehouse | 15.900 | 15.000 | 0.602 | 2.000 | -1.398 |
| archive | 17.925 | 15.000 | 0.233 | 2.000 | -1.767 |
| bus | 2.615 | 4.000 | 2.160 | 2.000 | +0.160 |

Nine historical buildings exceed their declared height envelope: library, research lab, depot, review office, council, arts studio, engineering workshop, release gatehouse and archive. The largest relative excesses are research lab (about 39.6%) and council (about 37.3%). This is not evidence that the models should all be normalized: survey geometry and update correct bounds, cameras, collisions and grading together.

The original review office exceeds its rotated declared XZ bounding envelope by 0.899 m in X; historical terrain exceeds by 0.902 m in Z. Other observed historical model XZ AABBs fit their declared rotated envelope. Geometry bounds include all visible tagged geometry, including authored ornamental parts; they are not floor-only footprints.

## Historical assets still present in the current review settlement

| Model | Minimum world Y m | Deck Y m | Difference m | Outside pad square X / Z m |
| --- | ---: | ---: | ---: | --- |
| depot | 2.083 | 2.100 | -0.017 | 0.000 / 0.000 |
| review-office | 2.328 | 2.350 | -0.022 | 0.505 / 0.000 |
| triage | 2.050 | 2.050 | -0.000 | 0.000 / 0.000 |
| garden | 1.475 | 1.900 | -0.425 | 0.000 / 0.000 |
| council | -0.880 | 2.000 | -2.880 | 0.000 / 0.000 |
| engineering-workshop | 0.442 | 2.250 | -1.808 | 0.000 / 0.000 |
| release-gatehouse | 0.752 | 2.150 | -1.398 | 0.000 / 0.000 |
| archive | 0.383 | 2.150 | -1.767 | 0.000 / 0.000 |
| bus | 2.360 | 2.200 | +0.160 | 0.000 / 0.000 |

Deck positions and pad diameters come from `scripts/lunar-city/settlement-layout.mjs` (`DECK_OFFSET = 1.65`). Council has visible geometry 2.88 m below its deck, engineering workshop 1.808 m, archive 1.767 m, release gatehouse 1.398 m, and garden 0.425 m. Depot/review-office deviations are about two centimetres; bus geometry begins 0.16 m above deck. These are minimum geometry points, not surveyed feet, doorway thresholds or usable walking surfaces.

The current review office AABB exceeds the 24 m pad square by 0.505 m in X. Other listed historical assets fit that square. Square containment does **not** establish containment on a circular pad or in an oriented road-clearance hull. Those need exact vertex/ground-footprint checks against final authored pad/hull geometry; this census does not claim collision acceptance. Existing review-building replacements were excluded from resizing and this mismatch table.

## Characters and remaining limits

`leaders` is a pack containing multiple placed inhabitants and shared far representations. Its aggregate bounds must not be treated as the height of one leader. Raw focus captures preserve leader-owner groups where present. The hidden worker source pack is likewise not an inhabitant. Raw `demoWorkers` are actual exact-owner local-demo clones; `focusedWorkers` captures their focus-selected near representations when available. Both overview and focus-selected actual demo clones measured 1.200 m after the concurrent worker factory repair; focused clones had five visible meshes and 78 excluded hidden meshes. Current runtime worker focus height is 1.2 m. Final samples are 2026-09-08T04:44:06.818Z (review) and 04:44:17.723Z (historical), both with zero page errors. The original 11 m defect was not observed in this final run. Historical leader focus still yielded an ownerless shared-surface aggregate (7.154 m high), so individual legacy leader height is not established by that aggregate.

This was read-only asset inspection in headless Chromium. No geometry, asset scale, manifest, navigation mesh, live gateway, Blender scene or foreground browser was changed. The script adds temporary selection options only inside its own disposable test page to exercise existing focus handlers. Initial attempts failed on macOS sandbox launch permission and an invalid URI substitution; those did not produce accepted measurements. The retained receipt comes from the successful unchanged-manifest request-routing run.
