# Leader locomotion readiness — 2026-09-08

Actual Babylon imports and weighted-target motion checks cover all eight manifest leaders, all eight preserved source rigs, and six contact candidates. This is clip/ground evidence, not visual acceptance. No rig was promoted by this audit.

| Leader | Current manifest walk | Preserved source rig | Contact-copy status |
|---|---|---|---|
| owl | Unavailable: static mesh, no skin/groups | Walk motion exists; visual review remains pending | Held: 78.7mm maximum absolute edge growth in Blender receipt. Babylon ground minimum +0.173mm. |
| elephant | Unavailable: static mesh, no skin/groups | Walk motion exists; visual review remains pending | Held: 114.7mm maximum absolute edge growth in Blender receipt. Babylon ground minimum +0.146mm. |
| cat | Usable weighted walk clip | Walk motion exists; visual review remains pending | Original selected walk penetrated 12.475mm. Preserved contact correction passes: minimum +0.148mm, maximum gap0.200mm, normalized deformation change 1.513µm; eligible for bounded review-pack replacement only. |
| fox | Unavailable: static mesh, no skin/groups | Walk motion exists; visual review remains pending | Held: 29.4mm maximum absolute edge growth in Blender receipt. Babylon ground minimum +0.151mm. |
| capybara | Unavailable: static mesh, no skin/groups | Walk motion exists; visual review remains pending | Held: 37.5mm maximum absolute edge growth in Blender receipt. Babylon ground minimum +0.161mm. |
| lion | Unavailable: static mesh, no skin/groups | Walk motion exists; visual review remains pending | Held: 27.7mm maximum absolute edge growth in Blender receipt. Babylon ground minimum +0.150mm. |
| beaver | Unavailable: static mesh, no skin/groups | Walk motion exists; visual review remains pending | Held: 32.5mm maximum absolute edge growth in Blender receipt. Babylon ground minimum +0.165mm. |
| monkey | Unavailable: static mesh, no skin/groups | Walk motion exists; visual review remains pending | Held: documented hand/coat/skirt entanglement; no contact candidate promoted. |

Evidence: [actual import and ground samples](evidence/leader-readiness.json); source contact receipts live under each `apps/desktop/public/lunar-city/multiview-2026-09-07/<asset>/walk-contact-review/`. Hashes in the import receipt were audited from the served files immediately after import.

Ground sampling uses CPU-deformed whole-mesh vertices and world transforms at 41 evenly spaced clip frames. It does not isolate individual feet, guarantee between-sample contact, clear garment stretching, or certify facial/finger rig quality. Structural source export receipts explicitly say `skeletal_export_verified_visual_animation_review_pending`; all report `facialRig: false`.

Runtime staging places new leaders at authored interior homes; cached positions take precedence only when still physically valid. Seven static leaders stay still with explicit unavailable locomotion controls. Cat home/work/stroll uses actual paths and its real walk group. The separate contact-copy receipt clears its bounded ground correction, without clearing pre-existing deformation aesthetics.

Actual latest-source [leader lifecycle receipt](evidence/leader-life-runtime.json) verifies all eight authored home placements, cat work/home walking and idle arrival, outdoor stroll, reduced-motion holds, and changed-position renderer remount retention. Initialization was 2514ms in that headless run. [Cat contact roundtrip](evidence/cat-contact-roundtrip.json) records 41 CPU-skinned samples, geometry matching despite 16 additional export seam vertices, hashes and all-seven-clip channel comparisons.

Final corrected-cat run: [automatic projection receipt](evidence/leader-observed-life-runtime.json) verifies exact-owner observed work plays the real walk, idle releases it, stale/ambiguous ownership holds, and Automatic clears a local override. It identifies `review-cat-a94817c2f48a.glb`. The refreshed lifecycle receipt repeats home/work/stroll/hold/remount with the corrected pack. These are local synthetic-snapshot proofs, not live backend acceptance.
