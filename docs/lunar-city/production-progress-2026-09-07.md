# Lunar City production continuation — 2026-09-07

## Approved direction

Convert the approved 2×2 building, worker, and leader designs into polished editable Blender models; complete materials, textures, suitable rigs, animation, and game integration. Use the recovered Hunyuan3D-2mv multiview reconstruction process and work visibly in Blender. One Blender unit is one metre. The user confirmed eight leaders: elephant, owl, beaver, lion, capybara, cat, fox, and monkey. Capybara leads revenue; beaver leads architecture. The user explicitly confirmed the monkey card belongs in this roster. Preserve source cards and recovered source meshes.

## Verified recovery

- Recovered 195 asset/script files from preservation PR #26, commit `4dd951e205acd86608bba40b795b10e0431f194d`, into this existing workspace. No existing destination files were overwritten.
- All 40 archived PNG cards match the Downloads backups byte for byte. Forty cards include variants; this is not yet a count of unique approved assets.
- The newer Hunyuan set contains eight buildings and six workers. Direct GLB inspection finds no UV0 coordinates, materials, textures, skins, or animation clips in any of the fourteen files.
- Building reference triangle counts range from 433,952 to 1,022,686. Worker counts range from 363,504 to 1,896,674. These are source references, not runtime-ready meshes.
- Existing metadata explicitly rejects the earlier scene-crop TripoSR meshes for production. The procedural master-scene presence also does not establish visual acceptance.
- The saved Hunyuan review image shows several workers as boxes. Inspect actual connected components and compare all views before choosing cleanup versus reconstruction.

## First leader: owl archivist

Reference: `Codex Image Sep 4, 2026, 12_37_24 AM.png`.
Keep the stocky owl silhouette, layered brown plumage, cream facial discs, ear tufts, brass circular glasses, teal tailored coat, cream shirt, dark tie, brass buttons, leather elbow patches, feathered hands and talons. Separate the catalog book, pencil, and archive satchel so attachments and gestures remain possible. The reference side pose differs from the front pose; establish consistent anatomy and bind pose before animation.

The manual primitive study is paused. A new five-step, resolution-256 Hunyuan3D-2mv owl mesh has been generated from the masked front, side, and back cards. The top view remains a comparison reference. The generated mesh has 543,332 triangles and was imported into a separate Blender scene; its 1.5 metre height was verified. Materials, topology suitable for animation, rigging, and runtime integration remain incomplete. The height is an explicit stylized design choice, not a measurement inferred from the card.

## Completion evidence per asset

1. Identify the exact reference card and compare silhouette from front, side, rear, and plan.
2. Inspect topology, normals, loose components, scale, and moving-part separation.
3. Produce UVs and material/texture work appropriate to the card; verify a real Blender render.
4. For characters, rig and verify deformation plus idle, walk, and role-specific gestures.
5. Derive runtime detail levels and collision shapes from the accepted model; verify an actual GLB reimport.
6. Integrate with the existing game asset loader and animation state mapping, then inspect the running game.

## Earlier execution state (superseded by the continuation below)

Blender screen access has recovered. The card-derived owl is visible in `multiview-2026-09-07/owl-librarian/metric-review.blend`. Source geometry and reconstruction receipts are preserved beside it. A local Apple Metal batch is generating remaining leader geometry. This is shape generation, not a claim of finished game assets.

Preserve current Blender scenes. Add a separate scene and save a separate output file for each review.

## Reusable skill and live imports

Installed `card-to-3d` in `/Users/mikedemott/.codex/skills/card-to-3d`. Skill validation and an isolated real owl preparation pass succeeded. The installed Blender queue now serves the eight-leader batch, preserving source objects and adding separate cleanup copies with light reversible smoothing. Verified imports so far: cat-arts, elephant-memory, fox-scientist, owl-librarian. Saved workshop: `multiview-2026-09-07/leader-workshop.blend`. Queue stops after the finite batch or when Blender closes; materials, retopology and animation remain work in progress.

## Rear projection defect and full leader audit

User found a second face on the monkey rear. Rendered actual exported/reimported
Blender materials from the rear and side for all eight leaders (`audit-rear.png`
and `audit-side.png` in each multiview asset folder). Duplicate front artwork was
confirmed on monkey, cat, fox, and beaver; owl, elephant, capybara, and lion did
not show duplicate rear faces in those views. This is not a claim of full visual
acceptance: side seams, projection contamination and surface noise remain.

The cropped rear inputs are correct. The texture study selected front/back by
normalized global depth, so a projecting tail moved the entire head into the
front-selection region. Corrected projection uses camera-facing smoothed normals
and asserts that rear-facing surfaces have zero front contribution. Torso width
registration excludes sideways tail extension in the reference card. Versioned
v4 exports have been reimported at their recorded metre heights and visually
checked: all four now show rear fur and the correct rear coat artwork.

A further v5 review uses the side reference on the projecting tail, where front
and rear body projection previously painted coat artwork. Review remains pending
until actual v5 renders are inspected. Original source meshes and earlier review
exports are retained; no production-ready claim, no completed rigs yet.

V5 tail study reviewed and rejected as the default replacement: the spatial tail
region also included the monkey book belt, and the beaver side projection degraded
the otherwise-correct rear tail surface. Retain v4 for all four corrected leaders.
All four v4 rear renders visibly have no second face; front views remain intact.
Tail isolation, source registration, crown seams, bake misses and rigging remain
open. The game agent was instructed to refresh review copies from v4 only.

## Buildings, workers and rig continuation

All eight leaders now have rigged review exports and reimport receipts. Cat body
animation has also been exercised in the real game loader. These are not final
deformation acceptance: stress poses exposed sleeve/finger stretching in other
leaders. The rig script now welds coincident UV-split vertices before smoothing
weights, preserves loop UVs, and blends arm attachment weights into the torso.
The latest monkey stress review uses this repair; the remaining review exports
must be regenerated and inspected before promotion. Facial animation remains open.

All twenty worker cards have inspected masks and a finite multiview generation
queue. At this checkpoint eight have generation and color-projection receipts:
baseline, acceptance-release, archive-acquisition, arts-studio, ci-repair-triage,
content-studio, community-intake and control-plane-incidents. The queue has moved
to core-runtime-ux-repairs. Live imports retain the sources in the central
`leader-rig-workshop.blend`. Baseline has a 30k-triangle review mesh, a 20-bone
rig, seventeen exported clips and eighty-five sampled deformation checks. Other
worker finishing remains pending. Wheel, track, hover and multi-arm variants need
individual mechanical/deformation review; generic clips alone do not establish
acceptance.

All eight recovered building source meshes have been uniformly scaled and grounded,
with source hashes and explicit assumed dimensions. Heights in metres are owl 16,
elephant 24, cat 10, fox 12, capybara 16, lion 26, beaver 14 and monkey 12. All have
four-view color-projection receipts. UV/normal baking, roof/silhouette review,
material refinement, entrances and mechanisms remain pending. Building review
camera framing now includes the full bounding-box extent, and the bake script
includes a top view. Those changes are syntax-checked but not yet run in Blender.

The finite worker finish and building bake queues are prepared but not registered.
Computer Use currently reports the Mac is locked and cannot unlock it. Resume
visible Blender control after the user unlocks; do not start a hidden replacement
Blender process. Existing generation, projection and live-import queues continue
independently. The game/environment agent continues its separate integration work;
its evidence is maintained under `docs/lunar-city/game/`.

### After the user unlocked the Mac

Started both finite Blender queues through Computer Use and verified actual output.
All eight building material review exports and four-view renders are saved. Uniform
scale restoration after smoothing fixes the elephant building's 8mm height loss.
These remain review assets: the recovered cat mesh contains a backdrop sheet;
the elephant/tree, lion and capybara have coarse surfaces; beaver needs proper water
materials. Owl, fox and monkey were sent for opt-in game review placement, not final
acceptance. A fresh masked cat reconstruction is queued after worker generation;
all original building sources remain preserved.

Paused the worker finish queue after visual stress checks exposed gripper/chassis
cross-weighting. Fitting the CI robot's joints to its actual forward arm depths and
separating chassis and gripper weights reduced its stress audit from large stretched
edges to zero edges above 3x. The latest CI rig still needs finger mesh cleanup and a
new surface/export pass. Monkey wrist fitting remains under repair; the last render
still showed finger/coat stretching. The next lower-finger boundary correction is
edited but unexecuted because the Mac locked again.

Added mesh-edge stretch receipts and sampling at every authored keyframe to both
animation exporters. Earlier five-sample heartbeat checks aliased its sine wave
at zero crossings; the game agent verified actual heartbeat motion between those
samples. Baseline runtime checks now show independent cloned skeletons and real
state animation; wait is intentionally held. These runtime results do not accept
the unreviewed worker variants.

See `continuation-checkpoint-2026-09-07.json` for the last observed generation counts
and the exact outstanding operations. The Mac lock blocks further visible Blender
control; it does not cancel the already-running finite generation processes.

### 20:26 visible Blender continuation

All 20 worker material bakes are complete. All 20 rear renders were visually inspected for duplicated front faces. Cat studio reconstruction-v2 has completed baking and front/rear visual review; the unwanted background sheet is absent, while detail cleanup remains. Acceptance-release feet no longer inherit hand weights; leg depth was fitted from mesh samples, and torso weight transitions were smoothed. Its 17 clips were reexported with 221 deformation samples. Three edges still exceed 3x in the deliberately exaggerated stress pose; this is a review export, not final mechanical/topology acceptance. Monkey hand/coat separation and the specialist mechanical rigs remain unfinished.

#### Release-worker follow-up validation and input blocker

The 20:26 exported clips retain one edge above 3x during review (max 3.2365x), and walk can dip 27.6 mm below ground. Helmet blending and root ground-contact correction were prepared afterward, but are NOT validated or included in the saved model. Computer Use stopped delivering input; native clicks report windowNotFoundAtPosition and native paste timed out waiting for Blender to read the clipboard. Reconnecting and user focus restoration did not resolve it. Do not mark those prepared fixes complete.

### 20:52 input restored and release-worker contact correction

User restored native input. Helmet weights and ground contact correction executed in visible Blender; root ground compensation now keyed every frame to eliminate between-key foot dip. All 17 clips/1,037 integer frames evaluated: minimum Z +0.012 mm, max edge stretch2.7979x, no edges over3x. Walk and review renders visually inspected, and walking playback shown in Blender. Source workshop and review GLB saved. This is still review quality, not final mechanical joint/topology acceptance. Game delegation expanded by user to alerts, personalities, social interactions and all planned/partial game systems.

### 21:05 engineering worker rig review

Fitted higher hip/knee joints and actual leg depth, excluded fingertips from leg weights, kept held tool region on gripper, and fitted its forward wrist pivot from mesh samples. Initial generic stress audit had86 edges over3x; successive full animation review retained a tool-use seam until the actual forward wrist fit. Latest17clip/1,037frame result: maxedge2.2274x, zeroedgesOver3x, minimum ground depth only numerical -0.000026mm. Review GLB saved; surface fragments and final mechanical joint acceptance remain. Worker review exports now9; only asset-specific QA establishes readiness.

### 21:09 engineering fragment cleanup

Removed18 disconnected single-triangle components (54vertices) from the engineering review mesh. Original mesh geometry_0.294 retained with fake user; closed components preserved. Reexported and repeated all1,037frame checks: maxedge2.2274x, noedgesOver3x, ground within floating-point noise. Additional larger rough fragments and mechanical separation still require art cleanup. Current next model investigation is core-runtime wheel/stabilizer segmentation, using actual front/side card geometry rather than genericwheel masks.

### Core-runtime mechanical wheel study

A separate Blender study preserves the source and replaces four fused wheel regions with rigid tires and hubs matching the card, at a 0.12 m radius. The ground sweep covers 292 samples with minimum Z +0.009 mm after correcting tread corners. The tire-to-stationary-mesh sweep found nine interfering brace faces; bounded relief moved seven vertices (maximum 20.1 mm), after which all 144 sweep samples had zero triangle intersections. This does not establish enclosed-volume or final part-interface acceptance.

The relief currently lives in unsaved Blender state; the workshop on disk contains the preceding study. Reconstruct it by running lunar_core_wheel_clearance_review.py followed by lunar_core_wheel_clearance_fit.py; do not rerun the fit against the zero-intersection output alone. The spin-preview script is prepared, but native input failed before execution. No core worker export or promotion has occurred.

### September 8: worker static finish and building interiors

Visible Blender control works through the editor dropdown. The core worker now has a preserved static v3 study: locally welded helmet seams, fitted roof gear, separated three-finger probe, and visor colors baked onto the original surface. A separate visor cover was rejected for poor surface fit and remains hidden. GLB reimport confirms 24 meshes, 1.200000 m height and ground minimum zero. Body, tablet, fenders and joints still need cleanup; this static candidate does not replace the animated runtime asset.

Fourteen basic interior plans now have full-height and cutaway Blender exports (28 GLBs). Revision 2 preserves the first gallery, raises door clearance to 2.6 m and uses measured actor widths for circulation. The beaver workshop keeps an open river channel with a raised bridge and ramp; coplanar bank-floor overlap was removed and visually checked. Runtime leader home/work/idle movement, indoor worker job pickup and cutaway controls are being validated by the game agent.

### Leader walk contact and deformation audit

Imported six preserved source rigs into separate visible Blender scenes and corrected walk ground contact over 61 integer frames. Exported copies remain under each leader's `walk-contact-review` directory; original rigs are unchanged. Quarter-stride renders were inspected. An absolute edge-growth check confirmed meaningful garment/prop/leg-weight defects: owl 78.7 mm, elephant 114.7 mm, fox 29.4 mm, capybara 37.5 mm, lion 27.7 mm and beaver 32.5 mm. These are not merely microscopic seam ratios. None of the six copies is promoted into the runtime. Monkey remains held for hand/coat entanglement. A ground correction does not establish final deformation or walking acceptance.

### Final interior clearance revision

Revision 3 uses every source vertex transformed into actor space to measure radial clearance about the actual actor origin. This catches diagonal props during turns: archive circulation is 2.4 m, library 1.7 m and cat/fox interiors 1.6 m, with the other roles retaining their fitting widths. The visible v3 Blender gallery and 28 exports were regenerated and saved while preserving v1/v2 scenes.

The current runtime cat source was verified byte-identical to its saved rig export. Its 12.475 mm walking foot penetration received the same separate per-frame root-height correction; original geometry, weights and pose channels remain intact. Export-roundtrip review is required before replacing the review registration. This bounded contact correction does not accept the cat's existing garment deformation as finished.

### Built interior and resident behavior verification

The final review build at port 5180 includes v3 interiors, eight leaders initially placed at their home anchors, automatic observed-activity controls with local overrides/reset, and the corrected cat contact export. All fourteen built cutaways and Automatic reset passed browser checks with no page errors or external requests. The actual cat reached its home and desk with its walk group playing, took a town stroll, held position for pause/reduced motion, and restored exactly the same position on scene remount. The actual imported worker reached indoor job pickup and stayed there after polling. The final cutaway/leader slice reports 70 focused tests, renderer typecheck, scoped lint and a 24.21-second production build passing. The preceding combined suite passed 616 tests; final handoff receipts capture subsequent full-suite results.

Seven leaders remain stationary pending accepted walk rigs. Six preserved contact corrections passed exported ground checks but failed meaningful deformation review; the monkey retains its hand/coat problem. The cat's bounded contact correction passed 41 CPU-skinned samples and a normalized-pose comparison (maximum difference 1.513 micrometres). All other character/art and exterior-shell fitting limits remain open.

Final combined validation: 618 tests across 68 files, TypeScript and scoped lint pass. The built 5180 viewer passes all 14 cutaways; final indoor worker and cat automatic/lifecycle receipts pass. No required runtime/interior-slice check remains pending. Full asset polish and prototype shell fitting remain unfinished.
