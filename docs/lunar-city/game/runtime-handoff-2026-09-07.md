# Lunar City game integration — 2026-09-07

Recovered the operational city from the clean historical checkout at commit
`0d9a879ce487cd6886c742ab06828547882c708c` in
`/Users/mikedemott/.codex/worktrees/bf383511-ef21-4aee-aee8-e6528891e987/hermes-agent`.
Source: `docs/lunar-city-design-handoff.md`, `lunar-city-handoff-2026-09-01.md`,
`lunar-city-video-reference-roadmap.md`, and `apps/desktop/src/app/lunar-city`.
Historical claims are not treated as current acceptance evidence.

## Implemented and recovered

- Dedicated `/lunar-city` route through the existing desktop contribution system.
- Babylon scene and manifest loading, real GLBs, interior/occlusion controls, orbit/pan/zoom,
  focus/follow, reduced motion, quality tiers, lifecycle cleanup and context recovery.
- Identity-preserving profile/session/subagent/Kanban adapters, navigation, bounded animation
  scheduling, worker aggregation, leader dialogue, inspector and explicit command confirmation.
- Restored the required exact-owner plugin REST/socket and voice scope capabilities while
  preserving current TTS lease APIs. Failed fleet reads retain visuals with stale authority.
- Restored missing library/laboratory destinations to their existing authored navigation nodes.
- Registered Babylon ray picking, which the modular loader previously left as a no-op; verified a real rendered owl mesh click.
- Repaired missing status-badge world projection. Recovered decoration manifests now actually
  load their optional geometry and participate in the decoration quality switch.
- Opt-in `v2-review` asset pack loads all eight current card-derived leader material reviews
  independently of legacy leader fixtures, preserving scale/materials/hierarchy. Cat now has its real 23-bone rig and seven imported clips. The other seven remain static; no synthetic animation metadata is emitted. Baseline worker has a separately selectable 20-bone review rig with 17 imported states.
- Review leader close-up radius derives from the measured target height, without enlarging the
  geometry. The normal fixture camera remains compatible.

- Expanded the recovered warm regolith environment at metre scale, with 16 district sites,
  new revenue/publishing pads, reserved card-building footprints and 21 connected paths.
  Right-handed rendering now matches glTF placement and Recast coordinates; entrance ramps
  are pitched correctly and remain at or below 1:12. Navigation excludes building footprints.
- Normal idle plays actual skeletal animation; explicit reduced motion parks it. Unweighted
  control tracks and constant held poses do not consume continuous animation frames.
- Production Vite output includes the desktop and standalone review viewer plus only
  manifest-referenced runtime files. Authoring archives and obsolete exported copies are excluded.

## Run

From `apps/desktop`:

```sh
node scripts/lunar-city/build-review-pack.mjs
../../node_modules/.bin/vite --host 127.0.0.1 --port 5178 --strictPort
```

Open `http://127.0.0.1:5178/lunar-city-review.html` for the backend-free material viewer.
The full desktop route can select that same pack through the existing same-origin override:
`/?lunarCityManifest=/lunar-city/v2-review/world-manifest.v2.json#/lunar-city`.
The review page has no simulated tasks or operational mutations. Use the normal desktop
route with an authenticated exact-owner gateway for live work and dialogue.

Rebuild the review pack after a Blender export changes; it copies assets into an isolated
runtime pack and never edits the source models. Vite's filesystem watcher in this environment
has served stale transformed modules after edits; restart only this owned development server
when that occurs.

## Asset contract

- glTF coordinates: +Y up, Y=0 at feet/base, 1 unit = 1 metre. Source cards face +Z;
  retain glTF conversion-root transforms. Any artistic yaw goes on a separate placement root.
- Leader IDs: owl, elephant, cat, fox, capybara, lion, beaver, monkey. Capybara is revenue;
  beaver is architecture. Model IDs never replace exact connection/profile identity.
- Leader source folder IDs: owl-librarian, elephant-memory, cat-arts, fox-scientist,
  capybara-revenue, lion-steward, beaver-architect, monkey-poet.
- Final dialogue rig states: acknowledging, idle, listening, talking, thinking, unavailable.
  Existing authored pack clip names are `leader:<species>:<state>`. Root extras carry
  `leaderId` and `stateClips`; keep clip targets separate when merging surfaces.
- Workers use the 19 canonical `characterAssets.groupKits[].kitId` values in
  `public/lunar-city/v2/world-manifest.v2.json`. Baseline is a separate neutral model.
  Existing worker runtime clips include idle/listen/talk/think/walk/work and state variants;
  a standalone high-resolution worker GLB must still be adapted into the shared LOD/kit
  contract before replacing the fleet pack. Do not remove identity/aggregation budgets.

## Explicit remaining acceptance

The terrain/path layout is rebuilt. Owl library, fox observatory and monkey publishing buildings
now appear in the opt-in review pack at measured scale, with source hashes and pending quality
labels. Other building replacements await Blender review. These three are visibly darker than
the historical fixtures; shared lighting/material grading and surveyed entrances remain pending.

Baseline is selectable in the review viewer and can replace neutral near workers with the explicit
query parameter `lunarCityReviewWorkers=baseline` on the review pack. The factory independently
clones skeletons and animation targets, preserves historical mid/far geometry and kit-specific
workers, and retains existing identity/animation budgets. Two real GLB clones showed zero motion
cross-talk; actual near-to-mid-to-near transitions passed. Specialist reviewed variants/LODs remain
pending. The normal pack and default fleet remain unchanged.

The leader pack is review only. Cat uses rigged-review.glb; fox, beaver and monkey use material-review-v4.glb;
v5 was rejected. Other leaders retain their previously reviewed material exports. Exact hashes are
in v2-review/review-sources.json. New worker rigs with stretched equipment and buildings with
stretched slab textures are held out of the pack. No production art acceptance is implied.

No packaged Electron, authenticated multi-gateway action/chat, representative GPU benchmark,
or 30-minute stability acceptance is claimed. Live operations require the authenticated desktop
route; the standalone viewer intentionally has no operational mutations.

## Validation receipt

- Focused game plus supporting API suites: 582 tests across 45 files pass.
- Renderer TypeScript check passes. Scoped game lint passes after automatic formatting fixes.
- Production renderer build succeeds with runtime asset packaging enabled, approximately
  170 MB. Refreshed built-viewer WebGL smoke loads all eight leaders, baseline, CI and four reviewed buildings without page errors.
- Real Recast navigation evidence: evidence/navigation.json, 21 routes, maximum horizontal
  endpoint error 0.000002291 m and ground error 0.00164922 m; no footprint crossings.
- Cat skin/clip evidence and worker review evidence are recorded under evidence/. Check each
  receipt's timestamp; older screenshots predate the expanded layout. Baseline wait drives an unweighted control and is a held pose. Heartbeat has real motion;
  checks sample between quarter-cycle points to avoid sinusoidal aliasing. All 17 imported
  states are available, 16 have visible motion including normal idle, and clip transitions
  have no overlap.
- The latest environment overview is evidence/environment-overview.png. The newly added revenue pad remains empty pending its reviewed building replacement.

## Current review process

The source `.blend` files, card images and authoring exports belong to the Blender task.
Re-running `build-review-pack.mjs` uses only its explicitly selected exports. Never glob all
new `rigged-review.glb` or building exports into acceptance; several are intentionally held.
For subsequent approved assets, refresh this pack, restart the owned Vite preview, then run
focused skin/placement checks. The world-scene module is now near 2,000 lines; extract the
entity factory into its own module before extending that area further.

Current preview: `http://127.0.0.1:5178/lunar-city-review.html`.
Runtime-owned dev server remains available; foreground Blender was never controlled by this agent.

## Specialist fitting audit and CI review addition

The review-pack builder also selects the individually repaired `ci-repair-triage` worker export.
It appears as a second selectable worker in the standalone review viewer. The normal fleet and
`lunarCityReviewWorkers=baseline` selection remain unchanged. Its actual WebGL 17-state check
passed with 20 bones, exclusive state transitions, real ambient idle and heartbeat motion,
and an honestly held wait pose. See `evidence/worker-ci-repair-triage-animation.json`.
Two CI clones also showed 13.665 mm maximum coordinate displacement on the moving copy,
zero cross-talk on the stationary copy, and successful near/mid/near factory transitions
(`evidence/worker-ci-clones.json`). This direct factory check does not enable CI in the fleet.
This checks playback, not completed track locomotion: the current walk clip moves the chassis,
and the probe/finger geometry still needs cleanup. No specialist production acceptance follows.

The refreshed production renderer build includes the CI addition and reconstructed cat studio. [Worker fitting checklist](worker-rig-fitting-checklist-2026-09-07.md) records
all 20 designs and their individual mechanical requirements. The
[monkey separation audit](monkey-hand-coat-separation-2026-09-07.md) documents the next topology
repair; the runtime continues using its static v4 material review.


## Reconstructed cat studio and refreshed production build

The opt-in arts-studio model now uses
`building-multiview-2026-09-07/cat/reconstruction-v2/building-review.glb`; the original source
and default v2 model are preserved. The runtime copy and source hashes are recorded in
`v2-review/review-sources.json`. Its unwanted background sheet is absent in the inspected
built-viewer screenshot. Surface detail, material grading and surveyed doorway alignment
remain pending. The apparent foreground streetlight was traced to an upright terrain trim
ring and corrected in the clearance follow-up below.

Actual Babylon world bounds measured 10.000000 m tall, from Y=2.3499999 m to Y=12.3499999 m,
matching the district raised deck at 2.35 m with unit scale. Evidence is
`evidence/cat-studio-runtime.json` and `evidence/cat-studio-runtime.png`.

The refreshed production build completed in 21.56 seconds (about 170 MB). The actual built
viewer loaded 26 GLBs, including CI and cat studio, with no page errors; CI work was selected
and cat focus visually inspected. The fetched built cat GLB SHA matches the manifest. See
`evidence/built-viewer.json`, `evidence/built-ci-repair-triage.png` and
`evidence/built-arts-studio.png`. Renderer typecheck also passed. This remains browser/WebGL
validation of the production renderer bundle, not packaged Electron/live-gateway acceptance.


## Forecourt clearance correction

The object crossing the cat doorway was `terrain:district-ring:8`, not a streetlight. Actual
scene ray picks and pre-merge terrain bounds identified the shared error: Babylon creates a
horizontal torus, but the district builder rotated it upright. Cat's trim then spanned
Y=-3.66 to 8.42 m through the building. Removed that rotation for district trim and recessed
its top 1 cm below the deck, avoiding both pedestrian obstruction and coplanar flicker.
Building assets, known placement directions, doorway assumptions and navigation mesh are
unchanged. The historical default pack was not rewritten.

A real generated-mesh regression test failed before the fix and passes afterward for every
district. All three layout/clearance tests pass. The actual Recast check still passes all 21
routes: maximum ground error 1.650 mm, arrival error 0.0023 mm. The rebuilt viewer screenshot
`evidence/built-arts-studio.png` shows the cleared front approach. No unverified doorway
bounds were invented or building meshes altered.
