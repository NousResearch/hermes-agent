# Basic interiors and local character life

The local review viewer now has fourteen authored interior/courtyard plans bound to the actual exterior asset URI, SHA and placed transforms. These are new role-based designs, not recovered original floorplans or surveyed shell fits. Exterior source meshes remain preserved. Runtime interiors are enabled only for the same-origin `lunar-city/v2-review/world-manifest.v2.json` review pack.

## Shared design and Blender exports

Canonical data: `apps/desktop/public/lunar-city/interior-plans-v1/plans.json`. The existing folder name remains stable; the third Blender study is explicitly named **Lunar City Interior Studies v3**. `actor-clearances.json` records every GLB POSITION vertex through its node transforms, measuring radial distance about the actual actor origin including props at bind/rest pose. The clear widths are 1.5–2.4 m depending on the resident leader; all enclosed door headers clear 2.6 m and walls provide 3 m ceiling clearance. Workers remain 1.2 m; no actor was shrunk to fit a door.

The plans cover library, research lab, depot, review office, triage, council, arts studio, engineering workshop, release gatehouse, archive, publishing and revenue. Garden and bus are open-air layouts; the 3 m overall bus model does not silently acquire a 3 m enclosed room. Plans provide role-specific rooms, furniture, entry/work/home/rest/job-pickup anchors, and future expansion markers. Archive canopy rooms, telescope access and upper council levels remain future shell-specific designs.

The waterworks has raised bank floors at world Y=3.0 m, a narrow crossing over the six-metre channel, and an inclined entry ramp from the existing Y=2.2 m approach. Its bridge underside is Y=2.85 m, above the Y≈2.24 m river. The crossing spans only the channel, meeting bank floors at their edges; its top does not overlap bank tops. The ramp ends at the crossing edge. This preserves flowing-water space underneath rather than filling the channel.

Generator: `apps/desktop/scripts/lunar-city/create-interior-cutaways.py`. Run **inside visible Blender** with `runpy.run_path(absolute_path, run_name='__main__')`. It creates a dedicated scene, full-height and cutaway collections, a labelled gallery, camera and light. GLB export uses both `use_selection=True` and `use_active_scene=True`; this is essential in the multi-scene workshop. It exports 28 local-origin GLBs, keeps exterior scenes intact, and preserves prior receipts before replacing current review exports. The parent executed v3 visibly and confirmed the previous waterworks floor overlap resolved. Runtime boxes and Blender geometry share the same metre-space plan; Blender converts `(x,y,z)` to `(x,-z,y)` and GLB returns Y-up/+Z-front.

## Runtime behavior

Selecting a building exposes a reversible cutaway toggle. Only that exterior is hidden; authored floor, cutaway walls, furniture and ramps become visible. Closing/switching restores the original visibility, including after LOD or occlusion updates. No global collision-disable flag is used.

Matched building OBBs are replaced by full-height wall/header/furniture colliders and actual floor triangles. Unmatched buildings and tree colliders remain intact. Indoor visibility graphs join only through declared approaches to the real outdoor Recast query. Every returned segment must pass continuous ground coverage and swept body collision. The inclined waterworks ramp is represented by sloped triangles, not a teleport or flat-floor approximation. Authored adjoining floor vertices retain double precision in the combined safety geometry; converting each rotated rectangle to float32 separately had introduced sub-millimetre seams after removing visual floor overlap.

A worker's already-authoritative destination maps to the corresponding indoor job-pickup anchor when physically reachable. This changes the presentation of an existing assignment; it does not create, accept or claim a backend job. Project assignments retain their existing project-site path. Source polling continues to preserve the actual actor position.

Leader life uses measured imported bounds, actual walk-clip motion, and local home/work/idle choices. Dialogue, pause and reduced motion hold positions. Static models cannot silently slide. The current pack has only the cat's actual walk group; the other seven current leader GLBs were static. Initial placement uses each validated home anchor, while safe cached positions take precedence on remount. Availability is exposed explicitly; held rig exports require individual QA and are not promoted by this feature. See the subsequent leader readiness receipt for the final census and actual motion results.

## Verification

- `validate-interior-plans.py`: 14 plans, 63 authored circulation segments clear the worker envelope; positive solid dimensions; door minima; water void remains below the elevated bridge.
- `interior-navigation.test.ts`: all 14 authored pickup paths, solid walls, every measured leader's home-door-work route, sampled waterworks slope, and rejection of a nonexistent floor in the river channel.
- Full Lunar City UI suite: 618 tests across 68 files passed in 6.97 seconds after interior navigation, leader persistence, measured clearance and cutaway integration. Scoped lint and renderer TypeScript also passed; subsequent final receipts identify any later source changes.
- `smoke-interior-navigation.mjs`: real browser imports and outdoor navigation GLB; all 14 movement controllers arrived, every segment stayed grounded and collision-safe, and depot-to-studio route crossed districts. An actual imported worker reached the arts-studio indoor pickup and remained at precisely the same position after a new source poll. Evidence: `evidence/interior-navigation-runtime.json` (2026-09-08T07:13:15Z).
- Cutaway browser checks: all 14 toggles, reversible selection, library and waterworks screenshots, no page errors or external requests. Evidence under `evidence/interiors-source/` and `evidence/interiors-built/`; final build/leader follow-up receipts identify their exact capture stage.

## Remaining acceptance limits

These are basic single-storey interior prototypes. Exterior wall/door alignment, organic shells and visual furnishing need continued Blender review; bounds containment alone does not prove a room fits an irregular shell. Building geometry, character materials and several held rigs remain under the parent's asset QA. Rest-pose collision bounds are conservative but do not certify every animated prop sweep. No live gateway mutation, external message, invented work achievement, full browser-reload persistence or packaged desktop acceptance is claimed.


## Final runtime receipt, 2026-09-08

The final built viewer on port5180 completed its production build in24.21 seconds and passed all14 cutaway interactions. Seven static leaders have explicit unavailable movement controls; the cat supports manual home/work/stroll and a reversible Automatic option. The parent also reloaded and inspected this built viewer visibly. No page errors or external requests were reported.

The only character asset replacement is the reviewed cat contact copy, SHA256 `a94817c2f48a03349f03c56b69646830b587a5eaca1d6340568386223f979e84`. Its original remains preserved. Actual CPU skin evaluation across41 walk samples found minimum foot clearance+0.1483 mm and maximum gap0.1997 mm. After subtracting global translation, deformation differs from the original by at most1.513 micrometres; all seven clips remain. This fixes the known12.475 mm penetration without claiming the remaining rig/material review is complete. Six other contact studies and the monkey rig remain held.

`leader-observed-life-runtime.json` proves that an authoritative session matched by exact connection/profile to a uniquely mapped leader triggers desk travel with actual walk frames. Observed idle releases it; stale data and ambiguous shared models hold. No observed session means return home, not invented work. Manual preview choices override only local presentation, and Automatic removes that override. This uses the existing finite owner-to-model mapping, not species/role-name inference or backend job creation.

`leader-life-runtime.json` was rerun on the corrected cat: all eight initial home placements, work/home walk-to-idle transitions, outdoor stroll, exact reduced-motion hold and a meaningful moved-position remount all pass. Initial static residents use their own measured interior collision envelope. Position retention is bounded in-memory renderer state (10-minute freshness, eight world scopes), not durable full-browser/app-reload persistence.

`interior-navigation-runtime.json` was refreshed at2026-09-08T07:47:36Z against v3 and the corrected cat pack: all14 physical controller arrivals, cross-district routing, actual imported-worker indoor arrival and identical position after polling pass. `physics-runtime.json` separately records all21 outdoor routes,28 collider checks and actual worker movement/poll/pause/remount. Full source UI suite618/618 and renderer TypeScript pass; scoped lint is clean after import-spacing cleanup. The final built cutaway receipt is `evidence/interiors-built/receipt.json`.
