# Town physical layout

The review pack now emits physical boundaries from the final model placements. Building boxes use source bounds, metre scale, yaw and translated bounds centre after reviewed building registration. Tree cylinders use the rendered trunk radius and height; only the near LOD declares them, avoiding duplicates.

`navigation.colliders` belongs to the same generated manifest as the hashed navigation GLB. Runtime collision and floor validation is implemented in the world navigation layer; see its validation evidence for acceptance status. These are static solid boundaries for character movement, not a dynamic rigid-body simulation or final doorway/interior geometry.

Legacy depot, review office, triage, garden, release gatehouse and bus routing reservations now match their physical dimensions. Nearby district approaches and project pads were adjusted to stay outside these footprints. Ground and navigation use metres, with occupied walking surfaces at 2.2m. Large sea and sky backgrounds do not enlarge the occupied town.

The river follows both measured beaver channel ports, then passes beside the archive tree building. Its full sampled segments reserve 2.9m to unrelated building footprints. The outlet surface sinks into the sea beyond the coast, avoiding the earlier raised rectangular river tail. Final building water-port mesh cleanup and bridge art remain pending.

Road strips are unioned before triangulation. Regenerate after a layout change:

```sh
/private/tmp/lunar-landscape-tools/bin/python apps/desktop/scripts/lunar-city/build-road-surfaces.py
node apps/desktop/scripts/lunar-city/build-review-pack.mjs
node --test apps/desktop/scripts/lunar-city/settlement-layout.test.mjs apps/desktop/scripts/lunar-city/terrain-clearance.test.mjs apps/desktop/scripts/lunar-city/river-placement.test.mjs
```

The temporary Python environment uses Shapely 2.1.2. If it no longer exists, create a temporary venv and install `shapely==2.1.2`; the script path is reusable. The terrain generator verifies the checked-in road geometry hash against current routes before building, so outdated roads cannot silently ship.

All models remain review candidates. Surface, entrance and animation acceptance is separate from these layout checks.

The static review lineup also used the old 0.9m floor height. Leaders now stage at their role's authored approach and the two worker previews stage on separate road segments. All ten positions clear transformed building boxes with a 0.35m body margin. The latest pack contains 28 static colliders and 77,520 terrain triangles across both LODs, below its 90,000 triangle allowance.

Runtime acceptance now passes in `evidence/physics-runtime.json`: all 21 routes use swept collision and continuous floor validation, and an actual imported worker preserves position through polling, pause and world remount. The in-memory position cache lasts up to ten minutes and does not survive a browser restart. Focused tests also isolate boxes and trunks on a continuous floor and reject a 1cm floor gap. Dynamic rigid-body interactions and detailed held-tool collision are outside this movement implementation.

After that receipt, terrain-only edits corrected river triangle winding and lowered court pads 2cm beneath the road surface to avoid coplanar flicker. Navigation and collider placements were unchanged. Visual review is still required for overall landscape and model quality.
