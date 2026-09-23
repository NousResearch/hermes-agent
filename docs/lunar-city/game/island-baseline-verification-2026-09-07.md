# Island baseline verification — 2026-09-07

Baseline captured before the pending runtime physics wiring. No runtime, model or layout edits were made by this verification task. Source Vite was restarted on 5178 (owned session 44615); built server 5180 remained unchanged. Do not treat this baseline as validation of later physics edits.

- Production renderer build passed in 21.82 seconds: `vite build --outDir /private/tmp/lunar-city-renderer-build`; log `/private/tmp/lunar-island-renderer-build.log`.
- Built eight-building roster smoke passed at 2026-09-08T05:58:10.211Z: unique model/focus IDs, successful GLB loads, actual SHA matches and declared building height/scale checks. [Receipt](evidence/island-building-roster/receipt.json) and [built overview](evidence/island-building-roster/overview.png). No page errors.
- Source Recast navigation smoke passed at 2026-09-08T05:56:31.896Z: 21 routes, right-handed, max ground error 0.000000047684 m and max arrival error 0.000001738020 m. Non-node origins establish real Recast paths rather than exact-link fallback. [Receipt](evidence/navigation.json).
- Current reviewed navigation GLB is `models/navigation-5911a76dccc7.glb`; actual SHA-256 equals declared `5911a76dccc72b32b98f8dc2b6b65193a8dc0c5c22b2ab443f3e813b7d988b63`.
- Existing environment audit succeeded: `/private/tmp/lunar-environment-audit.json` and `/private/tmp/lunar-environment-overview.png`. This audit records loaded meshes/materials/lights; it is not a complete physical simulation test.

## Visual observations

The source overview was opened and inspected. Island shoreline and joined road network are visible. A dark straight river ribbon extends beyond the southeast island boundary into the surrounding blue water with a conspicuous rectangular end; the west inlet also has a straight dark segment. White roads and large pads remain visually high-contrast. These are review observations, not code-test failures or a request to normalize models. Root received the overview for art direction. Asset surface quality remains pending.

No command failed after the owned source-server restart. The initial attempt to stop parent session 1428 returned unknown process ID in this agent context; the parent stopped its own session before the new server was started. Headless Chromium ran with the normal macOS sandbox escalation; no native/CUA or foreground-browser surface was used. These are local source/built-viewer receipts, not packaged Electron, target hardware or authenticated live-gateway acceptance.
