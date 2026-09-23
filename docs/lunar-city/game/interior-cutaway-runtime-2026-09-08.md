# Authored interior cutaway runtime

The local review pack now exposes **See inside / Show exterior** when a building is selected. The ordinary Lunar City route also shows this control when explicitly using that review pack. These are prototype authored floorplans; shell fit and art acceptance remain separate.

The production loader bundles `public/lunar-city/interior-plans-v1/plans.json`; it does not fetch an arbitrary plan URL. `world/interior-plan-data.ts` requires each plan's model ID, URI, SHA-256 and position/rotation/scale to match the loaded manifest. Only the same-origin review manifest opts in. The source manifest SHA is retained on the cutaway root as provenance, while model bytes still pass the existing runtime digest verifier.

`world/interior-cutaway.ts` creates only the selected interior. It uses authored metre dimensions and exterior yaw/position without applying exterior scale twice. Walls retain their authored base elevation and are cut to 0.9 metres. Headers and ceilings are omitted; floor collars, desks, benches, planters and inclined ramp walking surfaces remain. It hides opaque exterior mesh visibility only, preserving mesh enablement and collision settings. LOD/occlusion updates reapply the cutaway. Closing, switching or disposal restores original exterior visibility and disposes the generated meshes/materials.

`setInteriorBuilding(id?)` and `getInteriorBuildings()` are forwarded through the world handle. Camera selection changes close the previous cutaway. The review UI also exposes Automatic observed activity and local leader home/desk/stroll intent via the separate leader-life controller. Automatic clears the manual override. These controls do not issue backend work commands. Seven current imported leaders lack usable walk clips and have disabled movement controls with an explicit explanation; the accepted cat supports movement.

## Verification

- 70 focused tests passed across the cutaway/binding/UI and existing create-world/world-scene/route suites after leader-life scene integration; renderer TypeScript also passed.
- The expanded Babylon NullEngine cutaway test checks ramp top endpoints, wall height/base, metre placement, portal collars, exact visibility restoration, late LOD handling and disposal.
- Actual source browser: all 14 authored plans opened and closed, selecting overview reset the toggle, and no page errors or external requests occurred.
- Inspected `evidence/interiors-source/library.png` and `evidence/interiors-source/engineering-workshop.png`: library rooms/desks and the waterworks bank rooms/catwalk/approach ramp are visible. Receipt: `evidence/interiors-source/receipt.json`.

Final production build passed in 24.21 seconds and refreshed port 5180. The final built browser receipt at `evidence/interiors-built/receipt.json` verifies all 14 interior toggles, selection reset, seven explained disabled movement controls, and the cat's home/desk/stroll choices plus Automatic reset after reselecting the actor. There were no page errors or external requests. The served manifest SHA-256 is `8e6ad2140e8f14c0446f2670473032c21b7233a145920aad2baff8575d69e6c1`; the accepted cat URI is `models/review-cat-a94817c2f48a.glb`. The receipt records that manifest bytes were sampled just after the UI run. Final focused 70 tests, renderer TypeScript, owned-file lint and diff checks passed.

Final built screenshots are `evidence/interiors-built/library.png` and `evidence/interiors-built/engineering-workshop.png`. The waterworks image was inspected: the beaver is at its interior home, the bank rooms and connecting catwalk are visible, and the river remains clear below. The separate navigation/leader agent owns its actual movement and authority receipts. These screenshots do not certify shell fit, animation beauty, packaged Electron performance or backend authority.
