# 3D Anatomy Explorer — design & plan

An interactive anatomical model on the Health page that ties into everything
medical: browse the body, toggle layers, select structures, and — when you're
learning a condition — see where on the body it's affected.

## Adaptive fidelity (one feature, three renderers)

All tiers share the **same data layer** (structure registry + condition→region
map + `hub:anatomy-highlight` event bus). Only the renderer swaps by device.

| Tier | Device | Renderer |
|------|--------|----------|
| **A — high detail** | Desktop/laptop w/ GPU | High-poly labelled atlas (Phase 3, on-demand load) |
| **B — compact 3D** | Phones/tablets w/ WebGL | Procedural three.js model, major structures per layer |
| **C — 2D map** | No WebGL / weak / offline | Layered interactive SVG body (front & back) |

Tier is auto-detected (WebGL2, `deviceMemory`, `hardwareConcurrency`, screen)
with a manual **Quality** override (Auto / 3D / 2D) so the user is never stuck.

## Data layer

- `public/anatomy/structures.json` — `{ id, name, layer, blurb, region, pos }`
  keyed to both the SVG region ids and the 3D mesh names.
- `public/anatomy/conditions.json` — `{ slug, name, structures[] }` mapping a
  condition to the structures it affects (SA-relevant conditions prioritised:
  TB, HIV, diabetes, hypertension, …). Curated & reviewable; "educational,
  verify clinically", matching how MedBot/drug already behave.

## Cross-widget integration

A window `CustomEvent("hub:anatomy-highlight", { detail: { slug | structures } })`
mirrors the existing `hub:medbot-ask` bridge. Any widget can drive the model:
- MedBot answer mentions a mapped condition → highlight it.
- Med Ed / OSCE station → highlight the relevant structures.
- Selecting a structure → **"Ask SA MedBot about this"** (reuses the bridge).

## Rendering (Tier B — compact 3D)

- three.js (vendored ESM core at `public/js/vendor/three/`, ~650 KB, lazy-loaded
  **only** when the widget mounts — never slows the rest of the dashboard).
- Procedural body built from primitives, grouped by layer (skin / muscle /
  skeleton / organs); each mesh named to a structure id.
- Minimal built-in orbit controls (drag-rotate, wheel/pinch-zoom) — no addon,
  so no import-map needed. Raycaster picking → emissive highlight + info panel.
- Layer toggles set `group.visible`; "ghost skin" = translucent skin material.
- Render loop pauses when the widget is unmounted (tab switch already unmounts).

## Rendering (Tier C — 2D SVG)

- Hand-authored front/back body SVG with clickable region groups sharing the
  same structure ids. Highlight = accent fill. Always works; drives e2e coverage
  (3D pixels can't be asserted reliably).

## Phasing

- **Phase 1 (this PR):** data layer + Tier C (2D) + Tier B (procedural 3D) +
  auto-detection + Quality override + MedBot bridge + curated conditions + tests.
- **Phase 2:** search-to-structure, view presets, ghost-skin, Med Ed/OSCE + drug
  bridges, more conditions, richer 3D organs.
- **Phase 3:** Tier A high-detail atlas via a Blender decimation pipeline
  (Z-Anatomy / BodyParts3D, CC-BY-SA) loaded on demand on capable devices;
  nervous/vascular layers; cross-section clipping plane.

## Attribution & licensing

- three.js — MIT (`public/js/vendor/three/LICENSE`).
- Phase-3 atlas assets — CC-BY-SA; attribution shown in the widget footer.

## Testing

- Unit: condition-map integrity (every referenced structure id exists).
- e2e: widget mounts; 2D regions render + click-highlight; layer toggle; Quality
  switch; `hub:anatomy-highlight` highlights the right region; MedBot bridge.
