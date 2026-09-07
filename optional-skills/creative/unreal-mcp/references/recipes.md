# Unreal MCP — Worked Recipes

Complete build sequences from a plain-English brief to a delivered capture.
Written against the discovery contract, because the live tool surface is
project-dependent: each step names the **capability to locate** (via
`list_toolsets` / `describe_toolset`) and the **exact values** to feed it —
not hardcoded tool names, which drift while the plugin is experimental.

## Recipe grammar

Every step = four parts:

    INTENT     what this step achieves
    DISCOVER   which toolset/tool capability to use (locate via describe_toolset)
    VALUES     the exact arguments/numbers (from scene-craft.md)
    VERIFY     the query or screenshot that proves it worked

Dispatch shape: `call_tool` with `toolset_name`, `tool_name`, and an
`arguments` object matching the described schema — result returns on the
same turn.

LIGHTING RULE for every recipe: set **Mobility = Movable** on every light
you spawn (Lumen GI ignores Static/Stationary lights — the #1 "why is GI
dead" cause).

Session preamble for every recipe (do once):

1. `list_toolsets` → note the qualified names (e.g.
   `editor_toolset.toolsets.scene.SceneTools`,
   `EditorToolset.EditorAppToolset`).
2. `describe_toolset` on each group you'll touch → cache schemas.
3. Query current level (`SceneTools.get_current_level`) and inventory the
   environment: `find_actors` for DirectionalLight, SkyAtmosphere,
   SkyLight, ExponentialHeightFog, PostProcessVolume, VolumetricCloud.
   **Configure existing environment actors; spawn only what's missing** —
   template levels ship with most of them, and duplicates compound into
   whiteouts.
4. Read the existing sun's `intensity` — it tells you the scene's exposure
   calibration (template worlds are often calibrated around `intensity: 10`,
   not physical lux; see pitfalls 12b before applying scene-craft absolute
   values).
5. Locate your verification path: `EditorAppToolset.CaptureViewport` with a
   `captureTransform` is the virtual camera — no viewport piloting needed.

Save the level + dirty packages after every phase marked 💾. One tool call
at a time throughout — no batching, ever.

---

## Recipe A — Daylight exterior clearing (blocking-first exterior)

Brief: "a sunny clearing with some rocks and a path"

**Phase 1 — environment shell**

- INTENT sky + sun + atmosphere exist and track each other.
  DISCOVER actor-spawn capability (spawn by class).
  VALUES spawn `SkyAtmosphere`, `SkyLight` (real-time capture),
  `DirectionalLight` at rotation (0, −55, 40) [roll, pitch, yaw],
  intensity 90,000 lux, temperature 5,800 K, "atmosphere sun light" on;
  `ExponentialHeightFog` density 0.008.
  VERIFY actor list shows all four; screenshot reads as daytime sky, not
  black (if black: exposure — see Phase 3).

**Phase 2 — ground & blocking**

- INTENT walkable ground plane.
  DISCOVER spawn-from-asset capability.
  VALUES `/Engine/BasicShapes/Plane.Plane` at (0,0,0), scale (100,100,1)
  → 100×100 m ground. Label `Ground`.
- INTENT rock cluster + path silhouette from primitives (placeholder for
  real assets if the project has none).
  VALUES 5–9 `/Engine/BasicShapes/Cube.Cube` at scattered locations within
  ±2,000 cm of origin, non-uniform scales between (1.5,1.5,1) and (4,3,2),
  yaws randomized 0–360°, sunk 10–30 cm into the ground so nothing floats.
  A path: 6–10 flattened cubes scale ≈(1.2,0.8,0.05) snaking through.
  VERIFY screenshot at eye height (camera z≈165) along the path axis: rocks
  read as varied, nothing floats, scale sane against the 180 cm yardstick
  (place one 180 cm-tall cylinder temporarily as a human stand-in, delete
  after checking). 💾

**Phase 3 — exposure & mood**

- INTENT deterministic exposure.
  DISCOVER PostProcessVolume spawn + property-set capability.
  VALUES PPV unbound=true, metering Manual, EV100 = 14.5.
  VERIFY screenshot: bright but not blown; shadows readable.

**Phase 4 — deliver**

- INTENT hero still.
  VALUES `HighResShot 3840x2160` from a framed viewpoint (see Recipe C
  Phase 2 for framing rules).
  VERIFY file exists in Saved/Screenshots; `vision_analyze` against brief;
  iterate lighting yaw/fog once if flat. 💾 Report actor labels + paths.

---

## Recipe B — Moody practical-lit interior

Brief: "a dim cozy room at night, warm lamp, blue moonlight through window"

**Phase 1 — room shell from primitives**

- VALUES floor: Cube at (0,0,−10) scale (6,6,0.2) → 6×6 m room. Four walls:
  cubes scale (6,0.2,3) / (0.2,6,3) positioned at ±300 on the respective
  axis, z=140 (walls 280 cm tall, sitting on the floor plane; keep tops at
  z≈290). Ceiling: cube scale (6,6,0.2) at z≈290 — spawn it LAST so you can
  screenshot the interior while open-topped. One window: leave a gap in a
  wall by using two shorter wall segments with a 120×120 cm opening at
  sill height 90 cm.
  VERIFY top-down + interior screenshots; door/window heights sane.

**Phase 2 — lighting (the point of this recipe)**

- INTENT kill the sun; interior reads as night.
  VALUES if the template level has a DirectionalLight: intensity → 0.05 lux
  temperature 4,300 K, pitch −20°, yawed to rake through the window (this
  is the "moon"). SkyLight intensity scale down to ≈0.05–0.1.
- INTENT warm practical.
  VALUES PointLight at lamp position (e.g. corner table, z≈120):
  800 lumens (or ≈64 candela), temperature 2,700 K, attenuation radius
  600 cm, source radius 10 cm (softer shadows).
- INTENT cool window rim.
  VALUES SpotLight outside the window aimed through it: 2,000 lumens,
  6,500–8,000 K if faking without moon; skip if the directional moon
  already rakes through visibly. Inner/outer cone 25°/50°.
- INTENT exposure for dim interior.
  VALUES PPV unbound, Manual, EV100 = 4.5; fog: ExponentialHeightFog
  density 0.015 + volumetric fog on; practical's volumetric scattering
  intensity 2–4 so the lamp glows.
  VERIFY screenshot from a corner at z≈160: warm pool around lamp, cool
  slash from window, deep-but-readable shadows. The warm/cool split IS the
  deliverable — iterate intensities (never move both at once) until it
  reads. 💾

**Phase 3 — dress & deliver**

- VALUES if Starter Content exists, swap primitives: `/Game/StarterContent/
  Props/SM_TableRound`, `SM_Chair`, `SM_Lamp_Ceiling`, materials
  `M_Wood_Pine` on floor, `M_Basic_Wall` on walls. Otherwise assign
  MaterialInstances with warm-neutral base colors to primitives.
  VERIFY final `HighResShot 3840x2160`, vision-check, 💾, report.

---

## Recipe C — Golden-hour cinematic still (camera craft)

Brief: "make it golden hour and give me a cinematic shot of <subject>"

**Phase 1 — relight for golden hour**

- VALUES DirectionalLight: intensity 12,000 lux, temperature 3,200 K,
  pitch −8°, yaw set so the sun is 30–60° OFF the camera axis behind the
  subject (rim + long shadows — never light flat from the camera).
  Fog density 0.02 + volumetric fog, sun volumetric scattering 2–6.
  PPV EV100 = 11.
  VERIFY screenshot: long shadows, warm rim on subject edges.

**Phase 2 — the camera**

- INTENT a framed shot WITHOUT touching the user's viewport.
  DISCOVER `EditorAppToolset.CaptureViewport` with `captureTransform` — a
  virtual camera; no CineCamera or viewport piloting needed for stills.
  VALUES position: subject-distance by lens-equivalent framing — for a
  prop/monument subject ~500–800 cm back, height 120–160 cm; rotation
  aimed so the subject sits on a thirds intersection, horizon in upper or
  lower third. Slight upward pitch (+2° to +5°) from below eye height
  reads heroic and guarantees sky/horizon in frame.
  For an actual CineCameraActor (user wants a camera in the level, DoF,
  or a Sequencer shot): spawn `/Script/CinematicCamera.CineCameraActor`,
  set focal/aperture/focus via ObjectTools on its CineCameraComponent,
  then capture with `captureTransform` matching its transform.
  VERIFY capture at viewport res first; iterate framing cheaply, then take
  the final.

**Phase 3 — deliver**

- VALUES `HighResShot 3840x2160` (or user's target res) through the
  piloted camera. For a sequence/turntable instead of a still: this needs a
  Level Sequence with a Camera Cut track + Movie Render Queue — treat as
  its own task; warn about first-render shader-compile stall.
  VERIFY read file, `vision_analyze`: rim light present? focus falloff on
  the right plane? horizon off-center? Iterate at most twice, then deliver
  MEDIA: path + what was changed. 💾

---

## Recipe D — Import an asset and populate the scene

Brief: "here's model.fbx / a Fab asset — put a ring of them around the fountain"

**Phase 1 — import**

- INTENT asset lands in `/Game/Imported` with no dialog stall.
  DISCOVER import capability (if none advertised: custom toolset wrapping
  `unreal.AssetImportTask` with `automated=True` — see tool-surface.md;
  the `automated` flag is what prevents a modal import dialog from
  freezing the whole MCP loop).
  VALUES destination `/Game/Imported`, save=true.
  VERIFY asset-exists query on the resulting long package name; spawn one
  instance at origin, screenshot, check scale against 180 cm yardstick —
  DCC exports are routinely 100× off (meters vs centimeters). Fix by
  actor scale or reimport with unit conversion.

**Phase 2 — populate**

- INTENT ring of N instances around a center C.
  VALUES for i in 0..N−1: angle θ=360·i/N, position = C + (r·cosθ,
  r·sinθ, 0) with r = fountain radius + clearance (e.g. 350 cm), yaw =
  θ+90° so each faces the center (or +270° to face outward — check one
  instance first and LOOK). Spawn one, verify facing, then loop the rest
  one call at a time.
  VERIFY count query matches N; screenshot from above (camera z≈1,500
  looking down) for spacing; eye-level screenshot for scale. 💾

**Phase 3 — deliver**

- Report: asset path, N instances with label prefix, level saved,
  overview + eye-level captures as MEDIA paths.

---

## When a recipe's capability is missing

If discovery shows no shipped tool for a step (no import tool, no
console-exec for HighResShot, etc.):

1. Say so plainly; don't fake the step.
2. Offer the custom-toolset path (tool-surface.md) — a 20-line Python
   toolset usually covers the gap; it needs `RefreshTools` + session
   restart to appear.
3. Or hand the user the one-liner to run in the editor's Python/console
   themselves, with exact text.

Never claim a phase done without its VERIFY evidence. The user can't check
the editor for you — the screenshots are the ground truth they see.
