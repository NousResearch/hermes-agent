# Unreal MCP — Scene-Craft Cheat Sheet

The numbers and conventions that make a scene read as *good* instead of
merely present. Sources: physical/photographic standards (stable), UE
conventions (stable), and practical ranges from production use (marked ≈).
UE-version-specific defaults drift; when a live schema or editor value
disagrees with this sheet, trust the editor and patch the sheet.

## Units & Conventions (bedrock — memorize)

| Thing | Convention |
|---|---|
| Distance | 1 Unreal Unit = **1 cm** |
| Axes | **Z-up**, X-forward, Y-right (left-handed) |
| Rotation | Rotator in **degrees**: Roll (around X), Pitch (around Y), Yaw (around Z) |
| Color | Linear RGBA, each channel 0–1 (`FLinearColor`) |
| Light color | Prefer `use_temperature` + Kelvin over tinting RGB |
| Scale | Multiplier per axis (1,1,1 = authored size) |

Directional-light aiming: the ROTATION points the light. Pitch −90° = sun
straight overhead (noon); pitch −5° to −15° = sun grazing the horizon
(golden hour); yaw picks the compass direction the light travels toward.

### Human-scale reference (sanity-check every layout against these)

| Reference | Size (cm) |
|---|---|
| Eye height (standing) | 160–175 |
| Door | 200–210 tall × 80–90 wide |
| Ceiling, residential | 240–300 |
| One building storey | 300–400 |
| Counter/desk height | 75–110 |
| Chair seat | 45 |
| Stair riser / tread | ≈18 / ≈28 |
| UE default mannequin | ≈180 tall |
| Car | ≈450 long × 180 wide × 145 tall |

If a "house" door comes out 400 cm tall, the scene reads as toy/giant.
Always place one human-scale object early as a yardstick.

## Content Paths

| Root | Meaning |
|---|---|
| `/Game/...` | Project content (Content/ folder) |
| `/Engine/...` | Engine-shipped content, present in every project |
| `/Script/Module.Class` | Native classes (e.g. `/Script/Engine.PointLight`) |

Long package name form: `/Game/Props/SM_Chair.SM_Chair` (package.object).

Always-available engine primitives (great for blocking before real assets):

    /Engine/BasicShapes/Cube.Cube          (100×100×100 cm at scale 1)
    /Engine/BasicShapes/Sphere.Sphere      (100 cm diameter)
    /Engine/BasicShapes/Cylinder.Cylinder  (100 cm ⌀ × 100 cm)
    /Engine/BasicShapes/Cone.Cone
    /Engine/BasicShapes/Plane.Plane        (100×100 cm, single-sided)

Their default material is plain grey; assign a MaterialInstance for anything
presentational. If the project has Starter Content, useful packs live under
`/Game/StarterContent/` (Props, Materials like `M_Basic_Wall`, `M_Wood_Pine`,
`M_Metal_Steel`, Particles). Query before assuming Starter Content exists.

Common actor classes for spawning: `StaticMeshActor`, `PointLight`,
`SpotLight`, `RectLight`, `DirectionalLight`, `SkyLight`,
`ExponentialHeightFog`, `SkyAtmosphere`, `VolumetricCloud`,
`PostProcessVolume`, `CameraActor`, `CineCameraActor`, `PlayerStart`.

## Lighting — physically based values

UE5 lights default to physical units (directional in lux, point/spot in
candela or lumens, exposure in EV100). Use real-world values; they
compose correctly with exposure instead of fighting it.

### Sun (DirectionalLight, lux)

**Calibration check first (live-verified):** template levels often ship a
sun at `intensity: 10` with auto-exposure tuned around it — physical lux
values below will blow such a scene to white. Read the existing sun's
intensity; if it's single/double digits, scale moods RELATIVE to it (noon =
template value, golden hour ≈ 0.5–0.7×, overcast ≈ 0.3×, night ≈ 0.01×)
and rely on temperature + pitch for the mood. The absolute table applies
when you own the whole exposure chain (manual EV100 + physical values
everywhere):

| Condition | Intensity (lux) | Pitch | Temperature |
|---|---|---|---|
| Noon, clear | 75,000–120,000 | −60° to −90° | 5,500–6,000 K |
| Afternoon | 40,000–75,000 | −30° to −50° | 5,000–5,500 K |
| Golden hour | 5,000–20,000 | −5° to −15° | 2,800–3,500 K |
| Overcast | 5,000–20,000 (soft) | −45° ± | 6,500–7,500 K |
| Blue hour / dusk | 10–100 | −2° to +5° | 8,000–12,000 K |
| Full-moon night | 0.05–0.3 | −30° to −60° | 4,000–4,500 K (cool-blue read comes from exposure + grade) |

Overcast: also drop directional shadow contrast (soften via larger source
angle) and let the sky light dominate.

### Sky light

One SkyLight per level, Real-Time Capture (SLS Captured Scene) when using
SkyAtmosphere — it then tracks the sun automatically. Don't stack multiple
sky lights; don't leave a stale static capture after big lighting changes
(recapture if not real-time).

### Local lights (point/spot/rect)

Rules of thumb in lumens (candela ≈ lumens/(4π) for a point light):

| Source | Lumens |
|---|---|
| Candle flame | 10–15 (≈1,850 K) |
| 40 W incandescent equiv. | 450 (2,700 K) |
| 60 W equiv. | 800 (2,700–3,000 K) |
| 100 W equiv. | 1,600 (3,000 K) |
| Bright ceiling fixture | 2,000–4,000 (3,000–4,000 K) |
| Fluorescent tube / office | 2,500–5,000 (4,000–5,000 K) |
| Streetlight (sodium) | 5,000–15,000 (≈2,000 K, orange) |
| Car headlight | 1,000–1,500 each (4,300–6,000 K) |
| Campfire | 100–300, flicker (1,700–2,000 K) |

Spot cone: inner 20–35°, outer 40–60° for a natural falloff. Attenuation
radius: keep tight (a few hundred cm for practicals) — giant radii cost
performance and flatten the scene. Cast-shadow off for pure fill lights.

### Color temperature vocabulary (Kelvin)

1,700–1,900 match/candle · 2,700 warm bulb · 3,200 tungsten studio ·
3,500 golden hour · 4,300 moonlight-read · 5,600 daylight/flash ·
6,500 overcast · 7,500–10,000 shade/blue hour.
Warm subject + cool ambient (or inverse) is the cheapest way to make a
shot read "lit" instead of "flat".

### Exposure (PostProcessVolume — the #1 "why is it black/white" knob)

Auto-exposure fights deterministic lighting reads. For agent-driven work,
prefer **manual exposure** in a PPV:

1. Spawn/locate a PostProcessVolume, set **Infinite Extent (Unbound) = true**.
2. Metering Mode = Manual, then set Exposure Compensation ≈ 0 and EV100 to
   the scene value:

| Scene | EV100 |
|---|---|
| Bright sun exterior | 14–16 |
| Overcast exterior | 11–13 |
| Golden hour | 10–12 |
| Bright interior (day, windows) | 7–9 |
| Dim practical-lit interior | 4–6 |
| Street at night | 2–4 |
| Moonlit exterior | −2 to 0 |

If you keep auto-exposure instead (Metering: Auto Histogram), clamp it:
Min/Max EV100 within ±2 of the target so it can't swim. Symptom table:
scene renders black with lights present → EV100 too high for the light
levels; blown white → EV100 too low.

### Global illumination & reflections

UE5 defaults: **Lumen** GI + Lumen reflections, no lightmass bake needed —
lighting is live; just keep "Allow Static Lighting" defaults alone.
**Critical: Lumen GI only considers lights with Movable mobility.** Spawned
lights can default to Stationary/Static and then contribute nothing to GI —
set Mobility = Movable explicitly on every light you place, and check
mobility first when "GI isn't working". Metal/mirror surfaces read
correctly only with something to reflect: give the scene a sky and
surroundings before judging materials.

### Fog & atmosphere

- **SkyAtmosphere** for a physically-plausible sky (sun disk, horizon
  gradient); pairs with directional light "Atmosphere Sun Light = true".
- **ExponentialHeightFog**: density default 0.02. ≈ Practical ranges:
  0.005–0.015 subtle depth cue · 0.02–0.05 moody/morning · 0.05–0.2 heavy.
  Enable **Volumetric Fog** on it for light shafts through it; then dial
  per-light "Volumetric Scattering Intensity" (1–10) on the key lights.
- **VolumetricCloud** for real sky clouds (exterior only; costs GPU).
- Night sky: drop fog density, add faint cool fill (skylight at low
  intensity) so shadows aren't pure black.

## Mood recipes (compact)

| Mood | Sun/Key | Sky | Fog | EV100 | Grade notes |
|---|---|---|---|---|---|
| Crisp noon | 100k lux, pitch −70°, 5,800 K | Real-time capture | 0.005 | 15 | Neutral |
| Golden hour | 10k lux, pitch −8°, 3,200 K | Real-time capture | 0.02 + volumetric | 11 | Warm key, long shadows: rotate yaw for rim/side light |
| Overcast | 10k lux soft, 7,000 K | Dominant | 0.01 | 12 | Low contrast, saturation carries color |
| Night, moonlit | 0.25 lux, 4,300 K + practicals ≈800 lm 2,700 K | Very low | 0.015 | −1 to 1 | Cool ambient vs warm practicals |
| Horror interior | No sun; 1–2 practicals, hard shadows | Minimal | 0.03–0.06 volumetric | 4–5 | Single motivated key, deep blacks |
| Sci-fi corridor | Rect lights 2,000 lm 6,500–8,000 K + colored accents | None | 0.02 volumetric | 6 | Complementary accent pair (cyan/orange) |

## Camera & framing (CineCameraActor)

Use CineCameraActor (not plain Camera) for anything presentational — it has
real filmback/lens/DoF controls.

| Intent | Focal length | Aperture |
|---|---|---|
| Establishing / interior wide | 18–28 mm | f/5.6–8 |
| Neutral "human eye" | 35–50 mm | f/4 |
| Portrait / subject isolation | 85–135 mm | f/1.4–2.8 |
| Compression (stacked background) | 100–200 mm | f/2.8–5.6 |

- Filmback: default 16:9 digital film (≈23.76 × 13.365 mm) is fine; leave it.
- Focus: Manual focus distance = distance camera→subject in cm; shallow DoF
  needs long lens + wide aperture + subject far from background.
- Placement: eye-level ≈ 155–170 cm for neutral shots; below ≈ 100 cm =
  heroic/imposing; high angle = diminishing. Keep the horizon off
  dead-center; put subjects on thirds. Slight camera pitch (−2° to −8°)
  usually beats perfectly level for interiors.
- Aspect/eye candy: enable camera's "Constrain Aspect Ratio" for clean
  letterboxed stills.

## Capture & render

- **Viewport screenshot**: console `HighResShot 1` (viewport res),
  `HighResShot 2` (2×), or `HighResShot 3840x2160`. Output:
  `<Project>/Saved/Screenshots/<Platform>/`. Filenames auto-increment
  (`HighresScreenshot00000.png`).
- To frame from a camera: pilot/possess the CineCamera (or set viewport to
  its view) before HighResShot; verify by screenshotting first at 1×.
- **Movie Render Queue** (MRQ) is the quality path for finals/sequences:
  needs a Level Sequence with the camera bound (Camera Cut track); renders
  PNG/EXR sequences or stills at arbitrary resolution with anti-aliasing
  temporal sample counts. First render after opening a project stalls on
  shader compilation — warn the user, don't declare it hung.
- Judge results by looking: read the file back and `vision_analyze` every
  capture against the brief.

## Editor Python quick reference

Custom toolsets and any shipped Python-execution tool speak the `unreal`
module. Canonical entry points (verify names against the live editor —
Epic migrates libraries to subsystems over time):

```python
import unreal

# Actors (EditorActorSubsystem supersedes EditorLevelLibrary for these)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
actors = eas.get_all_level_actors()
actor  = eas.spawn_actor_from_class(unreal.PointLight, unreal.Vector(0, 0, 200))
mesh_a = eas.spawn_actor_from_object(
    unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube"),
    unreal.Vector(0, 0, 50))
actor.set_actor_label("Key Light")
actor.set_actor_location(unreal.Vector(100, 0, 250), False, True)
actor.set_actor_rotation(unreal.Rotator(0, -30, 45), True)   # roll, pitch, yaw
eas.destroy_actor(actor)

# Assets
unreal.EditorAssetLibrary.does_asset_exist("/Game/Props/SM_Chair")
unreal.EditorAssetLibrary.list_assets("/Game/Props", recursive=True)
unreal.EditorAssetLibrary.save_directory("/Game", only_if_is_dirty=True)

# Level save
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
les.save_current_level()

# Undo-friendly mutation
with unreal.ScopedEditorTransaction("Agent: dress set") as trans:
    ...  # property edits inside are one undo step

# Import (FBX/textures)
task = unreal.AssetImportTask()
task.filename = "/abs/path/model.fbx"
task.destination_path = "/Game/Imported"
task.automated = True      # suppresses the import dialog — critical for MCP
task.save = True
unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

# Editor property access works on anything reflected
light_comp = actor.get_component_by_class(unreal.PointLightComponent)
light_comp.set_editor_property("intensity", 800.0)
light_comp.set_editor_property("use_temperature", True)
light_comp.set_editor_property("temperature", 2700.0)
```

`set_editor_property`/`get_editor_property` with snake_case names is the
universal fallback when a dedicated setter doesn't exist — property names
match what the Details panel shows (spaces removed).
