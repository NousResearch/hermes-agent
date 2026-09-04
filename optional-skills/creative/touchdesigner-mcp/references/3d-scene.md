# 3D Scene Reference

Lighting rigs, shadows, IBL/cubemaps, multi-camera, and PBR materials. For wireframe rendering and feedback TOPs see `operator-tips.md`. For instancing geometry see `geometry-comp.md`. For shader code see `glsl.md`.

---

## Anatomy of a 3D Scene

```
[Geometry COMP]    ← contains SOPs (the shapes)
[Material]         ← Phong/PBR/GLSL/Constant MAT
[Light COMPs]      ← point/directional/spot/area/environment
[Camera COMP]      ← view position, FOV
        │
        ▼
   [Render TOP]    ← combines geo + lights + camera into a 2D image
        │
        ▼
   [post-FX chain] ← bloomTOP, glsl shaders, etc.
        │
        ▼
   [windowCOMP]    ← actual display
```

Render TOP is the heart. It takes an explicit `geometry` path, an explicit `camera` path, and lights via the lights table or an envlight reference.

---

## Minimal Scene

```python
# Geometry
geo = root.create(geometryCOMP, 'scene_geo')
sphere = geo.create(sphereSOP, 'shape')
sphere.par.rad = 1.0; sphere.par.rows = 64; sphere.par.cols = 64

# Material — start with PBR
mat = root.create(pbrMAT, 'mat')
mat.par.basecolorr = 0.7; mat.par.basecolorg = 0.7; mat.par.basecolorb = 0.7
mat.par.metallic = 0.0
mat.par.roughness = 0.4

geo.par.material = mat.path

# Camera
cam = root.create(cameraCOMP, 'cam1')
cam.par.tx = 0; cam.par.ty = 0; cam.par.tz = 4
cam.par.fov = 45
cam.par.near = 0.1; cam.par.far = 100

# Key light
key = root.create(lightCOMP, 'key_light')
key.par.lighttype = 'point'
key.par.tx = 3; key.par.ty = 3; key.par.tz = 3
key.par.dimmer = 1.5

# Render
render = root.create(renderTOP, 'render1')
render.par.outputresolution = 'custom'
render.par.resolutionw = 1920; render.par.resolutionh = 1080
render.par.camera = cam.path
render.par.geometry = geo.path
render.par.lights = key.path                 # single light path; for multi, see below
render.par.bgcolorr = 0; render.par.bgcolorg = 0; render.par.bgcolorb = 0
```

For multiple lights, leave `par.lights` blank — Render TOP scans the network for all `lightCOMP` and `envlightCOMP` ops by default. To restrict to specific lights, set `par.lights = '/project1/key_light /project1/fill_light'` (space-separated paths).

---

## Light Types

| Type | What | Common params |
|---|---|---|
| `point` | Omnidirectional, falls off with distance | `dimmer`, `coneangle` (n/a), `attenuation` |
| `directional` | Parallel rays, infinite distance (sun) | `dimmer`, light's rotation only matters |
| `spot` | Cone, falls off with distance + angle | `coneangle`, `conedelta`, `dimmer` |
| `cone` | Like spot but harder edge | same |
| `area` | Rectangular soft light source | `sizex`, `sizey` |

For all: `colorr`, `colorg`, `colorb`, `tx/ty/tz`, `rx/ry/rz`, `dimmer`.

### Three-Point Lighting (Studio Setup)

```python
# Key — main light, ~45° front
key = root.create(lightCOMP, 'key')
key.par.lighttype = 'point'
key.par.tx = 4; key.par.ty = 3; key.par.tz = 4
key.par.dimmer = 1.5
key.par.colorr = 1.0; key.par.colorg = 0.95; key.par.colorb = 0.85

# Fill — softer, opposite side
fill = root.create(lightCOMP, 'fill')
fill.par.lighttype = 'area'
fill.par.tx = -4; fill.par.ty = 2; fill.par.tz = 3
fill.par.dimmer = 0.5
fill.par.colorr = 0.7; fill.par.colorg = 0.8; fill.par.colorb = 1.0
fill.par.sizex = 4; fill.par.sizey = 4

# Rim/back — outline from behind
rim = root.create(lightCOMP, 'rim')
rim.par.lighttype = 'spot'
rim.par.tx = 0; rim.par.ty = 4; rim.par.tz = -4
rim.par.coneangle = 30
rim.par.dimmer = 1.0

# Optional: ambient lift to prevent pure-black shadows
amb = root.create(ambientlightCOMP, 'ambient')
amb.par.dimmer = 0.15
```

---

## Shadows

Spot and directional lights cast shadows when `par.shadowtype != 'none'`.

```python
key.par.shadowtype = 'softshadow'        # 'none' | 'hardshadow' | 'softshadow'
key.par.shadowsize = 1024                # shadow map resolution
key.par.shadowsoftness = 0.02            # softshadow only
```

**Tips:**
- Soft shadows are GPU-expensive. Start with `shadowsize = 1024` and only go higher (2048/4096) if shadow edges look pixelated at your resolution.
- Set the spot light's `near`/`far` to JUST contain the scene. Wider range = wasted shadow map precision.
- Multiple shadow-casting lights compound cost. Limit to 1-2 in real-time work; pre-bake the rest into the materials.

---

## Image-Based Lighting (IBL) / Environment Light

For realistic PBR materials you need a cubemap for reflections.

```python
# Environment light from an HDR
env = root.create(envlightCOMP, 'env')
env.par.envmap = '/project1/cube_in'         # path to a TOP that produces a cubemap
env.par.envlightmap = ...                    # diffuse irradiance map (often same as envmap)
env.par.dimmer = 1.0

# Cubemap source — option A: built-in cubeTOP from 6 faces
cube = root.create(cubeTOP, 'cube_in')
# (assign 6 face TOPs)

# Option B: HDR equirectangular → cubemap conversion
# Use a moviefileinTOP loading .hdr or .exr, then projectTOP type='cubemapfromequirect'
hdr = root.create(moviefileinTOP, 'hdr_src')
hdr.par.file = '/path/to/environment.hdr'

proj = root.create(projectTOP, 'cube_proj')
proj.par.projecttype = 'cubemapfromequirect'
proj.inputConnectors[0].connect(hdr)
```

PBR materials sample the environment automatically when `envlightCOMP` is in the scene. Verify param names with `td_get_par_info(op_type='envlightCOMP')` — TD versions vary.

---

## PBR Material Setup

```python
mat = root.create(pbrMAT, 'pbr_metal')
mat.par.basecolorr = 0.95; mat.par.basecolorg = 0.65; mat.par.basecolorb = 0.4
mat.par.metallic = 1.0
mat.par.roughness = 0.25
mat.par.specularlevel = 0.5
mat.par.emitcolorr = 0; mat.par.emitcolorg = 0; mat.par.emitcolorb = 0

# Texture maps
mat.par.basecolormap = '/project1/textures/albedo'         # TOP path
mat.par.metallicroughnessmap = '/project1/textures/mr'      # G=roughness, B=metallic (glTF convention)
mat.par.normalmap = '/project1/textures/normal'
mat.par.emitmap = '/project1/textures/emit'
mat.par.occlusionmap = '/project1/textures/ao'
```

**Material idioms:**

| Look | metallic | roughness | basecolor |
|---|---|---|---|
| Brushed steel | 1.0 | 0.4 | (0.7, 0.7, 0.7) |
| Polished gold | 1.0 | 0.1 | (1.0, 0.85, 0.4) |
| Plastic | 0.0 | 0.5 | mid-saturated |
| Rubber | 0.0 | 0.9 | dark |
| Glass | 0.0 | 0.05 | (1, 1, 1), low alpha + transmission |
| Glowing emitter | 0.0 | 1.0 | dark, high `emitcolor` |

For glass/transmission, recent TD versions support `transmission` in PBR; older versions need glslMAT.

---

## Multi-Camera Setups

For comparison views, instant replay, multi-screen mapping, etc.

```python
# Camera A — main scene
cam_a = root.create(cameraCOMP, 'cam_main')
cam_a.par.tz = 5

# Camera B — orbiting top-down
cam_b = root.create(cameraCOMP, 'cam_top')
cam_b.par.ty = 6; cam_b.par.rx = -90

# Render each via separate Render TOPs
render_a = root.create(renderTOP, 'render_main')
render_a.par.camera = cam_a.path
render_a.par.geometry = geo.path

render_b = root.create(renderTOP, 'render_top')
render_b.par.camera = cam_b.path
render_b.par.geometry = geo.path
```

Composite both with a `multiplyTOP`/`compositeTOP` for picture-in-picture, or route to separate `windowCOMP`s for multi-display.

### Camera animation

Drive camera params via expressions (orbit), animationCOMP (waypoint), or LFO (oscillation):

```python
# Orbiting camera
cam_a.par.tx.mode = ParMode.EXPRESSION
cam_a.par.tx.expr = "cos(absTime.seconds * 0.3) * 6"
cam_a.par.tz.mode = ParMode.EXPRESSION
cam_a.par.tz.expr = "sin(absTime.seconds * 0.3) * 6"
cam_a.par.lookat = '/project1/scene_geo'        # auto-aim at target
```

`par.lookat` is the simplest "always look at target" mechanism.

### Depth of field

PBR + Render TOP supports DOF when `par.dof = 'on'`.

```python
render.par.dof = 'on'
render.par.focusdistance = 5.0
render.par.aperture = 0.05         # blur strength
render.par.bokehshape = 'hexagon'
```

DOF is GPU-heavy. Render at lower res then upscale for performance.

---

## Common Pitfalls

1. **Render TOP shows black** — most common cause: no light. Even with PBR you need at least one `lightCOMP` or `envlightCOMP`. Add an `ambientlightCOMP` at low dimmer as a safety net.
2. **Material doesn't appear** — `geo.par.material` must be a string PATH, not the material op itself. Use `mat.path`, not `mat`.
3. **Lights ignored** — by default Render TOP picks up ALL `lightCOMP`s in the network. If you have leftover lights from another scene, they leak in. Set `par.lights` explicitly.
4. **PBR looks flat** — without an `envlightCOMP` providing reflections, PBR materials look like Phong. Add one even if you don't have an HDR (use a `constantTOP` cubemap as fallback).
5. **Shadow acne / striping** — increase `par.shadowbias` slightly. Tune per-light.
6. **Camera inside geometry** — if `cam.par.tz` is INSIDE a sphere, you see the inside (or nothing if backface culled). Move the camera further out.
7. **Light range too small** — point lights have implicit attenuation. Far-away geometry receives little light. Increase `par.dimmer` or move lights closer.
8. **Multiple cameras conflict** — one render TOP = one camera. Don't try to share. Use multiple render TOPs.
9. **Wrong handedness** — TD is right-handed Y-up. Imported assets from Z-up apps (Blender, Maya in Z-up) need a 90° X rotation on the geo COMP.
10. **Cooking budget** — PBR + IBL + shadows + DOF at 1080p60 is fine on modern GPUs but 4K + 4 lights + soft shadows + DOF will tank. Profile via `td_get_perf` and downgrade settings before adding more.

---

## Quick Recipes

| Goal | Recipe |
|---|---|
| Studio portrait | 3-point rig (key + fill + rim) + ambient + PBR mat + DOF |
| Outdoor daylight | One directional `lightCOMP` (sun) + envlight (sky HDR) + soft shadows |
| Dramatic / film noir | Single spot light from upper side, hard shadows, deep ambient = 0.05 |
| Abstract / dreamy | Multiple area lights at low dimmer, no shadows, `bloomTOP` post |
| Product render | Three-point + IBL + neutral PBR + `bgcolorr=g=b=1` (white seamless) |
| Game-style | Phong MAT + 1-2 lights + no IBL + flat ambient (cheap, stylized) |
| Wireframe + solid | Two render TOPs (one with wireframeMAT, one with PBR), composite via `addTOP` |
| Orbiting camera | `par.lookat` + expressions on tx/tz using sin/cos |
