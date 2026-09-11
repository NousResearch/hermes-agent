# Projection Mapping Reference

Multi-window output, surface mapping, edge blending, and projector calibration patterns for installation/event work.

For HUD layouts and on-screen panel grids, see `layout-compositor.md`. For wireframe/test-pattern generation, see `operator-tips.md`.

---

## Window COMP — Output to a Display

The `windowCOMP` is how TD pushes pixels to a real display.

```python
win = root.create(windowCOMP, 'output_window')
win.par.winop = '/project1/final_out'   # path to the TOP being displayed
win.par.winw = 1920
win.par.winh = 1080
win.par.winoffsetx = 0                  # screen-space offset
win.par.winoffsety = 0
win.par.borders = False                 # no chrome
win.par.alwaysontop = True
win.par.cursor = False                  # hide cursor in fullscreen
win.par.justify = 'fillaspect'          # 'fill' | 'fitaspect' | 'fillaspect' | 'native'
win.par.winopen.pulse()                 # OPEN the window
```

To target a specific physical display, set `par.location`:

```python
win.par.location = 'secondary'          # 'primary' | 'secondary' | 'monitor1' | 'monitor2' | ...
```

Or set absolute coordinates using `winoffsetx/y` matched to your OS display layout.

**Always pulse `winopen` — setting params alone doesn't open the window.**

---

## Multi-Window Output

For multi-projector or multi-display setups, create one `windowCOMP` per output, each pointing at a different TOP.

```python
for i, screen_top in enumerate(['out_left', 'out_center', 'out_right']):
    w = root.create(windowCOMP, f'win_{i}')
    w.par.winop = f'/project1/{screen_top}'
    w.par.winw = 1920; w.par.winh = 1080
    w.par.winoffsetx = i * 1920
    w.par.winoffsety = 0
    w.par.borders = False
    w.par.alwaysontop = True
    w.par.cursor = False
    w.par.winopen.pulse()
```

For ultra-wide single-output spans, use ONE windowCOMP at e.g. 5760×1080 spanning three projectors via the GPU's mosaic/spanning mode (Nvidia Mosaic, AMD Eyefinity), then split content via `cropTOP` per screen inside TD.

---

## 4-Point Corner Pin (Quad Warp)

The simplest projection mapping primitive — warping a rectangle onto a quadrilateral.

```python
# Source content
src = op('/project1/scene_out')

# Manual: cornerPinTOP (TD has this built-in)
cp = root.create(cornerPinTOP, 'corner_pin')
cp.par.tlx = 0.05; cp.par.tly = 0.10    # top-left (normalized 0-1)
cp.par.trx = 0.95; cp.par.try = 0.08    # top-right
cp.par.brx = 0.93; cp.par.bry = 0.92    # bottom-right
cp.par.blx = 0.07; cp.par.bly = 0.94    # bottom-left
cp.inputConnectors[0].connect(src)
```

Alternative: use a `geometryCOMP` with a `gridSOP` and bend the verts in vertex GLSL. More flexible (curved surfaces) but more setup.

Verify TD 2025.32 param names with `td_get_par_info(op_type='cornerPinTOP')`.

---

## Bezier / Mesh Warp (Curved Surfaces)

For non-flat surfaces (domes, columns, curved walls), use a subdivided mesh and per-vertex displacement.

### Pattern: Grid Mesh + GLSL Displacement

```python
# Subdivided grid in a geo
geo = root.create(geometryCOMP, 'warp_geo')
grid = geo.create(gridSOP, 'warp_grid')
grid.par.rows = 32          # higher = smoother curve
grid.par.cols = 32
grid.par.sizex = 2; grid.par.sizey = 2

# Texture the source onto it
mat = root.create(constMAT, 'warp_mat')      # use constMAT for unlit projection
mat.par.maptop = '/project1/scene_out'        # source TOP

geo.par.material = mat.path

# Render to a TOP that goes to the projector window
cam = root.create(cameraCOMP, 'cam_proj')
cam.par.tz = 4

render = root.create(renderTOP, 'projection_out')
render.par.camera = cam.path
render.par.geometry = geo.path
render.par.outputresolution = 'custom'
render.par.resolutionw = 1920; render.par.resolutionh = 1080
```

For per-vertex offsets, write a vertex GLSL on the constMAT (or use `glslMAT`) and read displacement values from a CHOP via uniform.

Calibration is iterative: render a checkerboard from `scene_out`, project it, photograph the projection, manually nudge corner/grid points until aligned.

---

## Edge Blending (Multi-Projector Overlap)

When two projectors overlap, the overlap region is twice as bright. Blend by ramping each projector's edge alpha to 0 across the overlap zone.

### GLSL Edge Blend Shader

Per-projector output pass that fades the inside edge to black:

```glsl
// edge_blend_pixel.glsl
out vec4 fragColor;
uniform float uBlendLeft;     // overlap width on left edge (0-0.5, 0=no blend)
uniform float uBlendRight;
uniform float uGamma;          // typically 2.2 — perceptual ramp

void main() {
    vec2 uv = vUV.st;
    vec4 col = texture(sTD2DInputs[0], uv);

    float aL = (uBlendLeft  > 0.0) ? smoothstep(0.0, uBlendLeft, uv.x) : 1.0;
    float aR = (uBlendRight > 0.0) ? smoothstep(0.0, uBlendRight, 1.0 - uv.x) : 1.0;
    float a = pow(aL * aR, uGamma);

    fragColor = TDOutputSwizzle(vec4(col.rgb * a, 1.0));
}
```

Apply this to each overlap-touching projector's output. Tune `uBlendLeft` / `uBlendRight` to match your physical overlap.

For top/bottom blends or cylindrical setups, extend the shader with `uBlendTop` / `uBlendBottom`.

---

## Calibration Patterns

Useful test patterns for aligning projectors. Build a `switchTOP` selecting one of these, route to all projector windows during setup.

```python
# Solid white — for brightness/uniformity check
white = root.create(constantTOP, 'cal_white')
white.par.colorr = 1.0; white.par.colorg = 1.0; white.par.colorb = 1.0

# Centered crosshair — for keystone alignment
gridcross = root.create(textTOP, 'cal_cross')
gridcross.par.text = '+'
gridcross.par.fontsizex = 200

# Fine grid — for warp/mesh alignment (use rampTOP + math + threshold, or build via GLSL)
# Color bars for projector color calibration
bars = root.create(rampTOP, 'cal_bars')
bars.par.type = 'horizontal'
```

Or use the bundled `testpatternTOP` if your TD version includes it.

---

## Projection Audit Workflow

When debugging a multi-screen setup:

1. Render a unique color and label per output (`textTOP` saying "LEFT", "CENTER", "RIGHT").
2. Check that each window is sourcing the correct path: `td_get_operator_info(path='/project1/win_0')`.
3. Verify display assignment: walk to each projector and confirm visually.
4. Check resolution: physical projector native res vs. TD output res — mismatches cause scaling artifacts.
5. Cook flag: `td_get_perf` — if a window's source TOP isn't cooking, the projector shows last frame frozen.

---

## Pitfalls

1. **Window won't open** — you forgot `winopen.pulse()`. Setting params alone doesn't open it.
2. **Wrong display** — `par.location='secondary'` depends on OS display order. Set `winoffsetx/y` to absolute coords as a more reliable override.
3. **Cursor visible** — set `par.cursor = False` BEFORE opening, or close+reopen.
4. **Black projection** — usually a cooking issue. Verify `final_out` TOP is cooking via `td_get_perf`. Check `td_get_errors` recursively from `/`.
5. **Tearing / vsync** — `windowCOMP` honors `par.vsync`. For projection always set `vsync='vsync'` (default). Tearing means GPU is over-budget — reduce render resolution.
6. **Aspect mismatch** — projector native is often 1920×1200 (16:10) not 1080. Use `justify='fitaspect'` or render at native projector res.
7. **Non-Commercial license** — caps total resolution at 1280×1280. For real installation work you need Commercial. Pro license adds 4K+.
8. **Multiple monitors on macOS** — `windowCOMP` honors macOS Spaces. Disable Spaces or pin TD to a specific display in System Settings before showtime.

---

## Quick Recipes

| Goal | Approach |
|---|---|
| Single fullscreen output | One `windowCOMP`, `justify='fillaspect'`, `winopen.pulse()` |
| 3-projector wide span | 3 `windowCOMP` + per-output `cropTOP` from one wide source |
| Single quad surface | `cornerPinTOP` → `windowCOMP` |
| Curved/dome | Subdivided gridSOP with vertex GLSL → `renderTOP` → `windowCOMP` |
| Edge blend overlap | GLSL fade shader per projector → `windowCOMP` |
| Calibration mode | `switchTOP` between scene and test patterns, hot-key triggered |
