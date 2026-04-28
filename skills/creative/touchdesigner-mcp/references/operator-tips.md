# Operator Tips

## Wireframe Rendering Pattern

Reusable setup for wireframe geometry on black background:

```python
# 1. Material
mat = root.create(wireframeMAT, 'wire_mat')
mat.par.colorr = 1.0; mat.par.colorg = 0.0; mat.par.colorb = 0.0
mat.par.linewidth = 3

# 2. Geometry COMP
geo = root.create(geometryCOMP, 'my_geo')
geo.par.rx.expr = 'absTime.seconds * 30'
geo.par.ry.expr = 'absTime.seconds * 45'
geo.par.material = mat.path  # NOTE: 'material' not 'mat'

# 3. Shape inside the geo
box = geo.create(boxSOP, 'cube')
box.par.sizex = 1.5; box.par.sizey = 1.5; box.par.sizez = 1.5

# 4. Camera
cam = root.create(cameraCOMP, 'cam1')
cam.par.tx = 0; cam.par.ty = 0; cam.par.tz = 4; cam.par.fov = 45

# 5. Render TOP
render = root.create(renderTOP, 'render1')
render.par.outputresolution = 'custom'
render.par.resolutionw = 1280; render.par.resolutionh = 720
render.par.bgcolorr = 0; render.par.bgcolorg = 0; render.par.bgcolorb = 0
render.par.camera = cam.path
render.par.geometry = geo.path

# 6. Output null
out = root.create(nullTOP, 'out1')
out.inputConnectors[0].connect(render.outputConnectors[0])
```

**Key rules:**
- Class names: `wireframeMAT` not `wireframeMat` (all-caps suffix)
- Geometry SOPs/POPs go INSIDE the geo comp
- Material: `geo.par.material` not `geo.par.mat`
- Render geometry: `render.par.geometry = geo.path` (string path)
- `wireframeMAT.par.wireframemode = 'topology'` for clean wireframe (vs `'tesselated'` for triangle edges)
- Alternative: Use `renderTOP.par.overridemat` instead of per-geo material

## Feedback TOP

### Basic Structure

```
input (initial state) ──┐
                        ├──→ feedback_top ──→ processing ──→ null_out
                        │                                        ↑
                        └── par.top = 'null_out' ────────────────┘
```

### Setup Pattern

```python
# 1. Processing chain
glsl = root.create(glslTOP, 'sim')
null_out = root.create(nullTOP, 'null_out')
glsl.outputConnectors[0].connect(null_out.inputConnectors[0])

# 2. Feedback referencing null_out
feedback = root.create(feedbackTOP, 'feedback')
feedback.par.top = 'null_out'

# 3. Black initial state
const_init = root.create(constantTOP, 'const_init')
const_init.par.colorr = 0; const_init.par.colorg = 0; const_init.par.colorb = 0

# 4. Wire: initial → feedback, feedback → processing
feedback.inputConnectors[0].connect(const_init)
glsl.inputConnectors[0].connect(feedback)

# 5. Reset to apply initial state
feedback.par.resetpulse.pulse()
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Not enough sources specified" | No input connected | Connect initial state TOP |
| Unexpected initial pattern | Wrong initial state | Use Constant TOP (black) |

### Tips

1. Use float format for simulations: `glsl.par.format = 'rgba32float'`
2. Reset after setup: `feedback.par.resetpulse.pulse()`
3. Match resolutions — feedback, processing, and initial state must match
4. Soft boundary prevents edge artifacts:
   ```glsl
   float edge = 3.0 * texel.x;
   float bx = smoothstep(0.0, edge, uv.x) * smoothstep(0.0, edge, 1.0 - uv.x);
   float by = smoothstep(0.0, edge, uv.y) * smoothstep(0.0, edge, 1.0 - uv.y);
   value *= bx * by;
   ```

### Use Cases
- **Wave Simulation** — R=height, G=velocity, black initial state
- **Cellular Automata** — white=alive, black=dead, random noise initial state
- **Trail / Motion Blur** — blend current frame with feedback, black initial
