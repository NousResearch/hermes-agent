# Layout Compositor Reference

Patterns for building modular multi-panel grids — useful for HUD interfaces, data dashboards, and multi-source visual composites.

## Layout Approaches

| Approach | Best For | Notes |
|----------|----------|-------|
| `layoutTOP` | Fixed grid, quick setup | GPU, simple tiling |
| Container COMP + `overTOP` | Full control, mixed-size panels | More setup, very flexible |
| GLSL compositor | Procedural / BSP-style | Most powerful, more complex |

---

## layoutTOP

Built-in grid compositor — fastest path for uniform tile grids.

```python
layout = root.create(layoutTOP, 'layout1')
layout.par.resolutionw = 1920
layout.par.resolutionh = 1080
layout.par.cols = 3
layout.par.rows = 2
layout.par.gap = 4
```

Connect inputs (up to cols×rows):
```python
layout.inputConnectors[0].connect(op('panel_radar'))
layout.inputConnectors[1].connect(op('panel_wave'))
layout.inputConnectors[2].connect(op('panel_data'))
```

**Variable-width columns:** Not directly supported. Use overTOP approach for non-uniform grids.

---

## Container COMP Grid

Build each element as its own `containerCOMP`. Compose with `overTOP`:

```python
def create_panel(root, name, width, height, x=0, y=0):
    panel = root.create(containerCOMP, name)
    panel.par.w = width
    panel.par.h = height
    panel.viewer = True
    return panel

# Composite with overTOP chain
over1 = root.create(overTOP, 'over1')
over1.inputConnectors[0].connect(panel_radar)
over1.inputConnectors[1].connect(panel_wave)
over1.par.topx2 = 0
over1.par.topy2 = 512
```

**Tip:** Use a `resolutionTOP` before each `overTOP` input if panels are different sizes.

---

## Panel Dividers (GLSL)

```glsl
out vec4 fragColor;
uniform vec2 uGridDivisions;   // e.g. vec2(3, 2) for 3 cols, 2 rows
uniform float uLineWidth;      // pixels
uniform vec4 uLineColor;       // e.g. vec4(0.0, 1.0, 0.8, 0.6) for cyan

void main() {
    vec2 res = uTDOutputInfo.res.zw;
    vec2 uv = vUV.st;
    vec4 bg = texture(sTD2DInputs[0], uv);

    float lineW = uLineWidth / res.x;
    float lineH = uLineWidth / res.y;

    float vDiv = 0.0;
    for (float i = 1.0; i < uGridDivisions.x; i++) {
        float x = i / uGridDivisions.x;
        vDiv = max(vDiv, step(abs(uv.x - x), lineW));
    }

    float hDiv = 0.0;
    for (float i = 1.0; i < uGridDivisions.y; i++) {
        float y = i / uGridDivisions.y;
        hDiv = max(hDiv, step(abs(uv.y - y), lineH));
    }

    float line = max(vDiv, hDiv);
    vec4 result = mix(bg, uLineColor, line * uLineColor.a);
    fragColor = TDOutputSwizzle(result);
}
```

---

## Element Library Pattern

Each visual element lives in its own `baseCOMP` as a reusable `.tox`:

### Standard Interface
```
inputs:
  - in_audio   (CHOP)  — audio envelope / beat data
  - in_data    (CHOP)  — optional data stream
  - in_control (CHOP)  — intensity, color, speed params

outputs:
  - out_top    (TOP)   — rendered element
```

### Network Structure
```
/project1/
  audio_bus/          ← all audio analysis (see audio-reactive.md)
  elements/
    elem_radar/       ← baseCOMP with out_top
    elem_wave/
    elem_data/
  compositor/
    layout1           ← layoutTOP or overTOP chain
    dividers1         ← GLSL divider lines
    postfx/           ← bloom → chrom → CRT stack (see postfx.md)
      null_out        ← final output
  output/
    windowCOMP        ← full-screen output
```

**Key principle:** Elements don't know about each other. The compositor assembles them. Audio bus is referenced by all elements but lives separately.
