---
name: touchdesigner-mcp
description: "Control a running TouchDesigner instance via twozero MCP — create operators, set parameters, wire connections, execute Python, build real-time visuals. 36 native tools."
version: 1.1.0
author: kshitijk4poor
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [TouchDesigner, MCP, twozero, creative-coding, real-time-visuals, generative-art, audio-reactive, VJ, installation, GLSL]
    related_skills: [native-mcp, ascii-video, manim-video, hermes-video]

---

# TouchDesigner Integration (twozero MCP)

## CRITICAL RULES

1. **NEVER guess parameter names.** Call `td_get_par_info` for the op type FIRST. Your training data is wrong for TD 2025.32.
2. **If `tdAttributeError` fires, STOP.** Call `td_get_operator_info` on the failing node before continuing.
3. **NEVER hardcode absolute paths** in script callbacks. Use `me.parent()` / `scriptOp.parent()`.
4. **Prefer native MCP tools over td_execute_python.** Use `td_create_operator`, `td_set_operator_pars`, `td_get_errors` etc. Only fall back to `td_execute_python` for complex multi-step logic.
5. **Call `td_get_hints` before building.** It returns patterns specific to the op type you're working with.

## Architecture

```
Hermes Agent -> MCP (Streamable HTTP) -> twozero.tox (port 40404) -> TD Python
```

36 native tools. Free plugin (no payment/license — confirmed April 2026).
Context-aware (knows selected OP, current network).
Hub health check: `GET http://localhost:40404/mcp` returns JSON with instance PID, project name, TD version.

## Setup (Automated)

Run the setup script to handle everything:

```bash
bash "${HERMES_HOME:-$HOME/.hermes}/skills/creative/touchdesigner-mcp/scripts/setup.sh"
```

The script will:
1. Check if TD is running
2. Download twozero.tox if not already cached
3. Add `twozero_td` MCP server to Hermes config (if missing)
4. Test the MCP connection on port 40404
5. Report what manual steps remain (drag .tox into TD, enable MCP toggle)

### Manual steps (one-time, cannot be automated)

1. **Drag `~/Downloads/twozero.tox` into the TD network editor** → click Install
2. **Enable MCP:** click twozero icon → Settings → mcp → "auto start MCP" → Yes
3. **Restart Hermes session** to pick up the new MCP server

After setup, verify:
```bash
nc -z 127.0.0.1 40404 && echo "twozero MCP: READY"
```

## Environment Notes

- **Non-Commercial TD** caps resolution at 1280×1280. Use `outputresolution = 'custom'` and set width/height explicitly.
- **Codecs:** `prores` (preferred on macOS) or `mjpa` as fallback. H.264/H.265/AV1 require a Commercial license.
- Always call `td_get_par_info` before setting params — names vary by TD version (see CRITICAL RULES #1).

## Workflow

### Step 0: Discover (before building anything)

```
Call td_get_par_info with op_type for each type you plan to use.
Call td_get_hints with the topic you're building (e.g. "glsl", "audio reactive", "feedback").
Call td_get_focus to see where the user is and what's selected.
Call td_get_network to see what already exists.
```

No temp nodes, no cleanup. This replaces the old discovery dance entirely.

### Step 1: Clean + Build

**IMPORTANT: Split cleanup and creation into SEPARATE MCP calls.** Destroying and recreating same-named nodes in one `td_execute_python` script causes "Invalid OP object" errors. See pitfalls #11b.

Use `td_create_operator` for each node (handles viewport positioning automatically):

```
td_create_operator(type="noiseTOP", parent="/project1", name="bg", parameters={"resolutionw": 1280, "resolutionh": 720})
td_create_operator(type="levelTOP", parent="/project1", name="brightness")
td_create_operator(type="nullTOP", parent="/project1", name="out")
```

For bulk creation or wiring, use `td_execute_python`:

```python
# td_execute_python script:
root = op('/project1')
nodes = []
for name, optype in [('bg', noiseTOP), ('fx', levelTOP), ('out', nullTOP)]:
    n = root.create(optype, name)
    nodes.append(n.path)
# Wire chain
for i in range(len(nodes)-1):
    op(nodes[i]).outputConnectors[0].connect(op(nodes[i+1]).inputConnectors[0])
result = {'created': nodes}
```

### Step 2: Set Parameters

Prefer the native tool (validates params, won't crash):

```
td_set_operator_pars(path="/project1/bg", parameters={"roughness": 0.6, "monochrome": true})
```

For expressions or modes, use `td_execute_python`:

```python
op('/project1/time_driver').par.colorr.expr = "absTime.seconds % 1000.0"
```

### Step 3: Wire

Use `td_execute_python` — no native wire tool exists:

```python
op('/project1/bg').outputConnectors[0].connect(op('/project1/fx').inputConnectors[0])
```

### Step 4: Verify

```
td_get_errors(path="/project1", recursive=true)
td_get_perf()
td_get_operator_info(path="/project1/out", detail="full")
```

### Step 5: Display / Capture

```
td_get_screenshot(path="/project1/out")
```

Or open a window via script:

```python
win = op('/project1').create(windowCOMP, 'display')
win.par.winop = op('/project1/out').path
win.par.winw = 1280; win.par.winh = 720
win.par.winopen.pulse()
```

## MCP Tool Quick Reference

**Core (use these most):**
| Tool | What |
|------|------|
| `td_execute_python` | Run arbitrary Python in TD. Full API access. |
| `td_create_operator` | Create node with params + auto-positioning |
| `td_set_operator_pars` | Set params safely (validates, won't crash) |
| `td_get_operator_info` | Inspect one node: connections, params, errors |
| `td_get_operators_info` | Inspect multiple nodes in one call |
| `td_get_network` | See network structure at a path |
| `td_get_errors` | Find errors/warnings recursively |
| `td_get_par_info` | Get param names for an OP type (replaces discovery) |
| `td_get_hints` | Get patterns/tips before building |
| `td_get_focus` | What network is open, what's selected |

**Read/Write:**
| Tool | What |
|------|------|
| `td_read_dat` | Read DAT text content |
| `td_write_dat` | Write/patch DAT content |
| `td_read_chop` | Read CHOP channel values |
| `td_read_textport` | Read TD console output |

**Visual:**
| Tool | What |
|------|------|
| `td_get_screenshot` | Capture one OP viewer to file |
| `td_get_screenshots` | Capture multiple OPs at once |
| `td_get_screen_screenshot` | Capture actual screen via TD |
| `td_navigate_to` | Jump network editor to an OP |

**Search:**
| Tool | What |
|------|------|
| `td_find_op` | Find ops by name/type across project |
| `td_search` | Search code, expressions, string params |

**System:**
| Tool | What |
|------|------|
| `td_get_perf` | Performance profiling (FPS, slow ops) |
| `td_list_instances` | List all running TD instances |
| `td_get_docs` | In-depth docs on a TD topic |
| `td_agents_md` | Read/write per-COMP markdown docs |
| `td_reinit_extension` | Reload extension after code edit |
| `td_clear_textport` | Clear console before debug session |

**Input Automation:**
| Tool | What |
|------|------|
| `td_input_execute` | Send mouse/keyboard to TD |
| `td_input_status` | Poll input queue status |
| `td_input_clear` | Stop input automation |
| `td_op_screen_rect` | Get screen coords of a node |
| `td_click_screen_point` | Click a point in a screenshot |
| `td_screen_point_to_global` | Convert screenshot pixel to absolute screen coords |

The table above covers the 32 tools used in typical creative workflows. The remaining 4 tools (`td_project_quit`, `td_test_session`, `td_dev_log`, `td_clear_dev_log`) are admin/dev-mode utilities — see `references/mcp-tools.md` for the full 36-tool reference with complete parameter schemas.

## Key Implementation Rules

**GLSL time:** No `uTDCurrentTime` in GLSL TOP. Use the Values page:
```python
# Call td_get_par_info(op_type="glslTOP") first to confirm param names
td_set_operator_pars(path="/project1/shader", parameters={"value0name": "uTime"})
# Then set expression via script:
# op('/project1/shader').par.value0.expr = "absTime.seconds"
# In GLSL: uniform float uTime;
```

Fallback: Constant TOP in `rgba32float` format (8-bit clamps to 0-1, freezing the shader).

**Feedback TOP:** Use `top` parameter reference, not direct input wire. "Not enough sources" resolves after first cook. "Cook dependency loop" warning is expected.

**Resolution:** Non-Commercial caps at 1280×1280. Use `outputresolution = 'custom'`.

**Large shaders:** Write GLSL to `/tmp/file.glsl`, then use `td_write_dat` or `td_execute_python` to load.

**Vertex/Point access (TD 2025.32):** `point.P[0]`, `point.P[1]`, `point.P[2]` — NOT `.x`, `.y`, `.z`.

**Extensions:** `ext0object` format is `"op('./datName').module.ClassName(me)"` in CONSTANT mode. After editing extension code with `td_write_dat`, call `td_reinit_extension`.

**Script callbacks:** ALWAYS use relative paths via `me.parent()` / `scriptOp.parent()`.

**Cleaning nodes:** Always `list(root.children)` before iterating + `child.valid` check.

## Recording / Exporting Video

```python
# via td_execute_python:
root = op('/project1')
rec = root.create(moviefileoutTOP, 'recorder')
op('/project1/out').outputConnectors[0].connect(rec.inputConnectors[0])
rec.par.type = 'movie'
rec.par.file = '/tmp/output.mov'
rec.par.videocodec = 'prores'  # Apple ProRes — NOT license-restricted on macOS
rec.par.record = True   # start
# rec.par.record = False  # stop (call separately later)
```

H.264/H.265/AV1 need Commercial license. Use `prores` on macOS or `mjpa` as fallback.
Extract frames: `ffmpeg -i /tmp/output.mov -vframes 120 /tmp/frames/frame_%06d.png`

**TOP.save() is useless for animation** — captures same GPU texture every time. Always use MovieFileOut.

### Before Recording: Checklist

1. **Verify FPS > 0** via `td_get_perf`. If FPS=0 the recording will be empty. See pitfalls #38-39.
2. **Verify shader output is not black** via `td_get_screenshot`. Black output = shader error or missing input. See pitfalls #8, #40.
3. **If recording with audio:** cue audio to start first, then delay recording by 3 frames. See pitfalls #19.
4. **Set output path before starting record** — setting both in the same script can race.

## Audio-Reactive GLSL (Proven Recipe)

### Correct signal chain (tested April 2026)

```
AudioFileIn CHOP (playmode=sequential)
  → AudioSpectrum CHOP (FFT=512, outputmenu=setmanually, outlength=256, timeslice=ON)
  → Math CHOP (gain=10)
  → CHOP to TOP (dataformat=r, layout=rowscropped)
  → GLSL TOP input 1 (spectrum texture, 256x2)

Constant TOP (rgba32float, time) → GLSL TOP input 0
GLSL TOP → Null TOP → MovieFileOut
```

### Critical audio-reactive rules (empirically verified)

1. **TimeSlice must stay ON** for AudioSpectrum. OFF = processes entire audio file → 24000+ samples → CHOP to TOP overflow.
2. **Set Output Length manually** to 256 via `outputmenu='setmanually'` and `outlength=256`. Default outputs 22050 samples.
3. **DO NOT use Lag CHOP for spectrum smoothing.** Lag CHOP operates in timeslice mode and expands 256 samples to 2400+, averaging all values to near-zero (~1e-06). The shader receives no usable data. This was the #1 audio sync failure in testing.
4. **DO NOT use Filter CHOP either** — same timeslice expansion problem with spectrum data.
5. **Smoothing belongs in the GLSL shader** if needed, via temporal lerp with a feedback texture: `mix(prevValue, newValue, 0.3)`. This gives frame-perfect sync with zero pipeline latency.
6. **CHOP to TOP dataformat = 'r'**, layout = 'rowscropped'. Spectrum output is 256x2 (stereo). Sample at y=0.25 for first channel.
7. **Math gain = 10** (not 5). Raw spectrum values are ~0.19 in bass range. Gain of 10 gives usable ~5.0 for the shader.
8. **No Resample CHOP needed.** Control output size via AudioSpectrum's `outlength` param directly.

### GLSL spectrum sampling

```glsl
// Input 0 = time (1x1 rgba32float), Input 1 = spectrum (256x2)
float iTime = texture(sTD2DInputs[0], vec2(0.5)).r;

// Sample multiple points per band and average for stability:
// NOTE: y=0.25 for first channel (stereo texture is 256x2, first row center is 0.25)
float bass = (texture(sTD2DInputs[1], vec2(0.02, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.05, 0.25)).r) / 2.0;
float mid  = (texture(sTD2DInputs[1], vec2(0.2, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.35, 0.25)).r) / 2.0;
float hi   = (texture(sTD2DInputs[1], vec2(0.6, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.8, 0.25)).r) / 2.0;
```

See `references/network-patterns.md` for complete build scripts + shader code.

## Operator Quick Reference

| Family | Color | Python class / MCP type | Suffix |
|--------|-------|-------------|--------|
| TOP | Purple | noiseTOP, glslTOP, compositeTOP, levelTop, blurTOP, textTOP, nullTOP | TOP |
| CHOP | Green | audiofileinCHOP, audiospectrumCHOP, mathCHOP, lfoCHOP, constantCHOP | CHOP |
| SOP | Blue | gridSOP, sphereSOP, transformSOP, noiseSOP | SOP |
| DAT | White | textDAT, tableDAT, scriptDAT, webserverDAT | DAT |
| MAT | Yellow | phongMAT, pbrMAT, glslMAT, constMAT | MAT |
| COMP | Gray | geometryCOMP, containerCOMP, cameraCOMP, lightCOMP, windowCOMP | COMP |

## Security Notes

- MCP runs on localhost only (port 40404). No authentication — any local process can send commands.
- `td_execute_python` has unrestricted access to the TD Python environment and filesystem as the TD process user.
- `setup.sh` downloads twozero.tox from the official 404zero.com URL. Verify the download if concerned.
- The skill never sends data outside localhost. All MCP communication is local.

## References

| File | What |
|------|------|
| `references/pitfalls.md` | Hard-won lessons from real sessions |
| `references/operators.md` | All operator families with params and use cases |
| `references/network-patterns.md` | Recipes: audio-reactive, generative, GLSL, instancing |
| `references/mcp-tools.md` | Full twozero MCP tool parameter schemas |
| `references/python-api.md` | TD Python: op(), scripting, extensions |
| `references/troubleshooting.md` | Connection diagnostics, debugging |
| `references/glsl.md` | GLSL uniforms, built-in functions, shader templates |
| `references/postfx.md` | Post-FX: bloom, CRT, chromatic aberration, feedback glow |
| `references/layout-compositor.md` | HUD layout patterns, panel grids, BSP-style layouts |
| `references/operator-tips.md` | Wireframe rendering, feedback TOP setup |
| `references/geometry-comp.md` | Geometry COMP: instancing, POP vs SOP, morphing |
| `references/audio-reactive.md` | Audio band extraction, beat detection, envelope following |
| `references/animation.md` | LFOs, timers, keyframes, easing, expression-driven motion |
| `references/midi-osc.md` | MIDI/OSC controllers, TouchOSC, multi-machine sync |
| `references/particles.md` | POPs and legacy particleSOP — emission, forces, collisions |
| `references/projection-mapping.md` | Multi-window output, corner pin, mesh warp, edge blending |
| `references/external-data.md` | HTTP, WebSocket, MQTT, Serial, TCP, webserverDAT |
| `references/panel-ui.md` | Custom params, panel COMPs, button/slider/field, panelExecuteDAT |
| `references/replicator.md` | replicatorCOMP — data-driven cloning, layouts, callbacks |
| `references/dat-scripting.md` | Execute DAT family — chop/dat/parameter/panel/op/executeDAT |
| `references/3d-scene.md` | Lighting rigs, shadows, IBL/cubemaps, multi-camera, PBR |
| `scripts/setup.sh` | Automated setup script |

---

> You're not writing code. You're conducting light.
