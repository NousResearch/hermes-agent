---
name: touchdesigner
description: "Control a running TouchDesigner instance programmatically — create operators, set parameters, wire connections, execute Python, build real-time visuals. Covers: GLSL shaders, audio-reactive, generative art, video processing, instancing, and live performance."
version: 3.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [TouchDesigner, MCP, creative-coding, real-time-visuals, generative-art, audio-reactive, VJ, installation, GLSL]
    related_skills: [native-mcp, ascii-video, manim-video, hermes-video]
    security:
      allow_network: true
      allow_install: true
      allow_config_write: true
---

# TouchDesigner Integration

## Architecture

Hermes Agent -> HTTP REST (curl) -> TD WebServer DAT (port 9981) -> TD Python environment.

The agent controls a **running TouchDesigner instance** via a REST API on port 9981. It does NOT generate .toe files from scratch.

## First-Time Setup (one-time, persists across sessions)

### 1. Verify TD is running and check for existing API

```bash
lsof -i :9981 -P -n | grep LISTEN   # TD listening?
curl -s --max-time 5 http://127.0.0.1:9981/api/td/server/td  # API working?
```

If HTTP 200 + JSON → skip to **Discovery**. Setup is already done.

### 2. If no API: deploy the custom handler

The user must paste ONE line into TD Textport (Alt+T / Dialogs > Textport and DATs):

```
exec(open('PATH_TO_SKILL/scripts/custom_api_handler.py').read())
```

Copy this to their clipboard with `pbcopy`. This creates a WebServer DAT + callback handler pair in `/project1` that implements the REST API. No external dependencies.

**Why not the official .tox?** The `mcp_webserver_base.tox` from 8beeeaaat/touchdesigner-mcp frequently fails to import its Python modules after drag-drop (relative path resolution issue). Our custom handler is self-contained and more reliable. See `references/pitfalls.md` #1-2.

### 3. Save the project to persist the API

After the handler is running, save the project so the API auto-starts on every future TD launch:

```python
td_exec("project.save(os.path.expanduser('~/Documents/HermesAgent.toe'))")
```

TD auto-opens the last saved project on launch. From now on, `open /Applications/TouchDesigner.app` → port 9981 is live → agent can connect immediately.

To launch TD with this project explicitly:
```bash
open /Applications/TouchDesigner.app ~/Documents/HermesAgent.toe
```

### 4. Optional: Configure Hermes MCP

Add under `mcp_servers:` in the user's Hermes config:
```yaml
touchdesigner:
  command: npx
  args: ["-y", "touchdesigner-mcp-server@latest"]
  env:
    TD_API_URL: "http://127.0.0.1:9981"
  timeout: 120
```

This is optional — the agent works fully via `curl` to the REST API using `execute_code`. MCP tools are a convenience layer.

## Talking to TD (the td_exec pattern)

All communication uses this pattern in `execute_code`:

```python
import json, shlex
from hermes_tools import terminal

API = "http://127.0.0.1:9981"
def td_exec(script):
    payload = json.dumps({"script": script})
    cmd = f"curl -s --max-time 15 -X POST -H 'Content-Type: application/json' -d {shlex.quote(payload)} '{API}/api/td/server/exec'"
    r = terminal(cmd, timeout=20)
    return json.loads(r['output'])

# Returns: {"result": <value>, "stdout": "...", "stderr": "..."}
```

For large GLSL shaders: write to a temp file, then `td_exec("op('...').text = open('/tmp/shader.glsl').read()")`.

## Workflow

### Step 0: Discovery (MANDATORY — never skip)

**Never hardcode parameter names.** They change between TD versions. Run this first:

```python
td_exec("""
import sys
info = {'version': str(app.version), 'platform': sys.platform}
root = op('/project1')
for name, optype in [('glslTOP', glslTOP), ('constantTOP', constantTOP),
                      ('blurTOP', blurTOP), ('textTOP', textTOP),
                      ('levelTOP', levelTOP), ('compositeTOP', compositeTOP),
                      ('transformTOP', transformTOP), ('feedbackTOP', feedbackTOP),
                      ('windowCOMP', windowCOMP)]:
    n = root.create(optype, '_d_' + name)
    kw = ['color','size','font','dat','alpha','opacity','resolution','text',
          'extend','operand','top','pixel','format','win','type']
    info[name] = [p.name for p in n.pars() if any(k in p.name.lower() for k in kw)]
    n.destroy()
result = info
""")
```

Use the returned param names for ALL subsequent calls. Store them in your session context.

### Step 1: Clean + Build

Build the entire network in ONE `td_exec` call (batching avoids round-trip overhead and ensures TD advances frames between calls):

```python
td_exec("""
root = op('/project1')
keep = {'api_server', 'api_handler'}
for child in list(root.children):  # snapshot before destroying
    if child.name not in keep and child.valid:
        child.destroy()

# Create nodes, set params (using discovered names), wire, verify
...
result = {'nodes': len(list(root.children)), 'errors': [...]}
""")
```

### Step 2: Wire connections

```python
gl.outputConnectors[0].connect(comp.inputConnectors[0])
```

### Step 3: Verify

```python
for c in list(root.children):
    e = c.errors(); w = c.warnings()
    if e: print(c.name, 'ERR:', e)
```

### Step 4: Display

```python
win = root.create(windowCOMP, 'display')
win.par.winop = out.path    # discovered param name
win.par.winw = 1280; win.par.winh = 720
win.par.winopen.pulse()
```

## Key Implementation Rules

**Always clean safely:** `list(root.children)` before iterating + `child.valid` check.

**GLSL time:** No `uTDCurrentTime` in TD 099. Feed time via 1x1 Constant TOP.
**CRITICAL: must use `rgba32float` format** — the default 8-bit format clamps values to 0-1, so `absTime.seconds % 1000.0` becomes 1.0 and the shader appears frozen:
```python
t = root.create(constantTOP, 'time_driver')
t.par.format = 'rgba32float'  # ← REQUIRED or time is stuck at 1.0
t.par.outputresolution = 'custom'
t.par.resolutionw = 1
t.par.resolutionh = 1
t.par.colorr.expr = "absTime.seconds % 1000.0"
t.par.colorg.expr = "int(absTime.seconds / 1000.0)"
t.outputConnectors[0].connect(glsl.inputConnectors[0])
# In GLSL: vec4 td = texture(sTD2DInputs[0], vec2(.5)); float t = td.r + td.g*1000.;
```

**Feedback TOP:** Use `top` parameter reference (not direct input wire). The "Not enough sources" error resolves after first cook. The "Cook dependency loop" warning is expected.

**Resolution:** Non-Commercial caps at 1280×1280. Use `outputresolution = 'custom'`.

**Large shaders:** Write GLSL to `/tmp/file.glsl`, then `td_exec("op('shader').text = open('/tmp/file.glsl').read()")`.

**WebServer DAT quirk:** Response body goes in `response['data']` not `response['body']`. Request POST body comes as bytes in `request['data']`.

## Recording / Exporting Video

To capture TD output as video or image sequence for external use (e.g., ASCII video pipeline):

### Movie Recording (recommended)

```python
# Put a Null TOP before the recorder (official best practice)
rec = root.create(moviefileoutTOP, 'recorder')
null_out.outputConnectors[0].connect(rec.inputConnectors[0])

rec.par.type = 'movie'
rec.par.file = '/tmp/output.mov'
rec.par.videocodec = 'mjpa'  # Motion JPEG — works on Non-Commercial

# Start/stop recording (par.record is a toggle, NOT .record() method)
rec.par.record = True   # start
# ... wait ...
rec.par.record = False  # stop
```

**H.264/H.265 require a Commercial license** — use `mjpa` (Motion JPEG) or `prores` on Non-Commercial. Extract frames afterward with ffmpeg if needed:
```bash
ffmpeg -i /tmp/output.mov -vframes 120 /tmp/frames/frame_%06d.png
```

### Image Sequence Export

```python
rec.par.type = 'imagesequence'
rec.par.imagefiletype = 'png'
rec.par.file.expr = "'/tmp/frames/out' + me.fileSuffix"  # fileSuffix is REQUIRED
rec.par.record = True
```

### Pitfalls

- **Race condition:** When setting `par.file` and starting recording in the same script, use `run("...", delayFrames=2)` so the file path is applied before recording begins.
- **TOP.save() is useless for animation:** Calling `op('null1').save(path)` in a loop or rapid API calls captures the same GPU texture every time — TD doesn't cook new frames between save calls. Always use MovieFileOut for animated output.
- See `references/pitfalls.md` #25-27 for full details.

## Audio-Reactive GLSL (Proven Recipe)

Complete chain for music-driven visuals: AudioFileIn → AudioSpectrum → Math (boost) → Resample (256) → CHOP To TOP → GLSL TOP (spectrum sampled per-pixel). See `references/network-patterns.md` Pattern 3b for the full working recipe with shader code.

## Audio-Reactive Visuals

The most powerful TD workflow for the agent: play an audio file, analyze its spectrum, and drive a GLSL shader in real-time. The agent builds the entire signal chain programmatically.

**Signal chain:**
```
AudioFileIn CHOP → AudioSpectrum CHOP → Math CHOP (gain=5)
  → Resample CHOP (256) → CHOP To TOP (spectrum texture)
                                  ↓ (GLSL input 1)
  Constant TOP (rgba32float, time) → GLSL TOP → Null TOP → MovieFileOut
        (input 0)
```

**Key technique:** The spectrum becomes a 256×1 texture. In GLSL, `texture(sTD2DInputs[1], vec2(x, 0.0)).r` samples frequency at position x (0=bass, 1=treble). This lets the shader react per-pixel to different frequency bands.

**Smoothing is critical:** Raw FFT jitters. Use `Math CHOP` gain to boost weak signal, then the GLSL shader's own temporal integration (via feedback or time-smoothed params) handles visual smoothing.

See `references/network-patterns.md` Pattern 9b for the complete build script + shader code.

## Operator Quick Reference

| Family | Color | Examples | Suffix |
|--------|-------|----------|--------|
| TOP | Purple | noiseTop, glslTop, compositeTop, levelTop, blurTop, textTop, nullTop, feedbackTop, renderTop | TOP |
| CHOP | Green | audiofileinChop, audiospectrumChop, mathChop, lfoChop, constantChop | CHOP |
| SOP | Blue | gridSop, sphereSop, transformSop, noiseSop | SOP |
| DAT | White | textDat, tableDat, scriptDat, webserverDAT | DAT |
| MAT | Yellow | phongMat, pbrMat, glslMat, constMat | MAT |
| COMP | Gray | geometryComp, containerComp, cameraComp, lightComp, windowCOMP | COMP |

See `references/operators.md` for full catalog. See `references/network-patterns.md` for recipes.

## References

| File | What |
|------|------|
| `references/pitfalls.md` | **READ FIRST** — 31 hard-won lessons from real sessions |
| `references/operators.md` | All operator families with params and use cases |
| `references/network-patterns.md` | Recipes: audio-reactive, generative, video, GLSL, instancing |
| `references/mcp-tools.md` | MCP tool schemas (optional — curl works without MCP) |
| `references/python-api.md` | TD Python: op(), scripting, extensions |
| `references/troubleshooting.md` | Connection diagnostics, param debugging, performance |
| `scripts/custom_api_handler.py` | Self-contained REST API handler for TD WebServer DAT |
