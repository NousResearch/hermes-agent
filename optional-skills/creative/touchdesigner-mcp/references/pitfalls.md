# TouchDesigner MCP — Pitfalls & Lessons Learned

Hard-won knowledge from real TD sessions. Read this before building anything.

## Parameter Names

### 1. NEVER hardcode parameter names — always discover

Parameter names change between TD versions. What works in one build may not work in another. ALWAYS use td_get_par_info to discover actual names from TD.

The agent's LLM training data contains WRONG parameter names. Do not trust them.

Known historical differences (may vary further — always verify):
| What docs/training say | Actual in some versions | Notes |
|---------------|---------------|-------|
| `dat` | `pixeldat` | GLSL TOP pixel shader DAT |
| `colora` | `alpha` | Constant TOP alpha |
| `sizex` / `sizey` | `size` | Blur TOP (single value) |
| `fontr/g/b/a` | `fontcolorr/g/b/a` | Text TOP font color (r/g/b) |
| `fontcolora` | `fontalpha` | Text TOP font alpha (NOT `fontcolora`) |
| `bgcolora` | `bgalpha` | Text TOP bg alpha |
| `value1name` | `vec0name` | GLSL TOP uniform name |

### 2. twozero td_execute_python response format

When calling `td_execute_python` via twozero MCP, successful responses return `(ok)` followed by FPS/error summary (e.g. `[fps 60.0/60] [0 err/0 warn]`), NOT the raw Python `result` dict. If you're parsing responses programmatically, check for the `(ok)` prefix — don't pattern-match on Python variable names from the script. Use `td_get_operator_info` or separate inspection calls to read back values.

### 3. When using td_set_operator_pars, param names must match exactly

Use td_get_par_info to discover them. The MCP tool validates parameter names and returns clear errors explaining what went wrong, unlike raw Python which crashes the whole script with tdAttributeError and stops execution. Always discover before setting.

### 4. Use `safe_par()` pattern for cross-version compatibility

```python
def safe_par(node, name, value):
    p = getattr(node.par, name, None)
    if p is not None:
        p.val = value
        return True
    return False
```

### 5. `td.tdAttributeError` crashes the whole script — use defensive access

If you do `node.par.nonexistent = value`, TD raises `tdAttributeError` and stops the entire script. Prevention is better than catching:
- Use `op()` instead of `opex()` — `op()` returns None on failure, `opex()` raises
- Use `hasattr(node.par, 'name')` before accessing any parameter
- Use `getattr(node.par, 'name', None)` with a default
- Use the `safe_par()` pattern from pitfall #3

```python
# WRONG — crashes if param doesn't exist:
node.par.nonexistent = value

# CORRECT — defensive access:
if hasattr(node.par, 'nonexistent'):
    node.par.nonexistent = value
```

### 6. `outputresolution` is a string menu, not an integer

```
menuNames: ['useinput','eighth','quarter','half','2x','4x','8x','fit','limit','custom','parpanel']
```
Always use the string form. Setting `outputresolution = 9` may silently fail.
```python
node.par.outputresolution = 'custom'  # correct
node.par.resolutionw = 1280; node.par.resolutionh = 720
```
Discover valid values: `list(node.par.outputresolution.menuNames)`

## GLSL Shaders

### 7. `uTDCurrentTime` does NOT exist in GLSL TOP

There is NO built-in time uniform for GLSL TOPs. GLSL MAT has `uTDGeneral.seconds` but that's NOT available in GLSL TOP context.

**PRIMARY — GLSL TOP Vectors/Values page:**
```python
gl.par.value0name = 'uTime'
gl.par.value0.expr = "absTime.seconds"
# In GLSL: uniform float uTime;
```

**FALLBACK — Constant TOP texture (for complex time data):**

CRITICAL: set format to `rgba32float` — default 8-bit clamps to 0-1:
```python
t = root.create(constantTOP, 'time_driver')
t.par.format = 'rgba32float'
t.par.outputresolution = 'custom'
t.par.resolutionw = 1; t.par.resolutionh = 1
t.par.colorr.expr = "absTime.seconds % 1000.0"
t.outputConnectors[0].connect(glsl.inputConnectors[0])
```

### 8. GLSL compile errors are silent in the API

The GLSL TOP shows a yellow warning triangle in the UI but `node.errors()` may return empty string. Check `node.warnings()` too, and create an Info DAT pointed at the GLSL TOP to read the actual compiler output.

### 9. TD GLSL uses `vUV.st` not `gl_FragCoord` — and REQUIRES `TDOutputSwizzle()` on macOS

Standard GLSL patterns don't work. TD provides:
- `vUV.st` — UV coordinates (0-1)
- `uTDOutputInfo.res.zw` — resolution
- `sTD2DInputs[0]` — input textures
- `layout(location = 0) out vec4 fragColor` — output

CRITICAL on macOS: Always wrap output with `TDOutputSwizzle()`:
```glsl
fragColor = TDOutputSwizzle(color);
```
TD uses GLSL 4.60 (Vulkan backend). GLSL 3.30 and earlier removed.

### 10. Large GLSL shaders — write to temp file

GLSL code with special characters can corrupt JSON payloads. Write the shader to a temp file and load it in TD:
```python
# Agent side: write shader to /tmp/shader.glsl via write_file
# TD side:
sd = root.create(textDAT, 'shader_code')
with open('/tmp/shader.glsl', 'r') as f:
    sd.text = f.read()
```

## Node Management

### 11. Destroying nodes while iterating `root.children` causes `tdError`

The iterator is invalidated when a child is destroyed. Always snapshot first:
```python
kids = list(root.children)  # snapshot
for child in kids:
    if child.valid:  # check — earlier destroys may cascade
        child.destroy()
```

### 11b. Split cleanup and creation into SEPARATE td_execute_python calls

Creating nodes with the same names you just destroyed in the SAME script causes "Invalid OP object" errors — even with `list()` snapshot. TD's internal references can go stale within one execution context.

**WRONG (single call):**
```python
# td_execute_python:
for c in list(root.children):
    if c.valid and c.name.startswith('my_'):
        c.destroy()
# ... then create my_audio, my_shader etc. in same script → CRASHES
```

**CORRECT (two separate calls):**
```python
# Call 1: td_execute_python — clean only
for c in list(root.children):
    if c.valid and c.name.startswith('my_'):
        c.destroy()

# Call 2: td_execute_python — build (separate MCP call)
audio = root.create(audiofileinCHOP, 'my_audio')
# ... rest of build
```

### 12. Feedback TOP: use `top` parameter, NOT direct input wire

The feedbackTOP's `top` parameter references which TOP to delay. Do NOT also wire that TOP directly into the feedback's input — this creates a real cook dependency loop.

Correct setup:
```python
fb = root.create(feedbackTOP, 'fb_delay')
fb.par.top = comp.path          # reference only — no wire to fb input
fb.outputConnectors[0].connect(xf)  # fb output -> transform -> fade -> comp
```

The "Cook dependency loop detected" warning on the transform/fade chain is expected.

### 13. GLSL TOP auto-creates companion nodes

Creating a `glslTOP` also creates `name_pixel` (Text DAT), `name_info` (Info DAT), and `name_compute` (Text DAT). These are visible in the network. Don't be alarmed by "extra" nodes.

### 14. The default project root is `/project1`

New TD files start with `/project1` as the main container. System nodes live at `/`, `/ui`, `/sys`, `/local`, `/perform`. Don't create user nodes outside `/project1`.

### 15. Non-Commercial license caps resolution at 1280x1280

Setting `resolutionw=1920` silently clamps to 1280. Always check effective resolution after creation:
```python
n.cook(force=True)
actual = str(n.width) + 'x' + str(n.height)
```

## Recording & Codecs

### 16. MovieFileOut TOP: H.264/H.265/AV1 requires Commercial license

In Non-Commercial TD, these codecs produce an error. Recommended alternatives:
- `prores` — Apple ProRes, **best on macOS**, HW accelerated, NOT license-restricted. ~55MB/s at 1280x720 but lossless quality. **Use this as default on macOS.**
- `cineform` — GoPro Cineform, supports alpha
- `hap` — GPU-accelerated playback, large files
- `notchlc` — GPU-accelerated, good quality
- `mjpa` — Motion JPEG, legacy fallback (lossy, use only if ProRes unavailable)

For image sequences: `rec.par.type = 'imagesequence'`, `rec.par.imagefiletype = 'png'`

### 17. MovieFileOut `.record()` method may not exist

Use the toggle parameter instead:
```python
rec.par.record = True   # start recording
rec.par.record = False  # stop recording
```

When setting file path and starting recording in the same script, use delayFrames:
```python
rec.par.file = '/tmp/new_output.mov'
run("op('/project1/recorder').par.record = True", delayFrames=2)
```

### 18. TOP.save() captures same frame when called rapidly

Use MovieFileOut for real-time recording. Set `project.realTime = False` for frame-accurate output.

### 19. AudioFileIn CHOP: cue and recording sequence matters

The recording sequence must be done in exact order, or the recording will be empty, audio will start mid-file, or the file won't be written.

**Proven recording sequence:**

```python
# Step 1: Stop any existing recording
rec.par.record = False

# Step 2: Reset audio to beginning
audio.par.play = False
audio.par.cue = True
audio.par.cuepoint = 0      # may need cuepointunit=0 too
# Verify: audio.par.cue.eval() should be True

# Step 3: Set output file path
rec.par.file = '/tmp/output.mov'

# Step 4: Release cue + start playing + start recording (with frame delay)
audio.par.cue = False
audio.par.play = True
audio.par.playmode = 2      # Sequential — plays once through
run("op('/project1/recorder').par.record = True", delayFrames=3)
```

**Why each step matters:**
- `rec.par.record = False` first — if a previous recording is active, setting `par.file` may fail silently
- `audio.par.cue = True` + `cuepoint = 0` — guarantees audio starts from the beginning, otherwise the spectrum may be silent for the first few seconds
- `delayFrames=3` on the record start — setting `par.file` and `par.record = True` in the same script can race; the file path needs a frame to register before recording starts
- `playmode = 2` (Sequential) — plays the file once. Use `playmode = 0` (Locked to Timeline) if you want TD's timeline to control position

## TD Python API Patterns

### 20. COMP extension setup: ext0object format is CRITICAL

`ext0object` expects a CONSTANT string (NOT expression mode):
```python
comp.par.ext0object = "op('./myExtensionDat').module.MyClassName(me)"
```
NEVER set as just the DAT name. NEVER use ParMode.EXPRESSION. ALWAYS ensure the DAT has `par.language='python'`.

### 21. td.Panel is NOT subscriptable — use attribute access

```python
comp.panel.select      # correct (attribute access, returns float)
comp.panel['select']   # WRONG — 'td.Panel' object is not subscriptable
```

### 22. ALWAYS use relative paths in script callbacks

In scriptTOP/CHOP/SOP/DAT callbacks, use paths relative to `scriptOp` or `me`:
```python
root = scriptOp.parent().parent()
dat = root.op('pixel_data')
```
NEVER hardcode absolute paths like `op('/project1/myComp/child')` — they break when containers are renamed or copied.

### 23. keyboardinCHOP channel names have 'k' prefix

Channel names are `kup`, `kdown`, `kleft`, `kright`, `ka`, `kb`, etc. — NOT `up`, `down`, `a`, `b`. Always verify with:
```python
channels = [c.name for c in op('/project1/keyboard1').chans()]
```

### 24. expressCHOP cook-only properties — false positive errors

`me.inputVal`, `me.chanIndex`, `me.sampleIndex` work ONLY in cook-context. Calling `par.expr0expr.eval()` from outside always raises an error — this is NOT a real operator error. Ignore these in error scans.

### 25. td.Vertex attributes — use index access not named attributes

In TD 2025.32, `td.Vertex` objects do NOT have `.x`, `.y`, `.z` attributes:
```python
# WRONG — crashes:
vertex.x, vertex.y, vertex.z

# CORRECT — index-based:
vertex.point.P[0], vertex.point.P[1], vertex.point.P[2]
# Or for SOP point positions:
pt = sop.points()[i]
pos = pt.P    # use P[0], P[1], P[2]
```

## Audio

### 26. Audio Spectrum CHOP output is weak — boost it

Raw output is very small (0.001-0.05). Use built-in boost: `spectrum.par.highfrequencyboost = 3.0`

If still weak, add Math CHOP in Range mode: `fromrangehi=0.05, torangehi=1.0`

### 27. AudioSpectrum CHOP: timeslice and sample count are the #1 gotcha

AudioSpectrum at 44100Hz with `timeslice=False` outputs the ENTIRE audio file as samples (~24000+). CHOP-to-TOP then exceeds texture resolution max and warns/fails.

**Fix:** Keep `timeslice = True` (default) for real-time per-frame FFT. Set `fftsize` to control bin count (it's a STRING enum: `'256'` not `256`).

If the CHOP-to-TOP still gets too many samples, set `layout = 'rowscropped'` on the choptoTOP.

```python
spectrum.par.fftsize = '256'      # STRING, not int — enum values
spectrum.par.timeslice = True     # MUST be True for real-time audio reactivity
spectex.par.layout = 'rowscropped'  # handles oversized CHOP inputs
```

**resampleCHOP has NO `numsamples` param.** It uses `rate`, `start`, `end`, `method`. Don't guess — always `td_get_par_info('resampleCHOP')` first.

### 28. CHOP To TOP has NO input connectors — use par.chop reference

```python
spec_tex = root.create(choptoTOP, 'spectrum_tex')
spec_tex.par.chop = resample  # correct: parameter reference
# NOT: resample.outputConnectors[0].connect(spec_tex.inputConnectors[0])  # WRONG
```

## Workflow

### 29. Always verify after building — errors are silent

Node errors and broken connections produce no output. Always check:
```python
for c in list(root.children):
    e = c.errors()
    w = c.warnings()
    if e: print(c.name, 'ERR:', e)
    if w: print(c.name, 'WARN:', w)
```

### 30. Window COMP param for display target is `winop`

```python
win = root.create(windowCOMP, 'display')
win.par.winop = '/project1/logo_out'
win.par.winw = 1280; win.par.winh = 720
win.par.winopen.pulse()
```

### 31. `sample()` returns frozen pixels in rapid calls

`out.sample(x, y)` returns pixels from a single cook snapshot. Compare samples with 2+ second delays, or use screencapture on the display window.

### 32. Audio-reactive GLSL: TD-side pipeline

For audio-synced visuals: AudioFileIn → AudioSpectrum(timeslice=True, fftsize='256') → Math(gain=5) → choptoTOP(par.chop=math, layout='rowscropped') → GLSL input. The shader samples `sTD2DInputs[1]` at different x positions for bass/mid/hi. Record the TD output with MovieFileOut.

**Key gotcha:** AudioFileIn must be cued (`par.cue=True` → `par.cuepulse.pulse()`) then uncued (`par.cue=False`, `par.play=True`) before recording starts. Otherwise the spectrum is silent for the first few seconds.

### 33. twozero MCP: prefer native tools

**Always prefer native MCP tools over td_execute_python:**
- `td_create_operator` over `root.create()` scripts (handles viewport positioning)
- `td_set_operator_pars` over `node.par.X = Y` scripts (validates param names)
- `td_get_par_info` over temp-node discovery dance (instant, no cleanup)
- `td_get_errors` over manual `c.errors()` loops
- `td_get_focus` for context awareness (no equivalent in old method)

Only fall back to `td_execute_python` for multi-step logic (wiring chains, conditional builds, loops).

### 34. twozero td_execute_python response wrapping

twozero wraps `td_execute_python` responses with status info: `(ok)\n\n[fps 60.0/60] [0 err/0 warn]`. Your Python `result` variable value may not appear verbatim in the response text. If you need to check results programmatically, use `print()` statements in the script — they appear in the response. Don't rely on string-matching the `result` dict.

### 35. Audio-reactive chain: DO NOT use Lag CHOP or Filter CHOP for spectrum smoothing

The Derivative docs and tutorials suggest using Lag CHOP (lag1=0.2, lag2=0.5) to smooth raw FFT output before passing to a shader. **This does NOT work with AudioSpectrum → CHOP to TOP → GLSL.**

What happens: Lag CHOP operates in timeslice mode. A 256-sample spectrum input gets expanded to 1600-2400 samples. The Lag averaging drives all values to near-zero (~1e-06). The CHOP to TOP produces a 2400x2 texture instead of 256x2. The shader receives effectively zero audio data.

**The correct chain is: Spectrum(outlength=256) → Math(gain=10) → CHOPtoTOP → GLSL.** No CHOP smoothing at all. If you need smoothing, do it in the GLSL shader via temporal lerp with a feedback texture.

Verified values with audio playing:
- Without Lag CHOP: bass bins = 5.0-5.4, mid bins = 1.0-1.7 (strong, usable)
- With Lag CHOP: ALL bins = 0.000001-0.00004 (dead, zero audio reactivity)

### 36. AudioSpectrum Output Length: set manually to avoid CHOP to TOP overflow

AudioSpectrum in Visualization mode with FFT 8192 outputs 22,050 samples by default (1 per Hz, 0–22050). CHOP to TOP cannot handle this — you get "Number of samples exceeded texture resolution max".

Fix: `spectrum.par.outputmenu = 'setmanually'` and `spectrum.par.outlength = 256`. This gives 256 frequency bins — plenty for visual FFT.

DO NOT set `timeslice = False` as a workaround — that processes the entire audio file at once and produces even more samples.

### 37. GLSL spectrum texture from CHOP to TOP is 256x2 not 256x1

AudioSpectrum outputs 2 channels (stereo: chan1, chan2). CHOP to TOP with `dataformat='r'` creates a 256x2 texture — one row per channel. Sample the first channel at `y=0.25` (center of first row), NOT `y=0.5` (boundary between rows):

```glsl
float bass = texture(sTD2DInputs[1], vec2(0.05, 0.25)).r;  // correct
float bass = texture(sTD2DInputs[1], vec2(0.05, 0.5)).r;   // WRONG — samples between rows
```

### 38. FPS=0 doesn't mean ops aren't cooking — check play state

TD can show `fps:0` in `td_get_perf` while ops still cook and `TOP.save()` still produces valid screenshots. The two most common causes:

**a) Project is paused (playbar stopped).** TD's playbar can be toggled with spacebar. The `root` at `/` has no `.playbar` attribute (it's on the perform COMP). The easiest fix is sending a spacebar keypress via `td_input_execute`, though this tool can sometimes error. As a workaround, `TOP.save()` always works regardless of play state — use it to verify rendering is actually happening before spending time debugging FPS.

**b) Audio device CHOP blocking the main thread (MOST COMMON).** An `audiodeviceoutCHOP` with `active=True` can consume 300-400ms/s (2000%+ of frame budget), stalling the cook loop at FPS=0. **`volume=0` is NOT sufficient** — the audio driver still blocks. Fix: `par.active = False`. This completely stops the CHOP from interacting with the audio driver. If you need audio monitoring, enable it only during short playback checks, then disable before recording.

Verified April 2026: disabling `audiodeviceoutCHOP` (`active=False`) restored FPS from 0 to 60 instantly, recovering from 2348% budget usage to 0.1%.

Diagnostic sequence when FPS=0:
1. `td_get_perf` — check if any op has extreme CPU/s (audiodeviceoutCHOP is the usual suspect)
2. If audiodeviceoutCHOP shows >100ms/s: set `par.active = False` immediately
3. `TOP.save()` on the output — if it produces a valid image, the pipeline works, just not at real-time rate
4. Check for other blocking CHOPs (audiodevin, etc.)
5. Toggle play state (spacebar, or check if absTime.seconds is advancing)

### 39. Recording while FPS=0 produces empty or near-empty files

This is the #1 cause of "I recorded for 30 seconds but got a 2-frame video." If TD's cook loop is stalled (FPS=0 or very low), MovieFileOut has nothing to record. Unlike `TOP.save()` which captures the last cooked frame regardless, MovieFileOut only writes frames that actually cook.

**Always verify FPS before starting a recording:**
```python
# Check via td_get_perf first
# If FPS < 30, do NOT start recording — fix the performance issue first
# If FPS=0, the playbar is likely paused — see pitfall #37
```

Common causes of recording empty video:
- Playbar paused (FPS=0) — see pitfall #37
- Audio device CHOP blocking the main thread — see pitfall #37b
- Recording started before audio was cued — audio is silent, GLSL outputs black, MovieFileOut records black frames that look empty
- `par.file` set in the same script as `par.record = True` — see pitfall #18

### 40. GLSL shader produces black output — test before committing to a long render

New GLSL shaders can fail silently (see pitfall #7). Before recording a long take, always:

1. **Write a minimal test shader first** that just outputs a solid color or pass-through:
```glsl
void main() {
    vec2 uv = vUV.st;
    fragColor = TDOutputSwizzle(vec4(uv, 0.0, 1.0));
}
```

2. **Verify the test renders correctly** via `td_get_screenshot` on the GLSL TOP's output.

3. **Swap in the real shader** and screenshot again immediately. If black, the shader has a compile error or logic issue.

4. **Only then start recording.** A 90-second ProRes recording is ~5GB. Recording black frames wastes disk and time.

Common causes of black GLSL output:
- Missing `TDOutputSwizzle()` on macOS (pitfall #8)
- Time uniform not connected — shader uses default 0.0, fractal stays at origin
- Spectrum texture not connected — audio values all 0.0, driving everything to black
- Integer division where float division was expected (`1/2 = 0` not `0.5`)
- `absTime.seconds % 1000.0` rolled over past 1000 and the modulo produces unexpected values

### 41. td_write_dat uses `text` parameter, NOT `content`

The MCP tool `td_write_dat` expects a `text` parameter for full replacement. Passing `content` returns an error: `"Provide either 'text' for full replace, or 'old_text'+'new_text' for patching"`.

If `td_write_dat` fails, fall back to `td_execute_python`:
```python
op("/project1/shader_code").text = shader_string
```

### 42. td_execute_python DOES return print() output — use it for debugging

`print()` statements in `td_execute_python` scripts appear in the MCP response text. This is the correct way to read values back from scripts. The response format is: printed output first, then `[fps X.X/X] [N err/N warn]` on a separate line.

However, the `result` variable (if you set one) does NOT appear verbatim — use `print()` for anything you need to read back:
```python
# CORRECT — appears in response:
print('value:', some_value)

# WRONG — not reliably in response:
result = some_value
```

For structured data, use dedicated inspection tools (`td_get_operator_info`, `td_read_chop`) which return clean JSON.

### 43. td_get_operator_info JSON is appended with `[fps X.X/X]` — breaks json.loads()

The response text from `td_get_operator_info` has `[fps 60.0/60]` appended after the JSON object. This causes `json.loads()` to fail with "Extra data" errors. Strip it before parsing:
```python
clean = response_text.rsplit('[fps', 1)[0]
data = json.loads(clean)
```

### 44. td_get_screenshot is unreliable — returns `{"status": "pending"}` and may never deliver

Screenshots don't complete instantly. The tool returns `{"status": "pending", "requestId": "..."}` and the actual file may appear later — or may NEVER appear at all. In testing (April 2026), screenshots stayed "pending" indefinitely with no file written to disk, even though the shader was cooking at 8-30fps.

**Do NOT rely on `td_get_screenshot` for frame capture.** For reliable frame capture, use MovieFileOut recording + ffmpeg frame extraction:
```bash
# Record in TD first, then extract frames:
ffmpeg -y -i /tmp/td_output.mov -t 25 -vf 'fps=24' /tmp/td_frames/frame_%06d.png
```

If you need a quick visual check, `td_get_screenshot` is worth trying (it sometimes works), but always have the recording fallback. There is no callback or completion notification — if the file doesn't appear after 5-10 seconds, it's not coming.

### 45. Heavy shaders cook below record FPS — many duplicate frames in output

A raymarched GLSL shader may only cook at 8-15fps even though MovieFileOut records at 60fps. The recording still works (TD writes the last-cooked frame each time), but the resulting file has many duplicate frames. When extracting frames for post-processing, use a lower fps filter to avoid redundant frames:
```bash
# Extract at 24fps from a 60fps recording of an 8fps shader:
ffmpeg -y -i /tmp/td_output.mov -t 25 -vf 'fps=24' /tmp/td_frames/frame_%06d.png
```
Check actual cook FPS with `td_get_perf` before committing to a long recording. If FPS < 15, the output will be a slideshow regardless of the recording codec.

### 46. Recording duration is manual — no auto-stop at audio end

MovieFileOut records until `par.record = False` is set. If audio ends before you stop recording, the file keeps growing with repeated frames. Always stop recording promptly after the audio duration. For precision: set a timer on the agent side matching the audio length, then send `par.record = False`. Trim excess with ffmpeg as a safety net:
```bash
ffmpeg -i raw.mov -t 25 -c copy trimmed.mov
```

### 47. AudioFileIn par.index stays at 0 in sequential mode — not a reliable progress indicator

When `audiofileinCHOP` is in `playmode=2` (sequential), `par.index.eval()` returns 0.0 even while audio IS actively playing and the spectrum IS receiving data. Do NOT use `par.index` to check playback progress in sequential mode.

**How to verify audio is actually playing:**
- Read the spectrum CHOP values via `td_read_chop` — if values are non-zero and CHANGE between reads 1-2s apart, audio is flowing
- Read the audio CHOP itself: non-zero waveform samples confirm the file is loaded and playing
- `par.play.eval()` returning True is necessary but NOT sufficient — it can be True with no audio flowing if cue is stuck

### 48. GLSL shader whiteout — clamp audio spectrum values in the shader

Raw spectrum values multiplied by Math CHOP gain can produce very large numbers (5-20+) that blow out the shader's lighting, producing flat white/grey. The shader MUST clamp audio inputs:

```glsl
float bass = texture(sTD2DInputs[1], vec2(0.05, 0.25)).r;
bass = clamp(bass, 0.0, 3.0);   // prevent whiteout
mids = clamp(mids, 0.0, 3.0);
hi = clamp(hi, 0.0, 3.0);
```

Discovered when gain=10 produced ~0.13 (too dark) during quiet passages but gain=50 produced ~9.4 (total whiteout). Fix: keep gain=10, use `highfreqboost=3.0` on AudioSpectrum, clamp in shader.

### 49. Non-Commercial TD records at 1280x1280 (square) — always crop in post

Even with `resolutionw=1280, resolutionh=720` on the GLSL TOP, Non-Commercial TD may output 1280x1280 to MovieFileOut. Always check dimensions with ffprobe and crop during extraction:

```bash
# Center-crop from 1280x1280 to 1280x720:
ffmpeg -y -i /tmp/td_output.mov -t 25 -r 24 -vf "crop=1280:720:0:280" /tmp/frames/frame_%06d.png
```

Large ProRes files (1-2GB) at 1280x1280 decode at ~3fps, so 25s of footage takes ~3 minutes to extract.

## Advanced Patterns (pitfalls 51+)

### 51. Connection syntax: use `outputConnectors`/`inputConnectors`, NOT `outputs`/`inputs`

```python
# CORRECT
src.outputConnectors[0].connect(dst.inputConnectors[0])
# WRONG — raises IndexError or AttributeError
src.outputs[0].connect(dst.inputs[0])
```

For feedback TOP, BOTH are required:
```python
fb.par.top = target.path
target.outputConnectors[0].connect(fb.inputConnectors[0])
```

### 52. moviefileoutTOP `par.input` doesn't resolve via Python in TD 2025.32460

Setting `moviefileoutTOP.par.input` programmatically does NOT work. All forms fail silently with "Not enough sources specified."

**Workaround — frame capture + ffmpeg:**
```python
out = op('/project1/out')
for i in range(300):
    delay = i * 5
    run(f"op('/project1/out').save('/tmp/frames/f_{i:04d}.png')", delayFrames=delay)
# Then: ffmpeg -y -framerate 30 -i /tmp/frames/f_%04d.png -c:v prores -pix_fmt yuv420p /tmp/output.mov
```

### 53. Batch frame capture — use `me.fetch`/`me.store` for state across calls

```python
start = me.fetch('cap_frame', 0)
for i in range(60):
    frame = start + i
    op('/project1/out').save(f'/tmp/frames/frame_{str(frame).zfill(4)}.png')
me.store('cap_frame', start + 60)
```
Call 5 times for 300 frames. Each picks up where the last left off.

### 54. GLSL TOP pixel shader requirements in TD 2025

```glsl
// REQUIRED — declare output
layout(location = 0) out vec4 fragColor;

void main() {
    vec3 col = vec3(1.0, 0.0, 0.0);
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
```
**Built-in uniforms available:** `uTDOutputInfo.res` (vec4), `uTDTimeInfo.seconds`, `sTD2DInputs[N]`.
**Auto-created DATs:** `name_pixel`, `name_vertex`, `name_compute` textDATs with example code.

### 55. TOP.save() doesn't advance time — identical frames in tight loops

`.save()` captures the current cooked frame without advancing TD's timeline:
```python
# WRONG — all frames identical
for i in range(300):
    op('/project1/out').save(f'frames/f_{i:04d}.png')

# CORRECT — use run() with delayFrames
for i in range(300):
    delay = i * 5
    run(f"op('/project1/out').save('frames/f_{i:04d}.png')", delayFrames=delay)
```
**NEVER use `time.sleep()` in TD** — it blocks the main thread and freezes the UI.

### 56. Feedback loop masks input changes — force switch during capture

With feedback TOP opacity 0.7+, the buffer dominates output. Switching input produces nearly identical frames.

**Fix — force switch index per capture:**
```python
for i in range(300):
    idx = (i // 8) % num_inputs
    delay = i * 5
    run(f"op('/project1/vswitch').par.index={idx}; op('/project1/out').save('f_{i:04d}.png')", delayFrames=delay)
```

### 57. Large td_execute_python scripts fail — split into incremental calls

10+ operator creations in one script cause timing issues. Split into 2-4 calls of 2-4 operators each. Within one call, `create()` handles work immediately. Across calls, `op('name')` may return `None` if the previous call hasn't committed.

### 58. MCP instance reconnection after project.load()

`project.load(path)` changes the PID. After loading, call `td_list_instances()` and use the new `target_instance`. For TOX files: import as child comp instead (doesn't disconnect).

### 59. TOX reverse-engineering workflow

```python
comp = root.loadTox(r'/path/to/file.tox')
comp.name = '_study_comp'
for child in comp.children:
    print(f'{child.name} ({child.OPType})')
# Use td_get_operators_info, td_read_dat, check custom params
```

### 60. sliderCOMP naming — TD appends suffix

TD auto-renames: `slider_brightness` → `slider_brightness1`. Always check names after creation.

### 61. create() requires full operator type suffix

```python
# CORRECT
proj.create('audiofileinCHOP', 'audio_in')
proj.create('glslTOP', 'render')

# WRONG — raises "Unknown operator type"
proj.create('audiofilein', 'audio_in')
proj.create('glsl', 'render')
```

### 62. Reparenting COMPs — use copyOPs, not connect()

Moving COMPs with `inputCOMPConnectors[0].connect()` fails. Use copy + destroy:
```python
copied = target.copyOPs([source])  # preserves internal wiring
source.destroy()
# Re-wire external connections manually after the move
```

### 63. Slider wiring — expressionCHOP with op() expressions crashes TD

```python
# CRASHES TD — don't do this
echop = root.create(expressionCHOP, 'slider_ctrl')
echop.par.chan0expr = 'op("/project1/controls/slider_brightness1").par.value0'

# WORKING — parameterCHOP as bridge
pchop = root.create(parameterCHOP, 'slider_vals')
pchop.par.ops = '/project1/controls'
pchop.par.parameters = 'value0'
pchop.par.custom = True
pchop.par.builtin = False
```