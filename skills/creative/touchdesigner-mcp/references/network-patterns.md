# TouchDesigner Network Patterns

Complete network recipes for common creative coding tasks. Each pattern shows the operator chain, MCP tool calls to build it, and key parameter settings.

## Audio-Reactive Visuals

### Pattern 1: Audio Spectrum -> Noise Displacement

Audio drives noise parameters for organic, music-responsive textures.

```
Audio File In CHOP -> Audio Spectrum CHOP -> Math CHOP (scale)
                                                |
                                                v (export to noise params)
                          Noise TOP -> Level TOP -> Feedback TOP -> Composite TOP -> Null TOP (out)
                                                        ^                |
                                                        |________________|
```

**MCP Build Sequence:**

```
1. td_create_operator(parent="/project1", type="audiofileinChop", name="audio_in")
2. td_create_operator(parent="/project1", type="audiospectrumChop", name="spectrum")
3. td_create_operator(parent="/project1", type="mathChop", name="spectrum_scale")
4. td_create_operator(parent="/project1", type="noiseTop", name="noise1")
5. td_create_operator(parent="/project1", type="levelTop", name="level1")
6. td_create_operator(parent="/project1", type="feedbackTop", name="feedback1")
7. td_create_operator(parent="/project1", type="compositeTop", name="comp1")
8. td_create_operator(parent="/project1", type="nullTop", name="out")

9. td_set_operator_pars(path="/project1/audio_in",
     properties={"file": "/path/to/music.wav", "play": true})
10. td_set_operator_pars(path="/project1/spectrum",
     properties={"size": 512})
11. td_set_operator_pars(path="/project1/spectrum_scale",
     properties={"gain": 2.0, "postoff": 0.0})
12. td_set_operator_pars(path="/project1/noise1",
     properties={"type": 1, "monochrome": false, "resolutionw": 1280, "resolutionh": 720,
                  "period": 4.0, "harmonics": 3, "amp": 1.0})
13. td_set_operator_pars(path="/project1/level1",
     properties={"opacity": 0.95, "gamma1": 0.75})
14. td_set_operator_pars(path="/project1/feedback1",
     properties={"top": "/project1/comp1"})
15. td_set_operator_pars(path="/project1/comp1",
     properties={"operand": 0})

16. td_execute_python: """
op('/project1/audio_in').outputConnectors[0].connect(op('/project1/spectrum'))
op('/project1/spectrum').outputConnectors[0].connect(op('/project1/spectrum_scale'))
op('/project1/noise1').outputConnectors[0].connect(op('/project1/level1'))
op('/project1/level1').outputConnectors[0].connect(op('/project1/comp1').inputConnectors[0])
op('/project1/feedback1').outputConnectors[0].connect(op('/project1/comp1').inputConnectors[1])
op('/project1/comp1').outputConnectors[0].connect(op('/project1/out'))
"""

17. td_execute_python: """
# Export spectrum values to drive noise parameters
# This makes the noise react to audio frequencies
op('/project1/noise1').par.seed.expr = "op('/project1/spectrum_scale')['chan1']"
op('/project1/noise1').par.period.expr = "tdu.remap(op('/project1/spectrum_scale')['chan1'].eval(), 0, 1, 1, 8)"
"""
```

### Pattern 2: Beat Detection -> Visual Pulses

Detect beats from audio and trigger visual events.

```
Audio Device In CHOP -> Audio Spectrum CHOP -> Math CHOP (isolate bass)
                                                    |
                                              Trigger CHOP (envelope)
                                                    |
                                              [export to visual params]
```

**Key parameter settings:**

```
# Isolate bass frequencies (20-200 Hz)
Math CHOP: chanop=1 (Add channels), range1low=0, range1high=10
           (first 10 FFT bins = bass frequencies with 512 FFT at 44100Hz)

# ADSR envelope on each beat
Trigger CHOP: attack=0.02, peak=1.0, decay=0.3, sustain=0.0, release=0.1

# Export to visual: Scale, brightness, or color intensity
td_execute_python: "op('/project1/level1').par.brightness1.expr = \"1.0 + op('/project1/trigger1')['chan1'] * 0.5\""
```

### Pattern 3: Multi-Band Audio -> Multi-Layer Visuals

Split audio into frequency bands, drive different visual layers per band.

```
Audio In -> Spectrum -> Audio Band EQ (3 bands: bass, mid, treble)
                              |
                    +---------+---------+
                    |         |         |
                 Bass      Mids     Treble
                  |          |         |
           Noise TOP   Circle TOP  Text TOP
           (slow,dark) (mid,warm)  (fast,bright)
                  |          |         |
                  +-----+----+----+----+
                        |         |
                   Composite  Composite
                        |
                       Out
```

### Pattern 3b: Audio-Reactive GLSL Fractal (Proven Recipe)

Complete working recipe. Plays an MP3, runs FFT, feeds spectrum as a texture into a GLSL shader where inner fractal reacts to bass, outer to treble.

**Network:**
```
AudioFileIn CHOP → AudioSpectrum CHOP (FFT=512, outlength=256)
    → Math CHOP (gain=10) → CHOP To TOP (256x2 spectrum texture, dataformat=r)
                                                                   ↓
Constant TOP (time, rgba32float) → GLSL TOP (input 0=time, input 1=spectrum) → Null → MovieFileOut
                                                                                        ↓
AudioFileIn CHOP → Audio Device Out CHOP                                          Record to .mov
```

**Build via td_execute_python (one call per step for reliability):**

```python
# Step 1: Audio chain
# td_execute_python script:
td_execute_python(code="""
root = op('/project1')
audio = root.create(audiofileinCHOP, 'audio_in')
audio.par.file = '/path/to/music.mp3'
audio.par.playmode = 0  # Locked to timeline
audio.par.volume = 0.5

spec = root.create(audiospectrumCHOP, 'spectrum')
audio.outputConnectors[0].connect(spec.inputConnectors[0])

math_n = root.create(mathCHOP, 'math_norm')
spec.outputConnectors[0].connect(math_n.inputConnectors[0])
math_n.par.gain = 5  # boost signal

resamp = root.create(resampleCHOP, 'resample_spec')
math_n.outputConnectors[0].connect(resamp.inputConnectors[0])
resamp.par.timeslice = True
resamp.par.rate = 256

chop2top = root.create(choptoTOP, 'spectrum_tex')
chop2top.par.chop = resamp  # CHOP To TOP has NO input connectors — use par.chop reference

# Audio output (hear the music)
aout = root.create(audiodeviceoutCHOP, 'audio_out')
audio.outputConnectors[0].connect(aout.inputConnectors[0])
result = 'audio chain ok'
""")

# Step 2: Time driver (MUST be rgba32float — see pitfalls #6)
# td_execute_python script:
td_execute_python(code="""
root = op('/project1')
td = root.create(constantTOP, 'time_driver')
td.par.format = 'rgba32float'
td.par.outputresolution = 'custom'
td.par.resolutionw = 1
td.par.resolutionh = 1
td.par.colorr.expr = "absTime.seconds % 1000.0"
td.par.colorg.expr = "int(absTime.seconds / 1000.0)"
result = 'time ok'
""")

# Step 3: GLSL shader (write to /tmp, load from file)
# td_execute_python script:
td_execute_python(code="""
root = op('/project1')
glsl = root.create(glslTOP, 'audio_shader')
glsl.par.outputresolution = 'custom'
glsl.par.resolutionw = 1280
glsl.par.resolutionh = 720

sd = root.create(textDAT, 'shader_code')
sd.text = open('/tmp/my_shader.glsl').read()
glsl.par.pixeldat = sd

# Wire: input 0 = time, input 1 = spectrum texture
op('/project1/time_driver').outputConnectors[0].connect(glsl.inputConnectors[0])
op('/project1/spectrum_tex').outputConnectors[0].connect(glsl.inputConnectors[1])
result = 'glsl ok'
""")

# Step 4: Output + recorder
# td_execute_python script:
td_execute_python(code="""
root = op('/project1')
out = root.create(nullTOP, 'output')
op('/project1/audio_shader').outputConnectors[0].connect(out.inputConnectors[0])

rec = root.create(moviefileoutTOP, 'recorder')
out.outputConnectors[0].connect(rec.inputConnectors[0])
rec.par.type = 'movie'
rec.par.file = '/tmp/output.mov'
rec.par.videocodec = 'mjpa'
result = 'output ok'
""")
```

**GLSL shader pattern (audio-reactive fractal):**
```glsl
out vec4 fragColor;

vec3 palette(float t) {
    vec3 a = vec3(0.5); vec3 b = vec3(0.5);
    vec3 c = vec3(1.0); vec3 d = vec3(0.263, 0.416, 0.557);
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    // Input 0 = time (1x1 rgba32float constant)
    // Input 1 = audio spectrum (256x2 CHOP To TOP, stereo — sample at y=0.25 for first channel)
    vec4 td = texture(sTD2DInputs[0], vec2(0.5));
    float t = td.r + td.g * 1000.0;

    vec2 res = uTDOutputInfo.res.zw;
    vec2 uv = (gl_FragCoord.xy * 2.0 - res) / min(res.x, res.y);
    vec2 uv0 = uv;
    vec3 finalColor = vec3(0.0);

    float bass = texture(sTD2DInputs[1], vec2(0.05, 0.25)).r;
    float mids = texture(sTD2DInputs[1], vec2(0.25, 0.25)).r;

    for (float i = 0.0; i < 4.0; i++) {
        uv = fract(uv * (1.4 + bass * 0.3)) - 0.5;
        float d = length(uv) * exp(-length(uv0));

        // Sample spectrum at distance: inner=bass, outer=treble
        float freq = texture(sTD2DInputs[1], vec2(clamp(d * 0.5, 0.0, 1.0), 0.25)).r;

        vec3 col = palette(length(uv0) + i * 0.4 + t * 0.35);
        d = sin(d * (7.0 + bass * 4.0) + t * 1.5) / 8.0;
        d = abs(d);
        d = pow(0.012 / d, 1.2 + freq * 0.8 + bass * 0.5);
        finalColor += col * d;
    }

    // Tone mapping
    finalColor = finalColor / (finalColor + vec3(1.0));
    fragColor = TDOutputSwizzle(vec4(finalColor, 1.0));
}
```

**Key insights from testing:**
- `spectrum_tex` (CHOP To TOP) produces a 256x2 texture — x position = frequency, y=0.25 for first channel
- Sampling at `vec2(0.05, 0.0)` gets bass, `vec2(0.65, 0.0)` gets treble
- Sampling based on pixel distance (`d * 0.5`) makes inner fractal react to bass, outer to treble
- `bass * 0.3` in the `fract()` zoom makes the fractal breathe with kicks
- Math CHOP gain of 5 is needed because raw spectrum values are very small

## Generative Art

### Pattern 4: Feedback Loop with Transform

Classic generative technique — texture evolves through recursive transformation.

```
Noise TOP -> Composite TOP -> Level TOP -> Null TOP (out)
                  ^      |
                  |      v
            Transform TOP <- Feedback TOP
```

**MCP Build Sequence:**

```
1. td_create_operator(parent="/project1", type="noiseTop", name="seed_noise")
2. td_create_operator(parent="/project1", type="compositeTop", name="mix")
3. td_create_operator(parent="/project1", type="transformTop", name="evolve")
4. td_create_operator(parent="/project1", type="feedbackTop", name="fb")
5. td_create_operator(parent="/project1", type="levelTop", name="color_correct")
6. td_create_operator(parent="/project1", type="nullTop", name="out")

7. td_set_operator_pars(path="/project1/seed_noise",
     properties={"type": 1, "monochrome": false, "period": 2.0, "amp": 0.3,
                  "resolutionw": 1280, "resolutionh": 720})
8. td_set_operator_pars(path="/project1/mix",
     properties={"operand": 27})  # 27 = Screen blend
9. td_set_operator_pars(path="/project1/evolve",
     properties={"sx": 1.003, "sy": 1.003, "rz": 0.5, "extend": 2})  # slight zoom + rotate, repeat edges
10. td_set_operator_pars(path="/project1/fb",
     properties={"top": "/project1/mix"})
11. td_set_operator_pars(path="/project1/color_correct",
     properties={"opacity": 0.98, "gamma1": 0.85})

12. td_execute_python: """
op('/project1/seed_noise').outputConnectors[0].connect(op('/project1/mix').inputConnectors[0])
op('/project1/fb').outputConnectors[0].connect(op('/project1/evolve'))
op('/project1/evolve').outputConnectors[0].connect(op('/project1/mix').inputConnectors[1])
op('/project1/mix').outputConnectors[0].connect(op('/project1/color_correct'))
op('/project1/color_correct').outputConnectors[0].connect(op('/project1/out'))
"""
```

**Variations:**
- Change Transform: `rz` (rotation), `sx/sy` (zoom), `tx/ty` (drift)
- Change Composite operand: Screen (glow), Add (bright), Multiply (dark)
- Add HSV Adjust in the feedback loop for color evolution
- Add Blur for dreamlike softness
- Replace Noise with a GLSL TOP for custom seed patterns

### Pattern 5: Instancing (Particle-Like Systems)

Render thousands of copies of geometry, each with unique position/rotation/scale driven by CHOP data or DATs.

```
Table DAT (instance data) -> DAT to CHOP -> Geometry COMP (instancing on) -> Render TOP
                                              + Sphere SOP (template geometry)
                                              + Constant MAT (material)
                                              + Camera COMP
                                              + Light COMP
```

**MCP Build Sequence:**

```
1. td_create_operator(parent="/project1", type="tableDat", name="instance_data")
2. td_create_operator(parent="/project1", type="geometryComp", name="geo1")
3. td_create_operator(parent="/project1/geo1", type="sphereSop", name="sphere")
4. td_create_operator(parent="/project1", type="constMat", name="mat1")
5. td_create_operator(parent="/project1", type="cameraComp", name="cam1")
6. td_create_operator(parent="/project1", type="lightComp", name="light1")
7. td_create_operator(parent="/project1", type="renderTop", name="render1")

8. td_execute_python: """
import random, math
dat = op('/project1/instance_data')
dat.clear()
dat.appendRow(['tx', 'ty', 'tz', 'sx', 'sy', 'sz', 'cr', 'cg', 'cb'])
for i in range(500):
    angle = i * 0.1
    r = 2 + i * 0.01
    dat.appendRow([
        str(math.cos(angle) * r),
        str(math.sin(angle) * r),
        str((i - 250) * 0.02),
        '0.05', '0.05', '0.05',
        str(random.random()),
        str(random.random()),
        str(random.random())
    ])
"""

9. td_set_operator_pars(path="/project1/geo1",
     properties={"instancing": true, "instancechop": "",
                  "instancedat": "/project1/instance_data",
                  "material": "/project1/mat1"})
10. td_set_operator_pars(path="/project1/render1",
     properties={"camera": "/project1/cam1", "geometry": "/project1/geo1",
                  "light": "/project1/light1",
                  "resolutionw": 1280, "resolutionh": 720})
11. td_set_operator_pars(path="/project1/cam1",
     properties={"tz": 10})
```

### Pattern 6: Reaction-Diffusion (GLSL)

Classic Gray-Scott reaction-diffusion system running on the GPU.

```
Text DAT (GLSL code) -> GLSL TOP (resolution, dat reference) -> Feedback TOP
                              ^                                       |
                              |_______________________________________|
                         Level TOP (out)
```

**Key GLSL code (write to Text DAT via td_execute_python):**

```glsl
// Gray-Scott reaction-diffusion
uniform float feed;    // 0.037
uniform float kill;    // 0.06
uniform float dA;      // 1.0
uniform float dB;      // 0.5

layout(location = 0) out vec4 fragColor;

void main() {
    vec2 uv = vUV.st;
    vec2 texel = 1.0 / uTDOutputInfo.res.zw;

    vec4 c = texture(sTD2DInputs[0], uv);
    float a = c.r;
    float b = c.g;

    // Laplacian (9-point stencil)
    float lA = 0.0, lB = 0.0;
    for(int dx = -1; dx <= 1; dx++) {
        for(int dy = -1; dy <= 1; dy++) {
            float w = (dx == 0 && dy == 0) ? -1.0 : (abs(dx) + abs(dy) == 1 ? 0.2 : 0.05);
            vec4 s = texture(sTD2DInputs[0], uv + vec2(dx, dy) * texel);
            lA += s.r * w;
            lB += s.g * w;
        }
    }

    float reaction = a * b * b;
    float newA = a + (dA * lA - reaction + feed * (1.0 - a));
    float newB = b + (dB * lB + reaction - (kill + feed) * b);

    fragColor = vec4(clamp(newA, 0.0, 1.0), clamp(newB, 0.0, 1.0), 0.0, 1.0);
}
```

## Video Processing

### Pattern 7: Video Effects Chain

Apply a chain of effects to a video file.

```
Movie File In TOP -> HSV Adjust TOP -> Level TOP -> Blur TOP -> Composite TOP -> Null TOP (out)
                                                                      ^
                                                          Text TOP ---+
```

**MCP Build Sequence:**

```
1. td_create_operator(parent="/project1", type="moviefileinTop", name="video_in")
2. td_create_operator(parent="/project1", type="hsvadjustTop", name="color")
3. td_create_operator(parent="/project1", type="levelTop", name="levels")
4. td_create_operator(parent="/project1", type="blurTop", name="blur")
5. td_create_operator(parent="/project1", type="compositeTop", name="overlay")
6. td_create_operator(parent="/project1", type="textTop", name="title")
7. td_create_operator(parent="/project1", type="nullTop", name="out")

8. td_set_operator_pars(path="/project1/video_in",
     properties={"file": "/path/to/video.mp4", "play": true})
9. td_set_operator_pars(path="/project1/color",
     properties={"hueoffset": 0.1, "saturationmult": 1.3})
10. td_set_operator_pars(path="/project1/levels",
     properties={"brightness1": 1.1, "contrast": 1.2, "gamma1": 0.9})
11. td_set_operator_pars(path="/project1/blur",
     properties={"sizex": 2, "sizey": 2})
12. td_set_operator_pars(path="/project1/title",
     properties={"text": "My Video", "fontsizex": 48, "alignx": 1, "aligny": 1})

13. td_execute_python: """
chain = ['video_in', 'color', 'levels', 'blur']
for i in range(len(chain) - 1):
    op(f'/project1/{chain[i]}').outputConnectors[0].connect(op(f'/project1/{chain[i+1]}'))
op('/project1/blur').outputConnectors[0].connect(op('/project1/overlay').inputConnectors[0])
op('/project1/title').outputConnectors[0].connect(op('/project1/overlay').inputConnectors[1])
op('/project1/overlay').outputConnectors[0].connect(op('/project1/out'))
"""
```

### Pattern 8: Video Recording

Record the output to a file. **H.264/H.265 require a Commercial license** — use Motion JPEG (`mjpa`) on Non-Commercial.

```
[any TOP chain] -> Null TOP -> Movie File Out TOP
```

```python
# Build via td_execute_python:
root = op('/project1')

# Always put a Null TOP before the recorder
null_out = root.op('out')  # or create one
rec = root.create(moviefileoutTOP, 'recorder')
null_out.outputConnectors[0].connect(rec.inputConnectors[0])

rec.par.type = 'movie'
rec.par.file = '/tmp/output.mov'
rec.par.videocodec = 'mjpa'  # Motion JPEG — works on Non-Commercial

# Start recording (par.record is a toggle — .record() method may not exist)
rec.par.record = True
# ... let TD run for desired duration ...
rec.par.record = False

# For image sequences:
# rec.par.type = 'imagesequence'
# rec.par.imagefiletype = 'png'
# rec.par.file.expr = "'/tmp/frames/out' + me.fileSuffix"  # fileSuffix REQUIRED
```

**Pitfalls:**
- Setting `par.file` + `par.record = True` in the same script may race — use `run("...", delayFrames=2)`
- `TOP.save()` called rapidly always captures the same frame — use MovieFileOut for animation
- See `pitfalls.md` #25-27 for full details

### Pattern 8b: TD → External Pipeline (FFmpeg / Python / Post-Processing)

Export TD visuals for use in another tool (ffmpeg, Python, ASCII art, etc.). This is the standard workflow when you need to composite TD output with external processing (ASCII conversion, Python shader chains, ML inference, etc.).

**Step 1: Record to video in TD**

```python
# Preferred: ProRes on macOS (lossless, Non-Commercial OK, ~55MB/s at 1280x720)
rec.par.videocodec = 'prores'
# Fallback for non-macOS: mjpa (Motion JPEG)
# rec.par.videocodec = 'mjpa'
rec.par.record = True
# ... wait N seconds ...
rec.par.record = False
```

**Step 2: Extract frames with ffmpeg**

```bash
# Extract all frames at 30fps
ffmpeg -y -i /tmp/output.mov -vf 'fps=30' /tmp/frames/frame_%06d.png

# Or extract a specific duration
ffmpeg -y -i /tmp/output.mov -t 25 -vf 'fps=30' /tmp/frames/frame_%06d.png

# Or extract specific frame range
ffmpeg -y -i /tmp/output.mov -vf 'select=between(n\,0\,749)' -vsync vfr /tmp/frames/frame_%06d.png
```

**Step 3: Process frames in Python**

```python
from PIL import Image
import os

frames_dir = '/tmp/frames'
output_dir = '/tmp/processed'
os.makedirs(output_dir, exist_ok=True)

for fname in sorted(os.listdir(frames_dir)):
    if not fname.endswith('.png'):
        continue
    img = Image.open(os.path.join(frames_dir, fname))
    # ... apply your processing ...
    img.save(os.path.join(output_dir, fname))
```

**Step 4: Mux processed frames back with audio**

```bash
# Create video from processed frames + audio with fade-out
ffmpeg -y \
  -framerate 30 -i /tmp/processed/frame_%06d.png \
  -i /tmp/audio.mp3 \
  -c:v libx264 -pix_fmt yuv420p -crf 18 \
  -c:a aac -b:a 192k \
  -shortest \
  -af 'afade=t=out:st=23:d=2' \
  /tmp/final_output.mp4
```

**Key considerations:**
- Use ProRes for the TD recording step to avoid generation loss during compositing
- Extract at the target output framerate (not TD's render framerate)
- For audio-synced content, analyze the audio file separately in Python (scipy FFT) to get per-frame features (rms, spectral bands, beats) and drive compositing parameters
- Always verify TD FPS > 0 before recording (see pitfalls #37, #38)

## Data Visualization

### Pattern 9: Table Data -> Bar Chart via Instancing

Visualize tabular data as a 3D bar chart.

```
Table DAT (data) -> Script DAT (transform to instance format) -> DAT to CHOP
                                                                      |
Box SOP -> Geometry COMP (instancing from CHOP) -> Render TOP -> Null TOP (out)
           + PBR MAT
           + Camera COMP
           + Light COMP
```

```python
# Script DAT code to transform data to instance positions
td_execute_python: """
source = op('/project1/data_table')
instance = op('/project1/instance_transform')
instance.clear()
instance.appendRow(['tx', 'ty', 'tz', 'sx', 'sy', 'sz', 'cr', 'cg', 'cb'])

for i in range(1, source.numRows):
    value = float(source[i, 'value'])
    name = source[i, 'name']
    instance.appendRow([
        str(i * 1.5),          # x position (spread bars)
        str(value / 2),        # y position (center bar vertically)
        '0',                   # z position
        '1', str(value), '1',  # scale (height = data value)
        '0.2', '0.6', '1.0'   # color (blue)
    ])
"""
```

### Pattern 9b: Audio-Reactive GLSL Fractal (Proven Recipe)

Audio spectrum drives a GLSL fractal shader directly via a spectrum texture input. Bass thickens inner fractal lines, mids twist rotation, highs light outer edges. **Always run discovery (SKILL.md Step 0) before using any param names from these recipes — they may differ in your TD version.**

```
Audio File In CHOP → Audio Spectrum CHOP (FFT=512, outlength=256)
    → Math CHOP (gain=10)
    → CHOP To TOP (spectrum texture, 256x2, dataformat=r)
                                          ↓ (input 1)
Constant TOP (rgba32float, time) → GLSL TOP (audio-reactive shader) → Null TOP
        (input 0)                    ↑
                              Text DAT (shader code)
```

**Build via td_execute_python (complete working script):**

```python
# td_execute_python script:
td_execute_python(code="""
import os
root = op('/project1')

# Audio input
audio = root.create(audiofileinCHOP, 'audio_in')
audio.par.file = '/path/to/music.mp3'
audio.par.playmode = 0  # Locked to timeline

# FFT analysis (output length manually set to 256 bins)
spectrum = root.create(audiospectrumCHOP, 'spectrum')
audio.outputConnectors[0].connect(spectrum.inputConnectors[0])
spectrum.par.fftsize = '512'
spectrum.par.outputmenu = 'setmanually'
spectrum.par.outlength = 256

# THEN boost gain on the raw spectrum (NO Lag CHOP — see pitfall #34)
math = root.create(mathCHOP, 'math_norm')
spectrum.outputConnectors[0].connect(math.inputConnectors[0])
math.par.gain = 10

# Spectrum → texture (256x2 image — stereo, sample at y=0.25 for first channel)
# NOTE: choptoTOP has NO input connectors — use par.chop reference!
spec_tex = root.create(choptoTOP, 'spectrum_tex')
spec_tex.par.chop = math
spec_tex.par.dataformat = 'r'
spec_tex.par.layout = 'rowscropped'

# Time driver (rgba32float to avoid 0-1 clamping!)
time_drv = root.create(constantTOP, 'time_driver')
time_drv.par.format = 'rgba32float'
time_drv.par.outputresolution = 'custom'
time_drv.par.resolutionw = 1
time_drv.par.resolutionh = 1
time_drv.par.colorr.expr = "absTime.seconds % 1000.0"
time_drv.par.colorg.expr = "int(absTime.seconds / 1000.0)"

# GLSL shader
glsl = root.create(glslTOP, 'audio_shader')
glsl.par.outputresolution = 'custom'
glsl.par.resolutionw = 1280; glsl.par.resolutionh = 720

shader_dat = root.create(textDAT, 'shader_code')
shader_dat.text = open('/tmp/shader.glsl').read()
glsl.par.pixeldat = shader_dat

# Wire: input 0=time, input 1=spectrum
time_drv.outputConnectors[0].connect(glsl.inputConnectors[0])
spec_tex.outputConnectors[0].connect(glsl.inputConnectors[1])

# Output + audio playback
out = root.create(nullTOP, 'output')
glsl.outputConnectors[0].connect(out.inputConnectors[0])
audio_out = root.create(audiodeviceoutCHOP, 'audio_out')
audio.outputConnectors[0].connect(audio_out.inputConnectors[0])

result = 'network built'
""")
```

**GLSL shader (reads spectrum from input 1 texture):**

```glsl
out vec4 fragColor;

vec3 palette(float t) {
    vec3 a = vec3(0.5); vec3 b = vec3(0.5);
    vec3 c = vec3(1.0); vec3 d = vec3(0.263, 0.416, 0.557);
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec4 td = texture(sTD2DInputs[0], vec2(0.5));
    float t = td.r + td.g * 1000.0;

    vec2 res = uTDOutputInfo.res.zw;
    vec2 uv = (gl_FragCoord.xy * 2.0 - res) / min(res.x, res.y);
    vec2 uv0 = uv;
    vec3 finalColor = vec3(0.0);

    float bass = texture(sTD2DInputs[1], vec2(0.05, 0.25)).r;
    float mids = texture(sTD2DInputs[1], vec2(0.25, 0.25)).r;
    float highs = texture(sTD2DInputs[1], vec2(0.65, 0.25)).r;

    float ca = cos(t * (0.15 + mids * 0.3));
    float sa = sin(t * (0.15 + mids * 0.3));
    uv = mat2(ca, -sa, sa, ca) * uv;

    for (float i = 0.0; i < 4.0; i++) {
        uv = fract(uv * (1.4 + bass * 0.3)) - 0.5;
        float d = length(uv) * exp(-length(uv0));
        float freq = texture(sTD2DInputs[1], vec2(clamp(d*0.5, 0.0, 1.0), 0.25)).r;
        vec3 col = palette(length(uv0) + i * 0.4 + t * 0.35);
        d = sin(d * (7.0 + bass * 4.0) + t * 1.5) / 8.0;
        d = abs(d);
        d = pow(0.012 / d, 1.2 + freq * 0.8 + bass * 0.5);
        finalColor += col * d;
    }

    float glow = (0.03 + bass * 0.05) / (length(uv0) + 0.03);
    finalColor += vec3(0.4, 0.1, 0.7) * glow * (0.6 + 0.4 * sin(t * 2.5));

    float ring = abs(length(uv0) - 0.4 - mids * 0.3);
    finalColor += vec3(0.1, 0.6, 0.8) * (0.005 / ring) * (0.2 + highs * 0.5);

    finalColor *= smoothstep(0.0, 1.0, 1.0 - dot(uv0*0.55, uv0*0.55));
    finalColor = finalColor / (finalColor + vec3(1.0));

    fragColor = TDOutputSwizzle(vec4(finalColor, 1.0));
}
```

**How spectrum sampling drives the visual:**
- `texture(sTD2DInputs[1], vec2(x, 0.0)).r` — x position = frequency (0=bass, 1=treble)
- Inner fractal iterations sample lower x → react to bass
- Outer iterations sample higher x → react to treble
- `bass * 0.3` on `fract()` scale → fractal zoom pulses with bass
- `bass * 4.0` on sin frequency → line density pulses with bass
- `mids * 0.3` on rotation speed → spiral twists faster during vocal/mid sections
- `highs * 0.5` on ring opacity → high-frequency sparkle on outer ring

**Recording the output:** Use MovieFileOut TOP with `mjpa` codec (H.264 requires Commercial license). See pitfalls #25-27.

## GLSL Shaders

### Pattern 10: Custom Fragment Shader

Write a custom visual effect as a GLSL fragment shader.

```
Text DAT (shader code) -> GLSL TOP -> Level TOP -> Null TOP (out)
                           + optional input TOPs for texture sampling
```

**Common GLSL uniforms available in TouchDesigner:**

```glsl
// Automatically provided by TD
uniform vec4 uTDOutputInfo;  // .res.zw = resolution

// NOTE: uTDCurrentTime does NOT exist in TD 099!
// Feed time via a 1x1 Constant TOP (format=rgba32float):
//   t.par.colorr.expr = "absTime.seconds % 1000.0"
//   t.par.colorg.expr = "int(absTime.seconds / 1000.0)"
// Then read in GLSL:
//   vec4 td = texture(sTD2DInputs[0], vec2(0.5));
//   float t = td.r + td.g * 1000.0;

// Input textures (from connected TOP inputs)
uniform sampler2D sTD2DInputs[1];  // array of input samplers

// From vertex shader
in vec3 vUV;  // UV coordinates (0-1 range)
```

**Example: Plasma shader (using time from input texture)**

```glsl
layout(location = 0) out vec4 fragColor;

void main() {
    vec2 uv = vUV.st;
    // Read time from Constant TOP input 0 (rgba32float format)
    vec4 td = texture(sTD2DInputs[0], vec2(0.5));
    float t = td.r + td.g * 1000.0;

    float v1 = sin(uv.x * 10.0 + t);
    float v2 = sin(uv.y * 10.0 + t * 0.7);
    float v3 = sin((uv.x + uv.y) * 10.0 + t * 1.3);
    float v4 = sin(length(uv - 0.5) * 20.0 - t * 2.0);

    float v = (v1 + v2 + v3 + v4) * 0.25;

    vec3 color = vec3(
        sin(v * 3.14159 + 0.0) * 0.5 + 0.5,
        sin(v * 3.14159 + 2.094) * 0.5 + 0.5,
        sin(v * 3.14159 + 4.189) * 0.5 + 0.5
    );

    fragColor = vec4(color, 1.0);
}
```

### Pattern 11: Multi-Pass GLSL (Ping-Pong)

For effects needing state across frames (particles, fluid, cellular automata), use GLSL Multi TOP with multiple passes or a Feedback TOP loop.

```
GLSL Multi TOP (pass 0: simulation, pass 1: rendering)
   + Text DAT (simulation shader)
   + Text DAT (render shader)
   -> Level TOP -> Null TOP (out)
      ^
      |__ Feedback TOP (feeds simulation state back)
```

## Interactive Installations

### Pattern 12: Mouse/Touch -> Visual Response

```
Mouse In CHOP -> Math CHOP (normalize to 0-1) -> [export to visual params]

# Or for touch/multi-touch:
Multi Touch In DAT -> Script CHOP (parse touches) -> [export to visual params]
```

```python
# Normalize mouse position to 0-1 range
td_execute_python: """
op('/project1/noise1').par.offsetx.expr = "op('/project1/mouse_norm')['tx']"
op('/project1/noise1').par.offsety.expr = "op('/project1/mouse_norm')['ty']"
"""
```

### Pattern 13: OSC Control (from external software)

```
OSC In CHOP (port 7000) -> Select CHOP (pick channels) -> [export to visual params]
```

```
1. td_create_operator(parent="/project1", type="oscinChop", name="osc_in")
2. td_set_operator_pars(path="/project1/osc_in", properties={"port": 7000})

# OSC messages like /frequency 440 will appear as channel "frequency" with value 440
# Export to any parameter:
3. td_execute_python: "op('/project1/noise1').par.period.expr = \"op('/project1/osc_in')['frequency']\""
```

### Pattern 14: MIDI Control (DJ/VJ)

```
MIDI In CHOP (device) -> Select CHOP -> [export channels to visual params]
```

Common MIDI mappings:
- CC channels (knobs/faders): continuous 0-127, map to float params
- Note On/Off: binary triggers, map to Trigger CHOP for envelopes
- Velocity: intensity/brightness

## Live Performance

### Pattern 15: Multi-Source VJ Setup

```
Source A (generative) ----+
Source B (video) ---------+-- Switch/Cross TOP -- Level TOP -- Window COMP (output)
Source C (camera) --------+
                           ^
                    MIDI/OSC control selects active source and crossfade
```

```python
# MIDI CC1 controls which source is active (0-127 -> 0-2)
td_execute_python: """
op('/project1/switch1').par.index.expr = "int(op('/project1/midi_in')['cc1'] / 42)"
"""

# MIDI CC2 controls crossfade between current and next
td_execute_python: """
op('/project1/cross1').par.cross.expr = "op('/project1/midi_in')['cc2'] / 127.0"
"""
```

### Pattern 16: Projection Mapping

```
Content TOPs ----+
                 |
Stoner TOP (UV mapping) -> Composite TOP -> Window COMP (projector output)
   or
Kantan Mapper COMP (external .tox)
```

For projection mapping, the key is:
1. Create your visual content as standard TOPs
2. Use Stoner TOP or a third-party mapping tool to UV-map content to physical surfaces
3. Output via Window COMP to the projector

### Pattern 17: Cue System

```
Table DAT (cue list: cue_number, scene_name, duration, transition_type)
    |
Script CHOP (cue state: current_cue, progress, next_cue_trigger)
    |
[export to Switch/Cross TOPs to transition between scenes]
```

```python
td_execute_python: """
# Simple cue system
cue_table = op('/project1/cue_list')
cue_state = op('/project1/cue_state')

def advance_cue():
    current = int(cue_state.par.value0.val)
    next_cue = min(current + 1, cue_table.numRows - 1)
    cue_state.par.value0.val = next_cue
    
    scene = cue_table[next_cue, 'scene']
    duration = float(cue_table[next_cue, 'duration'])
    
    # Set crossfade target and duration
    op('/project1/cross1').par.cross.val = 0
    # Animate cross to 1.0 over duration seconds
    # (use a Timer CHOP or LFO CHOP for smooth animation)
"""
```

## Networking

### Pattern 18: OSC Server/Client

```
# Sending OSC
OSC Out CHOP -> (network) -> external application

# Receiving OSC  
(network) -> OSC In CHOP -> Select CHOP -> [use values]
```

### Pattern 19: NDI Video Streaming

```
# Send video over network
[any TOP chain] -> NDI Out TOP (source name)

# Receive video from network
NDI In TOP (select source) -> [process as normal TOP]
```

### Pattern 20: WebSocket Communication

```
WebSocket DAT -> Script DAT (parse JSON messages) -> [update visuals]
```

```python
td_execute_python: """
ws = op('/project1/websocket1')
ws.par.address = 'ws://localhost:8080'
ws.par.active = True

# In a DAT Execute callback (Script DAT watching WebSocket DAT):
# def onTableChange(dat):
#     import json
#     msg = json.loads(dat.text)
#     op('/project1/noise1').par.seed.val = msg.get('seed', 0)
"""
```
