---
name: ascii-video
description: "Production pipeline for ASCII art video — any format. Converts video/audio/images/generative input into colored ASCII character video output (MP4, GIF, image sequence). Covers: video-to-ASCII conversion, audio-reactive music visualizers, generative ASCII art animations, hybrid video+audio reactive, text/lyrics overlays, real-time terminal rendering. Use when users request: ASCII video, text art video, terminal-style video, character art animation, retro text visualization, audio visualizer in ASCII, converting video to ASCII art, matrix-style effects, or any animated ASCII output."
---

# ASCII Video Production Pipeline

Full production pipeline for rendering any content as colored ASCII character video.

## Modes

| Mode | Input | Output | Read |
|------|-------|--------|------|
| **Video-to-ASCII** | Video file | ASCII recreation of source footage | `references/inputs.md` § Video Sampling |
| **Audio-reactive** | Audio file | Generative visuals driven by audio features | `references/inputs.md` § Audio Analysis |
| **Generative** | None (or seed params) | Procedural ASCII animation | `references/effects.md` |
| **Hybrid** | Video + audio | ASCII video with audio-reactive overlays | Both input refs |
| **Lyrics/text** | Audio + text/SRT | Timed text with visual effects | `references/inputs.md` § Text/Lyrics |
| **TTS narration** | Text quotes + TTS API | Narrated testimonial/quote video with typed text | `references/inputs.md` § TTS Integration |

## Stack

Single self-contained Python script per project. No GPU.

| Layer | Tool | Purpose |
|-------|------|---------|
| Core | Python 3.10+, NumPy | Math, array ops, vectorized effects |
| Signal | SciPy | FFT, peak detection (audio modes only) |
| Imaging | Pillow (PIL) | Font rasterization, video frame decoding, image I/O |
| Video I/O | ffmpeg (CLI) | Decode input, encode output segments, mux audio, mix tracks |
| Parallel | concurrent.futures / multiprocessing | N workers for batch/clip rendering |
| TTS | ElevenLabs API (or similar) | Generate narration clips for quote/testimonial videos |
| Optional | OpenCV | Video frame sampling, edge detection, optical flow |

## Pipeline Architecture (v2)

Every mode follows the same 6-stage pipeline. See `references/architecture.md` for implementation details, `references/scenes.md` for scene protocol, and `references/composition.md` for multi-grid composition and tonemap.

```
┌─────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌─────────┐   ┌────────┐
│ 1.INPUT  │→│ 2.ANALYZE │→│ 3.SCENE_FN │→│ 4.TONEMAP │→│ 5.SHADE  │→│ 6.ENCODE│
│ load src │  │ features  │  │ → canvas   │  │ normalize │  │ post-fx  │  │ → video │
└─────────┘   └──────────┘   └───────────┘   └──────────┘   └─────────┘   └────────┘
```

1. **INPUT** — Load/decode source material (video frames, audio samples, images, or nothing)
2. **ANALYZE** — Extract per-frame features (audio bands, video luminance/edges, motion vectors)
3. **SCENE_FN** — Scene function renders directly to pixel canvas (`uint8 H,W,3`). May internally compose multiple character grids via `_render_vf()` + pixel blend modes. See `references/composition.md`
4. **TONEMAP** — Percentile-based adaptive brightness normalization with per-scene gamma. Replaces linear brightness multipliers. See `references/composition.md` § Adaptive Tonemap
5. **SHADE** — Apply post-processing `ShaderChain` + `FeedbackBuffer`. See `references/shaders.md`
6. **ENCODE** — Pipe raw RGB frames to ffmpeg for H.264/GIF encoding

## Creative Direction

**Every project should look and feel different.** The references provide a vocabulary of building blocks — don't copy them verbatim. Combine, modify, and invent.

### Aesthetic Dimensions to Vary

| Dimension | Options | Reference |
|-----------|---------|-----------|
| **Character palette** | Density ramps, block elements, symbols, scripts (katakana, Greek, runes, braille), dots, project-specific | `architecture.md` § Character Palettes |
| **Color strategy** | HSV (angle/distance/time/value mapped), discrete RGB palettes, monochrome, complementary, triadic, temperature | `architecture.md` § Color System |
| **Color tint** | Warm, cool, amber, matrix green, neon pink, sepia, ice, blood, void, sunset | `shaders.md` § Color Grade |
| **Background texture** | Sine fields, noise, smooth noise, cellular/voronoi, video source | `effects.md` § Background Fills |
| **Primary effects** | Rings, spirals, tunnel, vortex, waves, interference, aurora, ripple, fire | `effects.md` § Radial / Wave / Fire |
| **Particles** | Energy sparks, snow, rain, bubbles, runes, binary data, orbits, gravity wells | `effects.md` § Particle Systems |
| **Shader mood** | Retro CRT, clean modern, glitch art, cinematic, dreamy, harsh industrial, psychedelic | `shaders.md` § Design Philosophy |
| **Grid density** | xs(8px) through xxl(40px), mixed per layer | `architecture.md` § Grid System |
| **Font** | Menlo, Monaco, Courier, SF Mono, JetBrains Mono, Fira Code, IBM Plex | `architecture.md` § Font Selection |
| **Mirror mode** | None, horizontal, vertical, quad, diagonal, kaleidoscope | `shaders.md` § Mirror Effects |
| **Transition style** | Crossfade, wipe (directional/radial), dissolve, glitch cut | `shaders.md` § Transitions |

### Per-Section Variation

Never use the same config for the entire video. For each section/scene/quote:
- Choose a **different background effect** (or compose 2-3)
- Choose a **different character palette** (match the mood)
- Choose a **different color strategy** (or at minimum a different hue)
- Vary **shader intensity** (more bloom during peaks, more grain during quiet)
- Use **different particle types** if particles are active

### Project-Specific Invention

For every project, invent at least one of:
- A custom character palette matching the theme
- A custom background effect (combine/modify existing ones)
- A custom color palette (discrete RGB set matching the brand/mood)
- A custom particle character set

## Workflow

### Step 1: Determine Mode and Gather Requirements

Establish with user:
- **Input source** — file path, format, duration
- **Mode** — which of the 6 modes above
- **Sections** — time-mapped style changes (timestamps → effect names)
- **Resolution** — default 1920x1080 @ 24fps; GIFs typically 640x360 @ 15fps
- **Style direction** — dense/sparse, bright/dark, chaotic/minimal, color palette
- **Text/branding** — easter eggs, overlays, credits, themed character sets
- **Output format** — MP4 (default), GIF, PNG sequence

### Step 2: Detect Hardware and Set Quality

Before building the script, detect the user's hardware and set appropriate defaults. See `references/optimization.md` § Hardware Detection.

```python
hw = detect_hardware()
profile = quality_profile(hw, target_duration, user_quality_pref)
log(f"Hardware: {hw['cpu_count']} cores, {hw['mem_gb']:.1f}GB RAM")
log(f"Render: {profile['vw']}x{profile['vh']} @{profile['fps']}fps, {profile['workers']} workers")
```

Never hardcode worker counts, resolution, or CRF. Always detect and adapt.

### Step 3: Build the Script

Write as a single Python file. Major components:

1. **Hardware detection + quality profile** — see `references/optimization.md`
2. **Input loader** — mode-dependent; see `references/inputs.md`
3. **Feature analyzer** — audio FFT, video luminance, or pass-through
4. **Grid + renderer** — multi-density character grids with bitmap cache; `_render_vf()` helper for value/hue field → canvas
5. **Character palettes** — multiple palettes chosen per project theme; see `references/architecture.md`
6. **Color system** — HSV + discrete RGB palettes as needed; see `references/architecture.md`
7. **Scene functions** — each returns `canvas (uint8 H,W,3)` directly. May compose multiple grids internally via pixel blend modes. See `references/scenes.md` + `references/composition.md`
8. **Tonemap** — adaptive brightness normalization with per-scene gamma; see `references/composition.md`
9. **Shader pipeline** — `ShaderChain` + `FeedbackBuffer` per-section config; see `references/shaders.md`
10. **Scene table + dispatcher** — maps time ranges to scene functions + shader/feedback configs; see `references/scenes.md`
11. **Parallel encoder** — N-worker batch clip rendering with ffmpeg pipes
12. **Main** — orchestrate full pipeline

### Step 4: Handle Critical Bugs

#### Font Cell Height (macOS Pillow)

`textbbox()` returns wrong height. Use `font.getmetrics()`:

```python
ascent, descent = font.getmetrics()
cell_height = ascent + descent  # correct
```

#### ffmpeg Pipe Deadlock

Never use `stderr=subprocess.PIPE` with long-running ffmpeg. Redirect to file:

```python
stderr_fh = open(err_path, "w")
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_fh)
```

#### Brightness — Use `tonemap()`, Not Linear Multipliers

ASCII on black is inherently dark. This is the #1 visual issue. **Do NOT use linear `* N` brightness multipliers** — they clip highlights and wash out the image. Instead, use the **adaptive tonemap** function from `references/composition.md`:

```python
def tonemap(canvas, gamma=0.75):
    """Percentile-based adaptive normalization + gamma. Replaces all brightness multipliers."""
    f = canvas.astype(np.float32)
    lo = np.percentile(f, 1)          # black point (1st percentile)
    hi = np.percentile(f, 99.5)       # white point (99.5th percentile)
    if hi - lo < 1: hi = lo + 1
    f = (f - lo) / (hi - lo)
    f = np.clip(f, 0, 1) ** gamma     # gamma < 1 = brighter mids
    return (f * 255).astype(np.uint8)
```

Pipeline ordering: `scene_fn() → tonemap() → FeedbackBuffer → ShaderChain → ffmpeg`

Per-scene gamma overrides for destructive effects:
- Default: `gamma=0.75`
- Solarize scenes: `gamma=0.55` (solarize darkens above-threshold pixels)
- Posterize scenes: `gamma=0.50` (quantization loses brightness range)
- Already-bright scenes: `gamma=0.85`

Additional brightness best practices:
- Dense animated backgrounds — never flat black, always fill the grid
- Vignette minimum clamped to 0.15 (not 0.12)
- Bloom threshold lowered to 130 (not 170) so more pixels contribute to glow
- Use `screen` blend mode (not `overlay`) when compositing dark ASCII layers — overlay squares dark values: `2 * 0.12 * 0.12 = 0.03`

#### Font Compatibility

Not all Unicode characters render in all fonts. Validate palettes at init:
```python
for c in palette:
    img = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(img).text((0, 0), c, fill=255, font=font)
    if np.array(img).max() == 0:
        log(f"WARNING: char '{c}' (U+{ord(c):04X}) not in font, removing from palette")
```

### Step 4b: Per-Clip Architecture (for segmented videos)

When the video has discrete segments (quotes, scenes, chapters), render each as a separate clip file. This enables:
- Re-rendering individual clips without touching the rest (`--clip q05`)
- Faster iteration on specific sections
- Easy reordering or trimming in post

```python
segments = [
    {"id": "intro", "start": 0.0, "end": 5.0, "type": "intro"},
    {"id": "q00", "start": 5.0, "end": 12.0, "type": "quote", "qi": 0, ...},
    {"id": "t00", "start": 12.0, "end": 13.5, "type": "transition", ...},
    {"id": "outro", "start": 208.0, "end": 211.6, "type": "outro"},
]

from concurrent.futures import ProcessPoolExecutor, as_completed
with ProcessPoolExecutor(max_workers=hw["workers"]) as pool:
    futures = {pool.submit(render_clip, seg, features, path): seg["id"]
               for seg, path in clip_args}
    for fut in as_completed(futures):
        fut.result()
```

CLI: `--clip q00 t00 q01` to re-render specific clips, `--list` to show segments, `--skip-render` to re-stitch only.

### Step 5: Render and Iterate

Performance targets per frame:

| Component | Budget |
|-----------|--------|
| Feature extraction | 1-5ms |
| Effect function | 2-15ms |
| Character render | 80-150ms (bottleneck) |
| Shader pipeline | 5-25ms |
| **Total** | ~100-200ms/frame |

**Fast iteration**: render single test frames to check brightness/layout before full render:
```python
canvas = render_single_frame(frame_index, features, renderer)
Image.fromarray(canvas).save("test.png")
```

**Brightness verification**: sample 5-10 frames across video, check `mean > 8` for ASCII content.

## References

| File | Contents |
|------|----------|
| `references/architecture.md` | Grid system, font selection, character palettes (library of 20+), color system (HSV + discrete RGB), `_render_vf()` helper, compositing, v2 effect function contract |
| `references/inputs.md` | All input sources: audio analysis, video sampling, image conversion, text/lyrics, TTS integration (ElevenLabs, voice assignment, audio mixing) |
| `references/effects.md` | Effect building blocks: 12 value field generators (`vf_sinefield` through `vf_noise_static`), 8 hue field generators (`hf_fixed` through `hf_plasma`), radial/wave/fire effects, particles, composing guide |
| `references/shaders.md` | 38 shader implementations (geometry, channel, color, glow, noise, pattern, tone, glitch, mirror), `ShaderChain` class, full `_apply_shader_step()` dispatch, audio-reactive scaling, transitions, tint presets |
| `references/composition.md` | **v2 core**: pixel blend modes (20 modes with implementations), multi-grid composition, `_render_vf()` helper, adaptive `tonemap()`, per-scene gamma, `FeedbackBuffer` with spatial transforms, `PixelBlendStack` |
| `references/scenes.md` | **v2 scene protocol**: scene function contract, `Renderer` class, `SCENES` table structure, `render_clip()` loop, beat-synced cutting, parallel rendering + pickling constraints, 4 complete scene examples, scene design checklist |
| `references/troubleshooting.md` | NumPy broadcasting traps, blend mode pitfalls, multiprocessing/pickling issues, brightness diagnostics, ffmpeg deadlocks, font issues, performance bottlenecks, common mistakes |
| `references/optimization.md` | Hardware detection, adaptive quality profiles (draft/preview/production/max), CLI integration, vectorized effect patterns, parallel rendering, memory management |
