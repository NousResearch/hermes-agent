# ☤ ASCII Video

Renders any content as colored ASCII character video. Audio, video, images, text, or pure math in, MP4/GIF/PNG sequence out. Full RGB color per character cell, 1080p 24fps default. No GPU.

Built for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Usable in any coding agent.

## What this is

A skill that teaches an agent how to build single-file Python renderers for ASCII video from scratch. The agent gets the full pipeline: grid system, font rasterization, effect library, shader chain, audio analysis, parallel encoding. It writes the renderer, runs it, gets video.

The output is actual video. Not terminal escape codes. Frames are computed as grids of colored characters, composited onto pixel canvases with pre-rasterized font bitmaps, post-processed through shaders, piped to ffmpeg.

## Modes

| Mode | Input | Output |
|------|-------|--------|
| Video-to-ASCII | A video file | ASCII recreation of the footage |
| Audio-reactive | An audio file | Visuals driven by frequency bands, beats, energy |
| Generative | Nothing | Procedural animation from math |
| Hybrid | Video + audio | ASCII video with audio-reactive overlays |
| Lyrics/text | Audio + timed text (SRT) | Karaoke-style text with effects |
| TTS narration | Text quotes + API key | Narrated video with typewriter text and generated speech |

## Pipeline

Every mode follows the same 6-stage path:

```
INPUT --> ANALYZE --> SCENE_FN --> TONEMAP --> SHADE --> ENCODE
```

1. **Input** loads source material (or nothing for generative).
2. **Analyze** extracts per-frame features. Audio gets 6-band FFT, RMS, spectral centroid, flatness, flux, beat detection with exponential decay. Video gets luminance, edges, motion.
3. **Scene function** returns a pixel canvas directly. Composes multiple character grids at different densities, value/hue fields, pixel blend modes. This is where the visuals happen.
4. **Tonemap** does adaptive percentile-based brightness normalization with per-scene gamma. ASCII on black is inherently dark. Linear multipliers don't work. This does.
5. **Shade** runs a `ShaderChain` (38 composable shaders) plus a `FeedbackBuffer` for temporal recursion with spatial transforms.
6. **Encode** pipes raw RGB frames to ffmpeg for H.264 encoding. Segments concatenated, audio muxed.

## Grid system

Characters render on fixed-size grids. Layer multiple densities for depth.

| Size | Font | Grid at 1080p | Use |
|------|------|---------------|-----|
| xs | 8px | 400x108 | Ultra-dense data fields |
| sm | 10px | 320x83 | Rain, starfields |
| md | 16px | 192x56 | Default balanced |
| lg | 20px | 160x45 | Readable text |
| xl | 24px | 137x37 | Large titles |
| xxl | 40px | 80x22 | Giant minimal |

Rendering the same scene on `sm` and `lg` then screen-blending them creates natural texture interference. Fine detail shows through gaps in coarse characters. Most scenes use two or three grids.

## Character palettes (20+)

Each sorted dark-to-bright, each a different visual texture. Validated against the font at init so broken glyphs get dropped silently.

| Family | Examples | Feel |
|--------|----------|------|
| Density ramps | ` .:-=+#@█` | Classic ASCII art gradient |
| Block elements | ` ░▒▓█▄▀▐▌` | Chunky, digital |
| Braille | ` ⠁⠂⠃...⠿` | Fine-grained pointillism |
| Dots | ` ⋅∘∙●◉◎` | Smooth, organic |
| Stars | ` ·✧✦✩✨★✶` | Sparkle, celestial |
| Half-fills | ` ◔◑◕◐◒◓◖◗◙` | Directional fill progression |
| Crosshatch | ` ▣▤▥▦▧▨▩` | Hatched density ramp |
| Math | ` ·∘∙•°±×÷≈≠≡∞∫∑Ω` | Scientific, abstract |
| Box drawing | ` ─│┌┐└┘├┤┬┴┼` | Structural, circuit-like |
| Katakana | ` ·ｦｧｨｩｪｫｬｭ...` | Matrix rain |
| Greek | ` αβγδεζηθ...ω` | Classical, academic |
| Runes | ` ᚠᚢᚦᚱᚷᛁᛇᛒᛖᛚᛞᛟ` | Mystical, ancient |
| Alchemical | ` ☉☽♀♂♃♄♅♆♇` | Esoteric |
| Arrows | ` ←↑→↓↔↕↖↗↘↙` | Directional, kinetic |
| Music | ` ♪♫♬♩♭♮♯○●` | Musical |
| Project-specific | ` .·~=≈∞⚡☿✦★⊕◊◆▲▼●■` | Themed per project |

Custom palettes are built per project to match the content.

## Color strategies

| Strategy | How it maps hue | Good for |
|----------|----------------|----------|
| Angle-mapped | Position angle from center | Rainbow radial effects |
| Distance-mapped | Distance from center | Depth, tunnels |
| Frequency-mapped | Audio spectral centroid | Timbral shifting |
| Value-mapped | Brightness level | Heat maps, fire |
| Time-cycled | Slow rotation over time | Ambient, chill |
| Source-sampled | Original video pixel colors | Video-to-ASCII |
| Palette-indexed | Discrete lookup table | Retro, flat graphic |
| Temperature | Warm-to-cool blend | Emotional tone |
| Complementary | Hue + opposite | Bold, dramatic |
| Triadic | Three equidistant hues | Psychedelic, vibrant |
| Analogous | Neighboring hues | Harmonious, subtle |
| Monochrome | Fixed hue, vary S/V | Noir, focused |

Plus 10 discrete RGB palettes (neon, pastel, cyberpunk, vaporwave, earth, ice, blood, forest, mono-green, mono-amber).

## Effects

### Backgrounds

| Effect | Description | Parameters |
|--------|-------------|------------|
| Sine field | Layered sinusoidal interference | freq, speed, octave count |
| Smooth noise | Multi-octave Perlin approximation | octaves, scale |
| Cellular | Voronoi-like moving cells | n_centers, speed |
| Noise/static | Random per-cell flicker | density |
| Video source | Downsampled video frame | brightness |

### Primary effects

| Effect | Description |
|--------|-------------|
| Concentric rings | Bass-driven pulsing rings with wobble |
| Radial rays | Spoke pattern, beat-triggered |
| Spiral arms | Logarithmic spiral, configurable arm count/tightness |
| Tunnel | Infinite depth perspective |
| Vortex | Twisting radial distortion |
| Frequency waves | Per-band sine waves at different heights |
| Interference | Overlapping sine waves creating moire |
| Aurora | Horizontal flowing bands |
| Ripple | Point-source concentric waves |
| Fire columns | Rising flames with heat-color gradient |
| Spectrum bars | Mirrored frequency visualizer |
| Waveform | Oscilloscope-style trace |

### Particle systems

| Type | Behavior | Character sets |
|------|----------|---------------|
| Explosion | Beat-triggered radial burst | `*+#@⚡✦★█▓` |
| Sparks | Short-lived bright dots | `·•●★✶*+` |
| Embers | Rising from bottom with drift | `·•●★` |
| Snow | Falling with wind sway | `❄❅❆·•*○` |
| Rain | Fast vertical streaks | `│┃║/\` |
| Bubbles | Rising, expanding | `○◎◉●∘∙°` |
| Data | Falling hex/binary | `01{}[]<>/\` |
| Runes | Mystical floating symbols | `ᚠᚢᚦᚱᚷᛁ✦★` |
| Orbit | Circular/elliptical paths | `·•●` |
| Gravity well | Attracted to point sources | configurable |
| Dissolve | Spread across screen, fade | configurable |
| Starfield | 3D projected, approaching | configurable |

## Shader pipeline

38 composable shaders, applied to the pixel canvas after character rendering. Configurable per section.

| Category | Shaders |
|----------|---------|
| Geometry | CRT barrel, pixelate, wave distort, displacement map, kaleidoscope, mirror (h/v/quad/diag) |
| Channel | Chromatic aberration (beat-reactive), channel shift, channel swap, RGB split radial |
| Color | Invert, posterize, threshold, solarize, hue rotate, saturation, color grade, color wobble, color ramp |
| Glow/Blur | Bloom, edge glow, soft focus, radial blur |
| Noise | Film grain (beat-reactive), static noise |
| Lines/Patterns | Scanlines, halftone |
| Tone | Vignette, contrast, gamma, levels, brightness |
| Glitch/Data | Glitch bands (beat-reactive), block glitch, pixel sort, data bend |

12 color tint presets: warm, cool, matrix green, amber, sepia, neon pink, ice, blood, forest, void, sunset, neutral.

7 mood presets for common shader combos:

| Mood | Shaders |
|------|---------|
| Retro terminal | CRT + scanlines + grain + amber/green tint |
| Clean modern | Light bloom + subtle vignette |
| Glitch art | Heavy chromatic + glitch bands + color wobble |
| Cinematic | Bloom + vignette + grain + color grade |
| Dreamy | Heavy bloom + soft focus + color wobble |
| Harsh/industrial | High contrast + grain + scanlines, no bloom |
| Psychedelic | Color wobble + chromatic + kaleidoscope mirror |

## Blend modes and composition

20 pixel blend modes for layering canvases: normal, add, subtract, multiply, screen, overlay, softlight, hardlight, difference, exclusion, colordodge, colorburn, linearlight, vividlight, pin_light, hard_mix, lighten, darken, grain_extract, grain_merge.

Mirror modes: horizontal, vertical, quad, diagonal, kaleidoscope (6-fold radial). Beat-triggered.

Transitions: crossfade, directional wipe, radial wipe, dissolve, glitch cut.

## Hardware adaptation

Auto-detects CPU count, RAM, platform, ffmpeg. Adapts worker count, resolution, FPS.

| Profile | Resolution | FPS | When |
|---------|-----------|-----|------|
| `draft` | 960x540 | 12 | Check timing/layout |
| `preview` | 1280x720 | 15 | Review effects |
| `production` | 1920x1080 | 24 | Final output |
| `max` | 3840x2160 | 30 | Ultra-high |
| `auto` | Detected | 24 | Adapts to hardware + duration |

`auto` estimates render time and downgrades if it would take over an hour. Low-memory systems drop to 720p automatically.

### Render times (1080p 24fps, ~180ms/frame/worker)

| Duration | 4 workers | 8 workers | 16 workers |
|----------|-----------|-----------|------------|
| 30s | ~3 min | ~2 min | ~1 min |
| 2 min | ~13 min | ~7 min | ~4 min |
| 5 min | ~33 min | ~17 min | ~9 min |
| 10 min | ~65 min | ~33 min | ~17 min |

720p roughly halves these. 4K roughly quadruples them.

## Known pitfalls

**Brightness.** ASCII characters are small bright dots on black. Most frame pixels are background. Linear `* N` multipliers clip highlights and wash out. Use `tonemap()` with per-scene gamma instead. Default gamma 0.75, solarize scenes 0.55, posterize 0.50.

**Render bottleneck.** The per-cell Python loop compositing font bitmaps runs at ~100-150ms/frame. Unavoidable without Cython/C. Everything else must be vectorized numpy. Python for-loops over rows/cols in effect functions will tank performance.

**ffmpeg deadlock.** Never `stderr=subprocess.PIPE` on long-running encodes. Buffer fills at ~64KB, process hangs. Redirect stderr to a file.

**Font cell height.** Pillow's `textbbox()` returns wrong height on macOS. Use `font.getmetrics()` for `ascent + descent`.

**Font compatibility.** Not all Unicode renders in all fonts. Palettes validated at init, blank glyphs silently removed.

## Requirements

◆ Python 3.10+
◆ NumPy, Pillow, SciPy (audio modes)
◆ ffmpeg on PATH
◆ A monospace font (Menlo, Courier, Monaco, auto-detected)
◆ Optional: OpenCV, ElevenLabs API key (TTS mode)

## File structure

```
├── SKILL.md                 # Modes, workflow, creative direction
├── README.md                # This file
└── references/
    ├── architecture.md      # Grid system, fonts, palettes, color, _render_vf()
    ├── effects.md           # Value fields, hue fields, backgrounds, particles
    ├── shaders.md           # 38 shaders, ShaderChain, tint presets, transitions
    ├── composition.md       # Blend modes, multi-grid, tonemap, FeedbackBuffer
    ├── scenes.md            # Scene protocol, SCENES table, render_clip(), examples
    ├── design-patterns.md   # Layer hierarchy, directional arcs, scene concepts
    ├── inputs.md            # Audio analysis, video sampling, text, TTS
    ├── optimization.md      # Hardware detection, vectorized patterns, parallelism
    └── troubleshooting.md   # Broadcasting traps, blend pitfalls, diagnostics
```

## Projects built with this

✦ 85-second highlight reel. 15 scenes (14×5s + 15s crescendo finale), randomized order, directional parameter arcs, layer hierarchy composition. Showcases the full effect vocabulary: fBM, voronoi fragmentation, reaction-diffusion, cellular automata, dual counter-rotating spirals, wave collision, domain warping, tunnel descent, kaleidoscope symmetry, boid flocking, fire simulation, glitch corruption, and a 7-layer crescendo buildup.

✦ Audio-reactive music visualizer. 3.5 min, 8 sections with distinct effects, beat-triggered particles and glitch, cycling palettes.

✦ TTS narrated testimonial video. 23 quotes, per-quote ElevenLabs voices, background music at 15% wide stereo, per-clip re-rendering for iterative editing.
