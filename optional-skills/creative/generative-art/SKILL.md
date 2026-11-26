---
name: generative-art
description: "Generative art and math animation: p5.js (interactive sketches, shaders, 3D) and Manim CE (3Blue1Brown-style math videos)."
version: 1.0.0
metadata:
  hermes:
    tags: [creative-coding, generative-art, p5js, manim, canvas, interactive, visualization, webgl, shaders, animation, math-videos, educational]
    related_skills: [ascii-video, excalidraw]
---

# Generative Art & Animation

## When to Use

Use this cluster when users request: generative art, creative coding, interactive visualizations, browser-based visual art, p5.js sketches, mathematical animations, 3Blue1Brown-style explainer videos, algorithm visualizations, or any programmatic animation with geometric or visual content.

| Tool | Best for | Output | Reference |
|------|----------|--------|-----------|
| **p5.js** | Interactive browser art, generative visuals, shaders, data viz, 3D scenes | HTML, PNG, GIF, MP4, SVG | `references/p5js.md` |
| **Manim CE** | Math explanations, algorithm walkthroughs, equation derivations, educational cinema | MP4 | `references/manim-video.md` |

## Quick Comparison

| Dimension | p5.js | Manim CE |
|-----------|-------|----------|
| **Domain** | Creative/generative art, data viz | Educational/math explanation |
| **Output** | Interactive canvas, images, video | Video (static renders) |
| **Stack** | Single HTML, no build step | Python script, LaTeX, ffmpeg |
| **Interactivity** | Mouse, keyboard, mic, touch | Non-interactive (playback) |
| **3D** | WebGL mode with GLSL | 3DScene with camera animation |
| **Audio** | p5.sound (FFT, mic) | External TTS muxing |
| **Export formats** | PNG, GIF, MP4, SVG, WebM | MP4, GIF, PNG stills |
| **Learning curve** | JS + creative concepts | Python + Manim API |

## Choosing the Right Tool

- **p5.js** when: the output is a live canvas, interactive experience, still image, or social-media loop. When the viewer *explores* rather than *watches*. When shaders, particle systems, or data viz are involved.
- **Manim CE** when: the output is a planned video explanation. When equations, derivations, or algorithm steps are the subject. When the narrative is scripted and linear. When it needs narration or music.

If unsure, ask: "Is this a *viewer experience* (p5.js) or a *prepared video* (Manim)?"

## Creative Standard

Both tools produce visual media. The shared standard: **first-render excellence is non-negotiable.** The output must be visually striking, conceptually clear, and aesthetically cohesive on first pass.

| Criterion | p5.js standard | Manim standard |
|-----------|----------------|----------------|
| **Frame** | Every frame rewards viewing — textured, layered, intentional | Every frame teaches — geometry before algebra |
| **Color** | Custom palette per project, never defaults | Shared palette across all scenes |
| **Motion** | Varied speeds, cohesive motion vocabulary | `self.wait()` after every reveal; pace for absorption |
| **Composition** | Intentional layout, never flat backgrounds | Opacity layering (1.0 / 0.4 / 0.15) |
| **Typography** | Custom font loading, `textToPoints()` | Monospace fonts, `font_size >= 18` |

---

## p5.js

See `references/p5js.md` for full reference.

p5.js produces interactive browser-based generative art, data visualizations, shaders, and 3D scenes as self-contained HTML files.

**Key capabilities:** particle systems, flow fields, noise fields, domain warping, GLSL shaders (WebGL mode), pixel manipulation, kinetic typography, audio-reactive visuals, SVG export, headless high-res rendering.

**Stack:** p5.js 1.11.3 (CDN) + optional p5.sound, p5.js-svg, CCapture.js, Puppeteer.

**Modes:** generative art, data visualization, interactive experience, animation/motion graphics, 3D scene, image processing, audio-reactive.

**Export:** PNG (`saveCanvas`), GIF (`saveGif`), frame sequence + ffmpeg for MP4, SVG, headless Puppeteer for batch.

**Reference:** `references/p5js.md`

---

## Manim Community Edition

See `references/manim-video.md` for full reference.

Manim CE produces 3Blue1Brown-style animated explanations, math proofs, and algorithm visualizations as MP4 videos.

**Key capabilities:** animated equations (`MathTex`), geometric constructions, scene orchestration (`FadeIn`, `Write`, `ReplacementTransform`), 3D surfaces, axes and plotting, custom `Mobjects`.

**Stack:** Python 3.10+, Manim CE v0.20+, LaTeX, ffmpeg.

**Modes:** concept explainer, equation derivation, algorithm visualization, data story, architecture diagram, paper explainer, 3D visualization.

**Export:** `-ql` draft (480p15), `-qm` medium (720p30), `-qh` production (1080p60). Stitch scenes with ffmpeg.

**Reference:** `references/manim-video.md`

---

## Divergent Creative Strategies

Use these when the user asks for experimental, unconventional, or unique output:

### For p5.js (generative art)
- **Conceptual Blending** — combine two visual systems (e.g., particle physics + handwriting)
- **SCAMPER** — transform a known pattern: substitute shapes, combine patterns, adapt to 3D, exaggerate scale, eliminate symmetry, reverse the simulation
- **Distance Association** — anchor on the user's concept, explore close/medium/far associations

### For Manim (math animation)
- **SCAMPER** — transform a standard explanation: substitute the visual metaphor, combine algebraic + geometric simultaneously, derive backward from result
- **Assumption Reversal** — reverse the most fundamental assumption of a standard visualization

## References

### p5.js References
| File | Contents |
|------|----------|
| `references/core-api.md` | Canvas setup, coordinate system, draw loop, `push()`/`pop()`, offscreen buffers, composition, `pixelDensity()`, responsive design |
| `references/shapes-and-geometry.md` | 2D primitives, `beginShape()`/`endShape()`, Bezier curves, `p5.Vector`, SDFs, SVG path conversion |
| `references/visual-effects.md` | Noise (Perlin, fractal, domain warp, curl), flow fields, particle systems, pixel manipulation, texture generation, feedback loops, reaction-diffusion |
| `references/animation.md` | Frame-based animation, easing, `lerp()`/`map()`, spring physics, state machines, timeline sequencing |
| `references/typography.md` | `text()`, `loadFont()`, `textToPoints()`, kinetic typography, text masks |
| `references/color-systems.md` | `colorMode()`, HSB/HSL/RGB, `lerpColor()`, palettes, `blendMode()`, gradients |
| `references/webgl-and-3d.md` | WEBGL renderer, 3D primitives, camera, lighting, GLSL shaders, framebuffers, post-processing |
| `references/interaction.md` | Mouse, keyboard, touch, DOM elements, sliders, audio input (FFT/amplitude), scroll-driven |
| `references/export-pipeline.md` | `saveCanvas()`, `saveGif()`, headless capture, ffmpeg, CCapture.js, SVG, per-clip architecture, platform export (fxhash) |
| `references/troubleshooting.md` | Performance profiling, common mistakes, WebGL debugging, memory leaks, CORS |
| `templates/viewer.html` | Interactive viewer: seed navigation, parameter sliders, PNG download |

### Manim CE References
| File | Contents |
|------|----------|
| `references/animations.md` | Core animations, rate functions, `.animate` syntax, timing |
| `references/mobjects.md` | Text, shapes, VGroup/Group, positioning, styling, custom mobjects |
| `references/visual-design.md` | 12 design principles, opacity layering, color palettes |
| `references/equations.md` | LaTeX in Manim, `TransformMatchingTex`, derivation patterns |
| `references/graphs-and-data.md` | Axes, plotting, BarChart, algorithm visualization |
| `references/camera-and-3d.md` | MovingCameraScene, ThreeDScene, 3D surfaces, camera control |
| `references/scene-planning.md` | Narrative arcs, layout templates, scene transitions, planning template |
| `references/rendering.md` | CLI reference, quality presets, ffmpeg, voiceover workflow |
| `references/troubleshooting.md` | LaTeX errors, animation errors, common mistakes |
| `references/animation-design-thinking.md` | When to animate vs show static, decomposition, pacing |
| `references/updaters-and-trackers.md` | ValueTracker, `add_updater`, `always_redraw`, time-based updaters |
| `references/paper-explainer.md` | Turning research papers into animations |
| `references/decorations.md` | `SurroundingRectangle`, Brace, arrows, `DashedLine`, Angle |
| `references/production-quality.md` | Pre-code, pre-render, post-render checklists, spatial layout |
