# Manim Community Edition Reference

> Full Manim CE production pipeline — see original at `~/.hermes/skills/.archive/creative/manim-video/`

## Quick Start

```python
from manim import *

BG = "#1C1C1C"
PRIMARY = "#58C4DD"
SECONDARY = "#83C167"
ACCENT = "#FFFF00"
MONO = "Menlo"

class Scene1_Introduction(Scene):
    def construct(self):
        self.camera.background_color = BG
        title = Text("Why Does This Work?", font_size=48, color=PRIMARY, weight=BOLD, font=MONO)
        self.add_subcaption("Why does this work?", duration=2)
        self.play(Write(title), run_time=1.5)
        self.wait(1.0)
        self.play(FadeOut(title), run_time=0.5)
```

## Key Patterns

- **Subtitles on every animation**: `self.add_subcaption("text", duration=N)` or `subcaption="text"` in `self.play()`
- **Shared color constants** at file top for cross-scene consistency
- **`self.camera.background_color`** set in every scene
- **Clean exits** — `FadeOut` all mobjects at scene end: `self.play(FadeOut(Group(*self.mobjects)))`
- **Monospace fonts only** — `font=MONO`; minimum `font_size=18`
- **Raw strings for LaTeX**: `MathTex(r"\frac{1}{2}")` — not `"\\frac{1}{2}"`
- **`buff >= 0.5`** for edge text
- **Never animate non-added mobjects**

## Modes

| Mode | Reference |
|------|-----------|
| Concept explainer | `references/scene-planning.md` |
| Equation derivation | `references/equations.md` |
| Algorithm visualization | `references/graphs-and-data.md` |
| Data story | `references/graphs-and-data.md` |
| Architecture diagram | `references/mobjects.md` |
| Paper explainer | `references/scene-planning.md` |
| 3D visualization | `references/camera-and-3d.md` |

## Render & Stitch

```bash
# Draft render
manim -ql script.py Scene1 Scene2

# Production render
manim -qh script.py Scene1 Scene2

# Stitch scenes
cat > concat.txt << 'EOF'
file 'media/videos/script/480p15/Scene1.mp4'
file 'media/videos/script/480p15/Scene2.mp4'
EOF
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy final.mp4
```

## Color Palettes

| Palette | Background | Primary | Secondary | Accent | Use case |
|---------|-----------|---------|-----------|--------|----------|
| **Classic 3B1B** | `#1C1C1C` | `#58C4DD` | `#83C167` | `#FFFF00` | General math/CS |
| **Warm academic** | `#2D2B55` | `#FF6B6B` | `#FFD93D` | `#6BCB77` | Approachable |
| **Neon tech** | `#0A0A0A` | `#00F5FF` | `#FF00FF` | `#39FF14` | Systems/architecture |
| **Monochrome** | `#1A1A2E` | `#EAEAEA` | `#888888` | `#FFFFFF` | Minimalist |

## Animation Timing

| Context | run_time | self.wait() after |
|---------|----------|-------------------|
| Title/intro appear | 1.5s | 1.0s |
| Key equation reveal | 2.0s | 2.0s |
| Transform/morph | 1.5s | 1.5s |
| Supporting label | 0.8s | 0.5s |
| "Aha moment" reveal | 2.5s | 3.0s |

## Reference Files

See `.archive/creative/manim-video/references/` for full content:
- `animations.md` — Core animations, rate functions, `.animate` syntax
- `mobjects.md` — Text, shapes, VGroup/Group, positioning, custom mobjects
- `visual-design.md` — 12 design principles, opacity layering, color palettes
- `equations.md` — LaTeX, `TransformMatchingTex`, derivation patterns
- `graphs-and-data.md` — Axes, plotting, BarChart, algorithm visualization
- `camera-and-3d.md` — MovingCameraScene, ThreeDScene, 3D surfaces
- `scene-planning.md` — Narrative arcs, layout templates, planning template
- `rendering.md` — CLI reference, quality presets, ffmpeg, voiceover
- `troubleshooting.md` — LaTeX errors, animation errors
- `animation-design-thinking.md` — When to animate vs show static
- `updaters-and-trackers.md` — ValueTracker, `add_updater`, `always_redraw`
- `paper-explainer.md` — Turning research papers into animations
- `decorations.md` — `SurroundingRectangle`, Brace, arrows, `DashedLine`
- `production-quality.md` — Pre-code, pre-render, post-render checklists

## Scripts

- `scripts/setup.sh` — Verify Python 3.10+, Manim CE v0.20+, LaTeX, ffmpeg