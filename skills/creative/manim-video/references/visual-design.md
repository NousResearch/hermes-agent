# Visual Design Principles

## 12 Core Principles

1. **Geometry Before Algebra** — Show the shape first, the equation second.
2. **Opacity Layering** — PRIMARY=1.0, CONTEXT=0.4, GRID=0.15. Direct attention through brightness.
3. **One New Idea Per Scene** — Each scene introduces exactly one concept.
4. **Spatial Consistency** — Same concept occupies the same screen region throughout.
5. **Color = Meaning** — Assign colors to concepts, not mobjects. If velocity is blue, it stays blue.
6. **Progressive Disclosure** — Show simplest version first, add complexity incrementally.
7. **Transform, Don't Replace** — Use Transform/ReplacementTransform to show connections.
8. **Breathing Room** — `self.wait(1.5)` minimum after showing something new.
9. **Visual Weight Balance** — Don't cluster everything on one side.
10. **Consistent Motion Vocabulary** — Pick a small set of animation types and reuse them.
11. **Dark Background, Light Content** — #1C1C1C to #2D2B55 backgrounds maximize contrast.
12. **Intentional Empty Space** — Leave at least 15% of the frame empty.

## Layout Templates

### FULL_CENTER
One main element centered, title above, note below.
Best for: single equations, single diagrams, title cards.

### LEFT_RIGHT
Two elements side by side at x=-3.5 and x=3.5.
Best for: equation + visual, before/after, comparison.

### TOP_BOTTOM
Main element at y=1.5, supporting content at y=-1.5.
Best for: concept + examples, theorem + cases.

### GRID
Multiple elements via `arrange_in_grid()`.
Best for: comparison matrices, multi-step processes.

### PROGRESSIVE
Elements appear one at a time, arranged DOWN with aligned_edge=LEFT.
Best for: algorithms, proofs, step-by-step processes.

### ANNOTATED_DIAGRAM
Central diagram with floating labels connected by arrows.
Best for: architecture diagrams, annotated figures.

## Color Palettes

### Classic 3B1B
```python
BG="#1C1C1C"; PRIMARY=BLUE; SECONDARY=GREEN; ACCENT=YELLOW; HIGHLIGHT=RED
```

### Warm Academic
```python
BG="#2D2B55"; PRIMARY="#FF6B6B"; SECONDARY="#FFD93D"; ACCENT="#6BCB77"
```

### Neon Tech
```python
BG="#0A0A0A"; PRIMARY="#00F5FF"; SECONDARY="#FF00FF"; ACCENT="#39FF14"
```

## Font Selection

Manim's default `Text()` uses the system's default sans-serif font, which often renders with poor kerning. Always specify a font explicitly.

### Recommended Fonts

| Use case | Font | Fallback |
|----------|------|----------|
| Body text, titles | `"Inter"`, `"SF Pro Display"` | `"Helvetica Neue"`, `"Arial"` |
| Code, terminal | `"JetBrains Mono"`, `"SF Mono"` | `"Menlo"`, `"Courier New"` |
| Math labels | Use `MathTex` (renders via LaTeX, not system fonts) | — |

```python
# Clean body text
title = Text("Gradient Descent", font_size=48, font="Inter", weight=BOLD)

# Monospaced code
code_label = Text("loss.backward()", font_size=24, font="JetBrains Mono")

# Math — always use MathTex, not Text
equation = MathTex(r"\nabla L = \frac{\partial L}{\partial w}")
```

### Font Availability

Not all fonts are installed on all systems. Manim falls back silently to a default if the font is missing. Use widely available fonts:
- **macOS**: SF Pro Display, SF Mono, Menlo, Helvetica Neue
- **Linux**: DejaVu Sans, Liberation Sans, Ubuntu, Noto Sans
- **Cross-platform**: Inter (install via Google Fonts), JetBrains Mono (install from jetbrains.com)

For maximum portability, use `"Helvetica Neue"` (body) and `"Menlo"` (code) — both available on macOS and have Linux equivalents.

### Fine-Grained Text Control

`Text()` does not support `letter_spacing` or kerning parameters. For fine control, use `MarkupText` with Pango attributes:

```python
# Letter spacing (Pango units: 1/1024 of a point)
MarkupText('<span letter_spacing="6000">HERMES</span>', font_size=18, font="Menlo")

# Bold specific words
MarkupText('This is <b>important</b>', font_size=24)

# Color specific words
MarkupText('Red <span foreground="#FF6B6B">warning</span>', font_size=24)
```

### Text Rendering Quality

Manim's text rendering quality depends heavily on output resolution. At `-ql` (480p), text kerning looks noticeably poor. Always preview text-heavy scenes at `-qm` (720p) or higher. See `references/rendering.md` for quality preset guidance.

## Visual Hierarchy Checklist

For every frame:
1. What is the ONE thing to look at? (brightest/largest)
2. What is context? (dimmed to 0.3-0.4)
3. What is structural? (dimmed to 0.15)
4. Enough empty space? (>15%)
5. All text readable at phone size?
