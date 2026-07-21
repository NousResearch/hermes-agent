# No-LaTeX Workaround for Manim

## Why This Exists

Manim's `MathTex` and `Tex` classes, as well as `Axes(include_numbers=True)`, require a working LaTeX distribution (`texlive`/`mactex`/`basictex`). On macOS these can be multiple GB and take 30+ minutes to install. This reference documents how to produce clean Manim videos without LaTeX.

## What You Can Use Without LaTeX

| Feature | Status | Notes |
|---------|--------|-------|
| `Text(...)` | ✅ Works | Any installed font, Unicode support |
| `MarkupText(...)` | ✅ Works | PangoMarkup for inline styling, `<span>` tags |
| `Paragraph(...)` | ✅ Works | Multi-line text blocks |
| `DecimalNumber(...)` | ✅ Works | Numeric displays |
| `Integer(...)` | ✅ Works | Integer displays |
| `Dot(...)` | ✅ Works | Points/circles |
| `Line(...)` | ✅ Works | With `set_stroke(opacity=N)` |
| `Rectangle(...)` | ✅ Works | All polygram shapes |
| `Circle(...)` | ✅ Works | All arc shapes |
| `Polygon(...)` | ✅ Works | |
| `Arc(...)` | ✅ Works | |
| `Arrow(...)` | ✅ Works | |
| `Brace(...)` | ✅ Works | |
| `VGroup(...)` | ✅ Works | Only VMobjects |
| `Group(...)` | ✅ Works | Any Mobject |
| `Axes(...)` | ⚠️ Partial | Set `include_numbers=False`, add manual labels |
| `NumberPlane(...)` | ✅ Works | Background grids |
| `ParametricFunction(...)` | ✅ Works | For custom curves |
| `MathTex(...)` | ❌ Requires LaTeX | No workaround for proper math typesetting |
| `Tex(...)` | ❌ Requires LaTeX | No workaround |
| `Matrix(...)` | ❌ Requires LaTeX | No workaround |
| `Axes(include_numbers=True)` | ❌ Requires LaTeX | Set to False, label manually |

## Pattern: Manual Axis Labels

```python
from manim import *

BG = "#2D2B55"
DIM = "#888888"
MONO = "Menlo"

class ManualAxesScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        axes = Axes(
            x_range=[0, 100, 10],
            y_range=[0, 100, 10],
            x_length=8,
            y_length=4,
            axis_config={"color": DIM, "include_numbers": False}
        )
        axes.move_to(ORIGIN)

        # Manual x-axis labels
        ax_labels = VGroup()
        for x_val in [0, 20, 40, 60, 80, 100]:
            pt = axes.coords_to_point(x_val, 0)
            label = Text(str(x_val), font_size=16, color=DIM, font=MONO)
            label.next_to([pt[0], axes.get_bottom()[1] - 0.2, 0], DOWN, buff=0)
            ax_labels.add(label)

        self.add(axes, ax_labels)
        self.wait()
```

## Pattern: Labeled Bar Chart

```python
# Sample data
tilts = [0, 10, 20, 30, 31, 40, 50, 60, 90]
production = [82, 90, 96, 99.5, 100, 99, 95, 88, 58]

# Create axes without LaTeX
axes = Axes(
    x_range=[-5, 95, 10],
    y_range=[50, 105, 10],
    x_length=10, y_length=4,
    axis_config={"color": DIM, "include_numbers": False}
)
axes.move_to([0, 0.5, 0])

# Manual labels for axes
for x_val in tilts:
    pt = axes.coords_to_point(x_val, 50)
    label = Text(str(x_val), font_size=14, color=DIM, font=MONO)
    label.move_to([pt[0], 0.5 - 0.3, 0])
    self.add(label)

for y_val in [50, 60, 70, 80, 90, 100]:
    pt = axes.coords_to_point(0, y_val)
    label = Text(str(y_val), font_size=14, color=DIM, font=MONO)
    label.move_to([pt[0] - 0.4, pt[1], 0])
    self.add(label)

# Bars (animated creation)
bar_width = 0.6
x_start = -4.5
for i, (tilt, prod) in enumerate(zip(tilts, production)):
    x_pos = x_start + i * 1.1
    bar_height = (prod - 50) / 55 * 4
    bar = Rectangle(
        width=bar_width, height=max(bar_height, 0.01),
        color="#6BCB77" if tilt == 31 else "#58C4DD",
        fill_color="#6BCB77" if tilt == 31 else "#58C4DD",
        fill_opacity=0.7 if tilt == 31 else 0.4
    )
    bar.move_to([x_pos, 0.5, 0], aligned_edge=DOWN)
    self.play(FadeIn(bar, scale=0.5), run_time=0.2)
```

## Unicode Math Characters Reference

These render correctly in `Text()` without LaTeX:

| Symbol | Unicode | Name |
|--------|---------|------|
| α | U+03B1 | Greek alpha |
| β | U+03B2 | Greek beta |
| θ | U+03B8 | Greek theta |
| θ | U+03B8 | Greek theta (variant) |
| π | U+03C0 | Greek pi |
| ∑ | U+2211 | Summation |
| ∏ | U+220F | Product |
| ∝ | U+221D | Proportional to |
| ∞ | U+221E | Infinity |
| ≈ | U+2248 | Approximately |
| ≠ | U+2260 | Not equal |
| ≡ | U+2261 | Identical |
| ≤ | U+2264 | Less or equal |
| ≥ | U+2265 | Greater or equal |
| × | U+00D7 | Multiplication |
| ÷ | U+00F7 | Division |
| ± | U+00B1 | Plus-minus |
| → | U+2192 | Right arrow |
| ← | U+2190 | Left arrow |
| ↑ | U+2191 | Up arrow |
| ↓ | U+2193 | Down arrow |
| ↔ | U+2194 | Left-right arrow |
| ⇒ | U+21D2 | Right double arrow |
| √ | U+221A | Square root |
| ° | U+00B0 | Degree |
| ′ | U+2032 | Prime (minute) |
| ″ | U+2033 | Double prime (second) |

## Checking if LaTeX is Available

```python
import subprocess, shutil
has_latex = shutil.which("latex") is not None
# or
try:
    subprocess.run(["latex", "--version"], capture_output=True, check=True)
    has_latex = True
except:
    has_latex = False
```

If LaTeX is not available, decide your rendering strategy:
- **Simple annotations/labels**: Use `Text()` with Unicode math (recommended)
- **Complex formulas**: Install `basictex` (~150MB) or use a system with LaTeX pre-installed
- **Bar charts/data**: Use manual axis labels as shown above
