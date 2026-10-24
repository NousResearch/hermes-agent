# Building Logos in SVG

How to turn a concept into clean, production-grade SVG code by hand (as an AI writing markup) — geometric,
minimal, and easy for printers, developers and vector editors to use.

## Contents
1. File conventions
2. Construction strategy: think in primitives
3. Paths: writing clean geometry
4. Negative space and compound shapes
5. Strokes vs fills
6. Type in SVG
7. Colour, gradients, variants
8. Common construction recipes
9. What to avoid in a master file
10. Outlines and boolean unions
11. Validation

---

## 1. File conventions

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"
     role="img" aria-labelledby="title">
  <title id="title">Harbor logo</title>
  <path fill="#0F7C80" d="…"/>
</svg>
```
- **viewBox first**: work on a clean integer canvas. 256 × 256 (or 512 × 512) for symbols; for lockups keep the
  height 256 and let the width follow (e.g. `0 0 960 256`). The bundled library uses the same convention (width
  256 or 512 in ~99 % of files).
- **Padding**: leave consistent padding inside the viewBox (≈ 4–8 % of the size for symbols) or crop tight and let
  the clear-space rule handle spacing — be consistent across the set.
- **Integer or 1–2 decimal coordinates**. Excess precision (`127.99999`) bloats files and hides misalignment.
- **Include `<title>`** for accessibility; keep IDs meaningful (`symbol`, `wordmark`).
- **Group logically**: `<g id="symbol">`, `<g id="wordmark">` so lockups can be recomposed.
- One file per variant (see §7), named predictably: `brand-logo-horizontal-color.svg`, `brand-symbol-black.svg`.

## 2. Construction strategy: think in primitives

Before writing path data, describe the construction in words:
- *"Circle r=96 centred at (128,128); remove a 45° wedge from the upper right; a smaller circle r=28 sits in the
  gap as a 'dot'."*
- *"Letter M from three 40-unit-wide bars on a 256 grid; the middle vertex rises to y=96; outer stems are extended
  4 units below the baseline for overshoot of the pointed vertex."*

Then choose the simplest element that expresses each part:
- `<circle>`, `<ellipse>`, `<rect rx>` for primitives (easy to read and adjust).
- `<path>` for everything else, and for the final merged silhouette.
- `<polygon>` for straight-edged shapes.

Pick a **unit** (e.g. 8 or 16 on a 256 canvas) and snap key dimensions to multiples of it; the mark will feel
systematic and gridding later is trivial. Keep a list of the radii and angles you use — reuse them.

## 3. Paths: writing clean geometry

- **Commands**: `M` move, `L`/`H`/`V` lines, `A` elliptical arc (perfect for circle segments), `C` cubic Bézier,
  `Q` quadratic, `Z` close. Uppercase = absolute (prefer while designing), lowercase = relative.
- **Arcs for circular geometry**: `A r r 0 largeArc sweep x y`. Circle-based curves keep radii consistent and
  are easy to verify. Example — a semicircle cap: `M 64 128 A 64 64 0 0 1 192 128`.
- **Béziers for organic curves**: place anchors at extrema (top, bottom, left, right of a curve) with handles
  horizontal or vertical there; this gives smooth, predictable curves with few points. A quarter circle as a cubic
  uses handle length ≈ 0.5523 × radius.
- **Few anchors**. Every extra point is a chance for a wobble. If a curve needs many points, split it into arcs.
- **Angles**: compute diagonal endpoints from exact angles. For 30°/60°: offsets use sin/cos (0.5, 0.866);
  for 45°: equal x and y offsets. Avoid "almost" angles like 44° or 2° off vertical (the audit script flags them).
- **Smooth corners (anti bone effect)**: instead of a plain arc joining a straight side, extend the curve's handles
  so curvature ramps gradually. A good squircle corner of size `s` starts ~1.3 × s before the corner, with handles
  ~0.6 × s.
- **Closed shapes**: always end filled paths with `Z`.

## 4. Negative space and compound shapes

- Put outer contour and holes in **one path** and use `fill-rule="evenodd"` — holes appear where subpaths overlap
  an odd number of times. Or wind holes in the opposite direction with the default `nonzero` rule.
  ```svg
  <path fill-rule="evenodd" d="M128 16 A112 112 0 1 1 127.9 16 Z  M128 80 A48 48 0 1 0 128.1 80 Z"/>
  ```
- For a hidden figure between two shapes, design the **negative shape first** (the arrow, the letter), then build the
  positive shapes around it; this guarantees the negative form is clean.
- Avoid `<mask>` and `<clipPath>` in the master unless essential — some tools and embroidery/cutting software handle
  them poorly. Bake the geometry into paths for final files. (In exploration, masks are fine for speed.)
- Where two filled shapes of the same colour touch, merge them into one path so there is no hairline seam when
  rendered or cut.

## 5. Strokes vs fills

- Explore with strokes (`stroke-width`, `stroke-linecap="round"`, `stroke-linejoin="round"`) — they're quick.
- In the master, **convert strokes to filled outlines** so the mark scales proportionally everywhere and survives
  tools that ignore stroke settings. As an AI without an outline tool, either construct the outlined geometry
  directly (offset curves by half the stroke width), or keep strokes but set `vector-effect` nowhere and verify that
  scaling the whole SVG scales stroke width proportionally (it does when the stroke is inside the scaled viewBox).
  Flag remaining strokes in the delivery notes so a vector editor can expand them.
- Monoline marks: pick stroke width relative to size (≥ 8 % of the mark's width if it must read at 24 px).

## 6. Type in SVG

- Final logos must not depend on installed fonts: `<text>` renders differently (or not at all) on other machines.
- Options, in order of preference:
  1. **Construct letterforms geometrically** as paths (ideal for short wordmarks, monograms, letterform symbols —
     and it forces ownable, custom letters).
  2. If the user has a font file and a vector tool, set the word, customise, and **convert to outlines**; paste the
     path data.
  3. For exploration/presentation only, use `<text>` with a clearly named font stack and a note that the wordmark
     must be outlined before delivery. Never ship `<text>` as the final master.
- When drawing letters: consistent stem width, overshoot on round letters, thinner horizontals, optical spacing
  (see `typography.md`).

## 7. Colour, gradients, variants

- Use exact HEX values from the palette; limit to the approved colours.
- Put colours directly on elements (`fill="#…"`) rather than in `<style>` blocks for maximum compatibility; optionally
  use `currentColor` for a UI-friendly one-colour variant.
- Gradients: define in `<defs>` with `gradientUnits="userSpaceOnUse"` and explicit coordinates so they render
  identically everywhere; always keep a flat-colour master alongside.
- **Variant set** to generate from the master (`scripts/export_variants.py` automates the colour swaps):
  - `*-color.svg` (full colour), `*-black.svg`, `*-white.svg` (reversed; consider slightly thinner geometry),
    `*-mono-<hex>.svg` (one brand colour).
  - Lockups: `horizontal`, `stacked`, `symbol`, `wordmark`.
  - App/favicon: symbol on a solid rounded-square or circle container, with the symbol optically sized
    (typically 60–70 % of the container).

## 8. Common construction recipes

**Perfect circle with a notch (open ring)**
```svg
<path fill="none" stroke="#111" stroke-width="32" stroke-linecap="round"
      d="M 201.5 54.5 A 104 104 0 1 0 232 128"/>   <!-- exploration; expand for master -->
```

**Rounded square container (app icon), 22 % radius**
```svg
<rect x="0" y="0" width="256" height="256" rx="56" fill="#0F7C80"/>
```

**Equilateral triangle (side 203.2, height 176), centred on its bounding box** — its visual mass (centroid)
sits low, so nudge it up a few units if it looks bottom-heavy inside a container.
```svg
<polygon points="128,40 229.6,216 26.4,216" fill="#111"/>
```

**Letter "A" as a letterform symbol** — flat apex, every edge on the same 1 : 2 slope, and a constant
48-unit stroke (legs, crossbar and apex all measure 48)
```svg
<path fill="#111" fill-rule="evenodd"
      d="M104 32 H152 L248 224 H200 L184 192 H72 L56 224 H8 Z  M96 144 H160 L128 80 Z"/>
```

**Speech bubble from a circle + tail (merged)**
```svg
<path fill="#111" d="M128 24 A104 104 0 1 1 61.6 208 L28 236 L37.9 180 A104 104 0 0 1 128 24 Z"/>
```

These are starting points; refine proportions and add the concept's twist.

## 9. What to avoid in a master file

| Avoid | Why | Instead |
|---|---|---|
| `<text>` | font dependence | outlined paths |
| `<image>` / embedded PNG | not vector, blurry | vector paths |
| `filter` (blur, shadow, glow) | inconsistent rendering, not printable | flat shapes; simulate shadow with a darker shape |
| many `<mask>`/`clipPath` | tool compatibility | bake into paths |
| transforms nested deeply | hard to edit, rounding errors | apply transforms to coordinates (one wrapper `translate/scale`, as `export_variants.py` writes, is fine) |
| 20+ colours / many gradients | poor reproduction, weak recall | ≤ 3 flat colours; stepped gradients |
| micro-details < 1/64 of size | vanish when small | merge or remove |
| off-by-a-degree angles | look accidental | exact angles |
| editor metadata, huge precision | bloat | clean markup |

## 10. Outlines and boolean unions (production)

Without a vector editor you can't boolean-unite shapes, but you can get close:
- Overlapping sub-paths inside **one** `<path>` with the default `nonzero` rule (all wound the same way) render as a
  seamless union on screen and in print — acceptable for web and most print masters.
- Cutters, vinyl plotters and embroidery software prefer truly merged outlines. If Inkscape is installed, it can
  expand strokes and union everything from the command line:
  ```bash
  inkscape logo.svg --actions="select-all:all;object-stroke-to-path;path-union;export-plain-svg;export-filename:logo-outlined.svg;export-do"
  ```
  Otherwise note in the handover that a designer should run *Outline Stroke* + *Unite* in a vector editor.

## 11. Validation

Run after every significant iteration:
```bash
python3 scripts/svg_audit.py path/to/logo.svg          # structure, colours, complexity, angles, text/raster checks
python3 scripts/preview_sheet.py path/to/logo.svg -o preview.html   # sizes, backgrounds, mono, blur, mirror, favicon
python3 scripts/render_png.py path/to/logo.svg --size 512 -o look.png # quick render to view
```
Open the preview in a browser and look at it (use the available browser/screenshot tool if you are an agent). The
audit compares the file's complexity with the reference library's distribution so you can see if a mark is unusually
complex for a logo.
