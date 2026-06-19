# Concept Diagram Archetypes

For educational and non-software visuals — physics, chemistry, math, biology,
physical objects, anatomy, floor plans, lifecycles, cross-sections, hub-spoke
systems. Flat, minimal, light/dark-aware. (Carried over from the former
`concept-diagrams` skill.)

Read `svg-diagrams.md` first for arrow markers, node groups, and coordinate
discipline. This file adds the educational color system and a library of archetypes
beyond the basic flowchart.

## Design philosophy

- **Flat**: no gradients, drop shadows, blur, glow, or neon.
- **Minimal**: show the essential. No decorative icons inside boxes.
- **Sentence case always.** Never Title Case, never ALL CAPS.
- **Two font sizes only**: `th` 14px/500 for titles, `ts` 12px/400 for subtitles &
  labels.
- **0.5px** stroke on node borders. `fill="none"` on every connector path.

## The 9-ramp educational color system

Color encodes **category/meaning**, never sequence. Use 2–3 ramps per diagram. Put
the class on a `<g>` or shape; the template CSS maps stops for light *and* dark mode
automatically (light: 50 fill / 600 stroke / 800 title; dark: 800 fill / 200 stroke /
100 title).

| Class | 50 | 200 | 400 | 600 | 800 |
|---|---|---|---|---|---|
| `c-purple` | #EEEDFE | #AFA9EC | #7F77DD | #534AB7 | #3C3489 |
| `c-teal`   | #E1F5EE | #5DCAA5 | #1D9E75 | #0F6E56 | #085041 |
| `c-coral`  | #FAECE7 | #F0997B | #D85A30 | #993C1D | #712B13 |
| `c-pink`   | #FBEAF0 | #ED93B1 | #D4537E | #993556 | #72243E |
| `c-gray`   | #F1EFE8 | #B4B2A9 | #888780 | #5F5E5A | #444441 |
| `c-blue`   | #E6F1FB | #85B7EB | #378ADD | #185FA5 | #0C447C |
| `c-green`  | #EAF3DE | #97C459 | #639922 | #3B6D11 | #27500A |
| `c-amber`  | #FAEEDA | #EF9F27 | #BA7517 | #854F0B | #633806 |
| `c-red`    | #FCEBEB | #F09595 | #E24B4A | #A32D2D | #791F1F |

Assignment rules: group nodes by category (same type → same color); `c-gray` for
neutral/structural (start, end, generic steps, users); reserve `c-blue`/`c-green`/
`c-amber`/`c-red` for semantic info/success/warning/error. The full template (with
the 7-stop ramps and the light/dark CSS) is `templates/diagram.html`.

## Layout constants

- ViewBox `0 0 680 H` (H = content + 40px buffer); safe area x 40→640.
- Single-line box 44px tall; two-line 56px; ≥60px gap between boxes.
- Inner padding 24px horizontal / 12px vertical. Container `rx` 16–20, node `rx` 8.
- Max nesting 2–3 levels (deeper is unreadable at 680px).

## Archetype library

Pick the shape that fits the subject. Each is hand-laid SVG using the ramps above.

**Flowchart / process** — `c-gray` start/end, one category color for steps,
`c-red` for error branches. Decision diamonds gate the flow (see `svg-diagrams.md`).

**Pipeline / data flow** (left→right) — `c-gray` sources, a category color for
processing stages, `c-teal` sinks. Straight horizontal edges on one row.

**Layered stack / exploded view** — vertical stack of full-width `<rect>`s, one ramp
stop darker per layer going down, labels to the side with leader lines. For "layers
of X" / "the N tiers of Y".

**Tree / hierarchy** — root at top center, children fanning down; edges are
`<line>`s or short Béziers. Same color per depth level.

**Quadrant / 2×2 matrix** — two crossing axis lines with arrowheads, four labeled
cells, axis labels in `ts`. For positioning / trade-off space.

**Before / after (comparison)** — two side-by-side panels sharing a column grid; use
`c-red`/`rust` accents on the "before" pain points and `c-green`/`olive` on the
"after" wins. A center divider or arrow shows the transition.

**Timeline / sequence** — a horizontal or vertical rail with dated/numbered nodes;
for UML-style sequence, vertical lifelines with horizontal message arrows labeled in
`ts`.

**Hub-spoke / system integration** — a central node with spokes to subsystems; use
distinct line styles per subsystem type (smart city, IoT, electricity grid).

**Cross-section / physical object / anatomy** — outline the object with `<path>`
(polygons, ellipses, Béziers for curves), fill regions with category colors, label
parts with `ts` + leader lines. For aircraft, turbines, cells, devices.

**Quantitative chart** — grouped bars as `<rect>`s on a baseline with axis ticks;
one ramp per series; values in `ts` above bars. Keep it flat — no 3D, no gradients.

## When to prefer this vs the dark-tech variant

Educational / scientific / physical subject → this (light, 9-ramp). Cloud / infra /
software system architecture → the dark token variant in `dark-tech.md`. When neither
fits cleanly, this educational look is the safe general-purpose default.
