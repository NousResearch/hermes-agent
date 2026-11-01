# Visual Techniques & Optical Corrections

The craft layer: how forms are perceived, and the small corrections that separate a good mark from a
perfect one. Read during development and refinement (phases 4–5), and when critiquing.

## Contents
1. Geometry, grids and the golden ratio
2. Balance (stability, proportion, composition, consistency, scalability)
3. Optical corrections: overshoot, bone effect, irradiation, same-size look
4. Composition dynamics between shapes
5. Symmetry vs. asymmetry
6. Solid vs. line
7. Sharp vs. round
8. Negative space & figure/ground
9. Visual paradoxes: impossible figures, ambiguity, motion
10. Gradients, light and shading
11. Dimension
12. Visibility devices (outline, container)
13. Patterns
14. Reviewing with fresh eyes

---

## 1. Geometry, grids and the golden ratio

- **See the world as simple geometry.** Train yourself to reduce objects to circles, squares, triangles and
  their parts. Once you feel the structure behind things, composition becomes easier and the results feel
  inevitable.
- **Construct from primitives.** Circles, arcs of consistent radii, straight lines at deliberate angles.
  Curves built from circle segments are easy to grid and reproduce; free-drawn curves are harder to justify
  and to fix.
- **Golden ratio (≈1.618) and Fibonacci proportions** (1, 2, 3, 5, 8, 13, 21 …) are useful as a standard of
  organisation — e.g. circle radii in a construction following 1 : 1.618 or Fibonacci multiples. Use them where
  they help; never let numbers overpower a form that already feels right. If a form feels right, the maths
  should not get in the way.
- **Whole-number angles.** Aim for clean angles: 0°, 15°, 30°, 45°, 60°, 90°. An element at 43° should be 45°;
  33.46° should become 30° or 35°. Lines that are 1–2° off horizontal/vertical read as mistakes, and clean
  angles show the elements were placed deliberately. (Colour values too: prefer whole-number CMYK percentages.)
- **Perfect primitives.** A circle must be a perfect circle and a square a perfect square — unless the deviation
  is an intentional optical correction.
- **Gridding is for finishing, not for starting.** Apply construction grids after the concept is approved to find
  and fix misalignments, uneven radii and stray angles. The changes are usually too small for viewers to notice
  consciously — but they feel the difference.
- **Don't over-grid.** Complex organic curves may not decompose into circles; it's fine to leave them ungridded.
  If something doesn't work within the grid, make it work outside the grid. A construction drawing cluttered with
  dozens of meaningless circles is a presentation gimmick, not rigour.

## 2. Balance

1. **Stability** — the mark should not feel accidentally tilted. There should be a sense of gravity; where the
   concept allows, a heavier or wider base feels grounded.
2. **Proportion** — aim for an overall footprint closer to a square than a long rectangle. Very tall or very wide
   symbols are awkward to use and to lock up with type. (Library data: ~76 % of standalone icons are between
   0.8 : 1 and 1.25 : 1.)
3. **Composition** — distribute elements and gaps evenly. Clusters in one area and emptiness in another create
   dissonance (unless tension is the intent).
4. **Consistency** — keep weights related. Very thick next to very thin reads as imbalance; in line marks, keep
   stroke width consistent throughout. Repeat radii and angles.
5. **Scalability** — every decision must survive enlargement and reduction.

## 3. Optical corrections

### Overshoot
Round and pointed shapes look smaller than flat-edged shapes of the same measured height. A circle placed
between squares of equal height looks too small; an `O` beside an `H` looks short. Let curves and apexes extend
slightly past the baseline/cap line/edge — typically **1–3 % of the height** for curves and more for sharp points
(triangles/apexes can need 3–6 %). Do it in the final stage, with one or two anchor tweaks. This is the
difference between great and perfect.

### Bone effect
Where a curve meets a straight line tangentially (a capsule: two semicircles joined by straight sides; rounded
rectangles; rounded triangles), the straight segment appears to pinch inward like a bone. Fixes, from quick to best:
1. Stretch the end curves into ellipses (softens but doesn't remove it).
2. Pull the curve handles so curvature ramps up gradually instead of jumping from zero to the full circle.
3. Use a smooth curvature transition (a "squircle"/superellipse-like corner): extra anchors that ease from straight
   to curved; copy the tuned curve to the symmetrical corners.
Not every instance needs fixing — sometimes the bone effect is expressive (bulkiness in a figure). Fixing it often
frees up interior space and makes rounded marks crisper at small sizes. In SVG, prefer continuous-curvature
corners (cubic Béziers with handles ≈ 0.55–0.65 of the corner size, extended past the tangent point) over plain
`rx` arcs when the corner is prominent.

### Irradiation: white on black looks bigger
A light shape on a dark background appears larger and bolder than the identical dark shape on a light background.
Consequences:
- A reversed (white-on-dark) logo looks heavier. Provide a **reversed version with slightly thinned strokes**
  (roughly 2–5 % of stroke weight), or scale the reversed version down slightly. Achieve it by offsetting the path
  inward (expand a thin stroke and subtract it from the shape).
- Test both versions side by side until they look equal; record which file to use in the guidelines.
- Thin white lines on dark grounds can visually "bloom" and fill in when printed; keep counters open.

### Same-size look for mixed shapes
Circles, squares and triangles of the same bounding box do not look the same size. Equalise by eye: circles
slightly larger than squares; triangles larger still. Same for icons in a set.

### Horizontal vs. vertical strokes
Horizontal strokes look heavier than vertical strokes of equal thickness. In lettering and geometric marks, make
horizontals slightly thinner (often 5–10 %) to look equal. Where strokes meet at acute angles, thin them near the
junction to avoid dark ink traps and blobs at small sizes.

### Optical centre
The optical centre of an area sits slightly above the geometric centre. A mark centred mathematically in a
container often looks low; nudge it up a little. Asymmetric shapes (a triangle pointing right, a play button) need
to be shifted toward their visual mass, not their bounding box.

## 4. Composition dynamics between shapes

Shapes interacting create feelings. First define the goal — harmony, tension, dynamism, balance, dissonance,
entropy — then arrange.
- Circle resting on a square of the same width → harmony, stability.
- Square balanced on a circle → tension, instability.
- Circle beside the apex of a triangle → motion.
- Circle exactly on top of a triangle's apex → balanced tension.
- Diagonals and tapering forms → movement; horizontals → calm; verticals → aspiration, strength.
- Density increasing across a form (sparse → dense lines or dots) → implied motion.

Years of playing with forms build the intuition; in the meantime, test variations side by side.

## 5. Symmetry vs. asymmetry

- Perfect mirror symmetry is stable and formal but often boring; repetition by mirroring can feel mechanical.
- A subtle asymmetry keeps the eye travelling: a change in colour on one side, a detail added or removed, an
  offset element, a mirrored-then-shifted part.
- Especially useful when the overall silhouette is symmetrical — let one detail break it.
- Symmetry still has its place for institutions and marks that must feel absolute (seals, public bodies).

## 6. Solid vs. line

- **Solid marks** look stable and strong; silhouettes stay clear when small and from a distance; bold shapes reproduce
  well across media.
- **Line (monoline/outline) marks** look light and elegant; good for passive or refined brands, UI iconography and
  interior signage. Downsides: weak from a distance and at small sizes unless strokes are thick enough; they are
  more vulnerable to bad reproduction.
- If a mark works both ways, choose one as **primary** and the other as secondary.
- In the final SVG, expand strokes into filled outlines (see `svg-construction.md`), so scaling does not change the
  line-to-size ratio unpredictably.

## 7. Sharp vs. round

- Sharp, angular forms read as assertive, authoritative, technical, sometimes threatening — we instinctively treat
  sharp objects with caution.
- Rounded forms read as friendly, inviting, soft, safe — we want to touch and hold them.
- Keep sharp elements out of the silhouette unless the brand needs edge (security, sport, performance, luxury
  precision). Most brands want to feel approachable, so rounded corners are a sensible default — but a slightly
  rounded corner (small radius) often reads more premium than fully pill-shaped forms.
- Mixing: a sharp exterior with soft interior details (or vice versa) can express duality ("secure but friendly").

## 8. Negative space & figure/ground

- Every silhouette has surrounding space. Treat the negative shapes (counters, gaps, space between elements) with the
  same care as the positive shapes — the inner shapes of letters echoing the outer form create coherence.
- Clever negative space (a hidden arrow, an animal between two shapes, a letter carved out of an object) creates a
  moment of discovery. Keep it simple: two elements with distinct silhouettes, conceptually related.
- Gaps must be large enough to survive reduction; a 1 px gap at 32 px disappears. As a rule of thumb, make the
  smallest gap ≥ 1/32 of the mark's width if it must read at favicon sizes.

## 9. Visual paradoxes

When something appears to be one thing and becomes another on closer inspection, the mind solves a small problem and
enjoys the solution. Three families:
1. **Impossible figures** — shapes that can't exist in 3D (reversed perspective, manipulated line connections,
   misaligned layers). Study the concept, don't copy famous figures; transpose the principle (e.g. a letter whose
   perspective switches direction halfway).
2. **Ambiguous forms** — one image that reads as two. Rare to succeed intentionally. The bigger risk is the
   *unintended* reading: after hours of work a designer stops seeing what is obvious to others (including sexual or
   offensive readings). Always ask fresh eyes, rotate the mark 90/180°, view it tiny and mirrored.
3. **Motion illusions** — tapering sweeps, speed lines, progressive repetition. For speed, run the form from thin to
   thick in the reading direction (left to right in Latin-script cultures); consider right-to-left audiences.

Other tools for turning a mundane shape into something memorable: layering, interlacing, overlapping with
transparency, juxtaposing flat and dimensional, and playing with colour at intersections.

## 10. Gradients, light and shading

- **Gradients are easy and therefore cliché.** They can soften a mark and add depth, but they reproduce poorly:
  vivid RGB gradients lose vigour in CMYK print and blur at small sizes. Unless the logo is digital-only, keep a flat
  version as the master and treat gradients as an optional enhancement.
- **Simplify gradation.** Replace a smooth gradient with a few distinct steps (3–5 flat bands of related hues). At
  small sizes the steps read as seamless; at large sizes they look crafted rather than default.
- **Light and shading for plain marks.** Simple is not always attractive — some marks look plain because they lack
  content. Introducing a light source (a fold, an overlap shadow, a highlight) can add depth and sophistication. If the
  mark is interesting in its reduced state, don't add light.
- **Few tones.** Many shades of grey kill sharpness at small sizes. Use at most highlight, one mid-tone, shadow and
  background ("light–dark" chiaroscuro reduction). For letterforms and abstract marks, one mid-tone applied to the
  parts that fold under or overlap others is usually enough.
- **Structure of light on a sphere**: highlight, midtone, core shadow, reflected light, cast shadow, occlusion shadow.
  Understand it to shade any form convincingly; then reduce to 2–3 tones.
- **Tapered strokes (engraving style)**: tonal gradation built from arrow-like strokes that thin toward the tip — a
  versatile way to render volume on organic forms. Time-consuming; reserve for brands where craft is the message.

## 11. Dimension

- Realistic 3D renders carry too much detail for logos.
- Dimension achieved by the simplest means — two or three flat faces, an isometric fold, an overlap — can be striking
  and widens the space of original solutions, since flat combinations of primitives are largely exhausted.
- Always keep a version that works flat and in one colour.

## 12. Visibility devices

A logo designed for light backgrounds may not invert well (a white swan inverted becomes a black swan — maybe desired,
maybe not). For busy, photographic or multicoloured backgrounds use a **graphic device**:
- An outline (a sufficiently thick light stroke around the silhouette), or
- A containing shape: circle for circular marks; square or rounded square for most others (the safest for rectangular
  formats and app icons). A basic container may not be the most beautiful, but it is functional.
- The device's proportions should match the mark's (square mark → square device; wide mark → wide device); mismatches
  create distracting empty space.

## 13. Patterns

- A grid is the heart of a pattern. Square grids are the most versatile but most common → sameness. Try triangles,
  hexagons, rotated squares (diamonds), or grids derived from the mark's angles.
- Plain tessellations are straightforward but get boring; prefer grids that allow variation.
- Distribute colour evenly; too much contrast disrupts flow, too little makes it invisible.
- Derive components from the mark when possible so the pattern and logo feel related without duplicating.
- Test at multiple scales (card, wall, vehicle) and, for print, keep ink coverage reasonable.

## 14. Reviewing with fresh eyes (the dialectical approach)

- **Save every meaningful iteration** side by side instead of overwriting. Designers often favour a lesser variant
  in the moment of excitement and lose the better one.
- **Compare pairs** directly and pick the stronger; then iterate from the winner.
- **Rest before deciding** — fatigue makes judgement less objective.
- **Mirror the mark.** Flipping horizontally reveals proportion problems the eye has adapted to (especially in
  animals and figures).
- **Blur/squint and shrink.** If the silhouette is unclear when blurred, the idea is not strong enough.
- **Physical aids**: for complex figures, references and simple 3D mock-ups (paper, clay, wire) reveal better
  viewpoints and inconsistencies.
