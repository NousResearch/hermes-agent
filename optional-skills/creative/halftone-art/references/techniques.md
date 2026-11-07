# Techniques — code, maths, and verified parameters

Every snippet here was rendered and inspected in a browser before being written down. Numbers
(cell sizes, contrast values, timings) are measured, not estimated.

---

## A. Canvas engine (recommended default)

`templates/halftone-core.js` — 18 KB, no dependencies, ES module.

### The four stages

```
1. SEPARATE   source pixels → ink coverage  t ∈ [0,1]   (1 = solid ink)
2. SCREEN     average t over one rotated cell
3. SOLVE      dot geometry whose AREA = t · cell²
4. OVERPRINT  paint plates at their own angles with multiply blending
```

**Stage 1 — coverage.** `t = 1 − luma` for a normal (dark-ink-on-light-paper) print. Optional
histogram stretch (`autoLevels`) first; without it, phone photos and scans that only occupy
luma 0.25–0.8 collapse into two dot sizes and the print looks like fog.

**Stage 2 — cell average.** Two details matter more than everything else in this file:

- **Bilinear** sampling of the tone map, not nearest-neighbour. Nearest sampling beats
  against the dot grid and produces low-frequency moiré that no amount of cell tuning removes.
- **Taps rotated with the screen angle.** A 4×4 tap square that stays axis-aligned while the
  grid rotates samples different parts of each cell, which aliases into visible cross-hatch.
  Rotate the tap offsets by the same angle as the screen.

```js
const reach = cell * 0.47;                 // roughly the cell's own footprint
for (a,b in 4×4 grid) {
  const u = ((a/3)*2 - 1) * reach;
  const v = ((b/3)*2 - 1) * reach;
  sum += bilerp(tone, mw, mh,
    cx + u*cos - v*sin,                    // rotate the tap, not the grid
    cy + u*sin + v*cos);
}
```

**Stage 3 — area, not radius.** A dot's *area* carries the tone:

| shape | solve from area A = t·cell² | note |
|---|---|---|
| round | r = √(A/π) | dots touch at t ≈ 0.785 — correct AM behaviour |
| square | side = √A | dot touches at t = 1 |
| diamond | diagonal = √(2A) | rotating the square gives the same coverage |
| ellipse | r = √(A/π), rx = 1.4r, ry = 0.68r | softer, more filmic |
| line | thickness = A / (1.12·cell) | lines connect into solid at high t |
| cross | arm = 0.8·cell, thickness = A / (1.7·arm) | grid at high t |
| ring | r = 0.9·√(A/π), stroke = 0.08·cell | outline dots; never flood |

**Flood rule.** Above `t ≈ 0.95` fill the whole cell. Circles that overlap still leave white
pinholes in the cell corners — real ink does not. Pinholes are the single most common tell of a
naive implementation (they look like sparkles in the shadows).

**Stage 4 — overprint.** Plate 2+ use `globalCompositeOperation = 'multiply'` on the same
canvas. Paper colour is painted first. That is exactly how translucent inks stack.

### Screen angles (degrees)

| model | angles |
|---|---|
| mono | 45 (canonical), 15 or 75 for a visibly different weave |
| duotone / riso | base ± 30 — e.g. 27 / 62. Equal angles stack the two plates into garbage |
| CMYK | 15 / 75 / 0 / 45 — the classic set that turns moiré into a rosette |
| line screen | 30–40 reads as engraving; 0 or 90 reads as scan lines |

### Measured performance (M-series laptop, Chrome, devicePixelRatio 1)

| job | time |
|---|---|
| 1440×900, cell 8, mono | 13 ms |
| 1440×900, cell 6, CMYK (4 plates) | 36 ms |
| 1440×900, cell 4, mono | 16 ms |
| 420×420, cell 5, mono | 2 ms |

Fine: render on load and on resize. Too slow: re-rendering a 1440×900 CMYK hero every frame.
For animation, animate the *tone function* in a smaller canvas or pre-render a handful of frames.

### Export for print / download

```js
halftoneToDataURL(img, { exportWidth: 3000, cell: 12 })  // cell in export pixels
```

Render off-screen at the export size with `pixelRatio: 1`; `cell` is then measured in output
pixels, so a 2× export looks the same, only sharper.

---

## B. CSS filter: screen + threshold (photo → dots, no JS)

The only pure-CSS route that produces *tone-varying* dots from a photograph.

```css
.plate {
  position: relative; isolation: isolate; overflow: hidden;
  filter: contrast(7.5) brightness(1.02);        /* ← the threshold */
}
.plate img {
  display: block; width: 100%; height: 100%; object-fit: cover;
  filter: grayscale(1) contrast(1.35) brightness(.96);
}
.plate::after {                                   /* ← the screen */
  content: ""; position: absolute; inset: -32%;
  background-image: radial-gradient(circle at 50% 50%, #000 0%, #555 42%, #fff 78%);
  background-size: 5px 5px;
  mix-blend-mode: screen;
  transform: rotate(45deg);
}
```

How to calibrate it, in order:

1. `background-size` = the cell. 4–6 px reads as newsprint, 8–10 px as poster.
2. The gradient stops set how *soft* the dots are. Tighter (0% → 40%) = smaller, crisper dots;
   wider = mushier, more "ink spread".
3. `contrast()` on the wrapper is the threshold. Below ~5 nothing snaps; above ~12 the image
   blows out to paper and you lose the highlights. 7–9 is the working band.
4. `inset: -32%` plus `rotate(45deg)` stops the rotated tile from leaving empty triangles.

**Known limits** — this is why the canvas engine exists:

- Threshold is global: one setting that works for a dark subject crushes a bright one.
- The dots are a uniform *overlay* screen; dot size is not solved from local tone, so midtones
  read as "screen at 50% opacity" rather than a real separation.
- Costly on large surfaces: `contrast()` + `blend` forces a repaint of the entire element on
  scroll in some engines. Keep it under ~600×600 or apply to a static hero.

Use it for: a single hero image, a card thumbnail, a hover effect — anything where per-image
calibration is acceptable and you want zero JS.

---

## C. CSS mask: dot grid over colour (textures, gradients, tints)

Uniform dots; tone comes from what is *underneath* them.

```css
.tex {                                  /* dots punched out of a colour field */
  background: linear-gradient(120deg, #0f4bd8, #ff5f8f 55%, #ffe600);
}
.tex::after {
  content: ""; position: absolute; inset: 0;
  background: #f2eee2;                  /* the paper showing through */
  -webkit-mask-image: radial-gradient(circle at 50% 50%, #000 44%, transparent 46%);
  mask-image: radial-gradient(circle at 50% 50%, #000 44%, transparent 46%);
  -webkit-mask-size: 9px 9px; mask-size: 9px 9px;
  -webkit-mask-repeat: round; mask-repeat: round;
}
```

- `mask-size` = **pitch**. `mask-repeat: round` avoids a half-dot seam at the right/bottom edge.
- The stop pair (`44%`/`46%`) is radius: 44% of half the tile = dot radius ≈ 0.22 × pitch.
  Radius ≈ 0.22–0.26 × pitch = classic; > 0.35 × pitch = dense black field; < 0.12 = dust.
- Fade the field with a second mask (a `linear-gradient` or an image), one per axis is not
  possible — combine with `mask-composite: intersect` if you need two.

This is the cheapest, most reliable halftone texture on the web. No JS, no image, sharp at any
DPR, and it composes with `mix-blend-mode`.

---

## D. SVG pattern (precise, scale-independent, animatable)

```html
<svg viewBox="0 0 300 300" preserveAspectRatio="none">
  <defs>
    <pattern id="dots" width="9" height="9" patternUnits="userSpaceOnUse"
             patternTransform="rotate(45)">
      <circle cx="4.5" cy="4.5" r="2.6" fill="#15140f"/>
    </pattern>
    <linearGradient id="fade" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0" stop-color="#fff" stop-opacity="1"/>
      <stop offset="1" stop-color="#fff" stop-opacity="0"/>
    </linearGradient>
    <mask id="m"><rect width="300" height="300" fill="url(#fade)"/></mask>
  </defs>
  <rect width="300" height="300" fill="#c8ff3d"/>
  <rect width="300" height="300" fill="url(#dots)" mask="url(#m)"/>
</svg>
```

- `patternTransform` rotates the screen — the SVG equivalent of `angle`.
- Mask with a gradient to fade the field; mask with a **photo** (or `<feImage>`) to get tone,
  though per-dot sizing still will not vary: coverage varies by *opacity*, not area. That is a
  different (and much softer) look than true halftone — use it deliberately.
- Infinite crispness when scaled, no JS, prints well. Good for logos, rules, section dividers.
- Animating `patternTransform` (rotate/translate over time) is cheap and legal — a rolling
  screen. Keep it subtle or disable it under `prefers-reduced-motion`.

---

## E. Halftone type (dot screen cut out of glyphs)

```css
.type b {
  color: #15140f;
  -webkit-mask-image: radial-gradient(circle at 50% 50%, #000 58%, transparent 60%);
  mask-image: radial-gradient(circle at 50% 50%, #000 58%, transparent 60%);
  -webkit-mask-size: 5px 5px; mask-size: 5px 5px;
  -webkit-mask-repeat: round; mask-repeat: round;
}
```

Rules that decide whether this works:

- **Never set a background on the masked element.** The mask eats the background box too, so
  `background: currentColor` turns the inline box into a solid dotted bar instead of letterforms.
  Colour the *text*, mask the *element*.
- 4–6 px pitch for display sizes (80–160 px type). At 5 px the dot radius should be ~55–60% of
  the cell, otherwise thin strokes break into disconnected specks.
- Bold/black weights only. Anything under 600 weight turns to dust at small sizes.
- Test at the real size. A 5 px screen at 160 px type is invisible at 60% zoom — that is fine
  for large display, fatal for a 32 px headline.
- For text inside a photo plate, invert: mask the photo and let the type punch through with
  `mask-composite`, or just place solid type over the halftone — solid-on-dots reads better
  than dots-on-dots.

---

## F. Misregistration (the "printed twice" charm)

```css
.plate { position: relative; background: #fdf3e3; }
.plate .ink { position: absolute; inset: 0; mix-blend-mode: multiply; }
.plate .ink.blue { color: #0f4bd8; transform: translate(-3px, -2px); }
.plate .ink.pink { color: #ff5f8f; transform: translate(3px, 2px); }
```

- Offsets of **1–3 px** read as printing; 8 px reads as a broken 3D effect.
- Both layers `multiply` so overlaps darken into a third colour (riso overprint).
- The canvas engine does the same thing with `misregister: 2` — per-plate random offsets so
  repeated renders do not look mechanically identical.
- Slight paper tint (`#fdf3e3`) does more for the print illusion than the offset does.

---

## G. Cutouts and die-cut stickers

`cutout: true` renders the same halftone but leaves everything outside the subject transparent.

### Pipeline

```
1. MASK      silhouette: alpha channel, or backdrop flooded from the frame corners
2. CLEAN     morphological closing (seals key leaks) + optional largest-blob
3. LAYER     paint paper + plates onto a scratch canvas  ← multiply overprint needs real paper
4. CLIP      layer.destination-in(mask)  → halftone only inside the silhouette
5. RIM       paint dilate(fillHoles(mask)) behind it, then punch the holes back out
6. COMPOSITE scratch layer onto the output canvas, which stays transparent
```

Why the scratch layer matters: `globalCompositeOperation = 'multiply'` only darkens against a
real backdrop. Blending onto transparency degrades to `source-over`, so a two-plate riso cutout
rendered directly would lose its overprint. Painting on paper first and clipping afterwards keeps
the ink behaviour identical to the rectangular plate.

### Two ways to get the mask

**`cutoutMode: 'alpha'` — exact.** Any PNG with transparency: Figma export, Photoshop, `remove.bg`,
an illustration you drew yourself. The mask is the alpha channel, thresholded at `cutoutAlpha`
(128). No tuning, no surprises. **Use this for anything real.**

**`cutoutMode: 'key'` — draft.** Samples the four corner patches (per-channel median), then
flood-fills inward, keeping only border-connected pixels within `cutoutTolerance` of that colour.

| setting | effect |
|---|---|
| `cutoutTolerance` 0.08–0.14 | tight — keeps hair and fine edges, leaves backdrop haze |
| 0.16–0.24 | general studio/product shots |
| 0.26+ | aggressive — starts eating the subject |
| `cutoutClose` 2–6 | seals the thin channels a flood leaks through in fur/fabric |
| `cutoutKeepLargest: true` | drops detached specks (good for textured backdrops) |

Measured on the sample photos: a c.1905 studio portrait keys cleanly at tolerance 0.16,
closing 5. A cat photographed against a **white sheet** leaks badly at every tolerance — light
fur and white paper are the same colour to a flood fill. That is not a bug to tune around; it is
the boundary of the technique.

### Rim, holes, and stock

- `cutoutOutline` (px, CSS) grows the silhouette and paints it in `cutoutOutlineColor` **behind**
  the artwork — a paper bleed, like a real die-cut sticker.
- Enclosed cut-outs (eyes, handles) must stay see-through: compute the rim from a hole-filled
  silhouette, then subtract the holes again.
- `cutoutFill: 'paper'` keeps the stock inside the shape (sticker on white vinyl).
  `cutoutFill: 'none'` knocks out the flat paper colour, leaving only dots (clear vinyl) — the
  antialiased dot edges survive because they are blends, not bars of paper.
- Clear vinyl needs an ink that reads on both light and dark surfaces. Dark ink on a dark page
  is an invisible sticker.

### Export

```js
canvas.toBlob((blob) => { /* download */ }, 'image/png');   // alpha is preserved
```

JPEG will flatten alpha to white. If you must ship JPEG, composite onto a known colour first.

## H. Tuning matrix

Starting points by subject and use. Tune `gain` first, then `contrast`, then `cell`.

| use | mode | cell (CSS px) | gain | contrast | notes |
|---|---|---|---|---|---|
| newsprint portrait | mono | 4–6 | 0.85–0.95 | 1.15–1.25 | `autoLevels: true`; paper `#f2eee2` |
| editorial hero plate | duotone | 7–9 | 0.95 | 1.1–1.2 | angles 27/62, `misregister: 2–3` |
| process-colour proof | cmyk | 6–8 | 1.0 | 1.0–1.1 | colour source only; angles 15/75/0/45 |
| pop / screenprint | mono | 11–16 | 1.05–1.15 | 1.35–1.6 | `levels: 3–4`, ink red on yellow |
| quiet UI texture | mono | 10–14 | 0.8 | 0.9 | `gain` low keeps it from vibrating behind text |
| engraving / line cut | line | 8–11 | 0.9 | 1.15–1.3 | angle 30–40 |
| poster / far viewing | any | 12–20 | 1.05 | 1.2–1.4 | dot becomes a graphic element, not a photo |

`gain` above 1.05 starts closing the highlights; below 0.8 the shadow areas go grey and weak.

**Second-plate curves (`plateCurves`).** The curve decides *where* the second ink appears, and
it is the difference between a riso print and confetti:

- `midtone` — spot colour lives in the faces and mid-tones. The safe default for a hero duotone.
- `shadows` — accent confined to the dark end. Reads punchy on simple subjects, but on fine dark
detail (fabric, hair, foliage) it sprays speckles through the suit instead of sitting behind it.
- `highlights` / `liner` — tint the light end; use for pale paper-on-paper washes, not for ink.

Pick `midtone` unless you have looked at the render and *want* shadow speckle.

---

## I. Browser support notes

- `mask-image` with `radial-gradient` — Chrome 120+, Safari 15.4+ (needs the `-webkit-` pair
  before Safari 17), Firefox 53+. Ship both properties.
- `mix-blend-mode` + `filter: contrast()` — universal, but each creates a compositing layer;
  dozens on one page will cost memory on mobile.
- `getImageData` on a canvas that drew a cross-origin image throws unless the response carries
  CORS headers **and** you set `img.crossOrigin = 'anonymous'` before `src`. Serve local files
  over HTTP; `file://` counts as cross-origin.
- `canvas.width` set from `getBoundingClientRect()` on a detached canvas returns 0 — pass
  explicit `width`/`height` for off-screen renders (contact sheets, exports).
