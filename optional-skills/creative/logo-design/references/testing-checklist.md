# Testing Checklist

A logo is only finished when it has survived these tests. Most can be run with
`scripts/svg_audit.py` (structure and geometry) and `scripts/preview_sheet.py` (visual tests); the rest need
judgement. Run them on every concept before presenting and again on the final artwork.

## 1. Scale
- [ ] **16 px favicon**: the core idea survives; no mush. If not, design a simplified small-size version
      (fewer elements, thicker strokes, bigger gaps) — detailed marks may keep a reduced companion.
- [ ] **24–32 px** (app lists, social avatars): recognisable at a glance.
- [ ] **Very large** (billboard, building): curves are smooth, no lumpy anchors, kerning holds up; flaws
      invisible at small sizes become glaring.
- [ ] **Minimum size**: shrink until the mark loses its identifying features; the size just above that is the
      documented minimum (in px for screen and mm for print).

## 2. Colour and value
- [ ] Works in **one-colour black** on white.
- [ ] Works in **one-colour white** on black — and doesn't look heavier than the black version (irradiation). If it
      does, provide a slightly thinned reversed file.
- [ ] Works on the **brand colour**, on **photos** and on **patterns** (use a graphic device — outline or container —
      if not).
- [ ] **Greyscale**: colour segments still separate by value; nothing relies on hue alone.
- [ ] Colours reproduce in **CMYK** and have spot/Pantone equivalents if vivid.

## 3. Form
- [ ] **Squint/blur test**: the silhouette alone is distinctive.
- [ ] **Mirror test**: flipping reveals no proportion errors you'd adapted to.
- [ ] **Rotation / unintended readings**: view rotated 90° and 180°, tiny, and at a distance; show it to someone who
      hasn't seen the brief. Check for accidental letters, body parts, offensive or sexual readings, political or
      religious symbols, and resemblance to hazard signs.
- [ ] **Optical corrections** done: overshoot, bone effect, horizontal stroke thinning, optical centre.
- [ ] **Geometry clean**: no near-miss angles, perfect primitives, consistent radii and stroke widths
      (`svg_audit.py` flags near-miss angles and tiny details).
- [ ] **Balance**: not accidentally tilted; weight distributed; proportions close to square for symbols.

- [ ] **Letter test**: every customised letter still reads as the intended letter at first glance (ask: "what
      letter is this?"). A K that reads as an h, or an N that reads as a lightning bolt, needs revising.
- [ ] **Junction check**: zoom in on every place strokes meet or overlap — no accidental notches, slivers, lumps,
      hairline gaps or ink traps.
- [ ] **Peer test**: next to 3–4 exemplary library marks at the same size, yours looks equally resolved.

## 4. Distinctiveness and originality
- [ ] **Shelf test**: placed among competitors (`preview_sheet.py --refs-industry <industry>`), it stands out
      rather than blending in.
- [ ] **Familiarity test**: it doesn't remind you (or anyone you ask) of an existing mark. If it feels familiar and
      it isn't yours, it's someone else's.
- [ ] **Library check**: search the bundled library for the same subject/technique
      (`search_library.py --subject <thing>`); make sure yours is clearly different.
- [ ] **Trademark check** (recommend to the user): search the relevant trademark databases and do a reverse image
      search before launch. You cannot give legal clearance — say so.

## 5. Meaning and fit
- [ ] The idea can be explained in **one sentence**.
- [ ] Tone matches the brand adjectives (sharp/round, heavy/light, warm/cool, classic/modern).
- [ ] It identifies rather than explains; it will still fit if the business expands.
- [ ] Cultural meaning of colours and symbols checked for the audience's cultures.
- [ ] It works without the tagline and, for symbols, eventually without the name.

## 6. Media and production
- [ ] One-colour printing, embroidery (no hairlines, gaps ≥ ~1 mm at chest-logo size), laser engraving, vinyl
      cutting, signage, dark mode, animation.
- [ ] App icon: optically sized inside the platform's rounded tile; no fine text.
- [ ] Social avatars: survives circular crop.
- [ ] Master file: vector only, text outlined, strokes expanded, no filters/raster, clean viewBox
      (`svg_audit.py` score ≥ ~90 with no FAIL).

## 7. Context
- [ ] Shown in 5–6 **realistic mockups relevant to the business** (a café's cup, not a gym bag), in a consistent
      style. Never judge a logo only in isolation on a white artboard.
- [ ] Lockups (horizontal, stacked, symbol-only) all tested at their intended sizes.

## Recording results
Summarise test outcomes when presenting: what passed, what was adjusted (e.g. "thinned the reversed version by 3 %;
opened the counter gap from 6 to 10 units so it survives 16 px"), and what the user must still do (trademark search,
Pantone proofing).
