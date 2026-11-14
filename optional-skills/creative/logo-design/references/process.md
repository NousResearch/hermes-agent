# The Design Process in Detail

The stages from brief to final artwork, adapted for an AI designer that works in SVG code as well as for a
human designer with a sketchbook. SKILL.md gives the short version; read this for the details of each stage.

## Contents
1. Stage map
2. Concepting
3. Mood boards and reference gathering
4. Exploration ("sketching") — three stages
5. Development in vector
6. Refinement and gridding
7. Final artwork checklist
8. Working habits

---

## 1. Stage map

| Stage | Output | Gate |
|---|---|---|
| Collaboration / discovery | Brief, criteria, schedule, decision-maker | Brief agreed |
| Concepting | Word map, 6–10 one-sentence concepts | Concepts are distinct and on-brief |
| Exploration | Many rough forms per concept | 2–4 directions worth developing |
| Development | Clean black vector versions of 3 concepts | Presentable at ~80 % finish |
| Presentation 1 (checkpoint) | 3 concepts in greyscale as one overview image + kit offer | User picks a direction and asks for the kit |
| Refinement | Optical corrections, gridding, colour, type, lockups | Approved final mark |
| System & applications | Palette, typefaces, variants, patterns, key applications | System approved |
| Production | Final files, specifications, guidelines | Delivered; audit plan |

Integrate existing equity (colours, shapes, heritage details) during exploration if it's a redesign.

## 2. Concepting

- Begin with the name and a detailed brief — from the brand's big idea down to how it operates.
- The most useful clues are **adjectives** describing the brand; they give abstract visual cues that can become
  symbols. Also mine: the name's letters and meaning, the promise, the audience's world, the product's process
  and materials, place and origin.
- If the name contains a visual (a letter + number with strong shapes, a word that is an object), the concept may be
  hiding in plain sight.
- Write each concept as **one sentence** ("A 'T' whose crossbar is a shield's top edge, for a logistics firm whose
  promise is protection"). If it can't be said in a sentence, it won't be understood in a glance.
- Aim for concepts that are clever *and* visually pleasing — marks with a clever idea and sound visual appeal make the
  strongest first impression.
- Consider the audience's demographics and culture (colourful/rounded for children; bold for some audiences;
  cultural symbols where relevant) — but the brand strategy decides; a children's product can have an adult identity.

## 3. Mood boards and reference gathering

- Not just logos: include nature, architecture, painting, materials, typography, photography, product details.
- Compartmentalise by direction — classic, futuristic/high-tech, colourful, monochrome — and by mark type
  (pictorial, letterform, monogram), each with relevant imagery. Clear boundaries make choices easier.
- Use the library for logo references by technique and subject (`scripts/search_library.py --technique
  negative-space --exemplary`). References are for learning construction and tone — never to be traced.
- Mood boards can be shared with the client to align on style, or kept private; clients can't always foresee how a
  direction will develop, so don't let an early mood preference lock the outcome.

## 4. Exploration — three stages

A human designer does this with pencil and tracing paper; as an AI, do the equivalent with quick descriptions and
rough SVG thumbnails. The logic is the same.

### Stage A — Initial (quantity over quality)
- Pour out ideas with no judgement about clarity, spacing, form or silhouette. Rough strokes, imperfect shapes,
  careless curves. The page should look like a battlefield of half-formed ideas.
- Quantity matters because it lets unexpected accidents happen. For an AI: list 15–30 micro-variations across
  the 6–10 concepts (different letter treatments, crops, containers, negative-space pairings, angles).
- Rough SVG thumbnails at 64–128 px are ideal: small size forces silhouette thinking.

### Stage B — Refinement (from ~30 % to ~60 %)
- Pick the most promising concept by gut feel; it may be only ~30 % of the final. The goal is to reach 50–60 %.
- Redraw it as a reference, then make many similar versions, each trying one new improvement: how elements
  interact, balanced flow, outline, proportion. Compare each version with the previous; keep what works, cut what
  doesn't.

### Stage C — Fine-tuning (clean and precise)
- Trace over the best version repeatedly with small improvements until the form is clean and precise. At the end,
  few formal changes should remain for the vector stage.

Designers who stop at the first acceptable result leave quality on the table. Keep adding and stripping.

## 5. Development in vector

- **Copy, then change.** Before every significant change, duplicate the current version and place it next to the
  previous. When something goes wrong, you can see exactly where. (In SVG work: save `concept-a-v1.svg`, `-v2.svg`
  … and compare them in `scripts/preview_sheet.py --compare`.)
- **Few anchor points.** Anchors have great power; too many make curves lumpy and jagged. Periodically delete
  unnecessary ones. Clean curves come from few, well-placed anchors with handles aligned to the curve's tangent
  (horizontal/vertical handles at extrema).
- **Build with shapes.** Use circles, rectangles and consistent radii to construct and to check curves and corners.
- **Black first.** Develop in solid black on white. Colour comes after the concept is chosen.
- Present at about **80 % finish** — refined enough to judge the idea, not so polished that effort is wasted on a
  direction that gets dropped.

## 6. Refinement and gridding

After a direction is approved:
1. **Alignment**: check that all horizontals and verticals are true (not 1–2° off). Snap near-standard angles to
   standard ones (43° → 45°).
2. **Primitives**: circles perfectly circular, squares square, consistent radii where intended.
3. **Consistency**: stroke weights, gap widths, corner radii, terminal angles.
4. **Optical corrections**: overshoot, bone effect, irradiation (reversed version), horizontal-stroke thinning,
   optical centre (see `visual-techniques.md`).
5. **Colour**: build palette; test on backgrounds (see `color.md`).
6. **Type & lockups**: finalise wordmark, spacing and all lockups (see `typography.md`).
7. **Scale tests**: see `testing-checklist.md`.

Don't grid organic curves that don't decompose into circles; leave them.

## 7. Final artwork checklist

- [ ] No duplicate, stray or unnecessary anchor points; no open paths in filled shapes.
- [ ] Angles accurate, especially on the outer silhouette.
- [ ] All text converted to outlines; no live fonts, no raster images, no filters or effects in the master.
- [ ] Strokes expanded to fills (unless a deliberately stroke-based variant).
- [ ] Shapes merged/compounded where they should be one shape (no hairline gaps or overlaps that show as seams).
- [ ] Artboard/viewBox tight to the mark with consistent padding; mark centred optically.
- [ ] Colours are exact brand values; one-colour and reversed versions provided.
- [ ] File set matches the agreed deliverables (see `presentation-delivery.md`).
- [ ] Working files archived; versions named clearly.

## 8. Working habits

- Carry a sketchbook (or a notes file): ideas arrive at odd moments and slip away quickly. Photograph interesting
  shapes, architecture, and the negative spaces in signage.
- Experiment outside the brief: work in the opposite of your usual style (colour if you work in black and white,
  lines if you work in solids, messy if you work clean). Accidents — a slipped stroke, a mis-set layer — can be
  exactly what a project was missing.
- Treat client feedback as valuable information, not an attack; clients usually know their business better than
  you. Don't fall in love with your creations, and don't demonise clients.
- Keep your process, not your mood, in charge: rested eyes, side-by-side comparisons, and returning to the brief.
