# Design language — what makes halftone read as *print* instead of a filter

Distilled from a study of four reference artefacts (a macro photograph of a CMYK rosette, a
1944 colour photolithographic poster, a c.1905 song-sheet halftone portrait, and modern riso
prints) plus the live renders in `templates/starter.html`.

---

## 0. Reference study — what to look at before you start

Four artefact types cover almost everything you need. Look at them at **100% crop**, not as
thumbnails; the dot structure is the whole lesson.

| look at | what it teaches |
|---|---|
| **Macro photo of a printed screen** (a CMYK rosette under magnification) | Dots are perfect circles of *different* diameters on a constant pitch, in four rotated grids. White paper shows between them everywhere except near solid. This kills the "dots at 50% opacity" instinct. |
| **A 1940s colour lithograph poster** | The actual composition grammar: flat spot-colour fields, halftone used *only* inside photographic panels, heavy display type, hairline rules, registration marks. Note how much of the page has no dots at all. |
| **A c.1900 newsprint / song-sheet portrait** | One black plate, coarse screen, brutal highlight clipping, and how much of the face is pure paper. Tonal discipline beats resolution. |
| **A modern riso print** | Two inks, visible misregistration of 1–3 mm at print size (≈1–3 px on screen), overprint darkening into a third colour, deliberate paper tint. |

Tooling: `web_search` for candidates, then download and inspect with `vision_analyze`, asking
specifically about *dot geometry, angle, spacing vs diameter, plate count, and paper colour*.
Reverse-engineering those four numbers is faster than any amount of parameter guessing.

## 1. The governing rule

**Tone lives in dot area, never in opacity or colour.** The moment a "halftone" layer uses a
50%-opacity dot pattern to fake midtone, it stops reading as printing and starts reading as a
texture overlay. Real screens are binary ink: the dot either exists at a given size or it does
not. Every decision below follows from that.

Corbollaries that repeat in every authentic reference:

- Highlights are **paper**, not pale dots. The lightest areas lose the dots entirely.
- Shadows are **solid ink** with clean merged edges, not dark-grey dots.
- Dot spacing is **constant**; only diameter changes. (Uniform spacing, varying diameter =
  amplitude modulation. This is the textural signature people recognise.)
- The screen has an **angle**. A 0°/90° screen on a pixel grid looks like a UI texture, not a
  print.

---

## 2. Dot scale — pick the cell from the display size, not from taste

Legibility of a halftone subject is a ratio: **cell ÷ display width**.

| intent | ratio | 320 px card | 900 px hero | 2400 px poster |
|---|---|---|---|---|
| photographic, legible | 1/70 … 1/90 | 3.5–4.5 px | 10–13 px | 27–34 px |
| photographic, graphic/coarse | 1/35 … 1/50 | 6–9 px | 18–26 px | 48–68 px |
| texture only (behind text) | 1/150+ | 2 px | 5–6 px | 12–16 px |
| dot as poster element | 1/15 … 1/25 | — | 36–60 px | 96–160 px |

Bands to avoid: **1/100 – 1/150**. Dots are too small to read as dots and too large to disappear —
the result is grey mush that shimmers on scroll. Either commit to photography (≤1/90) or commit
to texture (≥1/150).

Also: a 4 px cell is the practical floor for a streamable canvas render. Below that you are
fighting sub-pixel antialiasing more than you are designing.

---

## 3. Screen angles

| situation | angle | why |
|---|---|---|
| single black screen | **45°** | least visible to the eye; the historic default |
| wanting visible weave | 15° or 75° | reads as "deliberately printed" |
| two plates | **30–45° apart** | equal angles produce clumping, not overprint |
| four-colour process | **15 / 75 / 0 / 45** | turns interference into a rosette |
| line screen | **30–40°** | engraving; 0°/90° reads as scan lines |

Two plates at the *same* angle is the most common mistake in riso-style work: the second plate
simply duplicates the first and the image goes muddy at the midtones.

---

## 4. Inks and paper (copy these until you can justify your own)

Never `#000` on `#fff`. The fastest way to make a halftone look like a CSS filter is pure black
on pure white.

| set | ink(s) | paper | feel |
|---|---|---|---|
| Newsprint | `#171717` | `#f2eee2` | archive, editorial, cheap-and-proud |
| Riso duo | `#0f4bd8` + `#ff5f8f` | `#fdf3e3` | studio print, cultural poster |
| Pop | `#e8175d` | `#ffe95c` | comic / Lichtenstein / protest |
| Acid | `#111111` | `#c8ff3d` | club flyer, terminal, rave |
| Process | `#00aeef` `#ec008c` `#ffe600` `#1a1a1a` | `#f8f4ec` | proof, press check |
| Quiet | `#1d1d1d` | `#ffffff` | UI texture that stays out of the way |

Rules:

- **2–3 inks maximum.** The whole charm of riso is the constraint. Four is a process proof, not
  a palette.
- Paper is never pure white: `#f2eee2`–`#fdf3e3` is the working band. Warm paper + black ink is
  already 60% of the print illusion.
- Ink is never pure black: `#171717`–`#1d1d1d`. Add 4% warble and the dots stop looking like
  software.
- Overprint darkens: `mix-blend-mode: multiply`, always. Screen and normal blending are what
  make fake riso look like Pantone swatches glued together.

---

## 5. Composition recipes that actually work

**A — Editorial plate.** Halftone lives inside a bordered figure; everything around it is flat
colour, hairline rules, mono captions. This is the vintage-poster structure: photographic panels
inside flat spot-colour fields. Do not halftone the whole page.

```
[ hairline rule ───────────────────────── ]
[ flat colour field │ halftone plate 4:5 ]
[ mono caption ····· plate no. · screen  ]
```

**B — Hero band.** A full-bleed horizontal strip of dots at 10–25% of viewport height, with a
tone ramp or radial falloff. Type sits over it in a solid ink block or on a paper-coloured
panel. Keep the field's contrast below ~40% where text overlaps, or the text vibrates.

**C — Type-led.** One enormous word, masked into dots, everything else quiet: paper, a rule,
two lines of small mono. The word carries the page. Works at 5–6 px pitch on 100–200 px type and
nowhere else.

**D — Product card.** Photo halftoned at a coarse cell (1/35–1/50 of card width) so it reads as
a *print of a product* rather than a product shot, with flat colour behind. This is the strongest
use in e-commerce: it turns a stock photo into art direction.

**E — Specimen grid.** Same source, six cell sizes in a row, mono captions. Cheap, credible, and
it teaches the viewer what the effect is. Very good for a portfolio or a section divider.

**F — Sticker pack.** Halftone subjects cut to their silhouette on transparent stock, laid on a
coloured surface with a small tilt and a soft shadow. Two or three is a pack; six is a shop. Keep
the ink count low (one or two per sticker) and vary the stock — one paper-filled, one clear vinyl
— so the set reads as a series rather than a mistake.

---

## 6. Typography pairing

The references all pair a heavy display face with a mechanical caption face, and nothing else.

- **Display:** Archivo Black, Anton, Druk, Helvetica Black, Impact. Tight tracking
  (`-0.03em` to `-0.05em`), line-height `0.82–0.9`, uppercase.
- **Body:** a neutral grotesque (Archivo, Inter, Helvetica) at 14–17 px, `1.45–1.55` leading.
- **Caption / metadata:** mono (JetBrains Mono, IBM Plex Mono) at 9.5–11 px, `0.14–0.2em`
  tracking, uppercase — plate numbers, screen angles, cell sizes. This is what makes the page
  read as a *print studio* rather than a design experiment.

Never set body copy in dots. Never set halftone type below 40 px.

---

## 7. Anti-patterns (each one is instantly recognisable as fake)

1. **Pure black on pure white.** Reads as a CSS filter, not ink.
2. **Uniform dot size with opacity standing in for tone.** It is a texture, not a halftone —
   fine as a background, fatal as an image treatment.
3. **Dots at 0°.** No print artefact has an axis-aligned screen; it reads as a UI grid.
4. **White pinholes in the shadows.** Overlapping circles never quite fill the cell; flood at
   ~95% coverage.
5. **Perfect alignment between plates.** Zero misregistration is a tell. 1–3 px is the charm.
6. **Rainbow dots.** Colour-per-dot turns the image into confetti. One ink per plate.
7. **Halftoning everything.** If the whole layout is dots there is no figure/ground and the eye
   has nowhere to rest. One dotted element per viewport.
8. **Moiré on screen edges.** Two plates at similar angles, or non-integer cell sizes that beat
   against DPR. Check at 1×, 1.5× and 2× DPR.
9. **Retina mush.** Rendering the canvas at CSS pixel size and letting the browser upscale on a
   2× display. Always size the canvas by `dpr`.

---

## 8. Quick self-critique before shipping

Look at the render and answer:

- Squint: does the subject survive as a silhouette? If not, the cell is too large.
- Stand back: do the dots read as a *grey field* with structure? If they look like a checkerboard
  at a distance, the angle or the gain is wrong.
- Zoom to 100%: are the dots crisp circles with clean edges, and do the shadows go solid? If
  there are pinholes, flood them.
- Is there paper visible? A halftone with no clean paper is just noise.
- Would this survive being photocopied twice? Real print references always would.
