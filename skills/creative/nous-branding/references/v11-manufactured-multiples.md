# v11 Lane 8 — Manufactured Multiples

## Purpose

**Manufactured Multiples** is for repeated branded objects: capsules, cylinders, canisters, product modules, and ritual-industrial arrays. It frames `NOUS` as a mass-produced seal on objects that feel both commodity and archive.

Use for product-world, packaging, merch, supply-chain, or ritual-object visual metaphors.

## Primary References

| Role | File | Characteristics |
|---|---|---|
| Primary lane DNA | `image0 3.JPG` | repeated branded objects, product array rhythm |
| Industrial support | `style-industrial-duotone-grid.jpg` | blue/yellow grid, heavy halftone, industrial repetition |
| Xerox support | `style-cyan-xerox-poster.jpg` | thresholding, printed wordmark treatment |
| Grain support | `style-rough-grain.jpg` | analog screenprint damage |

## Core Visual Cues

- Rows of repeated branded capsules/cylinders/canisters/modules
- Production-line grid, factory documentation, flash-photo object array
- Repeated `NOUS` seals, serial marks, labels, stamped packaging
- Blue/yellow industrial duotone, heavy halftone, ink scuffs, product label imperfection
- Commodity object meets ritual/archive object

## Palette

- **Cobalt Industrial Blue** — `#123E8C`
- **Acid Yellow Cream** — `#E6D85A`
- **Xerox Black** — `#071017`
- **Aged Label Cream** — `#E2D7B8`
- Accent: **Warning Red/Orange** — `#C94D29`

## Reference Ordering

```bash
-i image0\ 3.JPG style-industrial-duotone-grid.jpg style-cyan-xerox-poster.jpg style-rough-grain.jpg
```

`image0 3.JPG` controls repeated-object grammar. Industrial print refs add palette/repetition discipline.

## Prompt Block

```text
STYLE TARGET: Manufactured Multiples — a repeated industrial product array of branded capsules/cylinders/canisters/modules, not a glossy ecommerce product render.
Use production-line rhythm, factory documentation, flash-photo object arrays, repeated `NOUS` seals, serial marks, stamped labels, worn packaging, and commodity-object-meets-ritual-object mood.
Use cobalt/navy blue, acid yellow/cream, xerox black, one warning red/orange accent, heavy halftone, screenprint grain, paper fibers, dusty flash, xerox edges, ink scuffs, imperfect labels, and misregistration.
Avoid luxury packaging ads, pharmaceutical cleanliness, clean 3D mockups, sterile ecommerce layouts, rainbow palettes, and floating logo overlays.
```

## Prompt Templates

### Production Array

```text
Original Nous Research manufactured multiples image: rows of repeated branded capsules or cylinders in an industrial grid, each object stamped with a small `NOUS` seal and serial marks. Blue/yellow halftone print, dusty flash-photo product archive.
[Manufactured Multiples Prompt Block]
--ar 16:9
```

### Ritual Commodity Modules

```text
Industrial product array of uncanny ritual modules for Nous Research. Repeated canisters, worn labels, NOUS seals, serial numbers as texture, factory documentation, heavy screenprint grain and xerox scuffs.
[Manufactured Multiples Prompt Block]
--ar 16:9
```

## Post-Processing

Use `--mode imprint --intensity 0.6-0.7`. Higher intensity helps break glossy product cleanliness.

## QA Checklist

Passes if repetition/object array is dominant, labels feel printed, palette is industrial duotone, and the result avoids ecommerce/3D cleanliness.

## Proof

- `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-v11-manufactured-multiples-codex-final.png`
