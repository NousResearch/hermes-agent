# v11 Lane 5 — Network Diagram Primitive

## Purpose

**Network Diagram Primitive** is for low-resolution civic/infrastructure network graphics: white pixel glyphs, dense connection lattices, agent-society maps, and screenprinted public-system diagrams on electric blue fields.

It is diagrammatic and primitive, not cinematic human-AI mysticism or polished SaaS visualization.

## Primary References

| Role | File | Characteristics |
|---|---|---|
| Primary lane DNA | `Hermes-together-reference.png` | community/network semantics, connected figures |
| Community support | `Nous-Discord-reference .png` | dense social signal energy |
| Mind/network support | `Hermes-mind-reference.png` | neural/network motifs |
| Print discipline | `style-cyan-xerox-poster.jpg` | xerox print texture and scanlines |

## Core Visual Cues

- Electric blue/deep navy field
- White/pixel human or agent glyphs, small square nodes
- Dense connection lattice / social graph / infrastructure map
- Low-res system diagram, civic network poster, public-service map
- `NOUS` as infrastructure label, legend stamp, or system-map title
- Rough halftone, scanlines, paper fibers, xerox thresholding, registration wobble

## Palette

- **Electric Civic Blue** — `#136BFF`
- **Deep Diagram Navy** — `#061A33`
- **Off-White Ink** — `#E8EEF0`
- **Cyan Paper** — `#C8E9F0`
- Accent: **Registration Orange** — `#D46A2A`

## Reference Ordering

```bash
-i Hermes-together-reference.png Nous-Discord-reference\ .png Hermes-mind-reference.png style-cyan-xerox-poster.jpg
```

The first three refs provide network/community semantics. Print ref adds xerox discipline.

## Prompt Block

```text
STYLE TARGET: Network Diagram Primitive — a low-resolution civic infrastructure / agent-society diagram on an electric blue field, not a polished infographic and not a cinematic neural network.
Use primitive white pixel-like human/agent glyphs, square nodes, dense connection lattice, system-map paths, stamped legends, crop marks, and early-computer diagram grammar.
Make `NOUS` appear as an embedded infrastructure label, map title, legend stamp, or printed systems mark.
Use electric blue, deep navy, off-white ink, sparse cyan/orange accents, screenprint halftone, scanlines, paper fibers, xerox thresholding, imperfect borders, and plate misregistration.
Avoid modern SaaS network visualizations, glossy cyberpunk HUDs, 3D glowing neurons, realistic people, rainbow palettes, and clean corporate diagrams.
```

## Prompt Templates

### Agent Society Map

```text
Original Nous Research agent society map: primitive white glyphs and square nodes connected by dense lattice lines on electric blue field. `NOUS` appears as central infrastructure label and legend stamp. Screenprinted civic network poster.
[Network Diagram Primitive Prompt Block]
--ar 16:9
```

### Public Systems Poster

```text
Low-res public-service infrastructure diagram for Nous Research. Pixel figures, node clusters, route lines, stamped legend, rough halftone, cyan xerox scanlines, one orange registration mark.
[Network Diagram Primitive Prompt Block]
--ar 16:9
```

## Post-Processing

Use `--mode imprint --intensity 0.55-0.65`. Higher intensity helps avoid clean vector polish but may muddy fine node lines.

## QA Checklist

Passes if it reads as primitive network/system diagram, `NOUS` is embedded as label/legend, and it avoids glossy neural/SaaS aesthetics.

## Proof

- `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-v11-network-diagram-primitive-codex-final.png`
