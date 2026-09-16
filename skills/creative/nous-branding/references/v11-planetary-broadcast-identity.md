# v11 Lane 7 — Planetary Broadcast Identity

## Purpose

**Planetary Broadcast Identity** is for satellite-era public-service brand graphics: cropped planet disks, large institutional `NOUS` typography, orbital lines, telemetry marks, and cold broadcast-card framing.

It extends Nous from local artifacts into planetary-scale communications identity.

## Primary References

| Role | File | Characteristics |
|---|---|---|
| Primary lane DNA | `image0 2.JPG` | cropped globe/planet disk, institutional space-age identity |
| Network support | `Hermes-together-reference.png` | planetary-scale connectivity semantics |
| Community support | `Nous-Discord-reference .png` | signal/community density |
| Print discipline | `style-cyan-xerox-poster.jpg` | cyan xerox, scanlines, thresholding |

## Core Visual Cues

- Cropped Earth/globe/planet disk entering from one edge
- Oversized vertical/sideways `NOUS` typography integrated into border
- Satellite-era broadcast ticks, orbital lines, telemetry marks
- Cold War public-service identity card / transmission poster
- Teal/navy/ice palette, halftone globe, screenprinted broadcast texture

## Palette

- **Broadcast Navy** — `#062235`
- **Institutional Teal** — `#0E6F78`
- **Ice Cyan** — `#D7EFF2`
- **Xerox Black** — `#020B10`
- Accent: **Signal Orange** — `#D0612D`

## Reference Ordering

```bash
-i image0\ 2.JPG Hermes-together-reference.png Nous-Discord-reference\ .png style-cyan-xerox-poster.jpg
```

`image0 2.JPG` controls planetary broadcast composition. Support refs add connectivity/signal meaning.

## Prompt Block

```text
STYLE TARGET: Planetary Broadcast Identity — a satellite-era institutional communications card / Cold War broadcast poster, not a glossy space render or modern startup ad.
Use a large cropped globe/planet disk, oversized vertical or sideways `NOUS` typography integrated into the border, orbital lines, broadcast ticks, telemetry marks, public-service framing, and halftone planet texture.
Use teal/navy/ice palette, washed cyan paper, xerox black, sparse orange/cyan signal accents, scanlines, paper fibers, ink scuffs, plate wobble, imperfect registration, and screenprinted satellite-era graphic design.
Avoid glossy NASA render, photoreal planet wallpaper, clean corporate vector poster, rainbow nebula, and floating logo overlays.
```

## Prompt Templates

### Satellite Transmission Card

```text
Original Nous Research planetary broadcast identity card: cropped halftone Earth disk entering from left, oversized vertical `NOUS` border typography, telemetry marks, orbital lines, cyan xerox stock, satellite-era public-service design.
[Planetary Broadcast Identity Prompt Block]
--ar 16:9
```

### Public Communications Poster

```text
Cold-war broadcast poster for Nous Research planetary cognition infrastructure. Large planet disk, institutional `NOUS` wordmark, signal ticks, transmission grid, teal/navy/ice print palette, xerox scanlines.
[Planetary Broadcast Identity Prompt Block]
--ar 16:9
```

## Post-Processing

Use `--mode imprint --intensity 0.55-0.65`. Preserve large `NOUS` typography and planet silhouette.

## QA Checklist

Passes if it reads as satellite-era planetary broadcast identity with embedded `NOUS`, halftone globe, and constrained cold palette.

## Proof

- `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-v11-planetary-broadcast-identity-codex-final.png`
