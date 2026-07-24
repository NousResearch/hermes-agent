# v11 Lane 9 — Motorsport Sponsor Field Research

## Purpose

**Motorsport Sponsor Field Research** is a kinetic real-world sponsorship lane: rally/motorsport field operations, dust, speed, credible decal placement, number plates, pit boards, crew/field-site language, and `NOUS` as team/sponsor identity.

The key correction is **sponsor energy**. This is not merely a car with a logo; it should feel like NOUS backs a credible field-research rally team.

## Primary References

| Role | File | Characteristics |
|---|---|---|
| Primary lane DNA | `Nous-car-reference.png` | car/field atmosphere, real-world placement seed |
| Industrial support | `style-industrial-duotone-grid.jpg` | sponsor poster grit, duotone field discipline |
| Grain support | `style-rough-grain.jpg` | analog film/screenprint damage |

## Core Visual Cues

- Rally/motorsport vehicle in motion, dust, technical course, field-site atmosphere
- Credible sponsor livery: doors, hood, quarter panels, vinyl seams, decals
- Number plate, auxiliary logo stack, safety/inspection stickers, pit boards, trackside banners
- `NOUS` as team/sponsor mark integrated into livery/signage
- Analog motorsport photography, field documentation, screenprinted sponsor poster finish

## Palette

- **Dust Cream** — `#D8C6A0`
- **Asphalt Black** — `#121417`
- **Faded Cobalt** — `#1D4F8C`
- **Sun-Bleached Tan** — `#B9905D`
- Accent: **Signal Red-Orange** — `#D44A2F`

## Reference Ordering

```bash
-i Nous-car-reference.png style-industrial-duotone-grid.jpg style-rough-grain.jpg
```

`Nous-car-reference.png` controls vehicle/field mood. Print refs add sponsor-poster grit.

## Prompt Block

```text
STYLE TARGET: Motorsport Sponsor Field Research — credible rally/motorsport sponsor-livery field photography, not a luxury car ad and not a random car with a floating logo.
Use low-angle 3/4 action, dust plume, scuffed bodywork, mud, motion blur, technical course, pit board or trackside vinyl banner, number plate, auxiliary sponsor-logo stack, safety stickers, inspection decals, quarter-panel marks, and realistic vinyl livery seams.
Make `NOUS` appear as the backer/team sponsor identity on doors, hood, signage, or pit-board surfaces, distressed like real vinyl/ink.
Use dusty cream, asphalt black, faded navy/cobalt, sun-bleached tan, one red-orange signal accent, analog film grain, halftone breakup, paper fibers, ink scuffs, scanline haze, and registration wobble.
Avoid showroom lighting, luxury advertising, clean 3D renders, glossy cars, floating logo overlays, neon cyberpunk, and rainbow palettes.
```

## Prompt Templates

### Rally Field Team

```text
Original Nous Research motorsport sponsor field-research image: rally car in motion on dusty technical course, large distressed `NOUS` door/hood livery, number plate, sponsor stack, pit-board banner, analog field photography printed as sponsor poster.
[Motorsport Sponsor Field Research Prompt Block]
--ar 16:9
```

### Expedition Pit Board

```text
Documentary motorsport field scene for a Nous-backed research rally team. Scuffed vehicle, mud, decals, number plate, crew/pit-board signage with `NOUS`, dust and motion blur, screenprinted analog sponsor-poster finish.
[Motorsport Sponsor Field Research Prompt Block]
--ar 16:9
```

## Post-Processing

Use `--mode imprint --intensity 0.55-0.65`; keep sponsor placement visible while degrading glossy surfaces.

## QA Checklist

Passes if `NOUS` reads as credible sponsor/team identity, livery placement is plausible, dust/field motion is present, and it avoids luxury/showroom aesthetics.

## Proof

- `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-v11-motorsport-sponsor-field-research-codex-final.png`
