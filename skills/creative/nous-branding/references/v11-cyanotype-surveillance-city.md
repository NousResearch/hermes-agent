# v11 Lane 4 — Cyanotype Surveillance City

## Purpose

**Cyanotype Surveillance City** is for photographic urban anomaly / evidence-still graphics: foggy city, blue monochrome, surveillance archive, blown lights, CRT/object silhouette, Polaroid or film-border framing.

Use when the brand image should feel like a recovered analog surveillance photograph, not a cinematic cyberpunk illustration.

## Primary References

| Role | File | Characteristics |
|---|---|---|
| Primary lane DNA | `Computersaysno-reference .png` | blue urban/CRT anomaly, photographic evidence mood |
| Registration support | `style-blue-registration-character.jpg` | deep blue print, orange marks, scratches |
| Xerox support | `style-cyan-xerox-poster.jpg` | scanlines, thresholding, pale cyan stock |

## Core Visual Cues

- Foggy urban night, alley, rooftop, skyline, surveillance still
- Blue monochrome / cyanotype photographic look
- CRT-like object, signal box, or urban anomaly as subject
- Polaroid/film border, evidence label, frame numbers, accession marks
- `NOUS` as stamped evidence label or archive strip
- Blown-out lights, washed contrast, scanner dust, paper fibers, halftone grain

## Palette

- **Prussian Blue** — `#082B4C`
- **Cyanotype Blue** — `#155D8D`
- **Ice Cyan Paper** — `#CFE8EF`
- **Xerox Black** — `#071018`
- **Fog Gray** — `#A8B7BA`
- Accent: **Evidence Orange** — `#D66A2C`

## Reference Ordering

```bash
-i Computersaysno-reference\ .png style-blue-registration-character.jpg style-cyan-xerox-poster.jpg
```

`Computersaysno-reference .png` controls photographic/evidence mood. Print refs add degradation and registration.

## Prompt Block

```text
STYLE TARGET: Cyanotype Surveillance City — a recovered blue-monochrome analog surveillance photograph / evidence still, not glossy cyberpunk and not a clean poster.
Use foggy urban night, rooftop/alley/skylines, blown-out lights, washed CRT contrast, Polaroid or film-border framing, evidence strips, frame numbers, accession marks, and a small anomalous object or signal source.
Make `NOUS` appear as a stamped evidence label, archive strip, frame mark, or printed border text inside the artifact.
Use Prussian/cyanotype blue, ice cyan, fog gray, xerox black, one small orange registration/evidence accent, halftone grain, scanner dust, paper fibers, edge wear, photocopy breakup, localized exposure loss, and slight plate misregistration.
Avoid neon cyberpunk cityscapes, glossy UI, cinematic concept art, clean vector posters, rainbow palettes, and floating logo overlays.
```

## Prompt Templates

### Urban Evidence Still

```text
Original Nous Research cyanotype surveillance city image: foggy urban rooftop at night, distant lights blown through haze, small CRT-like signal box or anomaly in the foreground, Polaroid border and stamped `NOUS` evidence label.
[Cyanotype Surveillance City Prompt Block]
--ar 4:5
```

### Alley Signal Archive

```text
Recovered analog surveillance photograph of a narrow blue urban alley containing a strange signal cabinet. `NOUS` appears on the evidence strip with frame numbers and accession marks. Washed cyanotype, film border, scanner dust, xerox decay.
[Cyanotype Surveillance City Prompt Block]
--ar 4:5
```

## Post-Processing

Use `--mode imprint --intensity 0.55-0.65`. Keep contrast washed but preserve the evidence-label hierarchy.

## QA Checklist

Passes if it reads as blue analog surveillance/evidence photography, has an urban anomaly, has stamped `NOUS`, and avoids glossy neon cyberpunk.

## Proof

- `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-v11-cyanotype-surveillance-city-codex-final.png`
