# v11 Lane 6 — Veiled Classical Artifact

## Purpose

**Veiled Classical Artifact** is for intimate, fragmentary, archaeological/classical images: close-cropped marble/statue faces partly hidden by veil, mesh, xerox scrim, or archival fabric.

It is artifact-scanned and museum-fragment oriented, not radiant celestial mysticism or glossy classical rendering.

## Primary References

| Role | File | Characteristics |
|---|---|---|
| Primary lane DNA | `image 3.PNG` | veiled classical close crop, cold artifact mood |
| Classical support | `Hermes-classical-reference .png` | statue/classical/hermetic language |
| Registration support | `style-blue-registration-character.jpg` | blue-green print, scratches, orange marks |
| Print discipline | `style-rough-print.jpg` | aged tactile paper and scuffed ink |

## Core Visual Cues

- Close-cropped classical marble/statue face or bust fragment
- Partial concealment by veil, mesh, scrim, fabric, or xerox artifacting
- Museum-fragment scan, archaeological record, occult archival specimen
- Cold green-blue monochrome, halftone grain, damaged photographic surface
- `NOUS` as accession label, stamped border, archival mark, or specimen tag

## Palette

- **Artifact Blue-Green** — `#1C5F67`
- **Cold Stone Gray** — `#9BA8A4`
- **Deep Archive Blue** — `#061F2B`
- **Bone Paper** — `#D8D7C8`
- Accent: **Registration Orange** — `#C65B2D`

## Reference Ordering

```bash
-i image\ 3.PNG Hermes-classical-reference\ .png style-blue-registration-character.jpg style-rough-print.jpg
```

`image 3.PNG` controls the close-cropped veiled artifact mood. Support refs add classical subject and print decay.

## Prompt Block

```text
STYLE TARGET: Veiled Classical Artifact — an intimate museum-fragment / archaeological scan of a close-cropped classical statue face partly concealed by veil, mesh, scrim, or xerox damage, not a luminous fantasy portrait.
Use weathered marble, cold green-blue monochrome, damaged photographic surface, archival specimen border, accession marks, halftone grain, scanner dust, paper fibers, scratches, blur, exposure loss, and plate misregistration.
Make `NOUS` appear as a stamped accession strip, specimen label, archival mark, or printed border text.
Avoid modern fashion portraits, glossy statue renders, angelic/celestial glow, clean museum posters, rainbow palettes, and floating logos.
```

## Prompt Templates

### Museum Fragment Scan

```text
Original Nous Research veiled classical artifact: close-cropped weathered marble face partially hidden by xerox scrim and archival mesh. Small stamped `NOUS` accession label at edge, cold blue-green photocopy texture.
[Veiled Classical Artifact Prompt Block]
--ar 4:5
```

### Occult Archive Specimen

```text
Recovered occult archaeological specimen image: fragmented classical bust behind translucent veil, damaged surface, halftone grain, scanner dust, specimen border, `NOUS` archival stamp.
[Veiled Classical Artifact Prompt Block]
--ar 4:5
```

## Post-Processing

Use `--mode imprint --intensity 0.55-0.65`. Avoid destroying the face silhouette; text can degrade.

## QA Checklist

Passes if it feels like a veiled classical artifact scan with embedded `NOUS`, cold monochrome, and physical damage — not fantasy or fashion.

## Proof

- `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-v11-veiled-classical-artifact-codex-final.png`
