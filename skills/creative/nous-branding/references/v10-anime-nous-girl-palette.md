# v10 Anime Nous Girl Palette — NOUS Analog Cyber-Manga

## Why This Palette Exists

This reference family captures a separate Nous visual lane from the v9 technical-imprint system. It keeps the same analog/printed discipline, but centers a **stylized anime/manga girl mascot** as the brand carrier.

The look is **not** glossy anime, kawaii pastel, or clean character art. It is:

- manga portrait language
- xerox/poster degradation
- risograph and screenprint texture
- cyber-zine framing and marginalia
- institutional `NOUS` typography
- constrained ink palettes with occasional acid signal colors

Treat this as the right lane for mascot posts, community graphics, agent/persona identity, culture/meme images, and expressive brand illustrations.

---

## Canonical Reference Files

Canonical local directory: `~/Desktop/Nous-branding-refs/`

| File | Use for | Characteristics |
|---|---|---|
| `anime-nous-acid-mascot-card.png` | strongest brand DNA / acid mascot | acid green, black ink, violet misregistration, headphones, square zine frame, dense marginalia, `NOUS` badge |
| `anime-nous-portal-silhouette.png` | restrained editorial / portal mood | muted teal-gray, black silhouette, square frame, vertical type, soft scan grain |
| `anime-nous-manga-xerox-portrait.png` | manga face treatment | high-contrast black/white manga portrait, crushed blacks, halftone, cropped `NOUS` type |
| `anime-nous-blueprint-scene.jpg` | cinematic poster / environment | electric blue negative figure, deep navy, orange technical marks, large negative space |
| `anime-nous-opart-poster.png` | bold poster / vehicle/action energy | pale ice-blue paper, black xerox ink, horizontal op-art scanlines, top `NOUS` wordmark |
| `anime-nous-future-halftone.png` | delicate fine-line / future caption | black-white linework, hair dissolving into halftone pixels, tiny typewriter captions |

---

## Style Name

**NOUS Analog Cyber-Manga**

A brand lane built from anime-girl mascot imagery, underground zine printing, cybernetic marginalia, xerox degradation, and restrained institutional typography.

---

## Color Palette

Primary inks:
- **Xerox Black** — `#020403`
- **Ice Paper** — `#EAFBFF`
- **Warm White Paper** — `#F7F5EE`

Signal inks:
- **Acid Green** — `#49F45D`
- **Blueprint Blue** — `#1688FF`
- **Signal Orange** — `#FF5A1F`
- **Registration Violet** — `#7B4DFF`

Atmospheric neutrals:
- **Deep Navy** — `#062331`
- **Fog Teal Gray** — `#5F7874`

Palette/reference board assets:
- `assets/anime-nous-girl-palette.png` — color swatches only
- `assets/anime-nous-girl-reference-palette.png` — reference contact sheet + swatches

---

## Sub-Lanes

### 1. Acid Signal
Use for high-energy mascot graphics, culture posts, badges, announcements.

Prompt cues:
- acid green xerox anime mascot portrait
- headphones or cyber accessory silhouette
- square zine frame / nested border
- distressed photocopy ink and black silhouette shapes
- violet registration error
- cybernetic marginalia, small codes, machine labels
- bold `NOUS` label/badge printed into the image

References:
- `anime-nous-acid-mascot-card.png`
- `anime-nous-manga-xerox-portrait.png`
- optional `style-cyan-xerox-poster.jpg` for additional print discipline

### 2. Portal Minimal
Use when the graphic should be quieter, more editorial, and more brand-refined.

Prompt cues:
- minimal black anime silhouette
- muted teal-gray print field
- square frame with large negative space
- vertical bold typography
- soft grain and analog scan haze
- restrained cyber editorial poster

References:
- `anime-nous-portal-silhouette.png`
- `anime-nous-future-halftone.png`
- optional `style-minimal-stipple-border.jpg`

### 3. Future Halftone
Use for delicate posters, poetic/identity graphics, and clean-but-printed manga portraits.

Prompt cues:
- black and white manga girl portrait
- fine ink linework
- hair or silhouette dissolving into halftone pixels
- minimal poster margin
- tiny typewriter caption and date/location-like microtext
- future-zine print, delicate stipple, archival restraint

References:
- `anime-nous-future-halftone.png`
- `anime-nous-manga-xerox-portrait.png`
- optional `style-rough-print.jpg`

### 4. Blueprint Scene
Use for cinematic full-poster compositions, agent/persona scenes, and sci-fi interiors.

Prompt cues:
- anime girl as small central figure in a dark environment
- cyanotype/blueprint poster treatment
- electric-blue figure against deep navy field
- orange technical registration marks
- film scan scratches, large negative space
- archival sci-fi layout, `NOUS` as subtle printed mark

References:
- `anime-nous-blueprint-scene.jpg`
- `style-blue-registration-character.jpg`
- optional `anime-nous-portal-silhouette.png`

---

## Generation Reference Ordering

Use 2-3 references. Put the anime lane refs first; put broader print/brand refs last.

Examples:

```bash
# Acid mascot card
-i anime-nous-acid-mascot-card.png anime-nous-manga-xerox-portrait.png style-cyan-xerox-poster.jpg

# Portal minimal
-i anime-nous-portal-silhouette.png anime-nous-future-halftone.png style-minimal-stipple-border.jpg

# Future halftone portrait
-i anime-nous-future-halftone.png anime-nous-manga-xerox-portrait.png style-rough-print.jpg

# Blueprint scene
-i anime-nous-blueprint-scene.jpg style-blue-registration-character.jpg anime-nous-portal-silhouette.png
```

Rule: the first reference controls the lane. Later references should add restraint, texture, or print finish — not override the anime/manga identity.

---

## v10 Prompt Block

Append this block to anime Nous girl prompts:

```text
STYLE TARGET: NOUS Analog Cyber-Manga — anime/manga mascot language fused with xerox, risograph, screenprint, and cyber-zine design.
The character is an original anonymous stylized anime/manga figure, not a real person and not copied from any reference.
Use constrained ink palette only: xerox black, ice paper/warm paper, one signal ink such as acid green, blueprint blue, signal orange, or registration violet.
Make it feel physically printed: crushed blacks, halftone dots, stipple, photocopy noise, ink dropout, scan haze, paper grain, slight RGB/plate misregistration.
Typography should be institutional and printed into the artifact: NOUS badge, cropped wordmark, vertical type, tiny machine/code captions.
Avoid glossy anime rendering, soft painterly shading, pastel kawaii colors, 3D mascot look, fantasy costume detail, perfect generated typography, and clean corporate vector art.
Borrow only palette, print treatment, composition grammar, and mood from references; do not copy exact subject, pose, or layout.
```

---

## Prompt Templates

### Acid Mascot Card

```text
Original anime/manga mascot portrait for Nous Research, anonymous stylized girl with cyber-audio accessory silhouette, centered in nested square zine frame. Acid green and xerox black, with small violet registration errors. Dense cybernetic marginalia and small machine-code labels around the border. `NOUS` as a stamped badge and top wordmark, visibly printed into the paper.
[v10 Prompt Block]
--ar 1:1
```

### Portal Minimal Poster

```text
Minimal editorial poster: anonymous anime girl silhouette inside a square frame, muted fog teal-gray paper field, black figure, large negative space, vertical `PORTAL` or `NOUS` type along the side. Soft scan grain, restrained cyber-zine mood, no glossy rendering.
[v10 Prompt Block]
--ar 1:1
```

### Future Halftone Portrait

```text
Fine-line black-and-white manga portrait for Nous Research, centered with large white margins. Hair and shadow masses dissolve into halftone pixels and stipple. Tiny typewriter caption below: `nous is for the future`. Minimal archival poster layout, delicate but visibly printed.
[v10 Prompt Block]
--ar 4:5
```

### Blueprint Scene

```text
Cinematic cyber-manga poster: anonymous anime girl as a small seated figure in a dark interior, rendered as electric-blue cyanotype/blueprint ink against deep navy. Orange technical registration marks, film scan scratches, huge negative space, subtle printed `NOUS` mark at bottom.
[v10 Prompt Block]
--ar 4:5
```

---

## Post-Processing Advice

Use the existing postprocess script after generation:

```bash
python3 /Users/johann/.hermes/profiles/palantir/skills/creative/nous-branding/scripts/postprocess.py \
  output-raw.png output-final.png --mode imprint --intensity 0.55
```

Calibration:
- `0.45-0.55` for Future Halftone / Portal Minimal so fine manga lines survive.
- `0.6-0.75` for Acid Signal / Xerox Poster when the model outputs too clean.
- `0.55-0.65` for Blueprint Scene; avoid crushing all blue interior detail.

If text matters, manually typeset final `NOUS`/caption text after generation or regenerate until the text is acceptable. Treat generated microtext as texture unless explicitly cleaned.

---

## Anti-Patterns

Avoid:
- glossy anime renderings
- soft painterly shading
- pastel kawaii palettes
- 3D mascot looks
- fantasy costumes or magical-girl excess
- over-detailed backgrounds
- clean vector corporate illustration
- perfect synthetic typography floating above the image
- literal copies of reference character, pose, or frame layout

The lane works because it feels **printed, degraded, graphic, cybernetic, and slightly institutional**.
