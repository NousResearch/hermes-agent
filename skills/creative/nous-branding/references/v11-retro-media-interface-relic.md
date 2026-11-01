# v11 Lane 1 — Retro Media Interface Relic

## Purpose

**Retro Media Interface Relic** is the first expanded v11 candidate lane. Use it when the image should feel like a recovered software/media artifact: an old desktop player, control panel, terminal window, sampler, or broadcast utility that has been photocopied, scanned, reprinted, and absorbed into the Nous visual system.

This lane is for software/product/agent imagery where the **interface itself is the artifact**.

It is not generic cyberpunk UI and not a clean app mockup. It should feel like a degraded executable relic.

---

## Primary Reference

Canonical local directory: `~/Desktop/Nous-branding-refs/`

| Role | File | Characteristics |
|---|---|---|
| Primary lane DNA | `Hermes-player-reference.png` | old media-player/window chrome, playback controls, software nostalgia, monochrome collage, degraded photocopy texture |
| Print discipline | `style-cyan-xerox-poster.jpg` | pale cyan xerox paper, scanlines, thresholding, block wordmark |
| Technical/manual texture | `style-rough-print.jpg` | aged paper, scuffed ink, manual-cover distress |
| Optional event/product semantics | `Hermes-update-reference .png` | release/update language, version hierarchy, product announcement structure |

---

## Core Visual Cues

- Old desktop/media-player interface chrome
- Playback buttons, scrub bars, waveform strips, tiny menus, window title bars
- Pixelated iconography, monochrome collage, utility-panel layout
- `NOUS` as window title, player label, toolbar text, tape/file name, or stamped evidence label
- Washed CRT contrast, photocopied UI screenshot, scanner haze
- Halftone, dust, edge wear, paper fibers, ink dropout
- Constrained palette: mostly blue-gray/black/cream with one small signal color
- Interface screenshot treated as a **printed object**, not a live glossy UI

---

## Palette

Primary:
- **Window Chrome Gray** — `#B8C2C7`
- **CRT Blue Black** — `#061A24`
- **Washed Cyan Paper** — `#DCEAF0`
- **Xerox Black** — `#020403`
- **Aged UI Cream** — `#E8E0D4`

Signal accents:
- **Playback Red** — `#D44A2F`
- **Signal Green** — `#49F45D`
- **Electric UI Blue** — `#1688FF`

Use no more than one signal accent unless the user explicitly asks for a louder composition.

---

## Reference Ordering

Use 2-3 refs. The media-interface reference must come first.

```bash
# Standard retro media relic
-i Hermes-player-reference.png style-cyan-xerox-poster.jpg style-rough-print.jpg

# Release / product media relic
-i Hermes-player-reference.png Hermes-update-reference\ .png style-rough-print.jpg

# More aggressive xerox/software-zine treatment
-i Hermes-player-reference.png style-cyan-xerox-poster.jpg style-rough-grain.jpg
```

Rule: `Hermes-player-reference.png` controls the interface grammar. Print refs should add degradation only; they should not turn the output into a generic poster.

---

## Prompt Block

Append this block to Retro Media Interface Relic prompts:

```text
STYLE TARGET: Retro Media Interface Relic — a degraded old desktop/media-player UI artifact, not a clean app mockup and not generic cyberpunk.
The interface itself is the subject: window chrome, title bars, playback controls, scrub bars, waveform strips, tiny menus, utility buttons, file/tape labels.
Make it feel like a recovered software artifact that was screenshotted, photocopied, faxed, scanned, and reprinted on paper.
Use constrained palette: CRT blue-black, washed cyan/cream paper, gray UI chrome, xerox black, plus one small signal accent.
NOUS should appear as a window title, media-player label, file name, toolbar stamp, or evidence label printed into the artifact.
Apply visible halftone, CRT scan haze, paper grain, scanner dust, ink dropout, xerox thresholding, slight RGB/plate misregistration, worn edges, localized exposure loss, asymmetric stains, warped window alignment, and damaged title-bar labels.
The output should feel like found media, not an intentionally pristine UI composition; some icons, borders, waveform grids, and labels should be partially degraded or uneven.
Avoid glossy futuristic UI, smooth glassmorphism, modern SaaS dashboard polish, neon cyberpunk panels, perfect clean typography, over-legible modern controls, and random fake unreadable clutter.
Borrow only interface grammar, print treatment, palette, and mood from references; do not copy the exact reference layout.
```

---

## Prompt Templates

### Product Release Media Relic

```text
Original Nous Research product-release graphic as a recovered retro media-player window. The central interface shows a fictional media track titled `[release name]`, with playback controls, waveform strip, scrub bar, menu chrome, tiny utility buttons, and a window title reading `NOUS PLAYER`. Version number `[version]` appears as a stamped file label. The whole image looks like an old software screenshot photocopied onto washed cyan paper.
[Retro Media Interface Relic Prompt Block]
--ar 16:9
```

### Agent Control Panel

```text
Original Nous agent-control panel rendered as a degraded desktop utility window. Multiple small panes: status meter, tape deck controls, waveform strip, tiny terminal readout, and one large central preview pane with an abstract agent glyph. `NOUS` appears as the title-bar label and as a stamped toolbar mark. Photocopied UI screenshot, CRT scan haze, paper grain, worn edges.
[Retro Media Interface Relic Prompt Block]
--ar 4:5
```

### Broadcast Player Poster

```text
Original broadcast artifact poster for Nous Research: a large retro media player window floating on dark CRT blue-black, playback controls at bottom, rough waveform strip across the center, tiny menu labels and file metadata. One red record/playback signal light. `NOUS TRANSMISSION` appears in the window chrome, visibly printed and degraded.
[Retro Media Interface Relic Prompt Block]
--ar 1:1
```

---

## Post-Processing

Use the shared imprint postprocess pipeline:

```bash
python3 /Users/johann/.hermes/profiles/palantir/skills/creative/nous-branding/scripts/postprocess.py \
  output-raw.png output-final.png --mode imprint --intensity 0.55
```

Calibration:
- `0.45-0.55` if generated UI text or controls need to remain legible.
- `0.6-0.7` if the raw output is too glossy or modern.
- Avoid `0.8+` unless text fidelity is irrelevant; it can destroy UI labels.

If exact typography matters, manually typeset the final labels after generation. Treat most tiny UI text as texture.

---

## QA Checklist

Passes if:
- The interface/window is the main subject.
- It feels like old software/media UI, not modern SaaS.
- The output is physically degraded/printed.
- `NOUS` is integrated as a title/label/stamp, not floating overlay text.
- Palette stays constrained.

Fails if:
- It becomes a glossy cyberpunk HUD.
- It becomes a generic poster with a small UI element.
- It uses glassmorphism/modern dashboard styling.
- It has rainbow neon or too many accent colors.
- Controls/menu chrome are absent or unreadable as interface grammar.

---

## Known Defects From First Proofs

First proof passed the lane: interface was the main subject, palette was constrained, `NOUS` integrated as title/label, and the artifact read as physically printed/xeroxed.

Remaining defects to correct in future generations:
- UI/text was still too legible and intentionally designed; push toward found media.
- Degradation was too uniform; request localized exposure loss, stains, blur, dropouts, and asymmetric wear.
- Icons, waveform grids, borders, and labels remained too crisp; prompt for warped window alignment and damaged title-bar labels.
- Stronger post-processing at `--intensity 0.65` improved the relic feel, but raw generation should carry more uneven damage before postprocess.
