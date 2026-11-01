# v8 Pipeline Validation Notes

## What Changed (v7 → v8)

**Problem:** Generated images were "too clean" — smooth digital illustrations lacking the rough, analog, print-like texture of actual Nous/Hermes branding.

**Root cause:** GPT-image-2 produces clean digital art by default. A single brand reference was not enough to teach it the surface quality needed.

**Solution validated in v8:**
1. Use 3 reference images per generation: 2 rough texture refs + 1 brand ref
2. Rough texture refs go FIRST in `-i` flag order (primary style drivers)
3. Texture refs teach surface quality; brand ref teaches color/lighting/composition
4. Post-processing at intensity 0.7-0.8 (not 0.5)

## Texture Reference Files

| File | Dimensions | Key Characteristics |
|---|---|---|
| `style-rough-grain.jpg` | 800×800, 111KB | Texture variance ~1060, strong halftone, contrasty, visible grain |
| `style-rough-print.jpg` | 928×1232, 189KB | Warmer, darker, texture var ~1040, fine print detail |

Located at: `~/Desktop/Nous-branding-refs/`

## Measured Texture Comparison

| Image | Lum Std | Texture Var | Edge Density |
|---|---|---|---|
| Rough ref 1 | 112.0 | 1059 | 14.8 |
| Rough ref 2 | 82.6 | 1039 | 19.1 |
| Raw GPT-image-2 output | 52.4 | 418 | 0.05 |
| v8 generated + post-process (0.7) | 41-56 | 594-875 | 41-59 |

Raw GPT output has ~2.5× LESS texture variance than refs. Post-processing recovers some but not all of the gap. Texture refs in the prompt help the raw output start closer.

## Prompt Texture Language That Works

These phrases consistently produce rougher results:
- "HALFTONE DOT PATTERN, like risograph or screen printing"
- "INK BLEED at edges — colors slightly smeary, not crisp digital lines"
- "PAPER / CANVAS FIBER TEXTURE visible in flat areas"
- "SLIGHTLY DISTRESSED — not polished, not perfect"
- "WARM ANALOG FEEL — ink on paper, not pixels on screen"

Most important single instruction:
- "DO NOT MAKE IT LOOK LIKE A SMOOTH CLEAN AI-GENERATED ILLUSTRATION"

## Keep CLI Pitfall

Codex sessions hang on `keep now` when SQLite is corrupted (`unable to open database file`). The IMAGE STILL GENERATES — the hang is in post-generation reflection only. Workaround: collect PNG manually from `~/.codex/generated_images/<session-id>/ig_*.png`.

## Intensity Calibration

| Intensity | Effect | Use When |
|---|---|---|
| 0.3 | Subtle | Brand is already somewhat textured |
| 0.5 | Standard | Early passes, iterating |
| 0.7 | Strong | v8+ pipeline, confirmed on-brand |
| 0.8 | Heavy | Raw output very clean, needs maximum texture |
