---
name: risomorphism-1911
description: "Use when building high-fidelity ASCII outputs from images or video, especially when you need named presets, glyph diagnostics, preview rendering, or Herm eikon generation. Risomorphism 1911 — Ousia Research aesthetic pipeline."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ascii, eikon, chafa, braille, preview, diagnostics, terminal-art, video]
    related_skills: [ascii-art, ascii-video, herm-tui-skin-development]
---

# Risomorphism 1911

## Overview

This skill is the operational foundation for high-quality ASCII rendering work: still images, dense text-art outputs, preview renders, and animated Herm eikons.

It exists to prevent two common failures:
- treating ASCII generation like a one-shot toy conversion instead of a quality-controlled pipeline
- shipping dense but unreadable output because glyph selection, scaling, and preview QA were not handled deliberately

The companion standalone package should remain the productized public surface. This skill is the in-repo operator manual for Hermes.

## When to Use

Use this skill when you need to:
- convert images into named ASCII styles with reproducible settings
- generate or inspect Herm `.eikon` files
- choose between stroke-based ASCII, dense D30-style output, or Braille-detail output
- render high-resolution PNG/GIF previews for visual QA
- diagnose whether ASCII output is crisp, legible, and structurally sound
- package a repeatable ASCII workflow for reuse or publication

Do not use this skill for:
- simple pyfiglet/cowsay banner generation only
- generic web mockups that do not involve ASCII rendering quality
- cases where no preview or quality verification is needed

## Canonical Presets

### `stroke-clarity`
- Purpose: readable stroke-first ASCII with strong silhouette preservation
- Backend: `chafa --format=symbols`
- Baseline source scale: `48x48` before `48x24` output
- Best for: readable portraits, simple scenes, legibility-first demos

### `d30-dense`
- Purpose: heavier, denser premium-terminal texture
- Backend: `ascii-image-converter` or equivalent dense palette workflow
- Baseline source scale: integer-multiple high-res intermediary before downsample
- Best for: cybernetic density, premium HUD feeling, showcase texture boards

### `braille-detail`
- Purpose: maximize effective detail within a fixed cell budget
- Backend: Braille block mapping / dithering
- Baseline effective detail target: `96x48` inside a `48x24` character grid
- Best for: high-detail portraits or scenes where dot-matrix aesthetics are acceptable

### `eikon-motion`
- Purpose: preserve portrait motion richness in Herm-style animated avatars
- Backend: video → frame extraction → `chafa --format=symbols` → `.eikon`
- Baseline source scale: `384x384`
- Best for: subtle motion clips and multi-state eikons

## Core Workflow

1. Choose the preset based on legibility, density, and desired aesthetic.
2. Produce text output with reproducible dimensions.
3. Render a high-resolution preview PNG/GIF.
4. Run diagnostics before declaring success.
5. For Herm assets, validate `.eikon` state counts and motion quality.
6. Save both the raw text artifact and the visual proof asset.

## Diagnostics Standard

Check at minimum:
- dimensions are exactly correct
- unique glyph count is within the expected band for the preset
- heavy-glyph presence is sufficient for stroke-based modes
- low-information clutter does not dominate
- motion presets show meaningful frame-to-frame variance
- output is visually recognizable in the rendered preview

Use `scripts/diagnose-glyph-quality.py` for a fast inspection pass.

## Common Pitfalls

1. **Oversized source without perceptual benefit**
   Bigger inputs do not automatically yield clearer ASCII. Some pipelines over-average and become mush.

2. **Treating density as quality**
   Dense unreadable output is failure, not sophistication.

3. **Skipping preview rendering**
   Raw text alone is not enough for QA. Render a PNG/GIF and inspect it.

4. **Mixing preset identities**
   Stroke-based, D30-dense, and Braille-detail should remain distinct named modes with clear trade-offs.

5. **Shipping historical experiments as canonical defaults**
   Archive experiments can inform the package, but they should not define the public interface.

## Repo Integration

### Standalone package target
- `/Users/johann/projects/ascii-art-pipeline/`

### Hermes in-repo skill target
- `skills/creative/risomorphism-1911/`

### Docs generation
After changing this skill, regenerate docs from the Hermes repo root:

```bash
python website/scripts/generate-skill-docs.py
```

## Verification Checklist

- [ ] Preset name chosen deliberately
- [ ] Output dimensions validated
- [ ] Preview render generated
- [ ] Diagnostics run and reviewed
- [ ] Example saved with both text and visual artifacts
- [ ] Herm docs regenerated if the bundled skill changed
