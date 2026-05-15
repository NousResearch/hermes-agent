---
name: ascii-art-pipeline
description: Production-grade ASCII/Braille rendering for still images, video, and animated eikons with quality gates and integer scaling
author: Ousia Research (plntrprotocol)
maintainer: Anduril
license: MIT
tags: [ascii, rendering, video, eikon, quality-gate, scaling]
version: 1.0.0
hermes:
  integration: skill
  category: creative-tooling
  maturity: production
  dependencies:
    - Pillow>=10.0
    - numpy>=1.26
    - ffmpeg (optional, for video interpolation)
  entrypoint: ascii-pipeline
---

# ASCII Art Pipeline — Hermes Skill

Deterministic, quality-gated ASCII/Braille rendering for Hermes operations. Still images, video extraction, animated eikon generation. Pure-Python backend. Zero runtime binary dependencies. Scale 1–16× from base 48×24 grid.

---

## Overview

Risomorphism-1911 is the ASCII rendering foundation for Hermes ops. It provides:

- **4 curated presets** for different aesthetic and density requirements
- **Integer scaling** (`--scale N`, 1–16) on base 48×24 grid
- **Automatic quality verdicts** — rejects `low-contrast-garble-risk` before ship
- **Video pipeline** — motion-phase detection, optional interpolation, embedded HTML5 player
- **Edge-aware processing** — Laplacian-weighted downsampling + CLAHE preserves structural edges
- **Pure-Python runtime** — Pillow + NumPy only; ffmpeg optional for interpolation

This skill is **production-locked** after V6 edge-wash failure analysis. All outputs are deterministic and quality-gated.

---

## Installation

From the repository root (requires Python 3.11+):

```bash
pip install -e .
```

This installs the `ascii-pipeline` CLI entry point.

**Dependencies:**
- Required: `Pillow>=10.0`, `numpy>=1.26`
- Optional: `ffmpeg` (with `minterpolate` filter) for motion-compensated frame interpolation

---

## Skill Surface

### Core Commands

```bash
ascii-pipeline presets
```
List available presets: `stroke-clarity`, `d30-dense`, `braille-detail`, `eikon-motion`.

---

```bash
ascii-pipeline diagnose <text-file>
```
Run quality diagnostics on an existing ASCII art file. Output includes:
- `unique_glyphs`: count of distinct characters used
- `fill_ratio`: proportion of non-background cells
- `verdict`: `high-contrast` / `low-contrast-garble-risk` / `braille-dominant`

---

```bash
ascii-pipeline render-preview <image>
```
Quick side-by-side preview (original + ASCII) at default scale (48×24) with `stroke-clarity` preset. PNG output to `preview.png`.

---

```bash
ascii-pipeline render-image \
  --input <image-path> \
  --preset <preset-name> \
  [--scale N] [--fullsize] \
  --out <output.txt> \
  --preview-out <output.png> \
  --diagnostics-out <output.json>
```

Render a still image to ASCII or Braille.

**Flags:**
- `--preset`: one of `stroke-clarity`, `d30-dense`, `braille-detail` (still-image presets)
- `--scale N`: integer multiplier on base 48×24 grid; range 1–16. Default: 1
- `--fullsize`: alias for `--scale 4` (192×96, showcase master)
- `--out`: text output path
- `--preview-out`: PNG preview path
- `--diagnostics-out`: JSON metrics path

**Notes:**
- `d30-dense` at scale ≥8 uses edge-aware downsampling internally (slower but edge-preserving)
- `braille-detail` produces 4× effective resolution via Braille dot matrix
- All presets enforce quality gates; `low-contrast-garble-risk` verdict still writes files but flags failure

---

```bash
ascii-pipeline build-eikon-from-video \
  --video <mp4-path> \
  --fps <frames-per-second> \
  --states <state-count> \
  --id <eikon-id> \
  [--no-motion-phases] [--grid WxH] [--charset <preset-name>]
```

Build an animated eikon from video input.

**Flags:**
- `--video`: path to MP4 file (ffmpeg must be available for extraction)
- `--fps`: target frame rate (12, 24, 48 recommended; 48 uses interpolation if available)
- `--states`: number of motion states (typically 3: idle/thinking/speaking)
- `--id`: eikon identifier (used for output filenames)
- `--no-motion-phases`: disable motion-phase clustering; treat all frames as one state
- `--grid`: explicit grid size as `WxH`; default `192x96` (scale-4 of 48×24)
- `--charset`: preset to use; default `d30-dense` for eikon-motion

**Output:**
- `<id>.eikon` — binary eikon file (frame data + metadata)
- `<id>-player.html` — embedded HTML5 canvas player (base64-encoded data, works offline)
- If `--fps 48` and ffmpeg with `minterpolate` is available, motion-compensated interpolation is applied for smoother playback.

**Workflow:**
1. Extract frames with `ffmpeg` (temporary PNGs in `tmp/`)
2. Convert each frame to ASCII using the selected preset
3. Cluster frames into motion states via frame-delta percentile analysis
4. Optionally interpolate to target fps (requires ffmpeg `minterpolate`)
5. Package frames into `.eikon` binary + HTML5 player

---

## Presets

### stroke-clarity
High-contrast, bold strokes. Direct intensity→character mapping. No block processing. Best for posters, logos, clean linework. Edge-aware not applicable. Fastest.

### d30-dense
180-glyph block-mode preset. Divides source into 16×16 pixel blocks, selects character by weighted intensity vote. **Uses edge-aware Laplacian-weighted downsampling** to preserve edges at high scale. Atmospheric density. Slower at scale ≥8.

### braille-detail
Braille dot matrix (2×4 per character) with Floyd–Steinberg error diffusion. Produces 4× effective resolution. No block collapse; detail-preserving. Dense, tactile, high information density.

### eikon-motion
Video pipeline preset. Does not accept `--scale` directly; instead uses `--grid` and `--fps` in `build-eikon-from-video`. Internally uses `d30-dense` charset with frame clustering and optional interpolation. Outputs animated eikon + HTML player.

---

## Quality Gates

Every render produces a `verdict` in the diagnostics JSON:

| Verdict | Meaning | Action |
|---------|---------|--------|
| `high-contrast` | Production-safe. Edges clear, density appropriate. | Ship |
| `low-contrast-garble-risk` | Edge washout or over-averaging detected. Output unreadable. | Reject |
| `braille-dominant` | Braille preset active; 4× resolution achieved. | Accept (braille mode) |

The `d30-dense` preset at scale-16 previously failed with `low-contrast-garble-risk` due to majority-vote block collapse. This was corrected with edge-aware weighting; all regenerated scale-16 assets now pass `high-contrast`.

---

## Scaling Strategy

Base resolution: **48×24** (standard Herm avatar).

| Scale | Grid | Use case | Performance |
|-------|------|----------|-------------|
| 1 | 48×24 | Avatar, deployment, chat | Fast (~0.1s) |
| 2–4 | 96×48 – 192×96 | Showcase, poster, detail view | Moderate (~0.3–1.2s) |
| 8 | 384×192 | High-fidelity showcase, print | Slow (~4–8s) |
| 16 | 768×384 | Maximum poster, archival | Very slow (~15–30s), edge-aware mandatory |

All scales share the same preset pipeline. Intermediate grids scale automatically; e.g., `d30-dense` intermediate processing grid = `base_grid × scale`.

**Caution:** `d30-dense` at scale 16 is **resource-intensive** (6144×3072 intermediate before collapse). Use for showcase only. Avatar-scale (`--scale 1`) remains fast.

---

## Video Pipeline — Eikon Motion Extraction

The `build-eikon-from-video` command implements a complete video→animated eikon workflow:

1. **Frame extraction:** `ffmpeg -i input.mp4 -vf fps=... tmp/frame_%04d.png`
2. **ASCII conversion:** Each frame → ASCII using `d30-dense` preset at target grid
3. **Motion-phase detection:** Frame-delta percentile clustering into `--states` groups (default 3)
4. **Optional interpolation:** If `--fps 48` and `ffmpeg minterpolate` available, generate smooth in-between frames
5. **Packaging:** `.eikon` binary (frame indices + glyph arrays) + standalone HTML5 player

The HTML player:
- Self-contained (base64-encoded eikon data)
- No HTTP server / CORS required
- Canvas rendering, play/pause/step controls
- State label display (idle/thinking/speaking)

**Example output:**
```
owl-smooth.eikon          10.8 MB  (573 frames, 48 fps interpolated)
owl-player-embedded.html  14.2 MB  (offline-capable player)
```

---

## Edge-Aware Downsampling (V6 Fix)

**Problem (V6):** Factor-16 block downsampling with majority-vote averaging washed out edges. Diagnostic: `low-contrast-garble-risk`.

**Solution:** Edge-aware Laplacian-weighted block reduction + CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing.

- Laplacian gradient magnitude weights each pixel's contribution to block character selection
- Edges get higher weight → preserved in final character
- CLAHE stretches local contrast to prevent edge compression

This fix is **mandatory** for `d30-dense` at scale ≥8. All high-scale assets in `examples/` use this pipeline.

---

## Examples & Gallery

The repository includes curated examples:

```
examples/
├── animated-eikon/      # owl eikon suite (12fps, 24fps, 48fps smooth)
├── dense-field/         # dense-field still renders (all presets × scales)
├── portrait/            # portrait still renders (all presets × scales)
└── scene-cosmic-pyramid/
    ├── preset-renders/  # cosmic pyramid: stroke-clarity, d30-dense, braille-detail; multiple scales
    └── source/          # cosmic-pyramid.jpg
```

**Gallery:** Open `gallery/index.html` in a browser to view all 16 panels with verdicts and notes.

---

## Operator Recommendations

### When to use which preset

| Scenario | Preset | Scale | Verdict target |
|----------|--------|-------|----------------|
| Avatar / chat | `d30-dense` | 1 | `high-contrast` |
| Poster / showcase | `stroke-clarity` | 4 | `high-contrast` |
| Maximum detail still | `braille-detail` | 4–8 | `braille-dominant` |
| Animated eikon | `eikon-motion` (video) | 4 (192×96) | `high-contrast` per frame |

### Quality gate enforcement

Always run `ascii-pipeline diagnose` on outputs before shipping. If verdict is `low-contrast-garble-risk`:
- For `d30-dense`: increase scale to 4+ (more pixels per glyph) or switch to `stroke-clarity`
- For `stroke-clarity`: check source contrast; may need preprocessing
- For `braille-detail`: normally yields `braille-dominant`; accept if Braille is intended

### Video best practices

- Source MP4 should be high-contrast, well-lit, minimal motion blur
- Use `--fps 48` + `ffmpeg minterpolate` for smooth playback (requires ffmpeg compiled with `minterpolate` support)
- `--states 3` works for most speech/animation; adjust for more motion nuance
- Embed the player: copy `<id>-player.html` to target location; works offline

---

## Integration Notes for Hermes

- **Skill type:** CLI-based creative tool
- **Execution model:** Synchronous command execution; video pipeline may take 30–60s for long inputs
- **Output artifacts:** `.txt` (ASCII), `.png` (preview), `.json` (diagnostics), `.eikon` (video), `.html` (player)
- **Cache policy:** Eikon builds write intermediate frames to `tmp/`; clean with `rm -rf tmp/`
- **Resource budget:** Still images: <1 GB RAM; video: up to 2–3 GB during frame extraction; scale-16 still: ~4 GB peak
- **Determinism:** Given same input + flags + seed (fixed), outputs are byte-for-byte reproducible (PNG encoders may vary slightly across Pillow versions)
- **Error handling:** Non-zero exit on failure; diagnostics still written when possible; `verdict` field in JSON is authoritative

---

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Current status: **11 tests passing** covering CLI parsing, preset existence, scale flag behavior, diagnostics, and backend purity (no chafa dependency).

---

## License

MIT — permissive, matches Hermes ecosystem licensing.

---

## Upstreaming

This skill is intended for inclusion in the main Hermes agent repository under `skills/ascii-art-pipeline/`. Integration steps:

1. Copy entire repository contents into `skills/ascii-art-pipeline/` in hermes-agent
2. Ensure `pyproject.toml` dependencies are added to Hermes package metadata
3. CLI entry point `ascii-pipeline` auto-registered via Hermes skill loader
4. No Hermes core code changes required; pure skill addition

Contact: Anduril (@Anduril) or Ousia Research (plntrprotocol)

---

**Status:** Production-ready. Deploy.
