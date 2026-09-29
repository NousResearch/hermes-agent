# Perceptual Volume

## What It Measures

Perceptual volume (PV) quantifies how much color variety a palette contains. It's the volume of the convex hull formed by all palette colors in OKLCH 3D space.

Think of it as a "color footprint" — a palette with lots of diverse colors occupies more 3D space than a monochrome palette where all colors cluster together.

## PV Ranges

| Range | Meaning | Examples |
|-------|---------|----------|
| 0.000 – 0.001 | Near-monochrome | Black & white photographs, single-hue minimalist art |
| 0.001 – 0.005 | Limited palette | Muted films, desaturated stills, foggy landscapes |
| 0.005 – 0.020 | Moderate variety | Most well-composed images, film stills |
| 0.020 – 0.050 | Rich palette | Vibrant paintings, colorful cinematography |
| 0.050+ | Exceptional range | Rainbow compositions, highly saturated art |

## What PV Is NOT

- **Not a quality score.** A B&W photograph with PV=0 is correctly extracted. High PV doesn't mean "better."
- **Not chroma.** A palette can have low chroma (muted colors) but high PV if they span different hues and lightness levels.
- **Not a measure of "how many colors."** Two palettes with 15 spectrum colors can have very different PVs depending on how spread out those colors are.

## How It's Computed

1. Convert all spectrum colors to OKLCH coordinates (L, C, h)
2. Map to 3D Cartesian: `x = C * cos(h)`, `y = C * sin(h)`, `z = L`
3. Compute the convex hull of these 3D points
4. PV = volume of the convex hull

This is the same computation used for the 3D galaxy visualization — the "bubbles" you see around each theme in the color-space viewer.

## Using PV

### For Palette Selection
- Low PV (0–0.003): Use for subtle, understated themes. Works well for professional documents, minimal interfaces.
- Mid PV (0.003–0.015): The sweet spot for most applications. Enough variety to be interesting without overwhelming.
- High PV (0.015+): Bold, statement palettes. Best for creative presentations, marketing, attention-grabbing designs.

### For Theme Similarity
Compare PV values to find palettes with similar color range. Two themes with PV ≈ 0.008 will feel similar in "color richness" even if their hues are completely different.

### For Accessibility
PV ≤ 0.001 indicates potential contrast issues — the palette may not have enough lightness range for readable text. Consider using tinted variants (higher or lower tints) instead of the center swatch for text/background pairs.

## PV in the Data

Each palette's PV is available at:
- API: `GET /api/themes/{slug}` → `perceptual_volume`
- Brand JSON: top-level `perceptual_volume` field
- Memes index: top-level `pv` field