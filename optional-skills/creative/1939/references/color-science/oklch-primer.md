# OKLCH Color Primer

## Why OKLCH?

OKLCH is a perceptually uniform color space. This means:

- **Same numeric change = same perceived change.** Moving from L=0.5 to L=0.6 looks like the same brightness increase regardless of hue or chroma.
- **Chroma (saturation) is visually consistent.** C=0.2 looks equally saturated for red, blue, and green. In HSL, "100% saturation" looks wildly different across hues.
- **Hue is perceptually even.** A 60° rotation in OKLCH feels like the same color shift everywhere. In HSL, 60° from yellow looks nothing like 60° from blue.

The 1939 pipeline extracts colors in sRGB, converts to OKLCH, clusters by perceptual similarity, and assigns semantic roles. The result: palettes where colors "belong together" because they literally do in human perception.

## OKLCH Components

| Component | Meaning | Range |
|-----------|---------|-------|
| **L** (Lightness) | Perceived brightness | 0 (black) to 1 (white) |
| **C** (Chroma) | Perceived saturation/colorfulness | 0 (gray) to ~0.4 (vivid) |
| **h** (Hue) | Perceived hue angle | 0°–360° |

## How It Differs from HSL

| Aspect | HSL | OKLCH |
|--------|-----|-------|
| Lightness | Relative to RGB channel | Perceived by human eye |
| Saturation | 100% looks neon at any hue | Same chroma = same vividness |
| Hue spacing | Uneven (blue-green is 72° but looks small) | Even (same degree = same perceived shift) |
| Gradient quality | Banding, hue shifts | Smooth, even transitions |

**The practical difference:** In HSL, `hsl(0, 100%, 50%)` (red) and `hsl(240, 100%, 50%)` (blue) have the same "lightness" but red looks brighter. In OKLCH, the same L value actually looks the same brightness.

## Perceptual Volume

Each palette has a `perceptual_volume` (PV) score. This measures how much 3D volume the palette's colors occupy in OKLCH space — think of it as "how much color variety does this palette have?"

- **PV near 0**: Monochrome or near-monochrome (B&W photos, single-hue images)
- **PV 0.001–0.005**: Limited palette, few distinct colors
- **PV 0.005–0.02**: Moderate variety, typical for most themes
- **PV 0.02+**: Rich, diverse color palette

PV is computed as the volume of the convex hull of all palette colors in OKLCH space. It's not a quality score — a B&W photo correctly gets PV≈0.

## Converting Colors

The brand JSON files already contain all computed tint hex values — you do
not need to perform OKLCH conversions to use this skill. These formulas are
for understanding the pipeline, not for runtime use.

### Hex → OKLCH (stdlib only)

```python
import math

def hex_to_oklch(hex_color):
    """Convert #RRGGBB to (L, C, h) using stdlib only."""
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)]
    # sRGB → Linear RGB
    def linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = linear(r), linear(g), linear(b)
    # Linear RGB → XYZ (D65)
    x = 0.4124*r + 0.3576*g + 0.1805*b
    y = 0.2126*r + 0.7152*g + 0.0722*b
    z = 0.0193*r + 0.1192*g + 0.9505*b
    # XYZ → OKLab (Ottosson)
    l_ = math.copysign(abs(x + 0.39633777*y - 0.004072047*z) ** (1/3), x + 0.39633777*y - 0.004072047*z)
    a_ = math.copysign(abs(1.9779985*x - 2.428592*y + 0.4505937*z) ** (1/3), 1.9779985*x - 2.428592*y + 0.4505937*z)
    b_ = math.copysign(abs(0.025904*x + 0.78277177*y - 0.80867577*z) ** (1/3), 0.025904*x + 0.78277177*y - 0.80867577*z)
    L = 0.99999999*l_ + 0.39633779*a_ + 0.21580376*b_
    a = 4.658439*a_ - 1.999426*b_
    b_ok = 1.29130*a_ + 2.62705*b_
    C = math.sqrt(a**2 + b_ok**2)
    h = math.degrees(math.atan2(b_ok, a)) % 360
    return L, C, h
```

### OKLCH → Hex (stdlib only)

```python
def oklch_to_hex(L, C, h):
    """Convert (L, C, h) to #RRGGBB using stdlib only."""
    h_rad = math.radians(h)
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    # OKLab → XYZ (inverse Ottosson) — simplified for reference
    # Full implementation requires matrix inverse; for practical use,
    # use the pre-computed tints in the brand JSON files.
    pass
```

### The Delta System
Each role's tints are computed by adding a lightness delta to the center color's L value:
```
L_tint = L_center + delta
```
This preserves the hue (h) and slightly reduces chroma (C) at extreme lightness values to stay within the sRGB gamut.