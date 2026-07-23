# Tint Scale System

Each role in a 1939 palette has 10 tints forming a perceptual color scale. This document explains how they work and how to use them.

## What Are Tints?

A "tint" is a variation of a role's center color at a different lightness. The 1939 pipeline expands each role's single center color into 10 perceptually spaced variations using OKLCH color space.

The center color (index 4, "500-level") is the role's identity — the color that best represents what this role means. The other 9 tints give you variations for:

- **Hover/active states** (one step lighter or darker)
- **Gradient backgrounds** (smooth transitions across the scale)
- **Typography hierarchy** (lighter tints for secondary text, darker for emphasis)
- **Border and surface variations** (lighter tints for subtle borders, darker for strong dividers)

## The 10-Tint Scale

```
Index  Level   Typical Use
─────────────────────────────────────────────
  0    50      Lightest — subtle hints, dividers, disabled text
  1    100     Very light — hover backgrounds (dark mode), secondary text backgrounds
  2    200     Light — surface fills, card backgrounds, table alt rows
  3    300     Medium-light — borders, subtle separators
  4    500     CENTER — the role's true color, identity color
  5    600     Medium-dark — hover states on dark backgrounds
  6    700     Dark — secondary backgrounds, table headers (dark mode)
  7    800     Very dark — slide header bars (dark mode), strong borders
  8    900     Darkest — near-black/white extremes for special cases
  9    950     Maximum — absolute extreme, rarely used
```

## Perceptual Curves

Not all roles use the same lightness distribution. The `curve` field tells you the shape:

- **`dark`** (Background roles): The center is very dark. Light tints (0-3) are spread across a wide lightness range because the perceptual differences matter more. Dark tints (6-9) are compressed near black because they're almost indistinguishable.

- **`light` / `surface`** (Canvas roles): The center is very light. Dark tints (6-9) have more perceptual spread. Light tints (0-3) are compressed near white.

- **`standard`** (Text, Highlight roles): Even distribution across the lightness range. The center is mid-toned.

- **`saturated`** (Support, Chart roles): Similar to standard but with more chroma preservation at extremes.

## The Delta System

Each role stores `deltas` — the OKLCH lightness offset from center for each tint. Delta 0.00 means "same as center." Positive deltas are lighter, negative are darker.

For Background (dark curve, center ≈ 0.06):
```
Deltas: +0.32, +0.24, +0.16, +0.10, 0.00, -0.05, -0.10, -0.15, -0.20, -0.25
```
Index 0 is much lighter (+0.32), index 9 is much darker (-0.25) from center.

For Canvas (surface curve, center ≈ 0.85):
```
Deltas: similar range but inverted distribution
```

## How to Use Tints in Practice

### CSS Custom Properties
```css
:root {
  --accent: var(--highlight-500);           /* Center */
  --accent-hover: var(--highlight-400);     /* One step lighter */
  --accent-active: var(--highlight-600);   /* One step darker */
  --accent-subtle: var(--highlight-200);    /* Subtle background */
  --accent-border: var(--highlight-300);   /* Visible border */
}
```

### PowerPoint
```
Title bar:    Background tint 700 (index 6)
Heading text: Highlight center (index 4)  
Body text:    Text center (index 4)
Subtle bg:    Canvas tint 200 (index 2)
Border:       Muted tint 300 (index 3)
```

### Word/Documents
```
Heading 1:    Highlight center
Heading 2:    Support center
Body:         Text center
Table header:  Background tint 700 bg, Highlight tint 100 text
Link:         Support center
```

## Computing Tints from Center

If you have a center hex and need a specific tint level, use OKLCH:

1. Convert hex to OKLCH: `L, C, h = hex_to_oklch(hex)`
2. Apply the delta: `L_new = L + delta_for_level(level)`
3. Convert back: `hex_new = oklch_to_hex(L_new, C, h)`

The brand JSON files store the computed tint hex values directly in the `tints` array — you
do not need to recompute them from deltas.

**Important:** OKLCH deltas are perceptually uniform. A delta of +0.10 means the same perceived brightness increase whether you're starting from a dark color or a light one. This is why the tints look evenly spaced to the human eye even though the hex values are not arithmetic progressions.