# Palette Generation

## The scale convention

Design system palettes use a numeric scale from 50 (lightest) to 950 (darkest). The standard labels by palette size:

| Size | Labels |
| --- | --- |
| 5 | 100, 300, 500, 700, 900 |
| 9 | 50, 100, 200, 300, 500, 700, 800, 900, 950 |
| 11 | 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950 |

11 steps matches Tailwind's default scales; use 9 as a leaner default when the `400`/`600` in-betweens aren't needed.

## Algorithm

Given a base color with lightness (L), chroma percentage, and hue (H):

**Step 1. Lightness bounds:**

```
delta = 0.4
minL = max(0.05, baseL - delta)
maxL = min(0.95, baseL + delta)
```

Lightness is clamped to [0.05, 0.95] to avoid pure black/white which have zero chroma.

**Step 2. Distribute lightness** evenly from maxL (lightest, label 50) to minL (darkest, label 950).

**Step 3. Clamp chroma per step.** Each lightness level has a different maximum chroma for a given hue and color space:

```
maxChroma = findMaxChroma(step[i].L, hue, colorSpace)
step[i].C = (chromaPercentage / 100) * maxChroma
```

This ensures every step is within gamut. High-chroma base colors will have lower chroma at the lightest and darkest ends; this is correct and expected.

## CSS variable output

```css
:root {
  --color-50: oklch(0.971 0.012 250);
  --color-100: oklch(0.932 0.028 250);
  --color-200: oklch(0.882 0.048 250);
  --color-300: oklch(0.812 0.078 250);
  --color-500: oklch(0.623 0.188 250);
  --color-700: oklch(0.445 0.138 250);
  --color-800: oklch(0.362 0.108 250);
  --color-900: oklch(0.289 0.078 250);
  --color-950: oklch(0.215 0.048 250);
}
```

## Multi-hue palettes

When generating palettes for multiple hues, use the same **lightness** and **chroma percentage** for all. Same L guarantees equal perceived brightness. Same chroma percentage (not absolute chroma) guarantees equal vividness relative to each hue's maximum.

```css
:root {
  /* Same L, same C% (80% of max): different absolute C per hue */
  --blue-500: oklch(0.623 0.141 250);   /* 80% of max 0.176 */
  --green-500: oklch(0.623 0.157 145);  /* 80% of max 0.196 */
  --red-500: oklch(0.623 0.202 25);     /* 80% of max 0.253 */
}
```

Different hues have different max chroma at the same lightness. Using the same absolute C value across hues would make some appear more vivid than others.

## Dark mode

Start by swapping the light and dark semantic roles, then tune the mapped values for the dark appearance:

```css
:root {
  --color-bg: var(--color-50);
  --color-text: var(--color-950);
}

.dark {
  --color-bg: var(--color-950);
  --color-text: var(--color-50);
}
```

Do not mechanically reverse every palette step. Dark appearances often need different chroma and lightness spacing, and equal OKLCH steps do not guarantee that every foreground/background pair preserves its contrast. Recheck each pair and tune the dark tokens independently where needed.

## Why not HSL palettes?

**Hue drift:** `hsl(240, 80%, 20%)` and `hsl(240, 80%, 90%)` are not the same perceptual hue. The light variant shifts ~16° toward purple. OKLCH hue is stable.

**Brightness inconsistency:** `hsl(60, 100%, 50%)` (yellow) and `hsl(240, 100%, 50%)` (blue) have the same HSL lightness but wildly different perceived brightness.
