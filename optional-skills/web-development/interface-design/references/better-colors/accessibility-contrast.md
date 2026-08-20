# Accessibility & Contrast

Contrast is always measured between a **foreground color** (text, icon, or UI element) and the **background color** it sits on. When checking contrast, identify the background the element will be rendered against, typically the nearest parent's background color.

**Report, don't repaint.** When a check fails, report it (the failing foreground/background pair, its measured Lc or ratio, and the threshold it misses) and leave the colors unchanged. A project's colors are a design decision; only apply the fix below when the user asks for one.

## APCA thresholds (recommended)

APCA (Accessible Perceptual Contrast Algorithm) is more perceptually accurate than WCAG 2 and pairs naturally with oklch since both are grounded in perceptual lightness. Use APCA as the default.

Lc (Lightness Contrast) measures the perceived contrast between foreground and background. These levels are simplified from APCA's full font-size/weight lookup table:

| Content Type | Minimum | Preferred |
| --- | --- | --- |
| Body text (columns/blocks of text) | Lc 75 | Lc 90 |
| Non-body text (labels, headlines) | Lc 60 | Lc 75 |
| Large text (≥36px) | Lc 45 | Lc 60 |
| UI components | Lc 30 | n/a |

Lc 30 is also APCA's minimum for disabled and placeholder text; the absolute floor for non-text elements to be discernible at all is Lc 15.

APCA's Lc value is signed: positive means dark text on a light background, negative means light text on a dark background. Use the absolute value for threshold comparison.

## WCAG 2 thresholds (for legal compliance)

WCAG 2 is still required when making formal WCAG 2.x conformance claims. It uses a luminance ratio that can be both too strict and too lenient depending on the color pair.

| Content Type | AA | AAA |
| --- | --- | --- |
| Normal text (<24px / <18.5px bold) | 4.5:1 | 7:1 |
| Large text (>=24px / >=18.5px bold) | 3:1 | 4.5:1 |
| UI components & graphical objects | 3:1 | n/a |

WCAG defines "large text" in points: 18pt ≈ `24px`, or 14pt bold ≈ `18.5px`.

## Fixing contrast with oklch (on request)

In hex/rgb, fixing contrast means trial and error across three channels. In oklch, lightness (L) is the clearest first lever: adjust the L distance between the foreground and its background while preserving C and H when possible:

```css
/* Failing: text too close in lightness to its background (Lc ≈ 50) */
color: oklch(0.65 0.08 250);      /* foreground */
background: oklch(0.95 0.02 250); /* background */

/* Fix: darken the text, keep C and H unchanged (Lc ≈ 90) */
color: oklch(0.3 0.08 250);       /* foreground: more L distance */
background: oklch(0.95 0.02 250); /* background: unchanged */
```

Note that mid-lightness backgrounds cap the achievable contrast: on a background of L 0.75, even pure black text only reaches about Lc 60; body text needs a background near the light or dark extreme.

Adjust L first, then remeasure the rendered foreground/background pair. Chroma and hue can still affect the converted color, gamut mapping, and measured contrast; reduce C when needed to keep the adjusted color in gamut.

## Quick lightness gap guide

For body text (targeting |Lc| >= 75):

- **Light background (L > 0.9):** foreground L should be below 0.35
- **Dark background (L < 0.25):** foreground L should be above 0.9

The gap is asymmetric because APCA is polarity-aware: mirrored pairs don't score identically. These are approximations; always verify with an actual contrast calculation.

## Light vs dark color detection

A background counts as light when its oklch lightness exceeds 0.73, the APCA crossover on neutral backgrounds:

```
if L > 0.73 → use dark text on this background
if L <= 0.73 → use light text on this background
```

The crossover is higher than intuition suggests: in the 0.6–0.73 band the background already looks light, but white text still scores meaningfully higher than black.

## Hue drift detection

To detect hue drift in an existing HSL palette:

1. Convert each step to oklch
2. Compare the H values across steps
3. If the hue spread is greater than 10°, the palette has visible drift

```css
/* HSL blue ramp: hue shifts toward purple */
hsl(240, 80%, 20%)  →  oklch H ≈ 269
hsl(240, 80%, 50%)  →  oklch H ≈ 267
hsl(240, 80%, 90%)  →  oklch H ≈ 285  /* shifted 18° */
```
