# Contrast Ratios

Every 1939 palette includes 7 WCAG 2.1 contrast ratios measuring the readability of key role combinations. These ratios tell you whether specific color combinations are safe for text, UI elements, and data visualization.

## The 7 Pairs

| Pair | What It Measures | Safe Threshold |
|------|----------------|----------------|
| `text_on_background` | Body text on page background | ≥ 4.5:1 (AA), ≥ 3:1 (AA large text) |
| `highlight_on_background` | Headings/CTAs on background | ≥ 3:1 (AA large text) |
| `highlight_on_canvas` | Headings on surface | ≥ 3:1 (AA large text) |
| `support_on_background` | Secondary accent on background | ≥ 3:1 minimum |
| `chart1_on_background` | Primary data on background | ≥ 3:1 minimum |
| `chart2_on_background` | Secondary data on background | ≥ 3:1 minimum |
| `canvas_on_background` | Surface vs. backdrop distinction | Higher = more visual separation |

## How to Read Them

A contrast ratio of **5.76** (like Hugo's Mom's `highlight_on_background`) means the Highlight color is 5.76× brighter/darker than the Background. This easily passes WCAG AA for large text (≥ 3:1) and passes for normal text too (≥ 4.5:1).

A ratio of **2.21** (like Hugo's Mom's `text_on_background`) means Text is only 2.21× different from Background. This **fails** WCAG AA for normal text. In practice, Hugo's Mom uses Background (near-black) as the page background with Text (warm brown) — the ratio is low because they're in the same tonal range. This is intentional: the Highlight and Support colors carry the emphasis, while Text is meant to be subtle.

## Design Decisions Based on Contrast

### When text_on_background < 4.5:1
The palette's Text color doesn't pass on Background. Solutions:
- Use Canvas as the text surface instead (place Text on Canvas, not directly on Background)
- Use a lighter Text tint (Text-200) for better readability
- Use Highlight or Support for important text, not Text

### When highlight_on_canvas < 3:1
The heading color blends into the surface. Solutions:
- Use a darker or lighter Highlight tint
- Add a subtle background behind the heading (Highlight-200)
- Switch to Support as the heading accent

### When canvas_on_background > 5:1
Strong visual separation between surface and page — good for card-based layouts. Below 3:1 means surfaces and background blend together.

## In Practice

The `legend_text` field on each role gives you the optimal text color for labels ON that role's center swatch. It's pre-computed to meet WCAG contrast requirements. Always use `legend_text` when placing text on a colored background.

```json
{
  "Background": {
    "hex": "#0A0A0D",
    "legend_text": "#f0ede8"
  }
}
```

This means: on a `#0A0A0D` background, use `#f0ede8` for text. On the swatch strip, this is the color that labels each tint.