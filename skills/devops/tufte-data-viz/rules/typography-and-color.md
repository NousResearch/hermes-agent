# Typography and Color Reference

## Font hierarchy

| Element | Font family | Size | Weight | Color (light) | Color (dark) |
|---------|------------|------|--------|---------------|--------------|
| Chart title | ET Book, Palatino, Georgia, serif | 18–22px | 400 | #111 | #ddd |
| Subtitle / caption | ET Book, Palatino, Georgia, serif | 14–16px | 400 | #666 | #999 |
| Axis tick label | system-ui, -apple-system, sans-serif | 11–12px | 400 | #999 | #666 |
| Data label (on chart) | ET Book, Palatino, Georgia, serif | 12–14px | 400 | #333 | #bbb |
| Annotation | ET Book, Palatino, Georgia, serif | 12–13px | 400 italic | #333 | #bbb |
| Tooltip text | system-ui, sans-serif | 12–13px | 400 | #333 | #ccc |
| Source citation | system-ui, sans-serif | 10–11px | 400 | #aaa | #555 |

**Key rule**: Data marks and labels should always be visually equal to or larger than axis tick labels. If axis ticks are 12px, data labels must be >= 12px. The data is the priority.

## Font loading

### CSS — ET Book (Tufte's custom Bembo variant)

```css
@font-face {
  font-family: "ET Book";
  src: url("https://cdn.jsdelivr.net/gh/edwardtufte/tufte-css@gh-pages/et-book/et-book-roman-line-figures/et-book-roman-line-figures.woff") format("woff");
  font-weight: normal;
  font-style: normal;
  font-display: swap;
}

@font-face {
  font-family: "ET Book";
  src: url("https://cdn.jsdelivr.net/gh/edwardtufte/tufte-css@gh-pages/et-book/et-book-bold-line-figures/et-book-bold-line-figures.woff") format("woff");
  font-weight: bold;
  font-style: normal;
  font-display: swap;
}

@font-face {
  font-family: "ET Book";
  src: url("https://cdn.jsdelivr.net/gh/edwardtufte/tufte-css@gh-pages/et-book/et-book-display-italic-old-style-figures/et-book-display-italic-old-style-figures.woff") format("woff");
  font-weight: normal;
  font-style: italic;
  font-display: swap;
}
```

### CSS — Google Fonts fallback (when ET Book is unavailable)

```html
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
```

EB Garamond is the closest freely available web font to Bembo/ET Book.

### Python (matplotlib)

```python
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"]
plt.rcParams["font.size"] = 12
```

### React / CSS-in-JS

```typescript
const TUFTE_FONT = '"ET Book", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif';
const TUFTE_FONT_SANS = 'system-ui, -apple-system, sans-serif';
```

## Old-style (hanging) figures

For tabular data with numbers, use old-style figures. These have descenders (3, 4, 5, 7, 9 hang below the baseline) which makes columns of numbers easier to scan vertically.

```css
.data-table td {
  font-feature-settings: "onum" 1;
  font-variant-numeric: oldstyle-nums;
}
```

In matplotlib: Palatino and Georgia include old-style figure variants by default.

## Color palettes

### Light mode

| Role | Hex | CSS variable suggestion |
|------|-----|------------------------|
| Page/chart background | `#fffff8` | `--bg` |
| Text primary (titles, data labels) | `#111111` | `--text-1` |
| Text secondary (subtitles, annotations) | `#666666` | `--text-2` |
| Text tertiary (axis ticks, source) | `#999999` | `--text-3` |
| Axis line (bottom/left only) | `#cccccc` | `--axis` |
| Grid line (if absolutely needed) | `#eeeeee` | `--grid` |
| Default data series | `#666666` | `--series-default` |
| Highlight / accent | `#e41a1c` | `--highlight` |

### Dark mode

| Role | Hex | CSS variable suggestion |
|------|-----|------------------------|
| Page/chart background | `#151515` | `--bg` |
| Text primary | `#dddddd` | `--text-1` |
| Text secondary | `#999999` | `--text-2` |
| Text tertiary | `#666666` | `--text-3` |
| Axis line | `#444444` | `--axis` |
| Grid line | `#333333` | `--grid` |
| Default data series | `#999999` | `--series-default` |
| Highlight / accent | `#fc8d62` | `--highlight` |

### Categorical palette (maximum 4 colors)

Use only when the chart must distinguish multiple categories and direct labeling is not sufficient on its own.

| Name | Hex | Swatch |
|------|-----|--------|
| Steel blue | `#4e79a7` | Muted blue |
| Tangerine | `#f28e2b` | Warm orange |
| Coral | `#e15759` | Muted red |
| Sage | `#76b7b2` | Teal-green |

If more than 4 categories exist, use small multiples or group lesser categories into "Other."

### Sequential palette (single hue)

For ordered magnitude (low to high):
```
#f7fbff → #deebf7 → #c6dbef → #9ecae1 → #6baed6 → #3182bd → #08519c
```

### Diverging palette (below/above center)

For deviation from a midpoint (e.g., profit/loss, above/below average):
```
Negative: #e15759 → #f7c4c4 → #f0f0f0 (center) → #c0ddd6 → #76b7b2 :Positive
```

## Color rules

1. **Gray is your primary color.** Default all data to gray. Only add color when it carries meaning.
2. **Maximum 4 categorical colors.** Human visual working memory cannot reliably track more without constant legend reference.
3. **Ghosting**: De-emphasize secondary data by setting opacity to 0.2–0.3. The primary series stays at full opacity.
4. **No rainbow/spectral palettes.** They have no natural ordering, create false boundaries, and are inaccessible to colorblind readers.
5. **Smallest effective difference.** If you need to distinguish two things, make them barely distinguishable — not screaming. Dark gray vs. medium gray, not red vs. blue.
6. **Grayscale test.** The chart should be fully readable when printed in grayscale. If two series become indistinguishable, add direct labels (which you should have anyway).
7. **No loud colors over large areas.** A full bright-red bar chart assaults the eye. Use muted tones for area fills; save saturated color for single points of emphasis.
