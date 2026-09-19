# Chart Color Assignment

## The Rules

Chart1 and Chart2 are **data colors only.** They exist for charts, graphs, data series, and palette swatches. They are NOT for:

- Body text (use Text)
- Headings (use Highlight or Support)
- Links (use Support)
- Buttons (use Highlight for primary, Support for secondary)
- Borders (use Muted)

## When to Use Chart Colors

✅ **Bar chart series** — Chart1 for primary bar, Chart2 for comparison bar
✅ **Line chart series** — Chart1 for primary line, Chart2 for secondary line
✅ **Pie/donut segments** — Chart1 for the dominant slice, Chart2 for comparison
✅ **Heatmap categories** — Chart1 for one category, Chart2 for another
✅ **Swatch strips** — Chart tints as accent colors in palette previews
✅ **Badge/chip colors** — Chart1 for primary badges, Chart2 for secondary

❌ **Section headings** — Use Highlight instead
❌ **Body copy** — Use Text
❌ **Navigation links** — Use Support
❌ **Borders and dividers** — Use Muted

## Multi-Series Charts

For charts with more than 2 series, derive additional colors from the tint scales:

```
Series 1: Chart1 center (index 4)       — primary
Series 2: Chart2 center (index 4)       — secondary
Series 3: Support center (index 4)     — tertiary
Series 4: Highlight tint 200 (index 2) — quaternary
Series 5: Muted center (index 4)       — neutral fallback
```

Or use tint variations within Chart1:
```
Series 1: Chart1 tint 300 (index 3)    — lighter
Series 2: Chart1 center (index 4)      — standard
Series 3: Chart1 tint 700 (index 6)    — darker
```

## Chart Color Accessibility

Check contrast ratios when placing chart colors on backgrounds:

- `chart1_on_background: 3.75` — Chart1 on dark background passes WCAG AA for large text
- `chart2_on_background: 4.72` — Chart2 on dark background passes for normal text

If a chart color doesn't meet 3:1 against the background:
1. Use a **tint** instead of the center color (lighter for dark backgrounds, darker for light)
2. Add a **border or outline** around the chart element in a contrasting color
3. Add **direct labels** (text next to bars/lines) instead of relying on color alone

## In PowerPoint

```
Chart accent bar on slide:    Support center
Chart data series 1:          Chart1 center
Chart data series 2:         Chart2 center
Chart data series 3:          Support center (if needed)
Chart data series 4:          Highlight tint 200 (if needed)
Slide title:                  Highlight center (NOT a chart color)
Slide body text:              Text center
```

## In Word Documents

```
Chart/table header accent:     Support tint 300
Chart/table row highlight:    Chart1 tint 200 (subtle bg)
Chart series 1 (in images):   Chart1 center
Chart series 2 (in images):   Chart2 center
Table borders:                Muted tint 300
```