# Interactive & Accessible Visualization

Screen-first principles extending Tufte's foundation. The Tufte-clean default is the starting point; interactivity layers on top.

---

## A. Progressive Disclosure

Overview first, zoom and filter, details on demand.

| Layer | What shows | Example |
|---|---|---|
| **Default** | Tufte-clean overview — high data-ink, no tooltips visible. Must stand alone as static screenshot. | Line chart with direct labels, no gridlines |
| **Hover / tap** | Values, annotations, comparisons. Contextual crosshair (single horizontal + vertical line at cursor) gives precise reading without permanent gridlines (see rule 3). | Tooltip with value + annotation at cursor |
| **Click / drill** | Linked views, data tables, source attribution, downloadable data. | Brushing across small multiples |

**Affordances**: cursor `pointer` on interactive elements, subtle opacity shift on hover, focus rings for keyboard, `:active` feedback within 100ms on touch.

**Linked views**: Small multiples become linked — selecting in one panel highlights all others. Use shared state (React context, D3 dispatch, ECharts `connect`).

---

## B. Accessibility

### Contrast

- **Chart elements** vs. background: **3:1** minimum
- **Text in charts**: **4.5:1** minimum
- The Tufte palette meets both in light and dark modes

### Dual encoding

Never rely on color alone:

| Primary channel | Pair with |
|---|---|
| Line color | Dash pattern, stroke width, or direct label |
| Bar fill | Pattern fill (hatching) or value label |
| Dot color | Shape (`circle`, `square`, `triangle`, `diamond`) |

### Keyboard navigation

```
TAB → Focus chart    Arrows → Move between points    Enter → Detail    Escape → Dismiss
```

Wrap charts in a focusable container: `<div role="img" aria-label={summary} tabIndex={0} onKeyDown={handler}>`.

### Screen reader text alternatives

Choose one per chart:

1. **`role="img"` + `aria-label`** — state the key finding: `"Revenue grew 23% Q1–Q4, peaking at $4.2M in Q3"` (not "A line chart showing revenue")
2. **Companion `<table>`** — link via `aria-describedby`, hide with `sr-only` class
3. **Structured `<p>` description** — for charts needing more than one sentence but less than a full table

For matplotlib/static output: use descriptive alt text on `<img>` or `![alt](path)`.

### User preference media queries

| Query | Action |
|---|---|
| `prefers-reduced-motion: reduce` | Disable transitions, instant state changes |
| `prefers-color-scheme: dark` | Switch to dark palette |
| `prefers-contrast: more` | Increase stroke widths, boost contrast |

---

## C. Responsive Design

### Strategies

- **Fluid**: percentage width + SVG `viewBox` — simplest, works for most chart types
- **Adaptive**: breakpoint-based layout changes (swap column→horizontal bars at narrow widths)
- **Hybrid**: fluid within ranges, adaptive across breakpoints — recommended for dashboards

### Mobile adaptations (< 600px)

- Horizontal bars replace columns (labels read naturally)
- Reduce tick density (every 2nd/3rd, or first/last/midpoint only)
- Abbreviate labels ("January" → "Jan", "$1,200,000" → "$1.2M")
- Stack small multiples vertically with selector/swipe

### Touch targets

- Minimum: **44 x 44px**, **8px** spacing between targets (WCAG 2.5.5)
- **No hover-only information** — every hover must have tap/click/focus alternative

### Test breakpoints

| Width | Expect |
|---|---|
| 320px | Single column, horizontal bars, minimal ticks, touch-sized targets |
| 768px | Columns ok, 2-up small multiples, moderate ticks |
| 1440px+ | Full layout, all annotations, hover interactions |

---

## D. Animation & Transitions

| | Examples |
|---|---|
| **Permitted** | Data transitions (sort, filter, time scrub), enter/exit, zoom/pan, chart type morphing |
| **Forbidden** | Entrance "build" animations, bouncing, spinning, decorative particles |

**Duration**: 200–500ms state changes, 300ms enter/exit, ease-out for exits, ease-in-out for transforms.

Always respect `prefers-reduced-motion` — disable all non-essential animation.

---

## E. Dark Mode

The Tufte palette includes dark mode (#151515 bg, #999 series, #fc8d62 highlight). Rules:

- **Never invert** — design dark palette independently
- **Reduce saturation** 10–20% (bright colors vibrate on dark backgrounds)
- **Same contrast minimums** apply (3:1 elements, 4.5:1 text)
- **Respect `prefers-color-scheme`**, allow manual override

### Semantic tokens (CSS)

```css
:root {
  --tufte-bg: #fffff8; --tufte-text: #111; --tufte-text-secondary: #666;
  --tufte-axis: #ccc; --tufte-series-default: #666; --tufte-highlight: #e41a1c;
}
@media (prefers-color-scheme: dark) {
  :root {
    --tufte-bg: #151515; --tufte-text: #ddd; --tufte-text-secondary: #999;
    --tufte-axis: #444; --tufte-series-default: #999; --tufte-highlight: #fc8d62;
  }
}
```

---

## Library Quick Reference

All snippets below. Read only what you need for your target library.

### Recharts

```jsx
// Responsive container (always use)
<ResponsiveContainer width="100%" height={400}><LineChart data={data} /></ResponsiveContainer>

// Accessible wrapper
<div role="img" aria-label={summary} tabIndex={0} onKeyDown={handler}>
  <ResponsiveContainer>...</ResponsiveContainer>
</div>

// Dark mode via CSS vars
<XAxis stroke="var(--tufte-axis)" tick={{ fill: 'var(--tufte-text-secondary)' }} />
<Line stroke="var(--tufte-series-default)" animationDuration={prefersReducedMotion ? 0 : 400} />

// Linked small multiples
const [activeIndex, setActiveIndex] = useState(null);
<Line onMouseMove={(e) => setActiveIndex(e.activeTooltipIndex)} />
```

### Chart.js

```javascript
new Chart(ctx, { options: {
  responsive: true, maintainAspectRatio: true, aspectRatio: 1.5,
  animation: prefersReducedMotion ? false : { duration: 400, easing: 'easeOutQuart' }
}});
// Plugins: chartjs-plugin-a11y-legend (keyboard), chartjs-plugin-chart2music (sonification)

// Dark mode
const isDark = matchMedia('(prefers-color-scheme: dark)').matches;
Chart.defaults.color = isDark ? '#999' : '#666';
Chart.defaults.backgroundColor = isDark ? '#151515' : '#fffff8';
```

### ECharts

```javascript
// Responsive
const chart = echarts.init(container);
new ResizeObserver(() => chart.resize()).observe(container);

// Animation + reduced motion
option = { animation: !prefersReducedMotion, animationDuration: 400, animationEasing: 'cubicOut' };

// Dark theme
echarts.registerTheme('tufte-dark', {
  backgroundColor: '#151515', textStyle: { color: '#ddd' },
  categoryAxis: { axisLine: { lineStyle: { color: '#444' } } }
});
// echarts.init(container, 'tufte-dark')

// Linked views: echarts.connect([chart1, chart2])
```

### matplotlib

```python
# Dark mode context
TUFTE_DARK = {
    'figure.facecolor': '#151515', 'axes.facecolor': '#151515',
    'text.color': '#ddd', 'axes.edgecolor': '#444',
    'xtick.color': '#999', 'ytick.color': '#999',
}
with plt.rc_context(TUFTE_DARK):
    fig, ax = plt.subplots(figsize=(10, 10/1.5))

# Alt text: use descriptive ![alt](path) when embedding
# Linked brushing: use mpl_connect('pick_event', handler) across axes
```

### Plotly

```javascript
// Responsive
Plotly.newPlot(div, data, layout, { responsive: true });

// Dark mode template
const tufteDark = { layout: { paper_bgcolor: '#151515', plot_bgcolor: '#151515', font: { color: '#ddd' } } };
Plotly.newPlot(div, data, { ...layout, template: isDark ? tufteDark : tufteLight });
```

### CSS utilities

```css
@media (prefers-reduced-motion: reduce) {
  .chart-element { transition: none !important; animation: none !important; }
}
@media (prefers-contrast: more) {
  :root { --tufte-series-default: #444; --tufte-axis: #888; }
}
```
