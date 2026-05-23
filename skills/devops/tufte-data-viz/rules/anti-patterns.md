# Anti-Patterns: Detection and Fixes

## Quick reference

| Anti-pattern | Why it fails | Fix |
|-------------|-------------|-----|
| **Pie chart** | Humans are poor at comparing angles and areas. Two slices of 28% and 32% look identical. | Horizontal bar chart, sorted by value descending. |
| **3D effects** | Perspective distorts relative sizes. Back bars appear smaller than front bars of equal value. | Remove all 3D. Flat 2D only. |
| **Dual y-axes** | Implies correlation between two series that may not exist. Scale manipulation can make any two lines appear correlated. | Two stacked charts sharing the same x-axis (small multiples). |
| **Legend box** | Forces eyes to shuttle between the legend and the data. Increases cognitive load by ~40%. | Direct labels at the endpoint or on the data element itself. |
| **Heavy gridlines** | Compete with data for visual attention. Create a "cage" that imprisons the data. | Remove entirely, or horizontal-only at opacity 0.08–0.12. |
| **Rainbow palette** | No natural ordering. Creates false category boundaries. Inaccessible to 8% of males (colorblind). | Gray default + single accent, or 4-color muted palette. |
| **Gauge/speedometer** | Uses enormous screen area for a single number. No trend, no context, no comparison. | Single large number + sparkline + comparison text. |
| **Zebra-striped table** | Alternating row colors create moire vibration. The stripes become louder than the data. | Whitespace + thin horizontal rules every 3–5 rows. |
| **Gradient fills** | Add no information. Distract from data values. Make it harder to read precise heights. | Solid flat color (muted gray or accent). |
| **Rotated axis labels** | 45° or 90° text is hard to read. Forces head-tilting. | Horizontal bar chart (flip axes), or abbreviate labels. |
| **Truncated y-axis** | Starting y-axis above zero exaggerates differences. A 2% change looks like a 50% swing. | Include zero, or use a clear axis break indicator and note. |
| **Decorative borders** | Box around chart is pure chartjunk. Adds zero information. | Remove entirely. Let whitespace define the chart boundary. |
| **Area chart with high opacity** | Filled areas at 50%+ opacity obscure overlapping series and add visual weight that distorts perception. | Use lines only, or area with opacity 0.03–0.08. |
| **Exploded pie slices** | Pulling slices apart makes comparison even harder than standard pie. | Don't. Use a bar chart. |
| **Data point markers on every point** | Large circles on 50+ data points create a confetti effect that obscures the trend. | `dot={false}` or markers only at notable points (min, max, endpoint). |
| **Thick axis lines** | Heavy 2-3px axis lines draw attention away from the data. The 1+1=3 effect creates phantom visual weight. | 0.5–1px axis lines, or remove entirely if ticks provide orientation. |
| **Hover-only information** | Touch and keyboard users can never access it. Violates WCAG. | Tap/click/focus fallback for all hover content. |
| **Missing text alternative** | Screen readers announce nothing. The chart is invisible to blind users. | Add `aria-label` with key finding + data summary, or companion data table. |
| **Color as sole differentiator** | 8% of males have color vision deficiency. Series become indistinguishable. | Add shape, dash pattern, or direct label as second channel. |
| **Gratuitous entrance animation** | Chart "building" on load delays comprehension and is chartjunk in motion. | Remove, or gate behind `prefers-reduced-motion` check. |
| **Fixed pixel width** | Chart overflows or becomes unreadable on mobile. | Use percentage/viewBox width or breakpoint-based layout. |

## Per-library detection patterns

### Recharts

```
PATTERN                              FIX
<Legend />                          → Remove entirely; add direct labels
<CartesianGrid />                   → <CartesianGrid stroke="none" /> or remove
<Pie> or <PieChart>                 → <BarChart layout="vertical">
fill="url(#gradient...)"           → fill="#666" (solid gray)
<YAxis yAxisId="right"...>          → Two separate <LineChart> in a flex column
stroke="#..." (bright/loud color)   → stroke="#666" (gray) unless it's the highlight series
<Area fillOpacity={0.3+}            → <Line> or <Area fillOpacity={0.05}>
```

### ECharts

```
PATTERN                              FIX
type: 'pie'                         → type: 'bar' with horizontal orientation
splitLine: { show: true }           → splitLine: { show: false }
legend: { show: true }              → legend: { show: false } + label on series
grid: { borderWidth: N }            → grid: { show: false }
series.areaStyle: { opacity: 0.3+ } → Remove or opacity: 0.05
visualMap with rainbow colors        → Muted single-hue sequential palette
```

### Chart.js

```
PATTERN                              FIX
type: 'pie' / 'doughnut'           → type: 'bar' with indexAxis: 'y'
grid: { display: true }            → grid: { display: false }
border: { display: true } on scale  → border: { display: false }
plugins.legend.display: true        → display: false + chartjs-plugin-datalabels
backgroundColor: 'rgba(R,G,B,0.5)' → 'rgba(102,102,102,0.8)' (gray)
```

### matplotlib

```
PATTERN                              FIX
plt.pie(...)                        → plt.barh(...) sorted by value
ax.legend()                         → ax.annotate() at each line endpoint
ax.grid(True)                       → ax.grid(False) or ax.grid(axis='y', alpha=0.1)
fig.patch.set_facecolor('white')    → fig.patch.set_facecolor('#fffff8')
plt.xticks(rotation=45)            → Use plt.barh() or abbreviate tick labels
ax.set_frame_on(True)              → Remove top/right spines + range-frame
```

### Plotly

```
PATTERN                              FIX
go.Pie(...)                         → go.Bar(orientation='h', ...)
fig.update_layout(showlegend=True)  → showlegend=False + add annotations at endpoints
xaxis=dict(showgrid=True)          → showgrid=False
plot_bgcolor='white'               → plot_bgcolor='#fffff8'
template='plotly'                   → Use custom Tufte template
```

## The "Can I remove it?" test

For each non-data element, ask: "If I delete this, does the chart lose information?"

- **Gridlines**: Usually no. The data points themselves show position.
- **Legend**: No, if you label directly.
- **Axis title**: Sometimes no — if the chart title already specifies the axis ("Monthly Revenue in USD").
- **Top/right borders**: Always no.
- **Background color other than off-white**: Always no (decoration).
- **Tick marks**: Maybe — keep if needed for reading precision, remove if axis line is sufficient.
- **Tooltip border/background**: Always no.

If the answer is "no," remove it. If the answer is "maybe," try removing it and see if the chart still communicates.
