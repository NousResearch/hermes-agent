# Small Multiples, Sparklines, and Advanced Techniques

## Small multiples

Small multiples are "postage-stamp-sized illustrations indexed by category or label, sequenced over time" (Tufte). They replace complex multi-series charts with repeated, consistently scaled panels.

### When to use

- Comparing the same metric across 3+ categories
- Showing the same variable over time for different groups
- Any situation where a single chart would need >4 series or dual y-axes

### Rules

1. **Same scale across ALL panels.** Never let each panel auto-scale. This is the most common mistake — it makes visual comparison impossible.
2. **Label each panel title, not each data element.** One title per panel is sufficient.
3. **Shared axis labels.** X-axis labels only on the bottom row. Y-axis labels only on the leftmost column.
4. **No panel borders.** Use whitespace to separate panels. If borders are needed, use 1px #eee.
5. **Compact layout.** Panels should be close together so the eye can compare without head movement.

### Recharts layout

```tsx
const categories = ['Region A', 'Region B', 'Region C', 'Region D'];
const sharedDomain = [globalMin, globalMax]; // Compute from all data

<div style={{
  display: 'grid',
  gridTemplateColumns: 'repeat(2, 1fr)',
  gap: '8px 16px',
  background: '#fffff8',
  padding: 16,
}}>
  {categories.map((cat, i) => (
    <div key={cat}>
      <div style={{
        fontFamily: '"ET Book", Palatino, Georgia, serif',
        fontSize: 14, color: '#111', marginBottom: 4,
      }}>
        {cat}
      </div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data[cat]}>
          <CartesianGrid stroke="none" />
          <XAxis
            dataKey="x"
            tickLine={false}
            axisLine={i >= categories.length - 2 ? { stroke: '#ccc', strokeWidth: 0.5 } : false}
            tick={i >= categories.length - 2 ? { fontSize: 10, fill: '#999' } : false}
          />
          <YAxis
            domain={sharedDomain}
            tickLine={false}
            axisLine={false}
            tick={i % 2 === 0 ? { fontSize: 10, fill: '#999' } : false}
            width={i % 2 === 0 ? 35 : 5}
          />
          <Line dataKey="value" stroke="#666" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  ))}
</div>
```

### matplotlib layout

```python
import matplotlib.pyplot as plt

categories = ['Region A', 'Region B', 'Region C', 'Region D']
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

global_min = min(min(d) for d in all_data.values())
global_max = max(max(d) for d in all_data.values())

for ax, cat in zip(axes.flat, categories):
    ax.plot(x_data, all_data[cat], color='#666', linewidth=1.5)
    ax.set_title(cat, fontsize=14, fontfamily='serif', fontweight='normal', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(global_min * 0.95, global_max * 1.05)
    ax.tick_params(labelsize=10)

# Only show x-labels on bottom row, y-labels on left column
for ax in axes[0, :]:
    ax.tick_params(labelbottom=False)
for ax in axes[:, 1]:
    ax.tick_params(labelleft=False)

plt.tight_layout()
plt.show()
```

### ECharts layout (grid-based)

```typescript
// Use ECharts' multi-grid feature for synchronized small multiples
const option = {
  backgroundColor: '#fffff8',
  grid: categories.map((_, i) => ({
    left: `${(i % 2) * 50 + 8}%`,
    top: `${Math.floor(i / 2) * 50 + 8}%`,
    width: '38%',
    height: '35%',
  })),
  xAxis: categories.map((_, i) => ({
    gridIndex: i,
    type: 'category',
    data: xLabels,
    axisLine: { lineStyle: { color: '#ccc', width: 0.5 } },
    axisTick: { show: false },
    axisLabel: { show: Math.floor(i / 2) === 1, fontSize: 10, color: '#999' },
    splitLine: { show: false },
  })),
  yAxis: categories.map((_, i) => ({
    gridIndex: i,
    type: 'value',
    min: globalMin,
    max: globalMax,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { show: i % 2 === 0, fontSize: 10, color: '#999' },
    splitLine: { show: false },
  })),
  series: categories.map((cat, i) => ({
    type: 'line',
    xAxisIndex: i,
    yAxisIndex: i,
    data: allData[cat],
    lineStyle: { color: '#666', width: 1.5 },
    showSymbol: false,
  })),
};
```

## Sparklines

"Small, intense, simple, word-sized graphics with typographic resolution" — Tufte.

### Rules

1. **No axes, no labels, no gridlines.** The sparkline shows shape only.
2. **Mark min and max** with small colored dots (r=1.5). Min = coral (#e15759), max = blue (#4e79a7).
3. **Optionally mark the endpoint** with a gray dot.
4. **Same height as surrounding text.** Typically 16–24px tall.
5. **Embed inline** in text, table cells, or dashboards.

### SVG sparkline (inline)

```javascript
function sparklineSvg(data, { width = 80, height = 20, color = '#666' } = {}) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const n = data.length;

  const y = (v) => height - 2 - ((v - min) / range) * (height - 4);
  const x = (i) => (i / (n - 1)) * width;

  const points = data.map((v, i) => `${x(i)},${y(v)}`).join(' ');
  const minIdx = data.indexOf(min);
  const maxIdx = data.indexOf(max);

  return `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="vertical-align:middle">
    <polyline points="${points}" fill="none" stroke="${color}" stroke-width="1"/>
    <circle cx="${x(minIdx)}" cy="${y(min)}" r="1.5" fill="#e15759"/>
    <circle cx="${x(maxIdx)}" cy="${y(max)}" r="1.5" fill="#4e79a7"/>
    <circle cx="${x(n - 1)}" cy="${y(data[n - 1])}" r="1.5" fill="${color}"/>
  </svg>`;
}
```

### Recharts sparkline component

```tsx
import { ResponsiveContainer, LineChart, Line } from 'recharts';

function Sparkline({ data, dataKey = 'value', width = 80, height = 20 }) {
  return (
    <ResponsiveContainer width={width} height={height}>
      <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <Line
          type="monotone"
          dataKey={dataKey}
          stroke="#666"
          strokeWidth={1}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
```

### matplotlib sparkline

```python
def sparkline(ax, data, color='#666'):
    ax.plot(data, color=color, linewidth=1)
    ax.set_xlim(0, len(data) - 1)

    # Min/max markers
    min_i, max_i = data.index(min(data)), data.index(max(data))
    ax.plot(min_i, data[min_i], 'o', color='#e15759', markersize=2, zorder=5)
    ax.plot(max_i, data[max_i], 'o', color='#4e79a7', markersize=2, zorder=5)
    ax.plot(len(data)-1, data[-1], 'o', color=color, markersize=2, zorder=5)

    # Remove all chrome
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
```

## Slopegraphs

Show change between two time points across multiple categories. Good for before/after or year-over-year comparisons.

### Rules

1. **Label both endpoints** with category name and value.
2. **Lines colored gray** by default; highlight 1-2 key slopes with accent color.
3. **No axes** — the labels provide all necessary reference.
4. **Resolve overlapping labels** by nudging vertically.

### Implementation pattern (SVG/D3)

```javascript
function slopegraph(data, { width = 400, height = 500, margin = 40 } = {}) {
  // data: [{ name, start, end }]
  const yScale = d3.scaleLinear()
    .domain([d3.min(data, d => Math.min(d.start, d.end)),
             d3.max(data, d => Math.max(d.start, d.end))])
    .range([height - margin, margin]);

  const x1 = margin + 80;  // Left column
  const x2 = width - margin - 80;  // Right column

  const svg = d3.select('#slopegraph')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .style('background', '#fffff8');

  // Column headers
  svg.append('text').attr('x', x1).attr('y', margin - 12)
    .text('2024').attr('text-anchor', 'end').attr('fill', '#666').attr('font-size', 12);
  svg.append('text').attr('x', x2).attr('y', margin - 12)
    .text('2025').attr('text-anchor', 'start').attr('fill', '#666').attr('font-size', 12);

  data.forEach(d => {
    const color = d.highlight ? '#e41a1c' : '#999';

    // Slope line
    svg.append('line')
      .attr('x1', x1 + 4).attr('y1', yScale(d.start))
      .attr('x2', x2 - 4).attr('y2', yScale(d.end))
      .attr('stroke', color).attr('stroke-width', d.highlight ? 2 : 1);

    // Left label
    svg.append('text').attr('x', x1 - 8).attr('y', yScale(d.start))
      .attr('dy', '0.35em').attr('text-anchor', 'end')
      .attr('fill', color).attr('font-size', 13)
      .text(`${d.name} ${d.start}`);

    // Right label
    svg.append('text').attr('x', x2 + 8).attr('y', yScale(d.end))
      .attr('dy', '0.35em').attr('text-anchor', 'start')
      .attr('fill', color).attr('font-size', 13)
      .text(`${d.end} ${d.name}`);
  });
}
```

## Dot-dash plots (rug marks)

Show marginal distributions along axes of a scatter plot.

### matplotlib

```python
def rug_marks(ax, x_data, y_data, color='#999', size=4, alpha=0.4):
    """Add rug marks along axes."""
    y_lo = ax.get_ylim()[0]
    x_lo = ax.get_xlim()[0]
    ax.plot(x_data, [y_lo]*len(x_data), '|', color=color, markersize=size, alpha=alpha,
            clip_on=False)
    ax.plot([x_lo]*len(y_data), y_data, '_', color=color, markersize=size, alpha=alpha,
            clip_on=False)
```

### SVG/D3

```javascript
// X-axis rug marks
data.forEach(d => {
  svg.append('line')
    .attr('x1', xScale(d.x)).attr('x2', xScale(d.x))
    .attr('y1', height).attr('y2', height + 4)
    .attr('stroke', '#999').attr('stroke-width', 0.5).attr('opacity', 0.4);
});

// Y-axis rug marks
data.forEach(d => {
  svg.append('line')
    .attr('x1', -4).attr('x2', 0)
    .attr('y1', yScale(d.y)).attr('y2', yScale(d.y))
    .attr('stroke', '#999').attr('stroke-width', 0.5).attr('opacity', 0.4);
});
```
