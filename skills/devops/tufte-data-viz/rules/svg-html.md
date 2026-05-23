# SVG, D3, and HTML: Tufte Configuration

Load this file when generating raw SVG charts, D3.js visualizations, or HTML data tables.

## SVG chart base template

```html
<svg viewBox="0 0 750 500" xmlns="http://www.w3.org/2000/svg"
     style="font-family: 'ET Book', 'Palatino Linotype', Palatino, Georgia, serif;
            background: #fffff8;">
  <!-- Chart area: leave margins for labels -->
  <g transform="translate(60, 40)">
    <!-- Data goes here -->
  </g>
</svg>
```

## CSS for Tufte-styled SVG

```css
.tufte-chart {
  background: #fffff8;
  font-family: "ET Book", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
}

.tufte-chart .axis-line {
  stroke: #ccc;
  stroke-width: 0.5;
}

.tufte-chart .domain {
  display: none; /* Remove D3 axis domain line — use custom range-frame instead */
}

.tufte-chart .tick line {
  stroke: #ddd;
  stroke-width: 0.5;
}

.tufte-chart .tick text {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 11px;
  fill: #999;
}

.tufte-chart .data-line {
  fill: none;
  stroke: #666;
  stroke-width: 1.5;
}

.tufte-chart .data-line.highlight {
  stroke: #e41a1c;
  stroke-width: 2;
}

.tufte-chart .data-point {
  fill: #666;
  r: 2;
}

.tufte-chart .annotation {
  font-size: 12px;
  font-style: italic;
  fill: #333;
}

.tufte-chart .direct-label {
  font-size: 13px;
  fill: #111;
}

.tufte-chart .title {
  font-size: 18px;
  font-weight: 400;
  fill: #111;
}

.tufte-chart .subtitle {
  font-size: 13px;
  fill: #666;
}

.tufte-chart .reference-line {
  stroke: #ccc;
  stroke-width: 0.5;
  stroke-dasharray: 4 3;
}
```

## D3.js axis configuration

### Remove default domain line, keep only ticks

```javascript
// After creating axes:
svg.selectAll('.domain').remove();

// Or via CSS (preferred):
// .domain { display: none; }
```

### Range-frame axis

```javascript
const xScale = d3.scaleLinear()
  .domain([d3.min(data, d => d.x), d3.max(data, d => d.x)])
  .range([0, width]);

// Draw axis line only over data range
svg.append('line')
  .attr('x1', xScale(d3.min(data, d => d.x)))
  .attr('x2', xScale(d3.max(data, d => d.x)))
  .attr('y1', height)
  .attr('y2', height)
  .attr('stroke', '#ccc')
  .attr('stroke-width', 0.5);
```

### Minimal tick styling

```javascript
const xAxis = d3.axisBottom(xScale)
  .tickSize(3)        // Short inward ticks
  .tickPadding(8);

const yAxis = d3.axisLeft(yScale)
  .tickSize(3)
  .tickPadding(8);

// Apply and style
svg.append('g')
  .attr('transform', `translate(0, ${height})`)
  .call(xAxis)
  .call(g => g.selectAll('.domain').remove())
  .call(g => g.selectAll('.tick text')
    .attr('fill', '#999')
    .attr('font-size', '11px')
    .attr('font-family', 'system-ui, sans-serif'));
```

### Direct labeling (replace legend)

```javascript
// Label at the last data point of each series
series.forEach(s => {
  const lastPoint = s.data[s.data.length - 1];
  svg.append('text')
    .attr('x', xScale(lastPoint.x) + 8)
    .attr('y', yScale(lastPoint.y))
    .attr('dy', '0.35em')
    .attr('fill', s.color || '#111')
    .attr('font-size', '13px')
    .attr('font-family', '"ET Book", Palatino, Georgia, serif')
    .text(s.name);
});
```

### Annotation

```javascript
function annotate(svg, x, y, text, xScale, yScale) {
  const g = svg.append('g')
    .attr('transform', `translate(${xScale(x)}, ${yScale(y)})`);

  g.append('line')
    .attr('y2', -20)
    .attr('stroke', '#ccc')
    .attr('stroke-width', 0.5);

  g.append('text')
    .attr('y', -24)
    .attr('text-anchor', 'middle')
    .attr('font-size', '12px')
    .attr('font-style', 'italic')
    .attr('fill', '#333')
    .attr('font-family', '"ET Book", Palatino, Georgia, serif')
    .text(text);
}
```

## Inline SVG sparkline

A minimal sparkline for embedding in text or table cells:

```html
<svg viewBox="0 0 80 20" width="80" height="20" style="vertical-align: middle;">
  <polyline
    points="0,15 10,12 20,8 30,14 40,6 50,10 60,3 70,9 80,5"
    fill="none"
    stroke="#666"
    stroke-width="1"
  />
  <!-- Min point (red dot) -->
  <circle cx="60" cy="3" r="1.5" fill="#e15759" />
  <!-- Max point (blue dot) — find from data -->
  <circle cx="0" cy="15" r="1.5" fill="#4e79a7" />
  <!-- Endpoint -->
  <circle cx="80" cy="5" r="1.5" fill="#666" />
</svg>
```

### Sparkline generator function (JavaScript)

```javascript
function sparkline(data, width = 80, height = 20) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(' ');

  const minIdx = data.indexOf(min);
  const maxIdx = data.indexOf(max);
  const minX = (minIdx / (data.length - 1)) * width;
  const minY = height - 2;
  const maxX = (maxIdx / (data.length - 1)) * width;
  const maxY = 2;
  const endX = width;
  const endY = height - ((data[data.length - 1] - min) / range) * (height - 4) - 2;

  return `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="vertical-align:middle">
    <polyline points="${points}" fill="none" stroke="#666" stroke-width="1"/>
    <circle cx="${minX}" cy="${minY}" r="1.5" fill="#e15759"/>
    <circle cx="${maxX}" cy="${maxY}" r="1.5" fill="#4e79a7"/>
    <circle cx="${endX}" cy="${endY}" r="1.5" fill="#666"/>
  </svg>`;
}
```

## HTML data tables — Tufte style

### CSS

```css
.tufte-table {
  font-family: "ET Book", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 14px;
  color: #111;
  border-collapse: collapse;
  width: 100%;
  background: #fffff8;
}

.tufte-table th {
  font-weight: normal;
  color: #666;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 8px 12px;
  border-bottom: 1px solid #999;
  text-align: left;
}

.tufte-table td {
  padding: 6px 12px;
  border: none;
  font-feature-settings: "onum" 1;
  font-variant-numeric: oldstyle-nums;
}

/* Right-align numbers */
.tufte-table td.num {
  text-align: right;
  font-variant-numeric: tabular-nums oldstyle-nums;
}

/* Thin separator every 3-5 rows instead of zebra striping */
.tufte-table tr.separator td {
  border-top: 1px solid #e0e0e0;
}

/* No zebra striping. No cell borders. No background colors. */
/* The data IS the design. */
```

### HTML structure

```html
<table class="tufte-table">
  <thead>
    <tr>
      <th>Product</th>
      <th style="text-align: right">Revenue</th>
      <th style="text-align: right">Growth</th>
      <th>Trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Product A</td>
      <td class="num">$42,000</td>
      <td class="num">+12%</td>
      <td><!-- sparkline SVG here --></td>
    </tr>
    <tr>
      <td>Product B</td>
      <td class="num">$38,000</td>
      <td class="num">+8%</td>
      <td><!-- sparkline SVG here --></td>
    </tr>
    <!-- Add class="separator" every 3-5 rows -->
    <tr class="separator">
      <td>Product C</td>
      <td class="num">$27,000</td>
      <td class="num">+3%</td>
      <td><!-- sparkline SVG here --></td>
    </tr>
  </tbody>
</table>
```

Key rules for tables:
- **No zebra striping** — creates moire vibration
- **No cell borders** — use whitespace for structure
- **Right-align all numbers** — decimal alignment
- **Old-style figures** — `font-feature-settings: "onum" 1`
- **Thin separator rules** every 3-5 rows, not on every row
- **Header bottom border** only — 1px solid, slightly darker than separators
