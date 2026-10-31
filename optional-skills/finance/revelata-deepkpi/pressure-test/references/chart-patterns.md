# Chart Patterns Reference

Every chart in an Analysis Pressure Test report uses Chart.js 4.x on a dark background
with the blue-family palette. This file shows the patterns for common chart types
you'll encounter when visualizing SEC filing data.

## Palette

```js
const B1 = '#4FEAFF';  // cyan — primary series
const B2 = '#90caf9';  // light blue — secondary
const B3 = '#5b8def';  // mid blue — tertiary
const B4 = '#a78bfa';  // periwinkle — quaternary
const grid = { color: 'rgba(255,255,255,0.05)' };
```

## Series differentiation (when you can't use color alone)

When multiple series appear on the same chart, differentiate with ALL THREE:
shade, dash pattern, and point shape.

| Series | Color | borderDash | pointStyle |
|--------|-------|-----------|------------|
| 1st    | B1    | `[]` (solid) | `'circle'` |
| 2nd    | B2    | `[6,3]` | `'rect'` |
| 3rd    | B3    | `[2,3]` | `'triangle'` |
| 4th    | B4    | `[8,4,2,4]` | `'rectRounded'` |

## Common chart types

### Single-series bar (positive/negative shading)

Good for: annual volume growth, single KPI trend with sign changes.

```js
new Chart(document.getElementById('cN'), {
  type: 'bar',
  data: {
    labels: ['FY20','FY21','FY22','FY23','FY24','FY25'],
    datasets: [{
      data: [10, 6, -5, -10, -5, 7],
      backgroundColor: [10,6,-5,-10,-5,7].map(v => v >= 0 ? B1+'99' : B1+'33'),
      borderColor: [10,6,-5,-10,-5,7].map(v => v >= 0 ? B1 : B1+'88'),
      borderWidth: 1, borderRadius: 4, barPercentage: 0.6
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => c.parsed.y + '%' }}
    },
    scales: {
      y: { ticks: { callback: v => v + '%' }, grid },
      x: { grid: { display: false }}
    }
  }
});
```

### Multi-series line (segments over time)

Good for: comparing segment volume, margin trends across divisions.

```js
new Chart(document.getElementById('cN'), {
  type: 'line',
  data: {
    labels: ['FQ1 24','FQ2 24','FQ3 24','FQ1 25','FQ2 25'],
    datasets: [
      {
        label: 'Segment A',
        data: [38, 22, -4, -16, 2],
        borderColor: B1, backgroundColor: B1+'18',
        tension: 0.35, pointRadius: 5, borderWidth: 2.5,
        borderDash: [], pointStyle: 'circle'
      },
      {
        label: 'Segment B',
        data: [43, 4, -3, -18, -3],
        borderColor: B2, backgroundColor: B2+'18',
        tension: 0.35, pointRadius: 5, borderWidth: 2.5,
        borderDash: [6,3], pointStyle: 'rect'
      }
    ]
  },
  options: {
    responsive: true,
    plugins: { legend: { labels: { color: '#8b949e' } } },
    scales: {
      y: { grid, ticks: { color: '#8b949e' } },
      x: { grid: { display: false }, ticks: { color: '#8b949e' } }
    }
  }
});
```

