# Chart.js: Tufte Configuration

Load this file when generating any Chart.js chart code.

## Global defaults

Apply at app initialization to make every Chart.js chart Tufte-compliant by default:

```javascript
import { Chart } from 'chart.js';

// Global Tufte defaults
Chart.defaults.backgroundColor = '#fffff8';
Chart.defaults.color = '#999';
Chart.defaults.font.family = '"ET Book", "Palatino Linotype", Palatino, Georgia, serif';
Chart.defaults.font.size = 12;

// Remove grid and borders globally
Chart.defaults.scale.grid.display = false;
Chart.defaults.scale.border.display = false;
Chart.defaults.scale.ticks.padding = 8;

// Remove legend globally
Chart.defaults.plugins.legend.display = false;

// Minimal tooltip
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(255,255,248,0.95)';
Chart.defaults.plugins.tooltip.titleColor = '#666';
Chart.defaults.plugins.tooltip.bodyColor = '#111';
Chart.defaults.plugins.tooltip.borderWidth = 0;
Chart.defaults.plugins.tooltip.cornerRadius = 2;
Chart.defaults.plugins.tooltip.padding = 8;
Chart.defaults.plugins.tooltip.boxPadding = 0;
Chart.defaults.plugins.tooltip.displayColors = false;
Chart.defaults.plugins.tooltip.titleFont = { family: 'system-ui, sans-serif', size: 11 };
Chart.defaults.plugins.tooltip.bodyFont = { family: '"Palatino Linotype", Georgia, serif', size: 13 };

// Line defaults
Chart.defaults.elements.line.borderWidth = 1.5;
Chart.defaults.elements.line.tension = 0;
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.point.hoverRadius = 3;
Chart.defaults.elements.point.hoverBackgroundColor = '#111';

// Bar defaults
Chart.defaults.elements.bar.borderWidth = 0;
```

## Tufte plugin

A plugin that applies Tufte styling to the canvas:

```javascript
const tuftePlugin = {
  id: 'tufte',
  beforeDraw(chart) {
    const { ctx, width, height } = chart;
    ctx.save();
    ctx.fillStyle = '#fffff8';
    ctx.fillRect(0, 0, width, height);
    ctx.restore();
  },
};

Chart.register(tuftePlugin);
```

## Color constants

```javascript
const TUFTE = {
  bg: '#fffff8',
  text: '#111',
  textSecondary: '#666',
  textTertiary: '#999',
  axis: '#ccc',
  seriesDefault: '#666',
  highlight: '#e41a1c',
  categorical: ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'],
};
```

## Scale configuration

### X-axis (category)

```javascript
scales: {
  x: {
    grid: { display: false },
    border: { display: true, color: TUFTE.axis, width: 0.5 },
    ticks: {
      color: TUFTE.textTertiary,
      font: { family: 'system-ui, sans-serif', size: 11 },
      padding: 8,
    },
  },
}
```

### Y-axis (value)

```javascript
scales: {
  y: {
    grid: { display: false },
    border: { display: false },
    ticks: {
      color: TUFTE.textTertiary,
      font: { family: 'system-ui, sans-serif', size: 11 },
      padding: 8,
    },
  },
}
```

If light horizontal gridlines are needed:

```javascript
y: {
  grid: {
    display: true,
    color: 'rgba(0, 0, 0, 0.06)',
    lineWidth: 0.5,
    drawTicks: false,
  },
}
```

## Direct labeling with chartjs-plugin-datalabels

```javascript
import ChartDataLabels from 'chartjs-plugin-datalabels';
Chart.register(ChartDataLabels);

// In dataset config:
datasets: [{
  data: values,
  backgroundColor: TUFTE.seriesDefault,
  datalabels: {
    anchor: 'end',
    align: 'right',
    color: TUFTE.textSecondary,
    font: {
      family: '"Palatino Linotype", Palatino, Georgia, serif',
      size: 13,
    },
    formatter: (value) => `$${(value / 1000).toFixed(0)}k`,
  },
}]
```

For line charts — label only the last point:

```javascript
datalabels: {
  display: (context) => context.dataIndex === context.dataset.data.length - 1,
  anchor: 'end',
  align: 'right',
  color: TUFTE.highlight,
  font: { family: '"Palatino Linotype", Georgia, serif', size: 13 },
  formatter: (value, context) => `${context.dataset.label}: ${value.toLocaleString()}`,
}
```

## Annotation plugin for notable points

```javascript
import annotationPlugin from 'chartjs-plugin-annotation';
Chart.register(annotationPlugin);

plugins: {
  annotation: {
    annotations: {
      peak: {
        type: 'label',
        xValue: 'Jun',
        yValue: 6200,
        content: 'Peak: $6,200',
        font: {
          family: '"Palatino Linotype", Georgia, serif',
          size: 12,
          style: 'italic',
        },
        color: '#333',
        backgroundColor: 'transparent',
        position: 'start',
        yAdjust: -16,
      },
      avgLine: {
        type: 'line',
        yMin: 5200,
        yMax: 5200,
        borderColor: '#ccc',
        borderWidth: 0.5,
        borderDash: [4, 3],
        label: {
          display: true,
          content: 'Avg',
          position: 'end',
          font: { size: 11 },
          color: '#999',
          backgroundColor: 'transparent',
        },
      },
    },
  },
}
```

## Complete example: line chart

```javascript
import { Chart } from 'chart.js/auto';
import ChartDataLabels from 'chartjs-plugin-datalabels';

Chart.register(ChartDataLabels);

const ctx = document.getElementById('revenue-chart').getContext('2d');

new Chart(ctx, {
  type: 'line',
  data: {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'Target',
        data: [4000, 4200, 4400, 4600, 4800, 5000],
        borderColor: TUFTE.seriesDefault,
        borderWidth: 1,
        borderDash: [4, 3],
        pointRadius: 0,
        datalabels: {
          display: (ctx) => ctx.dataIndex === ctx.dataset.data.length - 1,
          anchor: 'end', align: 'right', color: TUFTE.textSecondary,
          font: { family: '"Palatino Linotype", Georgia, serif', size: 13 },
          formatter: (v, ctx) => `Target: ${v.toLocaleString()}`,
        },
      },
      {
        label: 'Revenue',
        data: [4200, 4800, 5100, 4900, 5600, 6200],
        borderColor: TUFTE.highlight,
        borderWidth: 2,
        pointRadius: 0,
        datalabels: {
          display: (ctx) => ctx.dataIndex === ctx.dataset.data.length - 1,
          anchor: 'end', align: 'right', color: TUFTE.highlight,
          font: { family: '"Palatino Linotype", Georgia, serif', size: 13 },
          formatter: (v, ctx) => `Revenue: ${v.toLocaleString()}`,
        },
      },
    ],
  },
  options: {
    responsive: true,
    aspectRatio: 1.5,
    scales: {
      x: {
        grid: { display: false },
        border: { display: true, color: TUFTE.axis, width: 0.5 },
        ticks: { color: TUFTE.textTertiary, font: { size: 11 } },
      },
      y: {
        grid: { display: false },
        border: { display: false },
        ticks: { color: TUFTE.textTertiary, font: { size: 11 } },
      },
    },
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Monthly Revenue vs. Target',
        align: 'start',
        font: { family: '"Palatino Linotype", Georgia, serif', size: 18, weight: 'normal' },
        color: TUFTE.text,
        padding: { bottom: 16 },
      },
    },
    layout: { padding: { right: 80 } },
  },
});
```

## Complete example: horizontal bar chart

```javascript
new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
    datasets: [{
      data: [42000, 38000, 27000, 19000, 12000],
      backgroundColor: TUFTE.seriesDefault,
      borderWidth: 0,
      barThickness: 20,
      datalabels: {
        anchor: 'end', align: 'right',
        color: TUFTE.textSecondary,
        font: { family: '"Palatino Linotype", Georgia, serif', size: 13 },
        formatter: (v) => `$${(v / 1000).toFixed(0)}k`,
      },
    }],
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    aspectRatio: 1.8,
    scales: {
      x: { display: false },
      y: {
        grid: { display: false },
        border: { display: false },
        ticks: {
          color: TUFTE.text,
          font: { family: '"Palatino Linotype", Georgia, serif', size: 13 },
        },
      },
    },
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Revenue by Product',
        align: 'start',
        font: { family: '"Palatino Linotype", Georgia, serif', size: 18, weight: 'normal' },
        color: TUFTE.text,
      },
    },
    layout: { padding: { right: 60 } },
  },
});
```
