/**
 * Tufte defaults and plugin for Chart.js.
 *
 * Usage:
 *   import { applyTufteDefaults, tuftePlugin, TUFTE } from './chartjs-tufte-plugin';
 *   applyTufteDefaults(Chart);
 *   Chart.register(tuftePlugin);
 */

export const TUFTE = {
  bg: '#fffff8',
  text: '#111',
  textSecondary: '#666',
  textTertiary: '#999',
  axis: '#ccc',
  seriesDefault: '#666',
  highlight: '#e41a1c',
  categorical: ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'],
  font: '"ET Book", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif',
  fontSans: 'system-ui, -apple-system, sans-serif',
};

/**
 * Plugin that paints the off-white background.
 */
export const tuftePlugin = {
  id: 'tufte-background',
  beforeDraw(chart) {
    const { ctx, width, height } = chart;
    ctx.save();
    ctx.fillStyle = TUFTE.bg;
    ctx.fillRect(0, 0, width, height);
    ctx.restore();
  },
};

/**
 * Apply Tufte defaults to Chart.js globally.
 */
export function applyTufteDefaults(Chart) {
  // Colors and fonts
  Chart.defaults.backgroundColor = TUFTE.bg;
  Chart.defaults.color = TUFTE.textTertiary;
  Chart.defaults.font.family = TUFTE.font;
  Chart.defaults.font.size = 12;

  // Remove grid and borders
  Chart.defaults.scale.grid.display = false;
  Chart.defaults.scale.border.display = false;
  Chart.defaults.scale.ticks.padding = 8;

  // Remove legend
  Chart.defaults.plugins.legend.display = false;

  // Minimal tooltip
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(255,255,248,0.95)';
  Chart.defaults.plugins.tooltip.titleColor = TUFTE.textSecondary;
  Chart.defaults.plugins.tooltip.bodyColor = TUFTE.text;
  Chart.defaults.plugins.tooltip.borderWidth = 0;
  Chart.defaults.plugins.tooltip.cornerRadius = 2;
  Chart.defaults.plugins.tooltip.padding = 8;
  Chart.defaults.plugins.tooltip.boxPadding = 0;
  Chart.defaults.plugins.tooltip.displayColors = false;
  Chart.defaults.plugins.tooltip.titleFont = {
    family: TUFTE.fontSans,
    size: 11,
  };
  Chart.defaults.plugins.tooltip.bodyFont = {
    family: TUFTE.font,
    size: 13,
  };

  // Line defaults
  Chart.defaults.elements.line.borderWidth = 1.5;
  Chart.defaults.elements.line.tension = 0;
  Chart.defaults.elements.point.radius = 0;
  Chart.defaults.elements.point.hoverRadius = 3;
  Chart.defaults.elements.point.hoverBackgroundColor = TUFTE.text;

  // Bar defaults
  Chart.defaults.elements.bar.borderWidth = 0;
}

/**
 * Example: create a Tufte line chart.
 *
 * Requires: chart.js/auto, chartjs-plugin-datalabels
 */
export function createTufteLineChart(ctx) {
  // Assumes Chart and ChartDataLabels are already registered

  return new Chart(ctx, {
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
            display: (ctx) =>
              ctx.dataIndex === ctx.dataset.data.length - 1,
            anchor: 'end',
            align: 'right',
            color: TUFTE.textSecondary,
            font: { family: TUFTE.font, size: 13 },
            formatter: (v, ctx) =>
              `Target: ${v.toLocaleString()}`,
          },
        },
        {
          label: 'Revenue',
          data: [4200, 4800, 5100, 4900, 5600, 6200],
          borderColor: TUFTE.highlight,
          borderWidth: 2,
          pointRadius: 0,
          datalabels: {
            display: (ctx) =>
              ctx.dataIndex === ctx.dataset.data.length - 1,
            anchor: 'end',
            align: 'right',
            color: TUFTE.highlight,
            font: { family: TUFTE.font, size: 13 },
            formatter: (v, ctx) =>
              `Revenue: ${v.toLocaleString()}`,
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
          ticks: {
            color: TUFTE.textTertiary,
            font: { family: TUFTE.fontSans, size: 11 },
          },
        },
        y: {
          grid: { display: false },
          border: { display: false },
          ticks: {
            color: TUFTE.textTertiary,
            font: { family: TUFTE.fontSans, size: 11 },
          },
        },
      },
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: 'Monthly Revenue vs. Target',
          align: 'start',
          font: {
            family: TUFTE.font,
            size: 18,
            weight: 'normal',
          },
          color: TUFTE.text,
          padding: { bottom: 16 },
        },
      },
      layout: { padding: { right: 80 } },
    },
  });
}
