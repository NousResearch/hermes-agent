from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from market_monitor.db import Database
from market_monitor.runners import render_structured_results


DASHBOARD_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self' http://127.0.0.1:* http://localhost:*;\" />
  <title>China EV Market Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #0b1020; color: #e8edf7; }
    h1, h2, h3 { margin: 0 0 16px; }
    .toolbar { margin: 16px 0 24px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
    select { padding: 8px 12px; border-radius: 8px; }
    .dataset { background: #151d36; border: 1px solid #283252; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    .panels { display:grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin-bottom: 16px; }
    .panel { background: #111830; border: 1px solid #223055; border-radius: 12px; padding: 12px; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #2a3558; font-size: 14px; }
    .muted { color: #9fb0d6; }
    .warnings { margin-top: 10px; color: #ffd27d; }
    .freshness { color: #7fd7a5; }
    .rank-list { margin: 0; padding-left: 18px; }
    .rank-list li { margin-bottom: 4px; }
    svg { width: 100%; height: 160px; display:block; }
    .line { fill: none; stroke: #72d5ff; stroke-width: 3; }
    .dot { fill: #ffd27d; }
    .axis-labels { display:flex; justify-content:space-between; font-size:12px; color:#9fb0d6; }
  </style>
</head>
<body>
  <h1>China EV Market Dashboard</h1>
  <div class=\"toolbar\">
    <label for=\"period-select\">Period:</label>
    <select id=\"period-select\"></select>
    <span id=\"generated-at\" class=\"muted\"></span>
    <span id=\"run-mode\" class=\"muted\"></span>
  </div>
  <div id=\"content\"></div>
  <script>
    function cell(text) {
      const td = document.createElement('td');
      td.textContent = text == null ? '' : String(text);
      return td;
    }
    function getApiBase() {
      const params = new URLSearchParams(window.location.search);
      return params.get('api');
    }
    async function loadPayload() {
      const apiBase = getApiBase();
      if (!apiBase) {
        const response = await fetch('data.json');
        return await response.json();
      }
      const base = apiBase.replace(/\/$/, '');
      const periodsResponse = await fetch(`${base}/periods`);
      const periodsPayload = await periodsResponse.json();
      const periods = periodsPayload.periods || [];
      const results = {};
      for (const period of periods) {
        const res = await fetch(`${base}/results/${encodeURIComponent(period)}`);
        results[period] = await res.json();
      }
      return {
        schema_version: periodsPayload.schema_version || '2',
        generated_at: periodsPayload.generated_at || null,
        periods,
        results,
      };
    }
    function buildSeriesMap(payload) {
      const seriesMap = new Map();
      for (const period of payload.periods) {
        const datasets = payload.results[period]?.datasets || [];
        for (const dataset of datasets) {
          for (const metric of dataset.metrics || []) {
            if (metric.value_numeric == null || metric.metric_type === 'ranking') continue;
            const key = `${dataset.dataset_id}::${metric.metric_name}::${metric.metric_scope}`;
            const entry = seriesMap.get(key) || [];
            entry.push({ period, value: metric.value_numeric, unit: metric.unit, label: metric.metric_name, scope: metric.metric_scope });
            seriesMap.set(key, entry);
          }
        }
      }
      return seriesMap;
    }
    function createSeriesChart(points) {
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.classList.add('series-chart');
      if (!points.length) return svg;
      const width = 320, height = 160, padding = 18;
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      const values = points.map(p => p.value);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max === min ? 1 : max - min;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('class', 'line');
      const commands = points.map((point, index) => {
        const x = padding + ((width - padding * 2) * index / Math.max(points.length - 1, 1));
        const y = height - padding - (((point.value - min) / span) * (height - padding * 2));
        return `${index === 0 ? 'M' : 'L'}${x},${y}`;
      });
      path.setAttribute('d', commands.join(' '));
      svg.appendChild(path);
      points.forEach((point, index) => {
        const x = padding + ((width - padding * 2) * index / Math.max(points.length - 1, 1));
        const y = height - padding - (((point.value - min) / span) * (height - padding * 2));
        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot.setAttribute('class', 'dot');
        dot.setAttribute('cx', x);
        dot.setAttribute('cy', y);
        dot.setAttribute('r', '3.5');
        svg.appendChild(dot);
      });
      return svg;
    }
    async function load() {
      const payload = await loadPayload();
      const select = document.getElementById('period-select');
      const generatedAt = document.getElementById('generated-at');
      const runMode = document.getElementById('run-mode');
      const seriesMap = buildSeriesMap(payload);
      generatedAt.textContent = payload.generated_at ? `generated: ${payload.generated_at}` : '';
      runMode.textContent = getApiBase() ? 'mode: live API' : 'mode: static bundle';
      payload.periods.forEach(period => {
        const option = document.createElement('option');
        option.value = period;
        option.textContent = period;
        select.appendChild(option);
      });
      function render(period) {
        const content = document.getElementById('content');
        while (content.firstChild) content.removeChild(content.firstChild);
        const datasets = payload.results[period]?.datasets || [];
        datasets.forEach(ds => {
          const wrapper = document.createElement('div');
          wrapper.className = 'dataset';
          const title = document.createElement('h2');
          title.textContent = ds.dataset_id;
          wrapper.appendChild(title);
          const source = document.createElement('div');
          source.className = 'muted';
          source.textContent = `source: ${ds.source_id}`;
          wrapper.appendChild(source);
          if (ds.freshness?.latest_published_at) {
            const freshness = document.createElement('div');
            freshness.className = 'freshness';
            freshness.textContent = `latest published: ${ds.freshness.latest_published_at} · metrics: ${ds.freshness.metric_count}`;
            wrapper.appendChild(freshness);
          }
          if (ds.warnings && ds.warnings.length) {
            const warnings = document.createElement('div');
            warnings.className = 'warnings';
            warnings.textContent = `warnings: ${ds.warnings.join(' | ')}`;
            wrapper.appendChild(warnings);
          }
          const panels = document.createElement('div');
          panels.className = 'panels';
          const numericSeriesMetric = ds.metrics.find(metric => metric.value_numeric != null && metric.metric_type !== 'ranking');
          if (numericSeriesMetric) {
            const chartPanel = document.createElement('div');
            chartPanel.className = 'panel';
            const chartTitle = document.createElement('h3');
            chartTitle.textContent = `${numericSeriesMetric.metric_name} trend`;
            chartPanel.appendChild(chartTitle);
            const seriesKey = `${ds.dataset_id}::${numericSeriesMetric.metric_name}::${numericSeriesMetric.metric_scope}`;
            const points = (seriesMap.get(seriesKey) || []).sort((a, b) => a.period.localeCompare(b.period));
            chartPanel.appendChild(createSeriesChart(points));
            const labels = document.createElement('div');
            labels.className = 'axis-labels';
            labels.appendChild(document.createTextNode(points[0]?.period || ''));
            labels.appendChild(document.createTextNode(points[points.length - 1]?.period || ''));
            chartPanel.appendChild(labels);
            panels.appendChild(chartPanel);
          }
          const rankingMetrics = ds.metrics.filter(metric => metric.metric_type === 'ranking').slice(0, 5);
          if (rankingMetrics.length) {
            const rankingPanel = document.createElement('div');
            rankingPanel.className = 'panel';
            const rankingTitle = document.createElement('h3');
            rankingTitle.textContent = 'Top rankings';
            rankingPanel.appendChild(rankingTitle);
            const list = document.createElement('ol');
            list.className = 'rank-list';
            rankingMetrics.forEach(metric => {
              const li = document.createElement('li');
              li.textContent = `${metric.metric_name} #${metric.ranking}: ${metric.value_text ?? metric.value_numeric ?? ''}`;
              list.appendChild(li);
            });
            rankingPanel.appendChild(list);
            panels.appendChild(rankingPanel);
          }
          if (panels.childNodes.length) wrapper.appendChild(panels);
          const table = document.createElement('table');
          const thead = document.createElement('thead');
          const headRow = document.createElement('tr');
          ['Metric','Scope','Type','Value','Unit','Rank','Latest','Revision'].forEach(label => {
            const th = document.createElement('th');
            th.textContent = label;
            headRow.appendChild(th);
          });
          thead.appendChild(headRow);
          table.appendChild(thead);
          const tbody = document.createElement('tbody');
          ds.metrics.forEach(metric => {
            const row = document.createElement('tr');
            row.appendChild(cell(metric.metric_name));
            row.appendChild(cell(metric.metric_scope));
            row.appendChild(cell(metric.metric_type));
            row.appendChild(cell(metric.value_text ?? metric.value_numeric));
            row.appendChild(cell(metric.unit));
            row.appendChild(cell(metric.ranking));
            row.appendChild(cell(metric.is_latest));
            row.appendChild(cell(metric.revision_no));
            tbody.appendChild(row);
          });
          table.appendChild(tbody);
          wrapper.appendChild(table);
          content.appendChild(wrapper);
        });
      }
      select.addEventListener('change', () => render(select.value));
      if (payload.periods.length) {
        select.value = payload.periods[0];
        render(payload.periods[0]);
      }
    }
    load();
  </script>
</body>
</html>
"""


def create_dashboard_payload(db: Database) -> dict:
    periods = [row[0] for row in db.query("SELECT DISTINCT period_label FROM observations WHERE is_latest = 1 ORDER BY period_label DESC")]
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "2",
        "generated_at": generated_at,
        "periods": periods,
        "results": {period: render_structured_results(db, period_label=period) for period in periods},
    }


def build_dashboard_bundle(*, db: Database, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = create_dashboard_payload(db)
    index_html = out_dir / "index.html"
    data_json = out_dir / "data.json"
    index_html.write_text(DASHBOARD_HTML, encoding="utf-8")
    data_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"index_html": index_html, "data_json": data_json}


def start_dashboard_server(*, bundle_dir: Path, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(bundle_dir))
    return ThreadingHTTPServer((host, port), handler)
