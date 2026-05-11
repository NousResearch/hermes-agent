"""Hermes operator dashboard — single-file web server.

Serves a live dashboard at http://localhost:7788 showing task history,
structured log events, slow tools, and loop detections.

Usage:
    python3 -m hermes_cli.dashboard            # default port 7788
    python3 -m hermes_cli.dashboard --port 8080
    hermes-ops serve                           # alias via ops.py CLI

No external dependencies — uses stdlib http.server only.
Data comes from ops.py which reads ~/.hermes/state.db and structured.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Embedded HTML dashboard (served at /)
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hermes Dashboard</title>
<style>
  :root {
    --bg: #0b0c10;
    --surface: #161922;
    --surface-2: #1b2030;
    --border: #2b3142;
    --border-soft: #24293a;
    --accent: #8b7cf7;
    --accent-2: #5fb0ff;
    --green: #3ecf8e;
    --red: #ff7b7b;
    --yellow: #f5c451;
    --blue: #6aa8ff;
    --muted: #7f8aa3;
    --text: #e8edf7;
    --text-dim: #a4afc3;
    --shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { background: var(--bg); }
  body {
    min-height: 100vh;
    background:
      radial-gradient(circle at top left, rgba(139, 124, 247, 0.12), transparent 28%),
      radial-gradient(circle at top right, rgba(95, 176, 255, 0.08), transparent 26%),
      linear-gradient(180deg, #0b0c10 0%, #0d1017 100%);
    color: var(--text);
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
    line-height: 1.5;
    letter-spacing: 0.01em;
    padding: 16px;
  }
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent 20%);
  }
  .shell {
    width: min(1460px, calc(100vw - 32px));
    margin: 0 auto;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 18px 20px;
    background: linear-gradient(180deg, rgba(27, 32, 48, 0.95), rgba(22, 25, 34, 0.96));
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
  }
  .brand { min-width: 0; }
  header h1 {
    font-size: 17px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.02em;
    margin-bottom: 3px;
  }
  .brand .subtle {
    color: var(--text-dim);
    font-size: 12px;
  }
  #status-bar {
    font-size: 11px;
    color: var(--muted);
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  .status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 9px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.03);
    color: var(--text-dim);
    white-space: nowrap;
  }
  .status-chip strong { color: var(--text); font-weight: 600; }
  .status-chip.live { color: var(--green); border-color: rgba(62, 207, 142, 0.28); background: rgba(62, 207, 142, 0.08); }
  .status-chip.live strong { color: var(--green); }
  #refresh-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    display: inline-block;
    transition: opacity 0.2s, transform 0.2s, background 0.2s;
    box-shadow: 0 0 0 4px rgba(62, 207, 142, 0.12);
  }
  #refresh-dot.stale { background: var(--yellow); box-shadow: 0 0 0 4px rgba(245, 196, 81, 0.10); }

  #error-banner {
    display: none;
    margin: 14px 0 0;
    padding: 12px 14px;
    border: 1px solid rgba(255, 123, 123, 0.22);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(69, 10, 10, 0.95), rgba(38, 15, 18, 0.96));
    color: #fca5a5;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    box-shadow: var(--shadow);
  }
  #error-banner.visible { display: flex; }
  #error-banner .message { font-size: 12px; line-height: 1.45; }
  #error-banner button {
    border: 1px solid rgba(255, 183, 183, 0.22);
    background: rgba(255, 255, 255, 0.06);
    color: #ffd2d2;
    border-radius: 999px;
    padding: 6px 12px;
    font: inherit;
    cursor: pointer;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-top: 16px;
  }
  .stat-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, rgba(27, 32, 48, 0.98), rgba(22, 25, 34, 0.98));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px 16px 14px;
    box-shadow: var(--shadow);
  }
  .stat-card::before {
    content: '';
    position: absolute;
    inset: 0 auto auto 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, rgba(139, 124, 247, 0.75), rgba(95, 176, 255, 0.18));
  }
  .stat-card .label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .stat-card .value {
    font-size: 28px;
    font-weight: 750;
    line-height: 1;
    font-variant-numeric: tabular-nums;
  }
  .stat-card .value.green { color: var(--green); }
  .stat-card .value.red { color: var(--red); }
  .stat-card .value.blue { color: var(--blue); }
  .stat-card .value.accent { color: var(--accent); }

  .section {
    margin-top: 16px;
    background: linear-gradient(180deg, rgba(24, 28, 41, 0.96), rgba(18, 21, 30, 0.98));
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .section h2 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    padding: 15px 18px 11px;
    border-bottom: 1px solid var(--border-soft);
    background: rgba(255, 255, 255, 0.01);
  }
  .section > :not(h2) { padding-left: 18px; padding-right: 18px; }

  table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
  }
  th {
    text-align: left;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--muted);
    padding: 10px 10px;
    border-bottom: 1px solid var(--border-soft);
    font-weight: 600;
    white-space: nowrap;
  }
  td {
    padding: 10px 10px;
    border-bottom: 1px solid rgba(43, 49, 66, 0.58);
    vertical-align: top;
    font-size: 12px;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: rgba(255, 255, 255, 0.03); }

  .badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .badge.completed { background: rgba(20, 83, 45, 0.48); color: var(--green); }
  .badge.failed { background: rgba(69, 10, 10, 0.62); color: var(--red); }
  .badge.interrupted { background: rgba(69, 26, 3, 0.62); color: var(--yellow); }
  .badge.running { background: rgba(30, 58, 95, 0.55); color: var(--blue); }
  .badge.unknown { background: rgba(30, 30, 42, 0.9); color: var(--muted); }

  .mono-dim { color: var(--text-dim); font-size: 11px; }
  .err-text { color: var(--red); font-size: 11px; }
  .event-type { color: var(--accent); }
  .tool-name { color: var(--accent-2); }
  .dur { color: var(--yellow); }

  #models {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding-bottom: 16px;
  }
  .model-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 6px 11px;
    font-size: 11px;
    color: var(--text);
  }
  .model-chip span {
    color: var(--accent);
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }

  .empty { color: var(--muted); padding: 16px; text-align: center; font-size: 12px; }
  .two-col { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-top: 16px; }

  .project-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    padding: 16px 18px 18px;
  }
  .project-card {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(27, 32, 48, 0.96), rgba(18, 21, 30, 0.98));
    padding: 14px 14px 13px;
  }
  .project-card.running {
    border-color: rgba(95, 176, 255, 0.35);
    box-shadow: inset 0 0 0 1px rgba(95, 176, 255, 0.12);
  }
  .project-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
  }
  .project-name {
    font-size: 14px;
    font-weight: 700;
    line-height: 1.25;
    color: var(--text);
    margin-bottom: 4px;
    word-break: break-word;
  }
  .project-subtitle {
    color: var(--text-dim);
    font-size: 11px;
  }
  .project-badges {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 6px;
  }
  .project-stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
    margin-top: 10px;
  }
  .project-stat {
    border: 1px solid rgba(43, 49, 66, 0.72);
    border-radius: 12px;
    padding: 8px 9px;
    background: rgba(255, 255, 255, 0.03);
  }
  .project-stat .k {
    display: block;
    font-size: 9px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 4px;
  }
  .project-stat .v {
    font-size: 12px;
    font-weight: 650;
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .project-footer {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(43, 49, 66, 0.6);
    color: var(--text-dim);
    font-size: 11px;
    line-height: 1.45;
    min-height: 2.8em;
  }
  .project-preview {
    color: var(--text-dim);
    font-size: 11px;
    margin-top: 6px;
  }
  .project-empty {
    padding: 18px 18px 20px;
    color: var(--muted);
    text-align: center;
    font-size: 12px;
  }
  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    width: 100%;
  }
  .section-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }
  .section-controls select,
  .section-controls input[type="text"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(80, 90, 120, 0.55);
    color: var(--text);
    border-radius: 10px;
    padding: 8px 10px;
    font-size: 12px;
  }
  .section-controls input[type="text"] { min-width: 180px; }
  .mini-btn {
    border: 1px solid rgba(80, 90, 120, 0.65);
    background: rgba(255, 255, 255, 0.05);
    color: var(--text);
    border-radius: 10px;
    padding: 8px 10px;
    font-size: 12px;
    cursor: pointer;
  }
  .mini-btn:hover { border-color: rgba(120, 180, 255, 0.7); }
  .mini-btn.danger:hover { border-color: rgba(255, 120, 120, 0.8); }
  .project-edit {
    display: flex;
    gap: 8px;
    margin-top: 10px;
  }
  .project-edit input {
    flex: 1;
    min-width: 0;
  }
  .project-meta {
    margin-top: 8px;
    color: var(--text-dim);
    font-size: 11px;
  }
  .chat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    padding: 16px 18px 18px;
  }
  .chat-card {
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(27, 32, 48, 0.96), rgba(18, 21, 30, 0.98));
    padding: 14px;
  }
  .chat-card.pinned { border-color: rgba(232, 193, 78, 0.45); box-shadow: inset 0 0 0 1px rgba(232, 193, 78, 0.12); }
  .chat-name { font-size: 14px; font-weight: 700; margin-bottom: 4px; word-break: break-word; }
  .chat-subtitle { color: var(--text-dim); font-size: 11px; line-height: 1.4; }
  .chat-preview { color: var(--text-dim); font-size: 11px; margin-top: 8px; min-height: 2.5em; }
  .chat-edit {
    display: flex;
    gap: 8px;
    margin-top: 10px;
  }
  .chat-edit input {
    flex: 1;
    min-width: 0;
  }
  .chat-meta {
    margin-top: 8px;
    color: var(--text-dim);
    font-size: 11px;
  }
  .chat-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .chat-actions .mini-btn.pinned { border-color: rgba(232, 193, 78, 0.75); }
  .chat-empty {
    padding: 18px 18px 20px;
    color: var(--muted);
    text-align: center;
    font-size: 12px;
  }

  @media (max-width: 1180px) {
    .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .two-col { grid-template-columns: 1fr; }
    .project-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 780px) {
    body { padding: 12px; }
    header { padding: 16px; border-radius: 16px; align-items: flex-start; flex-direction: column; }
    #status-bar { justify-content: flex-start; }
    .grid { grid-template-columns: 1fr; }
    .stat-card .value { font-size: 26px; }
    .section h2 { padding: 13px 16px 10px; }
    .section > :not(h2) { padding-left: 14px; padding-right: 14px; }
    th, td { padding-left: 8px; padding-right: 8px; }
    .project-grid { padding-left: 14px; padding-right: 14px; }
  }
  @media (max-width: 620px) {
    .brand .subtle { font-size: 11px; }
    #status-bar { gap: 6px; }
    .status-chip { width: 100%; justify-content: space-between; }
    .status-chip.live { width: auto; }
  }
</style>
</head>
<body>
<div class="shell">
<header>
  <div class="brand">
    <h1>Hermes Dashboard</h1>
    <div class="subtle">Live ops surface · local-first · low-latency refresh</div>
  </div>
  <div id="status-bar" aria-label="Dashboard status">
    <span class="status-chip live"><span id="refresh-dot"></span><strong>Live</strong></span>
    <span class="status-chip"><strong id="last-refresh">—</strong></span>
    <span class="status-chip"><strong id="task-count">—</strong></span>
    <span class="status-chip"><strong id="project-count">—</strong></span>
  </div>
</header>

<div id="error-banner" role="alert" aria-live="polite">
  <div class="message" id="error-banner-message">Dashboard refresh failed.</div>
  <button type="button" id="error-banner-retry">Retry now</button>
</div>

<div class="grid" id="stat-cards">
  <div class="stat-card"><div class="label">Total Tasks</div><div class="value accent" id="s-total">—</div></div>
  <div class="stat-card"><div class="label">Completed</div><div class="value green" id="s-completed">—</div></div>
  <div class="stat-card"><div class="label">Failed</div><div class="value red" id="s-failed">—</div></div>
  <div class="stat-card"><div class="label">Avg Tokens Out</div><div class="value blue" id="s-tokens">—</div></div>
</div>

<div class="section">
  <h2>
    <div class="section-header">
      <span>Projects</span>
      <div class="section-controls">
        <label class="mono-dim" for="project-sort">Sort</label>
        <select id="project-sort">
          <option value="pinned">Pinned first</option>
          <option value="recent" selected>Recent</option>
          <option value="title">Title</option>
        </select>
      </div>
    </div>
  </h2>
  <div id="projects" class="project-grid">
    <div class="project-empty">Loading…</div>
  </div>
</div>

<div class="section">
  <h2>
    <div class="section-header">
      <span>Chats</span>
      <div class="section-controls">
        <span class="mono-dim" id="chat-count">—</span>
        <button class="mini-btn" type="button" id="chat-prune" title="Delete older unpinned threads while keeping the newest five">Prune old unpinned threads</button>
      </div>
    </div>
  </h2>
  <div id="chats" class="chat-grid">
    <div class="chat-empty">Loading…</div>
  </div>
</div>

<div class="section">
  <h2>Models Used</h2>
  <div id="models"><span class="empty">Loading…</span></div>
</div>

<div class="two-col">
  <div class="section">
    <h2>Recent Tasks</h2>
    <table>
      <thead><tr><th>Status</th><th>Model</th><th>Step</th><th>Updated</th><th>Error</th></tr></thead>
      <tbody id="task-body"><tr><td colspan="5" class="empty">Loading…</td></tr></tbody>
    </table>
  </div>
  <div class="section">
    <h2>Recent Events</h2>
    <table>
      <thead><tr><th>Time</th><th>Event</th><th>Tool / Model</th><th>Duration</th></tr></thead>
      <tbody id="event-body"><tr><td colspan="4" class="empty">Loading…</td></tr></tbody>
    </table>
  </div>
</div>

<div class="two-col" style="padding-bottom: 16px;">
  <div class="section">
    <h2>Slow Tools (config threshold)</h2>
    <table>
      <thead><tr><th>Tool</th><th>Duration</th><th>Task</th></tr></thead>
      <tbody id="slow-body"><tr><td colspan="3" class="empty">Loading…</td></tr></tbody>
    </table>
  </div>
  <div class="section">
    <h2>Loop Detections</h2>
    <table>
      <thead><tr><th>Time</th><th>Tool</th><th>Type</th><th>Count</th></tr></thead>
      <tbody id="loop-body"><tr><td colspan="4" class="empty">Loading…</td></tr></tbody>
    </table>
  </div>
</div>
</div>

<script>
const FETCH_TIMEOUT_MS = 2500;
let refreshInFlight = false;
let refreshQueued = false;

const fmtTs = ts => {
  if (!ts) return '—';
  let d;
  if (typeof ts === 'number') d = new Date(ts * 1000);
  else d = new Date(ts);
  if (isNaN(d)) return String(ts).slice(0, 16);
  const pad = n => String(n).padStart(2, '0');
  return `${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
};

const statusBadge = s => {
  const raw = String(s ?? '?');
  const cls = ['completed','failed','interrupted','running'].includes(s) ? s : 'unknown';
  return `<span class="badge ${cls}" title="${esc(raw)}">${esc(raw)}</span>`;
};

const fmt = n => n >= 1000 ? `${(n/1000).toFixed(1)}s` : `${Math.round(n)}ms`;
const esc = s => String(s ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');
const clip = (s, n) => {
  const v = String(s ?? '');
  return v.length > n ? `${v.slice(0, n - 1)}…` : v;
};

const banner = document.getElementById('error-banner');
const bannerMessage = document.getElementById('error-banner-message');
const bannerRetry = document.getElementById('error-banner-retry');
const refreshDot = document.getElementById('refresh-dot');
const lastRefreshEl = document.getElementById('last-refresh');

function showBanner(message) {
  bannerMessage.textContent = message;
  banner.classList.add('visible');
}

function hideBanner() {
  banner.classList.remove('visible');
}

function setHealthy() {
  refreshDot.style.background = 'var(--green)';
  refreshDot.classList.remove('stale');
}

function setStale() {
  refreshDot.style.background = '';
  refreshDot.classList.add('stale');
}

async function fetchJSON(path, timeoutMs = FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetch(path, { signal: controller.signal, cache: 'no-store' });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error(`${path} ${r.status}${text ? `: ${text.slice(0, 120)}` : ''}`);
    }
    return await r.json();
  } finally {
    clearTimeout(timer);
  }
}

async function postJSON(path, body, timeoutMs = FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetch(path, {
      method: 'POST',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error(`${path} ${r.status}${text ? `: ${text.slice(0, 120)}` : ''}`);
    }
    return await r.json();
  } finally {
    clearTimeout(timer);
  }
}

function renderSummary(summary) {
  document.getElementById('s-total').textContent = summary.total ?? '—';
  document.getElementById('s-completed').textContent = summary.by_status?.completed ?? 0;
  document.getElementById('s-failed').textContent = (summary.by_status?.failed ?? 0) + (summary.by_status?.interrupted ?? 0);
  document.getElementById('s-tokens').textContent = summary.avg_tokens_out ? summary.avg_tokens_out.toLocaleString() : '—';

  const modelsEl = document.getElementById('models');
  const models = summary.models_used || {};
  modelsEl.innerHTML = Object.entries(models).length
    ? Object.entries(models).sort((a,b)=>b[1]-a[1]).map(([m,c]) =>
        `<div class="model-chip" title="${esc(m)}"><span>${esc(m)}</span><span>${c}</span></div>`).join('')
    : '<span class="empty">No model data yet</span>';

  document.getElementById('task-count').textContent = `${summary.total ?? 0} tasks total`;
}

function latestTaskStatus(project) {
  const latest = project.latest_task || {};
  return latest.status || 'unknown';
}

function getProjectSortMode() {
  return document.getElementById('project-sort')?.value || 'recent';
}

function sortProjects(projects) {
  const items = [...projects];
  const mode = getProjectSortMode();
  if (mode === 'title') {
    items.sort((a, b) => (a.title || a.display_name || '').localeCompare(b.title || b.display_name || ''));
  } else if (mode === 'pinned') {
    items.sort((a, b) => {
      const ap = a.is_pinned ? 0 : 1;
      const bp = b.is_pinned ? 0 : 1;
      if (ap !== bp) return ap - bp;
      return (b.last_active || b.started_at || 0) - (a.last_active || a.started_at || 0);
    });
  } else {
    items.sort((a, b) => (b.last_active || b.started_at || 0) - (a.last_active || a.started_at || 0));
  }
  return items;
}

async function saveProjectTitle(sessionId, inputEl) {
  const title = (inputEl.value || '').trim();
  await postJSON('/api/projects/title', { session_id: sessionId, title });
  await refresh();
}

async function toggleChatPin(sessionId, pinned) {
  await postJSON('/api/chats/pin', { session_id: sessionId, pinned });
  await refresh();
}

async function saveChatTitle(sessionId, input) {
  const title = (input?.value || '').trim();
  const res = await postJSON('/api/chats/title', { session_id: sessionId, title });
  if (input) {
    input.value = res?.session?.title || '';
  }
  await refresh();
}

async function deleteChat(sessionId) {
  const confirmed = window.confirm('Delete this chat thread forever? This cannot be undone, including for pinned threads.');
  if (!confirmed) return;
  await postJSON('/api/chats/delete', { session_id: sessionId });
  await refresh();
}

async function pruneChats() {
  const confirmed = window.confirm('Prune old unpinned chat threads? This permanently deletes history beyond the newest five unpinned threads, while leaving pinned threads alone.');
  if (!confirmed) return;
  await postJSON('/api/chats/prune', { keep: 5 });
  await refresh();
}

function renderProjects(projects) {
  const el = document.getElementById('projects');
  const sorted = sortProjects(projects || []);
  document.getElementById('project-count').textContent = `${sorted.length} projects visible`;
  el.innerHTML = sorted.length
    ? sorted.map(project => {
        const latest = project.latest_task || {};
        const status = latestTaskStatus(project);
        const counts = project.task_status_counts || {};
        const running = status === 'running';
        const title = project.title || '';
        return `<article class="project-card ${running ? 'running' : ''}">
          <div class="project-top">
            <div>
              <div class="project-name" title="${esc(project.title || project.session_id || '')}">${esc(project.display_name || project.title || project.session_id || 'Untitled')}</div>
              <div class="project-subtitle">
                ${esc(project.source || 'source: —')} · last active ${esc(fmtTs(project.last_active))}
              </div>
            </div>
            <div class="project-badges">
              ${statusBadge(status)}
              ${project.is_pinned ? '<span class="badge completed" title="Pinned thread">Pinned</span>' : ''}
              <span class="badge unknown" title="Session ID">${esc(clip(project.session_id || '—', 10))}</span>
            </div>
          </div>
          <div class="project-stats">
            <div class="project-stat"><span class="k">Messages</span><span class="v">${esc(project.message_count ?? 0)}</span></div>
            <div class="project-stat"><span class="k">Tasks</span><span class="v">${esc(project.task_count ?? 0)}</span></div>
            <div class="project-stat"><span class="k">Tool calls</span><span class="v">${esc(project.tool_call_count ?? 0)}</span></div>
            <div class="project-stat"><span class="k">Model</span><span class="v" title="${esc(latest.model_used || '—')}">${esc(clip(latest.model_used || '—', 20))}</span></div>
          </div>
          <div class="project-edit">
            <input type="text" value="${esc(title)}" placeholder="Rename project…" aria-label="Project title" data-session-id="${esc(project.session_id || '')}" />
            <button class="mini-btn js-save-project" type="button" data-session-id="${esc(project.session_id || '')}">Save</button>
          </div>
          <div class="project-meta">Manual title edits help you group and sort work. Empty titles are allowed.</div>
          <div class="project-footer">
            <div><strong>Latest step:</strong> ${esc(latest.current_step || '—')}</div>
            <div><strong>Updated:</strong> ${esc(fmtTs(latest.updated_at))}</div>
            <div class="project-preview" title="${esc(project.preview || '')}">${esc(project.preview || 'No preview available yet')}</div>
            ${counts.failed ? `<div class="project-preview"><strong>Failures:</strong> ${esc(counts.failed)}</div>` : ''}
          </div>
        </article>`;
      }).join('')
    : '<div class="project-empty">No projects yet — sessions with titles will appear here automatically.</div>';

  el.querySelectorAll('.js-save-project').forEach(btn => {
    btn.addEventListener('click', async () => {
      const sid = btn.dataset.sessionId;
      const input = el.querySelector(`input[data-session-id="${CSS.escape(sid)}"]`);
      try {
        await saveProjectTitle(sid, input);
      } catch (e) {
        showBanner(`Could not save project title: ${e.message || e}`);
      }
    });
  });
}

function renderChats(payload) {
  const el = document.getElementById('chats');
  const countEl = document.getElementById('chat-count');
  const list = Array.isArray(payload) ? payload : (payload?.items || []);
  const meta = Array.isArray(payload) ? {} : (payload?.meta || {});
  const pinnedCount = meta.pinned_count ?? list.filter(c => c.is_pinned).length;
  const visibleCount = meta.visible_count ?? list.length;
  const totalHistorical = meta.total_historical ?? list.length;
  countEl.textContent = `${totalHistorical} historical threads · ${visibleCount} visible · ${pinnedCount} pinned`;
  el.innerHTML = list.length
    ? list.map(chat => {
        const title = chat.title || chat.display_name || chat.session_id || 'Untitled';
        const pinned = !!chat.is_pinned;
        const titleInput = `chat-title-${esc(chat.session_id || '')}`;
        return `<article class="chat-card ${pinned ? 'pinned' : ''}">
          <div class="chat-name">${esc(title)}</div>
          <div class="chat-subtitle">
            ${pinned ? 'Pinned · ' : ''}${esc(chat.source || 'source: —')} · started ${esc(fmtTs(chat.started_at))} · last active ${esc(fmtTs(chat.last_active))}
          </div>
          <div class="chat-preview" title="${esc(chat.preview || '')}">${esc(chat.preview || 'No preview available yet')}</div>
          <div class="chat-edit">
            <input type="text" value="${esc(chat.title || '')}" placeholder="Rename thread…" aria-label="Chat thread title" data-session-id="${esc(chat.session_id || '')}" id="${esc(titleInput)}" />
            <button class="mini-btn js-save-chat" type="button" data-session-id="${esc(chat.session_id || '')}">Save title</button>
          </div>
          <div class="chat-meta">Manual titles make threads easier to group. Leave blank to clear the title.</div>
          <div class="chat-actions">
            <button class="mini-btn js-pin-chat ${pinned ? 'pinned' : ''}" type="button" data-session-id="${esc(chat.session_id || '')}" data-pinned="${pinned ? '0' : '1'}">${pinned ? 'Unpin thread' : 'Pin thread'}</button>
            <button class="mini-btn danger js-delete-chat" type="button" data-session-id="${esc(chat.session_id || '')}">Delete forever</button>
          </div>
        </article>`;
      }).join('')
    : '<div class="chat-empty">No historical chats yet — finished threads will appear here automatically.</div>';

  el.querySelectorAll('.js-save-chat').forEach(btn => {
    btn.addEventListener('click', async () => {
      const sid = btn.dataset.sessionId;
      const input = el.querySelector(`input[data-session-id="${CSS.escape(sid)}"]`);
      try {
        await saveChatTitle(sid, input);
      } catch (e) {
        showBanner(`Could not save chat title: ${e.message || e}`);
      }
    });
  });
  el.querySelectorAll('.js-pin-chat').forEach(btn => {
    btn.addEventListener('click', async () => {
      const sid = btn.dataset.sessionId;
      const pinned = btn.dataset.pinned === '1';
      try {
        await toggleChatPin(sid, pinned);
      } catch (e) {
        showBanner(`Could not update pin state: ${e.message || e}`);
      }
    });
  });
  el.querySelectorAll('.js-delete-chat').forEach(btn => {
    btn.addEventListener('click', async () => {
      try {
        await deleteChat(btn.dataset.sessionId);
      } catch (e) {
        showBanner(`Could not delete thread: ${e.message || e}`);
      }
    });
  });
}

function renderTasks(tasks) {
  const tb = document.getElementById('task-body');
  tb.innerHTML = tasks.length
    ? tasks.map(t => `<tr>
        <td>${statusBadge(t.status)}</td>
        <td class="mono-dim" title="${esc(t.model_used || '—')}">${esc(clip(t.model_used || '—', 28))}</td>
        <td class="mono-dim" title="${esc(t.current_step || '—')}">${esc(clip(t.current_step || '—', 28))}</td>
        <td class="mono-dim">${esc(fmtTs(t.updated_at))}</td>
        <td class="err-text" title="${esc(t.error_info || '')}">${esc(clip(t.error_info || '', 42) || '—')}</td>
      </tr>`).join('')
    : '<tr><td colspan="5" class="empty">No tasks yet — run hermes to populate</td></tr>';
}

function renderEvents(events) {
  const eb = document.getElementById('event-body');
  eb.innerHTML = events.length
    ? events.map(ev => `<tr>
        <td class="mono-dim">${esc(fmtTs(ev.ts))}</td>
        <td class="event-type" title="${esc(ev.event || '?')}">${esc(clip(ev.event || '?', 20))}</td>
        <td class="tool-name" title="${esc(ev.tool_name || ev.model || '')}">${esc(clip(ev.tool_name || ev.model || '', 26))}</td>
        <td class="dur">${esc(ev.duration_ms ? fmt(ev.duration_ms) : '—')}</td>
      </tr>`).join('')
    : '<tr><td colspan="4" class="empty">No events yet</td></tr>';
}

function renderSlowTools(slow) {
  const sb = document.getElementById('slow-body');
  sb.innerHTML = slow.length
    ? slow.map(ev => `<tr>
        <td class="tool-name" title="${esc(ev.tool_name || '?')}">${esc(clip(ev.tool_name || '?', 28))}</td>
        <td class="dur">${esc(fmt(ev.duration_ms || 0))}</td>
        <td class="mono-dim" title="${esc(ev.task_id || '')}">${esc(clip(ev.task_id || '', 10))}</td>
      </tr>`).join('')
    : '<tr><td colspan="3" class="empty">No slow tools above threshold</td></tr>';
}

function renderLoopEvents(loops) {
  const lb = document.getElementById('loop-body');
  lb.innerHTML = loops.length
    ? loops.map(ev => `<tr>
        <td class="mono-dim">${esc(fmtTs(ev.ts))}</td>
        <td class="tool-name" title="${esc(ev.tool_name || '?')}">${esc(clip(ev.tool_name || '?', 22))}</td>
        <td class="mono-dim" title="${esc(ev.loop_type || '?')}">${esc(clip(ev.loop_type || '?', 18))}</td>
        <td class="mono-dim">${esc(ev.count || '?')}</td>
      </tr>`).join('')
    : '<tr><td colspan="4" class="empty">No loops detected</td></tr>';
}

async function refresh() {
  if (refreshInFlight) {
    refreshQueued = true;
    return;
  }
  refreshInFlight = true;
  setStale();

  const requests = [
    ['summary', () => fetchJSON('/api/summary')],
    ['projects', () => fetchJSON('/api/projects?limit=8')],
    ['chats', () => fetchJSON('/api/chats?limit=12')],
    ['tasks', () => fetchJSON('/api/tasks?limit=15')],
    ['events', () => fetchJSON('/api/events?limit=20')],
    ['slow', () => fetchJSON('/api/slow')],
    ['loops', () => fetchJSON('/api/loops?limit=10')],
  ];

  try {
    const settled = await Promise.allSettled(requests.map(([, fn]) => fn()));
    const failed = [];

    settled.forEach((result, idx) => {
      const [name] = requests[idx];
      if (result.status === 'fulfilled') {
        if (name === 'summary') renderSummary(result.value);
        else if (name === 'projects') renderProjects(result.value);
        else if (name === 'chats') renderChats(result.value);
        else if (name === 'tasks') renderTasks(result.value);
        else if (name === 'events') renderEvents(result.value);
        else if (name === 'slow') renderSlowTools(result.value);
        else if (name === 'loops') renderLoopEvents(result.value);
      } else {
        failed.push(name);
      }
    });

    if (failed.length) {
      const suffix = failed.length === 1 ? 'endpoint failed' : 'endpoints failed';
      showBanner(`Dashboard refresh partial failure: ${failed.join(', ')} ${suffix}. Showing last good data.`);
      refreshDot.style.background = 'var(--red)';
    } else {
      hideBanner();
      setHealthy();
      lastRefreshEl.textContent = 'Updated ' + new Date().toLocaleTimeString();
    }
  } catch (e) {
    showBanner(`Dashboard refresh failed: ${e.message || e}. Showing last good data.`);
    refreshDot.style.background = 'var(--red)';
  } finally {
    refreshInFlight = false;
    if (refreshQueued) {
      refreshQueued = false;
      refresh();
    }
  }
}

bannerRetry.addEventListener('click', refresh);
document.getElementById('project-sort')?.addEventListener('change', () => refresh());
document.getElementById('chat-prune')?.addEventListener('click', async () => {
  try {
    await pruneChats();
  } catch (e) {
    showBanner(`Could not prune chats: ${e.message || e}`);
  }
});
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    """Handles GET requests for the dashboard UI and JSON API endpoints."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default access logs — too noisy for a local dashboard
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _same_origin_request(self) -> bool:
        """Allow local dashboard POSTs, reject explicit cross-origin POSTs."""
        origin = self.headers.get("Origin") or self.headers.get("Referer")
        if not origin:
            return True
        try:
            parsed = urlparse(origin)
            host = parsed.hostname or ""
        except Exception:
            return False
        return host in {"127.0.0.1", "localhost", "::1"}

    def _send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        def _int(key: str, default: int) -> int:
            try:
                return int(qs[key][0])
            except (KeyError, ValueError, IndexError):
                return default

        def _float(key: str, default: float) -> float:
            try:
                return float(qs[key][0])
            except (KeyError, ValueError, IndexError):
                return default

        def _str(key: str) -> str | None:
            try:
                return qs[key][0] or None
            except (KeyError, IndexError):
                return None

        # Import lazily so the module is usable even if ops.py has issues
        try:
            from hermes_cli import ops
        except ImportError:
            self._send_json({"error": "hermes_cli.ops not importable"}, 500)
            return

        if path == "/" or path == "/index.html":
            self._send_html(_HTML)

        elif path == "/api/summary":
            try:
                self._send_json(ops.task_summary(session_id=_str("session")))
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/projects":
            try:
                self._send_json(ops.recent_projects(
                    limit=_int("limit", 8),
                    source=_str("source"),
                ))
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/chats":
            try:
                self._send_json(ops.recent_chats(
                    limit=_int("limit", 12),
                    source=_str("source"),
                ))
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/tasks":
            try:
                self._send_json(ops.list_tasks(
                    limit=_int("limit", 20),
                    status=_str("status"),
                    session_id=_str("session"),
                ))
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/events":
            try:
                self._send_json(ops.recent_events(
                    limit=_int("limit", 50),
                    event_type=_str("type"),
                    task_id=_str("task"),
                ))
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/slow":
            # threshold_ms=None → ops reads from observability.slow_tool_threshold_ms in config
            explicit = _str("threshold")
            try:
                self._send_json(ops.slow_tools(
                    limit=_int("limit", 10),
                    threshold_ms=float(explicit) if explicit else None,
                ))
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/loops":
            try:
                self._send_json(ops.loop_events(limit=_int("limit", 20)))
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if not self._same_origin_request():
            self._send_json({"error": "cross-origin POST rejected"}, 403)
            return

        parsed = urlparse(self.path)
        path = parsed.path

        try:
            from hermes_cli import ops
        except ImportError:
            self._send_json({"error": "hermes_cli.ops not importable"}, 500)
            return

        data = self._read_json()
        if data is None:
            self._send_json({"error": "invalid JSON body"}, 400)
            return

        try:
            from hermes_state import SessionDB
        except ImportError:
            self._send_json({"error": "hermes_state not importable"}, 500)
            return

        db = SessionDB(db_path=ops._db_path())
        try:
            session_id = data.get("session_id")
            if path == "/api/projects/title":
                if not session_id:
                    self._send_json({"error": "session_id is required"}, 400)
                    return
                if not db.set_session_title(session_id, data.get("title")):
                    self._send_json({"error": "session not found"}, 404)
                    return
                self._send_json({"ok": True, "session": db.get_session(session_id)})
                return

            if path == "/api/chats/title":
                if not session_id:
                    self._send_json({"error": "session_id is required"}, 400)
                    return
                if not db.set_session_title(session_id, data.get("title")):
                    self._send_json({"error": "session not found"}, 404)
                    return
                self._send_json({"ok": True, "session": db.get_session(session_id)})
                return

            if path == "/api/chats/pin":
                if not session_id:
                    self._send_json({"error": "session_id is required"}, 400)
                    return
                pinned = bool(data.get("pinned", True))
                if not db.set_session_pinned(session_id, pinned=pinned):
                    self._send_json({"error": "session not found"}, 404)
                    return
                self._send_json({"ok": True, "session": db.get_session(session_id)})
                return

            if path == "/api/chats/delete":
                if not session_id:
                    self._send_json({"error": "session_id is required"}, 400)
                    return
                if not db.delete_session(session_id):
                    self._send_json({"error": "session not found"}, 404)
                    return
                self._send_json({"ok": True})
                return

            if path == "/api/chats/prune":
                try:
                    keep = int(data.get("keep", 5))
                except (TypeError, ValueError):
                    self._send_json({"error": "keep must be an integer"}, 400)
                    return
                deleted = db.prune_unpinned_historical_sessions(keep=keep, source=data.get("source"))
                self._send_json({"ok": True, "deleted": deleted})
                return

            self._send_json({"error": "not found"}, 404)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def serve(port: int = 7788, host: str = "127.0.0.1") -> None:
    """Start the dashboard HTTP server."""
    server = HTTPServer((host, port), DashboardHandler)
    url = f"http://{host}:{port}"
    print(f"  ⚕ Hermes Dashboard  →  {url}")
    print(f"  Auto-refreshes every 5s. Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hermes-dashboard",
        description="Hermes operator dashboard — live web UI for tasks and events",
    )
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on (default 7788)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1)")
    args = parser.parse_args(argv)
    serve(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
