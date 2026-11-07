import {clamp01, pad} from './util.js';

/**
 * Instrument-style HUD: header with stage/command/clock, annotations over the
 * viewport, a terminal log and a side panel (minimap + stats) at the bottom.
 * Everything is derived from `t` and the config's stages, so it's deterministic.
 *
 * cfg.stages: [{at, name, cmd, logs: [string]}]. A log line starting with "!"
 * is highlighted in the accent. {key} tokens are replaced from `facts`.
 */
export function createHud(root, cfg) {
  const h = cfg.hud;
  root.insertAdjacentHTML(
    'beforeend',
    `<div class="hud">
      <div class="ann"></div>
      <div class="wm"></div>
      <div class="hdr">
        <div class="row row1">
          <div class="title"><i class="bar"></i><span class="t"></span></div>
          <div class="sub"></div>
          <div class="right"></div>
        </div>
        <div class="row row2">
          <div class="stage accent"></div>
          <div class="cmd"></div>
          <div class="time accent"></div>
        </div>
        <div class="prog"><i></i></div>
      </div>
      <div class="bot">
        <div class="term">
          <div class="thead">
            <span class="host accent"></span><span class="path"></span>
            <span class="status accent"><i></i><span></span></span>
          </div>
          <div class="prompt"></div>
          <div class="logs"></div>
          <div class="tfoot"></div>
        </div>
        <div class="side">
          <div class="shead accent"></div>
          <canvas class="mini"></canvas>
          <div class="stats"></div>
          <div class="sfoot"></div>
        </div>
        <div class="baseline"><i></i></div>
      </div>
    </div>`,
  );
  const $ = (s) => root.querySelector(s);
  const el = {
    hud: $('.hud'),
    ann: $('.ann'),
    stage: $('.row2 .stage'),
    cmd: $('.row2 .cmd'),
    time: $('.row2 .time'),
    prog: $('.prog i'),
    prompt: $('.prompt'),
    logs: $('.logs'),
    tfoot: $('.tfoot'),
    status: $('.thead .status span'),
    stats: $('.stats'),
    sfoot: $('.sfoot'),
    base: $('.baseline i'),
    mini: $('.mini'),
  };
  $('.title .t').textContent = h.title;
  $('.sub').textContent = h.subtitle;
  $('.right').textContent = h.right;
  $('.wm').textContent = cfg.watermark;
  $('.thead .host').textContent = h.host;
  $('.thead .path').textContent = h.path;
  $('.shead').textContent = h.sideTitle;

  const stages = cfg.stages;
  const D = cfg.duration;
  const TYPE_CMD = 0.028;
  const TYPE_LOG = 0.011;

  // Log line schedule: each stage types its command, then its lines spread
  // over the stage.
  const lines = [];
  stages.forEach((s, i) => {
    const end = i + 1 < stages.length ? stages[i + 1].at : D;
    const t0 = s.at + 0.25 + s.cmd.length * TYPE_CMD + 0.2;
    const n = s.logs.length;
    const gap = Math.min(0.8, Math.max(0.22, (end - t0 - 0.4) / Math.max(1, n)));
    s.logs.forEach((text, j) => lines.push({at: t0 + j * gap, text, stage: i}));
  });

  const stageAt = (t) => {
    let i = 0;
    while (i + 1 < stages.length && stages[i + 1].at <= t) i++;
    return i;
  };
  const fill = (text, facts) => text.replace(/\{(\w+)\}/g, (m, k) => (k in facts ? facts[k] : m));
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');

  let lastStats = '';
  let lastLogs = '';
  const annPool = [];

  return {
    minimap: el.mini,
    stageAt,
    setVisible(v) {
      el.hud.classList.toggle('hidden', !v);
    },
    /** Viewport between header and bottom panel, in CSS px. */
    viewport() {
      const fs = parseFloat(getComputedStyle(document.documentElement).fontSize);
      return {top: 13.4 * fs, bottom: 30.5 * fs};
    },
    render(t, frame, info = {}) {
      const facts = {...cfg.facts, ...(info.facts || {})};
      const si = stageAt(t);
      const s = stages[si];
      const dt = t - s.at;
      const typed = Math.floor(Math.max(0, dt - 0.15) / TYPE_CMD);
      const cmd = fill(s.cmd, facts);
      const typing = typed < cmd.length;
      const caretOn = typing || Math.floor(t * 2.4) % 2 === 0;

      el.stage.textContent = `${pad(si + 1, 2)} / ${s.name}`;
      el.cmd.textContent = cmd.slice(0, typed);
      el.time.textContent = `${t.toFixed(2).padStart(5, '0')} / ${D}`;
      el.prog.style.width = `${(clamp01(t / D) * 100).toFixed(2)}%`;
      el.base.style.width = `${(clamp01(dt / ((stages[si + 1]?.at ?? D) - s.at)) * 100).toFixed(2)}%`;
      el.prompt.innerHTML = `<span class="accent">&gt;</span> ${esc(cmd.slice(0, typed))}${
        caretOn ? '<i class="caret"></i>' : ''
      }`;

      const shown = lines.filter((l) => l.at <= t);
      const rows = shown.slice(-h.logRows);
      const offset = shown.length - rows.length;
      const html = rows
        .map((l, j) => {
          const full = fill(l.text, facts);
          const hl = full.startsWith('!');
          const text = hl ? full.slice(1) : full;
          const n = Math.floor((t - l.at) / TYPE_LOG);
          const isLast = j === rows.length - 1 && n < text.length;
          return `<div><span class="ln">${pad(offset + j, 3)}</span><span class="${hl ? 'hl' : ''}">${esc(
            text.slice(0, n),
          )}</span>${isLast ? '<i class="caret"></i>' : ''}</div>`;
        })
        .join('');
      if (html !== lastLogs) {
        el.logs.innerHTML = html;
        lastLogs = html;
      }

      el.tfoot.textContent = `frame ${pad(frame, 4)}   delta ${pad((dt * 7.3) % 100, 2)}   /   ${s.name.toLowerCase()}`;
      el.status.textContent = info.status ?? (t >= D - 0.05 ? 'DONE' : 'RUNNING');
      el.sfoot.textContent = `view ${pad(si + 1, 2)} / ${pad(h.views, 2)}    seed ${cfg.seed} · t+${dt.toFixed(2)}s`;

      const stats = (info.stats || []).map((x) => `<div>${esc(x.label)}<b>${esc(String(x.value))}</b></div>`).join('');
      if (stats !== lastStats) {
        el.stats.innerHTML = stats;
        lastStats = stats;
      }

      const anns = info.annotations || [];
      while (annPool.length < anns.length) {
        const d = document.createElement('div');
        el.ann.appendChild(d);
        annPool.push(d);
      }
      annPool.forEach((d, i) => {
        const a = anns[i];
        if (!a) {
          d.style.display = 'none';
          return;
        }
        d.style.display = '';
        d.className = a.cls || '';
        d.style.left = `${a.x}px`;
        d.style.top = `${a.y}px`;
        d.style.opacity = a.opacity ?? 1;
        if (a.html !== undefined) d.innerHTML = a.html;
        else d.textContent = a.text;
        // keep labels inside the viewport (between header and bottom panel)
        const vp = this.viewport();
        const w = d.offsetWidth;
        const hh = d.offsetHeight;
        const x = Math.min(Math.max(a.x, 8), innerWidth - w - 40);
        const y = Math.min(Math.max(a.y, vp.top + hh / 2 + 8), innerHeight - vp.bottom - hh / 2 - 8);
        d.style.left = `${x}px`;
        d.style.top = `${y}px`;
      });
    },
  };
}
