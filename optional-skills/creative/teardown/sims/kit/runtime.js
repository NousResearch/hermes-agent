import {createDirector, DIRECTOR} from './director.js';
import {createHud} from './hud.js';
import {merge} from './util.js';

/** Config every sim starts from; a sim's defaults and the post config merge over it. */
export const BASE = {
  duration: 30,
  fps: 30,
  seed: 746337560,
  theme: 'dark',
  accent: '#f2b632',
  watermark: '',
  hud: {
    title: 'AGENT MESH',
    subtitle: '/ SIMULATION',
    right: 'VISUAL SIMULATION',
    host: 'agent@local',
    path: '~/session-01',
    sideTitle: 'LOCAL VIEW',
    logRows: 5,
    views: 10,
  },
  stages: [{at: 0, name: 'RUN', cmd: 'run', logs: []}],
  facts: {},
  director: DIRECTOR,
  story: [],
  beats: null,
};

const fromBase64 = (s) => decodeURIComponent(escape(atob(s.replace(/-/g, '+').replace(/_/g, '/'))));

/**
 * Config sources, later wins: BASE, the sim's defaults, window.__SIM_CONFIG
 * (injected by the capture script), ?config=<base64 json>, and a few plain
 * query params (?title= ?accent= ?seed= ?theme= ?duration=).
 */
export function loadConfig(defaults) {
  let cfg = merge(BASE, defaults);
  if (window.__SIM_CONFIG) cfg = merge(cfg, window.__SIM_CONFIG);
  const q = new URLSearchParams(location.search);
  if (q.get('config')) {
    try {
      cfg = merge(cfg, JSON.parse(fromBase64(q.get('config'))));
    } catch (e) {
      console.warn('bad ?config', e);
    }
  }
  if (q.get('title')) cfg = merge(cfg, {hud: {title: q.get('title')}});
  if (q.get('accent')) cfg.accent = '#' + q.get('accent').replace('#', '');
  if (q.get('seed')) cfg.seed = Number(q.get('seed'));
  if (q.get('theme')) cfg.theme = q.get('theme');
  if (q.get('duration')) cfg.duration = Number(q.get('duration'));
  if (q.has('nodirector')) cfg.director = {...cfg.director, enabled: false};
  return cfg;
}

/**
 * Boots a sim. `setup({cfg, canvas, hud, capture})` returns (or resolves to)
 * `{render(t) -> hudInfo, resize()}`. render must be a pure function of t.
 *
 * Live mode loops in real time. Keys: space pause, h hide HUD, f fullscreen,
 * r restart, ←/→ seek 2 s, 1–9 jump to a stage.
 * Capture mode (?capture or window.__CAPTURE) exposes window.SIM.render(t).
 */
export async function runSim(defaults, setup) {
  const cfg = loadConfig(defaults);
  const q = new URLSearchParams(location.search);
  const capture = !!window.__CAPTURE || q.has('capture');
  const root = document.documentElement;
  root.dataset.theme = cfg.theme;
  root.style.setProperty('--accent', cfg.accent);
  document.title = `${cfg.hud.title} ${cfg.hud.subtitle}`.trim();

  const scale = () => {
    root.style.fontSize = `${10 * Math.min(innerWidth / 1080, innerHeight / 1350)}px`;
  };
  scale();

  const app = document.getElementById('app');
  const canvas = document.createElement('canvas');
  canvas.id = 'gl';
  app.appendChild(canvas);
  // cfg.hud.enabled === false: no instrument chrome (UI demos, infographics).
  const hud =
    cfg.hud.enabled === false
      ? {minimap: null, stageAt: () => 0, setVisible() {}, viewport: () => ({top: 0, bottom: 0}), render() {}}
      : createHud(app, cfg);

  await Promise.all([
    document.fonts.load('400 20px "JetBrains Mono"'),
    document.fonts.load('600 20px "JetBrains Mono"'),
    document.fonts.load('800 20px "Inter"'),
  ]).catch(() => {});

  window.__HUDVP = () => hud.viewport();
  const sim = await setup({cfg, canvas, hud, capture});
  if (!capture && sim.warm) await sim.warm();
  const dir = createDirector(app, cfg);
  // The sim and HUD run on warped time (hero lands on the drop); the
  // director's cuts, punches and story run on real (music) time.
  const draw = (t) => {
    const tw = dir.warp(t);
    dir.pre(t);
    const info = sim.render(tw, dir.beat(t)) || {};
    hud.render(tw, Math.round(t * cfg.fps), info);
    dir.post(t, info);
  };

  let cur = 0;
  let paused = false;
  const resize = () => {
    scale();
    sim.resize?.();
    if (paused || capture) draw(cur);
  };
  addEventListener('resize', resize);
  resize();

  window.SIM = {
    cfg,
    duration: cfg.duration,
    fps: cfg.fps,
    ready: true,
    render: (t) => {
      cur = t;
      draw(t);
    },
  };
  if (capture) {
    draw(0);
    return;
  }

  let start = performance.now();
  const seek = (t) => {
    cur = ((t % cfg.duration) + cfg.duration) % cfg.duration;
    start = performance.now() - cur * 1000;
  };
  let hudOn = true;
  addEventListener('keydown', (e) => {
    if (e.key === ' ') {
      paused = !paused;
      if (!paused) seek(cur);
    } else if (e.key === 'h') {
      hudOn = !hudOn;
      hud.setVisible(hudOn);
    } else if (e.key === 'f') {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
    } else if (e.key === 'r') seek(0);
    else if (e.key === 'ArrowRight') seek(cur + 2);
    else if (e.key === 'ArrowLeft') seek(cur - 2);
    else if (/^[1-9]$/.test(e.key) && cfg.stages[+e.key - 1]) seek(Math.max(0, cfg.stages[+e.key - 1].at));
    else return;
    e.preventDefault();
  });
  const loop = (now) => {
    if (!paused) cur = Math.max(0, (now - start) / 1000) % cfg.duration;
    draw(cur);
    // the site page swaps its poster for the canvas on this
    if (!window.SIM_SHOWN) {
      window.SIM_SHOWN = true;
      dispatchEvent(new Event('simframe'));
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
