// STRANDBEEST SB-10, the console film: the beast on the dune (stage.js) under the SB-10 engineering sheet
// (console.js). An operator isolates one leg and letters its eleven bars, traces its foot, lights all twenty loops,
// opens the wind from 6 to 12.5 m/s and follows the walk from the front, along the spine and from above over the
// footprints. Wind drives the crank, the crank drives everything: theta(t) is the integral of the wind's crank speed.
// A pure function of t.
import {runSim} from '../kit/runtime.js';
import {CFG, legAt, legZ, phaseOf, S, STRIDE} from './beast.js';
import {createConsole, LABELS, VIEW_KEYS} from './console.js';
import {footLift} from './jansen.js';
import {createBeestStage} from './stage.js';

// the foot traces are off in this cut (no trace events); the console keeps the TRACE FEET button for the live page
export const SCRIPT = [
  {t: 0, view: 'cine'},
  {t: 3.4, iso: 1, view: 'leg'},
  {t: 9.6, iso: 0, view: 'front'},
  {t: 13.0, wind: [6, 12.5, 1.6]},
  {t: 16.8, view: 'side'},
  {t: 21.0, view: 'top'},
  {t: 25.8, view: 'cine'},
];
export const DURATION = 30.7;
export const DEFAULTS = {duration: DURATION, fps: 60, theme: 'light', hud: {enabled: false, title: 'beestsim', subtitle: ''}, director: {enabled: false}, watermark: ''};
export const HERO = {st: 0, side: 0};
const D2R = Math.PI / 180;
const smoother = (x) => {
  const u = Math.min(1, Math.max(0, x));
  return u * u * u * (u * (u * 6 - 15) + 10);
};
const ss = (a, b, t) => smoother((t - a) / (b - a));

// ------------------------------------------------------------ wind, crank and the distance the wind has run
const WIND_EV = SCRIPT.find((e) => e.wind);
export const windAt = (t) => {
  const [w0, w1, dur] = WIND_EV.wind;
  const base = w0 + (w1 - w0) * ss(WIND_EV.t, WIND_EV.t + dur, t);
  return base + 0.35 * Math.sin(t * 0.9) + 0.2 * Math.sin(t * 2.3 + 1);
};
// the crank's turn rate for a wind: 0.42 rev/s at 6 m/s, rising slower than the wind
export const rpsAt = (w) => 0.42 * Math.pow(Math.max(0.5, w) / 6, 0.75);
const T0 = -16;
const DT = 1 / 240;
const NTAB = Math.ceil((DURATION + 2 - T0) / DT);
const TH = new Float64Array(NTAB + 1);
const FLOW = new Float64Array(NTAB + 1);
{
  // integrate from t = 0 both ways so theta(0) = 0 and the body starts at the origin
  const i0 = Math.round(-T0 / DT);
  for (let i = i0 + 1; i <= NTAB; i++) {
    const t = T0 + (i - 0.5) * DT;
    TH[i] = TH[i - 1] + rpsAt(windAt(t)) * Math.PI * 2 * DT;
    FLOW[i] = FLOW[i - 1] + windAt(t) * DT;
  }
  for (let i = i0 - 1; i >= 0; i--) {
    const t = T0 + (i + 0.5) * DT;
    TH[i] = TH[i + 1] - rpsAt(windAt(t)) * Math.PI * 2 * DT;
    FLOW[i] = FLOW[i + 1] - windAt(t) * DT;
  }
}
const lookup = (A, t) => {
  const u = Math.min(NTAB - 1e-6, Math.max(0, (t - T0) / DT));
  const i = Math.floor(u);
  return A[i] + (A[i + 1] - A[i]) * (u - i);
};
export const thetaAt = (t) => lookup(TH, t);
export const flowAt = (t) => lookup(FLOW, t);

// ------------------------------------------------------------ operator state
const evAt = (key, t, init = 0) => {
  let v = init;
  for (const e of SCRIPT) if (e[key] !== undefined) v += (e[key] - v) * ss(e.t, e.t + 0.7, t);
  return v;
};
export const isoAt = (t) => evAt('iso', t);
const TRACE_ON = SCRIPT.find((e) => e.trace === 1)?.t ?? Infinity;
const TRACE_OFF = SCRIPT.find((e) => e.trace === 0)?.t ?? Infinity;
// when the hero's foot has drawn one full loop
const LOOP_DONE = (() => {
  if (!Number.isFinite(TRACE_ON)) return Infinity;
  const th0 = thetaAt(TRACE_ON);
  let t = TRACE_ON;
  while (thetaAt(t) - th0 < Math.PI * 2) t += 0.01;
  return t;
})();
export const traceState = (t) => ({
  heroTrace: t < TRACE_ON ? 0 : Math.min(1, (thetaAt(t) - thetaAt(TRACE_ON)) / (Math.PI * 2)),
  trace: ss(LOOP_DONE, LOOP_DONE + 0.9, t),
  traceFade: 1 - ss(TRACE_OFF, TRACE_OFF + 0.7, t),
  on: t >= TRACE_ON && t < TRACE_OFF + 0.2 ? 1 : 0,
});
export const ISO_T = SCRIPT.find((e) => e.iso === 1).t;
// the hero's letters come up one by one, starting 0.9 s after the leg is isolated
export const labelTimes = (start) => LABELS.map((_, i) => start + 0.9 + i * 0.3);

// ------------------------------------------------------------ views
const heroZ = legZ(HERO.st, HERO.side) * S;
// each view as a function of u, the seconds since it was switched on, and the body's x
export const VIEW_FN = {
  cine: (u, bx) => ({tx: bx + 0.05, ty: 1.1, tz: 0.1, az: 319.6 + 1.4 * u, el: 11, d: 9.2, sx: 14, sy: -58}),
  leg: (u, bx) => ({tx: bx - 0.34, ty: 0.74, tz: heroZ, az: 94 + 0.9 * u, el: 4, d: 5.6, sx: 0, sy: -40}),
  front: (u, bx) => ({tx: bx - 0.1, ty: 1.08, tz: 0, az: 196 + 1.1 * u, el: 7, d: 9.6, sx: -6, sy: -50}),
  side: (u, bx) => ({tx: bx + 0.1, ty: 0.95, tz: -0.2, az: 262 - 1.2 * u, el: 11, d: 9.6, sx: -6, sy: -58}),
  top: (u, bx) => ({tx: bx + 1.1, ty: 0.3, tz: 0, az: 192 + 1.6 * u, el: 60, d: 13.5, sx: 0, sy: -40}),
};
const VIEWS_LIST = SCRIPT.filter((e) => e.view).map((e) => [e.t, e.view]);
export const scriptView = (t) => VIEWS_LIST.filter(([t0]) => t0 <= t).pop();
const lerpDeg = (a, b, k) => a + ((((b - a + 540) % 360) - 180) * k);
export const viewAt = (t, bx, list = VIEWS_LIST) => {
  let i = 0;
  while (i < list.length - 1 && list[i + 1][0] <= t) i++;
  const cur = VIEW_FN[list[i][1]](t - list[i][0], bx);
  if (i === 0) return cur;
  const prev = VIEW_FN[list[i - 1][1]](t - list[i - 1][0], bx);
  const k = ss(list[i][0], list[i][0] + 1.9, t);
  const o = {};
  for (const key in cur) o[key] = key === 'az' ? lerpDeg(prev.az, cur.az, k) : prev[key] + (cur[key] - prev[key]) * k;
  // a move through a wide swing backs off and climbs so the camera never passes through the beast
  const swing = Math.min(1, Math.abs(((cur.az - prev.az + 540) % 360) - 180) / 90);
  o.el += 8 * Math.sin(Math.PI * k) * swing;
  o.d += Math.max(0, Math.max(prev.d, cur.d) + 2.5 * swing - o.d) * Math.sin(Math.PI * k);
  return o;
};
export const toCam = (c) => ({...c, az: c.az * D2R, el: c.el * D2R});

// ------------------------------------------------------------ footprints from the walk itself
export function footprints(bodyPos, t1 = DURATION) {
  const out = [];
  let r = 17;
  const rnd = () => (r = (r * 16807) % 2147483647) / 2147483647;
  for (let st = 0; st < CFG.stations; st++)
    for (let side = 0; side < 2; side++) {
      let on = false;
      let t0 = 0;
      for (let t = T0; t <= t1; t += 1 / 120) {
        const ps = thetaAt(t) + phaseOf(st);
        const lift = footLift(side ? Math.PI - ps : ps);
        if (!on && lift < 0.5) {
          on = true;
          t0 = t;
        } else if (on && lift >= 0.5) {
          on = false;
          const tm = (t0 + t) / 2;
          const th = thetaAt(tm);
          const J = legAt(st, side, th);
          out.push([bodyPos(th).x + J.F[0] * S, legZ(st, side) * S, t0, (rnd() - 0.5) * 0.25]);
        }
      }
    }
  return out;
}

// ------------------------------------------------------------ the operator's pointer
const REST = [[820, 430], [700, 480], [900, 380], [760, 520], [640, 420]];
export function makeCursor(ui) {
  const EV = SCRIPT.filter((e) => e.t > 0).map((e) => {
    const key = e.iso !== undefined ? 'iso' : e.trace !== undefined ? 'trace' : e.wind ? 'wind' : e.view;
    return {t: e.t, key, spot: key === 'wind' ? ui.knobAt(e.wind[0]) : ui.spots[key], drag: key === 'wind' ? e.wind[2] : 0};
  });
  const WP = [[0, 840, 560]];
  EV.forEach((e, i) => {
    WP.push([e.t - 0.16, ...e.spot]);
    if (e.drag) {
      for (let k = 0; k <= 12; k++) {
        const tt = e.t + (k / 12) * e.drag;
        WP.push([tt + 0.02, ...ui.knobAt(windAt(tt) - 0.35 * Math.sin(tt * 0.9) - 0.2 * Math.sin(tt * 2.3 + 1))]);
      }
      WP.push([e.t + e.drag + 0.3, ...ui.knobAt(SCRIPT.find((x) => x.wind).wind[1])]);
    } else WP.push([e.t + 0.24, ...e.spot]);
    WP.push([e.t + e.drag + 1.7, ...REST[i % REST.length]]);
  });
  WP.sort((a, b) => a[0] - b[0]);
  return (t) => {
    let i = 0;
    while (i < WP.length - 2 && t > WP[i + 1][0]) i++;
    const [t0, x0, y0] = WP[i];
    const [t1, x1, y1] = WP[i + 1];
    const lin = WP[i + 1][0] - WP[i][0] < 0.2;
    const k = lin ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : ss(t0, t1, t);
    const len = Math.hypot(x1 - x0, y1 - y0);
    const bow = lin ? 0 : Math.sin(Math.PI * k) * Math.min(26, len * 0.07);
    let press = false;
    let drag = false;
    let ring = 0;
    let rx = 0;
    let ry = 0;
    for (const e of EV) {
      if (t >= e.t - 0.07 && t < e.t + Math.max(0.09, e.drag + 0.1)) press = true;
      if (e.drag && t >= e.t && t < e.t + e.drag + 0.1) drag = true;
      if (t >= e.t && t < e.t + 0.55) {
        ring = 1 - (t - e.t) / 0.55;
        [rx, ry] = e.spot;
      }
    }
    return {x: x0 + (x1 - x0) * k + (len ? (-(y1 - y0) / len) * bow : 0), y: y0 + (y1 - y0) * k + (len ? ((x1 - x0) / len) * bow : 0), press, drag, ring, ringX: rx, ringY: ry};
  };
}

// ------------------------------------------------------------ console data for a frame
const CALLS = [
  {name: 'SAIL ARRAY', sub: '10 SAILS · 0.9 m² EACH', pt: (B) => [6, 66, B.zs(6)]},
  {name: 'CRANKSHAFT', sub: '10 THROWS · 36° APART', pt: (B) => [0, 0, B.zs(8)]},
  {name: 'JANSEN LEG', sub: '11 BARS · 1 CRANK · NO MOTOR', pt: (B, th) => {
    const J = legAt(3, 0, th);
    return [(J.C[0] + J.E[0]) / 2, (J.C[1] + J.E[1]) / 2, legZ(3, 0)];
  }},
  {name: 'RUBBER FOOT', sub: 'ON THE SAND FOR 155° OF CRANK', pt: (B, th) => [legAt(1, 0, th).F[0], legAt(1, 0, th).F[1], legZ(1, 0)]},
];
const CALL_VIEWS = {cine: 1, front: 1, side: 0, leg: 0, top: 0};
const FILM_SRC = {thetaAt: (t) => thetaAt(t), windAt: (t) => windAt(t)};

// src gives the crank angle and the wind at any recent time (the film's integral, or the live page's history);
// ops.labelStart is when the leg was last isolated, ops.trace whether the TRACE FEET button is lit
export function makeConsoleFrame(ui, st, src = FILM_SRC) {
  const {thetaAt, windAt} = src;
  const legLift = (s, side, th) => {
    const ps = th + phaseOf(s);
    return footLift(side ? Math.PI - ps : ps);
  };
  return (t, ops) => {
    const th = thetaAt(t);
    const wind = windAt(t);
    const rps = rpsAt(wind);
    const {view, since, iso, cursor} = ops;
    const LABEL_T = labelTimes(ops.labelStart ?? ISO_T);
    // the contact chart: 150 columns over the last 3 s, a run per stance
    const gait = [];
    for (let r = 0; r < 20; r++) {
      const s = r % 10;
      const side = r < 10 ? 0 : 1;
      const runs = [];
      let a = -1;
      for (let c = 0; c <= 150; c++) {
        const on = legLift(s, side, thetaAt(t - 3 + (c / 150) * 3)) < 0.6;
        if (on && a < 0) a = c / 150;
        if ((!on || c === 150) && a >= 0) {
          runs.push([a, c / 150]);
          a = -1;
        }
      }
      gait.push(runs);
    }
    let feetDown = 0;
    for (let r = 0; r < 20; r++) if (legLift(r % 10, r < 10 ? 0 : 1, th) < 0.6) feetDown++;
    const liftHist = [];
    const bodyHist = [];
    let bodyMax = 0;
    for (let i = 0; i <= 120; i++) {
      const tt = t - 4 + (i / 120) * 4;
      const tth = thetaAt(tt);
      liftHist.push(legLift(0, 0, tth) * 12);
      const low = st.bodyPos(tth).y;
      bodyHist.push((low - st.bodyPos(0).y) * 1000);
      bodyMax = Math.max(bodyMax, Math.abs((low - st.bodyPos(0).y) * 1000));
    }
    const windHist = Array.from({length: 97}, (_, i) => windAt(t - 12 + i * 0.125));
    // the holy numbers and the hero's letters
    const rows = {};
    const tags = [];
    const dims = [];
    const J = st.beast.out.heroJ;
    const z = st.beast.out.heroZ;
    const W = (u, v) => st.project(...st.beastWorld(u, v, z));
    LABELS.forEach(([k], i) => {
      const tl = LABEL_T[i];
      if (t >= tl) rows[k] = t - tl < 0.7 ? 2 : 1;
      const o = ss(tl, tl + 0.25, t) * iso * (view === 'leg' ? 1 : 0.0);
      if (o < 0.02 || !J) return;
      let at;
      if (k === 'a') at = W(-19, 0);
      else if (k === 'l') at = W(-38, -3.9);
      else if (k === 'm') at = W(J.A[0] / 2, J.A[1] / 2);
      else {
        const pair = {j: ['A', 'B'], k: ['A', 'C'], b: ['P', 'B'], c: ['P', 'C'], d: ['P', 'D'], e: ['B', 'D'], f: ['D', 'E'], g: ['C', 'E'], h: ['E', 'F'], i: ['C', 'F']}[k];
        at = W((J[pair[0]][0] + J[pair[1]][0]) / 2, (J[pair[0]][1] + J[pair[1]][1]) / 2);
      }
      if (at) tags.push({k, at, o, lit: rows[k] === 2});
      if (k === 'a') {
        const A = W(0, 0);
        const B = W(-38, 0);
        if (A && B) dims.push({a: A, b: B, o});
      }
      if (k === 'l') {
        const A = W(-38, 0);
        const B = W(-38, -7.8);
        if (A && B) dims.push({a: A, b: B, o});
      }
    });
    // callouts on the beast in the wide views
    const vw = (CALL_VIEWS[view] ?? 0) * ss(1.2, 1.9, since) * (1 - iso);
    const callouts = CALLS.map((c) => {
      if (vw < 0.02) return {o: 0};
      const at = st.project(...st.beastWorld(...c.pt(st.beast, th)));
      if (!at || at[0] < 40 || at[0] > 1560 || at[1] < 60 || at[1] > 720 || ui.covered(at[0], at[1])) return {o: 0};
      const dx = at[0] < 790 ? -70 : 70;
      if (ui.covered(at[0] + dx + (dx < 0 ? -170 : 170), at[1] - 64)) return {o: 0};
      return {...c, at, dx, dy: -64, o: vw};
    });
    ui.update({
      wind,
      rpm: rps * 60,
      speed: rps * Math.abs(STRIDE) * S,
      feetDown,
      dist: ops.dist ?? Math.abs(st.bodyPos(th).x - st.bodyPos(0).x),
      crankDeg: (th * 180) / Math.PI,
      leg: legAt(0, 0, th),
      rows,
      gait,
      throws: Array.from({length: 10}, (_, s) => th + phaseOf(s)),
      liftHist,
      bodyHist,
      liftMM: legLift(0, 0, th) * 12,
      bodyMM: Math.max(0.1, bodyMax),
      windHist,
      drag: cursor.drag,
      viewOn: Object.fromEntries(VIEW_KEYS.map((k) => [k, k === view ? 1 : 0])),
      iso,
      trace: ops.trace ?? traceState(t).on,
      tags,
      dims,
      callouts,
      cursor,
    });
  };
}

export function makeFilm() {
  return async ({cfg, canvas, capture}) => {
    const st = createBeestStage({canvas, palette: cfg.palette ?? 'dune', ink: cfg.ink ?? 0.7});
    const resize = () => st.resize(1600, 900, capture ? devicePixelRatio || 1 : Math.min(devicePixelRatio || 1, 2));
    resize();
    st.setPrints(footprints(st.bodyPos));
    const ui = createConsole(document.getElementById('app'));
    const cursorAt = makeCursor(ui);
    const frame = makeConsoleFrame(ui, st);
    window.BEEST = st;
    return {
      resize,
      render(t) {
        const th = thetaAt(t);
        const b = st.bodyPos(th);
        const iso = isoAt(t);
        const ts = traceState(t);
        st.frame(th, t, toCam(viewAt(t, b.x)), {wind: windAt(t), flow: flowAt(t) * 0.35, iso, hero: HERO, ...ts});
        const [t0, view] = scriptView(t);
        frame(t, {view, since: t - t0, iso, cursor: cursorAt(t)});
        return {};
      },
    };
  };
}

export const run = () => runSim(DEFAULTS, makeFilm());
