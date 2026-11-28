// CESSNA 337 SKYMASTER, the main landing gear film: the gear (gear.js) over the runway (stage.js) under the
// reference-style console (console.js). The operator flies two takeoffs and two landings (flight.js) and walks the
// views and display modes: the airframe ghost, the gearbox close-up with the labels on, the side tube spinning, the
// front, the x-ray plate, the plan from above, the exploded set, and an orbit for the last landing. A pure function of t.
import {runSim} from '../kit/runtime.js';
import {createConsole, DISPLAY} from './console.js';
import {phaseAt, SEQ, stateAt} from './flight.js';
import {AXIS, E0, G, J, LEG0} from './gear.js';
import {createCessnaStage} from './stage.js';
import * as THREE from 'three';

export const SCRIPT = [
  {t: 0, seq: 'takeoff', view: 'hero'},
  {t: 2.4, display: 'fuselage'},
  {t: 5.4, seq: 'landing'},
  {t: 5.9, view: 'gearbox'},
  {t: 6.5, display: 'labels'},
  {t: 8.7, view: 'sidetube'},
  {t: 10.9, view: 'front'},
  {t: 11.5, display: 'xray'},
  {t: 12.7, display: 'labels'},
  {t: 13.0, seq: 'takeoff'},
  {t: 13.4, view: 'plan'},
  {t: 16.2, display: 'exploded'},
  {t: 17.8, display: 'exploded'},
  {t: 18.3, view: 'hero'},
  {t: 18.7, display: 'orbit'},
  {t: 19.1, seq: 'landing'},
];
export const DURATION = 26.2;
export const DEFAULTS = {duration: DURATION, fps: 60, theme: 'light', hud: {enabled: false, title: 'cessna', subtitle: ''}, director: {enabled: false}, watermark: ''};
const D2R = Math.PI / 180;
const smoother = (x) => {
  const u = Math.min(1, Math.max(0, x));
  return u * u * u * (u * (u * 6 - 15) + 10);
};
const ss = (a, b, t) => smoother((t - a) / (b - a));

// ------------------------------------------------------------ the sequence, the display toggles at film time t
export const seqAt = (t) => {
  const e = SCRIPT.filter((x) => x.seq && x.t <= t).pop();
  return {seq: e.seq, tau: Math.min(SEQ[e.seq].dur, t - e.t)};
};
export const displayAt = (t) => {
  const out = Object.fromEntries(DISPLAY.map((k) => [k, 0]));
  for (const e of SCRIPT) if (e.display && e.t <= t) out[e.display] = out[e.display] ? 0 : 1;
  return out;
};
const lastFlip = (key, t) => SCRIPT.filter((e) => e.display === key && e.t <= t).pop()?.t ?? -9;
// an eased 0..1 for a display toggle
export const easedDisplay = (key, t, dur = 0.7) => {
  const on = displayAt(t)[key];
  const k = ss(lastFlip(key, t), lastFlip(key, t) + dur, t);
  return on ? k : 1 - k;
};
export const mechAt = (seq, tau) => {
  const S = SEQ[seq];
  return {stroke: S.stroke(tau), height: S.height(tau), speed: S.speed(tau)};
};
// the runway distance and the wheel spin, integrated over the film
const DT = 1 / 240;
const DIST = [0];
const ROLL = [0];
{
  let w = 0;
  for (let i = 1; i * DT <= DURATION + 1; i++) {
    const t = (i - 0.5) * DT;
    const {seq, tau} = seqAt(t);
    const m = mechAt(seq, tau);
    DIST.push(DIST[i - 1] + m.speed * DT);
    // the wheels roll with the runway, coast down once airborne
    w = m.height < 0.005 ? m.speed / G.wheelR : w * Math.exp(-DT / 1.2);
    ROLL.push(ROLL[i - 1] + w * DT);
  }
}
const look = (A, t) => {
  const u = Math.max(0, Math.min(A.length - 2, t / DT));
  const i = Math.floor(u);
  return A[i] + (A[i + 1] - A[i]) * (u - i);
};
export const distAt = (t) => look(DIST, t);
export const rollAt = (t) => look(ROLL, t);

// ------------------------------------------------------------ views (target relative to the gear), u = seconds on
export const VIEW_FN = {
  hero: (u) => ({tx: 0.1, ty: -0.36, tz: 0, az: 222 + 2.5 * u, el: 21, d: 4.8, sx: 0, sy: -44}),
  front: (u) => ({tx: 0, ty: -0.36, tz: 0, az: 4 + 2 * u, el: 7, d: 4.7, sx: 0, sy: -44}),
  plan: (u) => ({tx: 0, ty: -0.1, tz: -0.1, az: 180 + 1.5 * u, el: 86, d: 4.8, sx: 0, sy: -30}),
  top: (u) => ({tx: 0, ty: -0.25, tz: 0, az: 200 + 2 * u, el: 48, d: 5.0, sx: 0, sy: -36}),
  gearbox: (u) => ({tx: 0.06, ty: -0.14, tz: 0.0, az: 238 + 3 * u, el: 8, d: 2.0, sx: 0, sy: -36}),
  sidetube: (u) => ({tx: 0.38, ty: -0.36, tz: -0.04, az: 148 + 3 * u, el: 14, d: 2.8, sx: 0, sy: -36}),
};
const VIEWS = SCRIPT.filter((e) => e.view).map((e) => [e.t, e.view]);
const blend = (a, b, k) => {
  const o = {};
  for (const key in b) o[key] = key === 'az' ? a.az + ((((b.az - a.az + 540) % 360) - 180) * k) : a[key] + (b[key] - a[key]) * k;
  const swing = Math.min(1, Math.abs(((b.az - a.az + 540) % 360) - 180) / 90);
  o.d += Math.max(0, Math.max(a.d, b.d) + 0.6 * swing - o.d) * Math.sin(Math.PI * k);
  return o;
};
export const camAt = (t, orbit = 0) => {
  let i = 0;
  while (i < VIEWS.length - 1 && VIEWS[i + 1][0] <= t) i++;
  const cur = VIEW_FN[VIEWS[i][1]](t - VIEWS[i][0]);
  let o = cur;
  if (i > 0) {
    const k = ss(VIEWS[i][0], VIEWS[i][0] + 1.6, t);
    if (k < 1) o = blend(VIEW_FN[VIEWS[i - 1][1]](t - VIEWS[i - 1][0]), cur, k);
  }
  // the orbit: a steady turn from when it came on
  if (orbit > 0) o = {...o, az: o.az + 14 * Math.max(0, t - lastFlip('orbit', t)) * orbit};
  return o;
};
export const viewAt = (t) => VIEWS.filter(([a]) => a <= t).pop();

// ------------------------------------------------------------ the operator's pointer
const REST = [[860, 470], [760, 420], [940, 520], [820, 380]];
export function makeCursor(ui) {
  const EV = SCRIPT.filter((e) => e.t > 0).flatMap((e) => {
    const out = [];
    if (e.seq) out.push({t: e.t, spot: ui.spots[e.seq]});
    if (e.view) out.push({t: e.t + (e.seq ? 0.35 : 0), spot: ui.spots[e.view]});
    if (e.display) out.push({t: e.t, spot: ui.spots[e.display]});
    return out;
  });
  const WP = [[0, 900, 600]];
  EV.forEach((e, i) => {
    WP.push([e.t - 0.16, ...e.spot], [e.t + 0.22, ...e.spot]);
    WP.push([e.t + 1.4, ...REST[i % REST.length]]);
  });
  WP.sort((a, b) => a[0] - b[0]);
  return (t) => {
    let i = 0;
    while (i < WP.length - 2 && t > WP[i + 1][0]) i++;
    const [t0, x0, y0] = WP[i];
    const [t1, x1, y1] = WP[i + 1];
    const k = ss(t0, t1, t);
    const len = Math.hypot(x1 - x0, y1 - y0);
    const bow = Math.sin(Math.PI * k) * Math.min(24, len * 0.07);
    let press = false;
    let ring = 0;
    let rx = 0;
    let ry = 0;
    for (const e of EV) {
      if (t >= e.t - 0.07 && t < e.t + 0.09) press = true;
      if (t >= e.t && t < e.t + 0.55) {
        ring = 1 - (t - e.t) / 0.55;
        [rx, ry] = e.spot;
      }
    }
    return {x: x0 + (x1 - x0) * k + (len ? (-(y1 - y0) / len) * bow : 0), y: y0 + (y1 - y0) * k + (len ? ((x1 - x0) / len) * bow : 0), press, ring, ringX: rx, ringY: ry};
  };
}

// ------------------------------------------------------------ the labels: parts of the right gear, in the gear's frame
const V3 = THREE.Vector3;
const PARTS = [
  {name: 'HYDRAULIC ACTUATOR', sub: 'EXTENDS / RETRACTS THE GEAR', at: () => new V3(0, 0.06, -0.42)},
  {name: 'MAIN PLATE', sub: 'AIRFRAME ATTACHMENT', at: () => new V3(0.36, 0.0, 0.08)},
  {name: 'ROD-END CONNECTOR', sub: 'RUNS IN THE PLATE SLOT', at: (m) => new V3(0, 0.06, 0.32 - m.stroke * G.stroke)},
  {name: 'GEAR RACK', sub: 'ONE RACK, FORE AND AFT', at: (m) => new V3(0, -0.055, 0.12 - m.stroke * G.stroke)},
  {name: 'PINION', sub: '32 TEETH, 4 MM MODULE', at: () => new V3(0, G.shaftY - G.pinionR * 0.6, 0)},
  {name: 'ROTARY OUTPUT SHAFT', sub: 'DRIVES BOTH SIDES', at: () => new V3(-0.16, G.shaftY, 0)},
  {name: 'CLEVIS JOINT', sub: 'TRANSFERS ROTATION ONLY', at: () => J.clone()},
  {name: 'SIDE TUBE', sub: 'SPINS ABOUT ITS AXIS', at: () => J.clone().addScaledVector(AXIS, 0.17)},
  {name: 'SPRING LEG', sub: 'STEEL, FLEXES ON TOUCHDOWN', at: (m) => E0.clone().addScaledVector(LEG0, 0.22).sub(J).applyAxisAngle(AXIS, m.stroke * Math.PI).add(J)},
];
export function makeLabels(st) {
  return (m, o) => {
    if (o < 0.02) return [];
    const y0 = st.gear.root.position.y;
    const pts = PARTS.map((p) => {
      const q = p.at(m);
      const ex = m.explode ?? 0;
      return {p, at: st.project(q.x, q.y + y0, q.z), ex};
    }).filter((x) => x.at && x.at[0] > 320 && x.at[0] < 1300 && x.at[1] > 130 && x.at[1] < 745);
    // two columns, stacked by height so no two labels touch
    const L = pts.filter((x) => x.at[0] < 800).sort((a, b) => a.at[1] - b.at[1]);
    const R = pts.filter((x) => x.at[0] >= 800).sort((a, b) => a.at[1] - b.at[1]);
    const out = [];
    const stack = (col, x, w) => {
      let y = 150;
      for (const c of col) {
        y = Math.max(y, Math.min(700, c.at[1] - 10));
        out.push({name: c.p.name, sub: c.p.sub, at: c.at, label: [x, y], w, o});
        y += 34;
      }
    };
    stack(L, 340, 170);
    stack(R, 1110, 170);
    return out;
  };
}

export function makeConsoleFrame(ui, st) {
  const labels = makeLabels(st);
  return (t, s) => {
    const S = SEQ[s.seq];
    const m = s.mech;
    const climb = (S.height(Math.min(S.dur, s.tau + 0.05)) - S.height(s.tau)) / 0.05;
    ui.update({
      state: stateAt(m.stroke, m.height),
      stroke: m.stroke,
      seq: s.seq,
      prog: s.tau / S.dur,
      height: m.height,
      climb,
      roll: s.roll,
      view: s.view,
      display: s.display,
      paused: s.paused ?? false,
      from: S.from,
      to: S.to,
      phase: phaseAt(s.seq, s.tau),
      tau: s.tau,
      dur: S.dur,
      speed: s.speed ?? 1,
      speedK: s.speedK ?? 0.5,
      callouts: labels({...m, explode: s.explode}, s.labels),
      cursor: s.cursor,
    });
  };
}

export function makeFilm() {
  return async ({canvas, capture}) => {
    const st = createCessnaStage({canvas});
    const resize = () => st.resize(1600, 900, capture ? devicePixelRatio || 1 : Math.min(devicePixelRatio || 1, 2));
    resize();
    const ui = createConsole(document.getElementById('app'));
    const cursorAt = makeCursor(ui);
    const frame = makeConsoleFrame(ui, st);
    window.CS = st;
    return {
      resize,
      render(t) {
        const {seq, tau} = seqAt(t);
        const m = mechAt(seq, tau);
        const disp = displayAt(t);
        const explode = easedDisplay('exploded', t, 0.8);
        const cam = camAt(t, disp.orbit);
        st.frame(
          {stroke: m.stroke, roll: rollAt(t), height: m.height, dist: distAt(t), fuselage: easedDisplay('fuselage', t) * 0.9, xray: easedDisplay('xray', t), explode, section: !!disp.section},
          {...cam, az: cam.az * D2R, el: cam.el * D2R},
        );
        const [vt0, view] = viewAt(t);
        frame(t, {seq, tau, mech: m, roll: rollAt(t), view, display: disp, explode, labels: easedDisplay('labels', t, 0.5) * ss(vt0 + 1.1, vt0 + 1.7, t), cursor: cursorAt(t)});
        return {};
      },
    };
  };
}

export const run = () => runSim(DEFAULTS, makeFilm());
