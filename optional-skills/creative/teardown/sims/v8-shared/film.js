// V8 · Four-Stroke, the explorer film: the engine (engine.js) on the pale studio (stage.js) under the overlay
// (console.js). The crank runs at engine speed × playback (1/20: 700 rpm turns 210° a second on screen). An operator
// isolates cylinder 1, switches the tracking to cylinder 6 from the firing order, looks down the valley, opens the
// throttle to 2,400 rpm, then walks the side, front and hero views. A pure function of t.
import {runSim} from '../kit/runtime.js';
import {CAMS, createConsole, RPM_MAX} from './console.js';
import {bankXY, CYL, E, ORDER, psi} from './kin.js';
import {createV8Stage} from './stage.js';

// the film opens on the whole engine, a slow orbit and a gentle push in for 3.5 s, then a beat every 2.5-3 s
export const SCRIPT = [
  {t: 0, cam: 'open'},
  {t: 3.5, cam: 'cyl'},
  {t: 6.9, track: 6},
  {t: 9.7, cam: 'valley'},
  {t: 12.1, rpm: [700, 2400, 1.4]},
  {t: 14.7, cam: 'side'},
  {t: 17.3, cam: 'front'},
  {t: 19.9, cam: 'hero'},
  {t: 22.5, track: 1},
];
export const DURATION = 26;
export const DEFAULTS = {duration: DURATION, fps: 60, theme: 'light', hud: {enabled: false, title: 'v8d', subtitle: ''}, director: {enabled: false}, watermark: ''};
const D2R = Math.PI / 180;
const mm = (v) => v * 0.001;
const smoother = (x) => {
  const u = Math.min(1, Math.max(0, x));
  return u * u * u * (u * (u * 6 - 15) + 10);
};
const ss = (a, b, t) => smoother((t - a) / (b - a));
const PLAY = 20; // playback divider: 1/20 speed

// ------------------------------------------------------------ engine speed and the crank angle
const RPM_EV = SCRIPT.find((e) => e.rpm);
export const rpmAt = (t) => RPM_EV.rpm[0] + (RPM_EV.rpm[1] - RPM_EV.rpm[0]) * ss(RPM_EV.t, RPM_EV.t + RPM_EV.rpm[2], t);
const DT = 1 / 240;
const TH = [];
{
  let th = 12;
  for (let t = 0; t <= DURATION + 1; t += DT) {
    TH.push(th);
    th += ((rpmAt(t + DT / 2) / 60) * 360 * DT) / PLAY;
  }
}
export const thetaAt = (t) => {
  const u = Math.max(0, Math.min(TH.length - 2, t / DT));
  const i = Math.floor(u);
  return TH[i] + (TH[i + 1] - TH[i]) * (u - i);
};
export const trackAt = (t) => SCRIPT.filter((e) => e.track && e.t <= t).pop()?.track ?? 1;

// ------------------------------------------------------------ views, each a function of u (seconds since it came on)
const cylCentre = (c) => {
  const cy = CYL[c - 1];
  const [x, y] = bankXY(cy.bank, 0, 170);
  return [mm(x), mm(y), mm(cy.z)];
};
export const VIEW_FN = {
  // the opening: a quarter turn round the whole engine, easing in and out, pushing in a little
  open: (u) => {
    const k = smoother(u / 4.1);
    return {tx: 0, ty: 0.11, tz: -0.02, az: 4 + 54 * k, el: 18 + 9 * k, d: 2.2 - 0.25 * k, sx: -50, sy: 10};
  },
  hero: (u) => ({tx: 0, ty: 0.11, tz: -0.02, az: 46 + 5 * u, el: 26, d: 1.95, sx: -50, sy: 10}),
  cyl: (u, c) => {
    const [x, y, z] = cylCentre(c);
    const L = CYL[c - 1].bank === 'L';
    return {tx: x * 0.72, ty: y * 0.72 + 0.02, tz: z, az: (L ? 16 : -16) + (L ? 1 : -1) * 5.5 * u, el: 9 + 1.2 * u, d: 1.18, sx: -40, sy: 14};
  },
  valley: (u) => ({tx: 0, ty: 0.14, tz: 0, az: 14 + 7 * u, el: 60, d: 2.35, sx: -40, sy: -28}),
  side: (u) => ({tx: 0, ty: 0.1, tz: 0, az: 100 - 5 * u, el: 12 + 1.5 * u, d: 1.95, sx: -40, sy: 10}),
  front: (u) => ({tx: 0, ty: 0.12, tz: 0, az: -6 + 5 * u, el: 10 + 2 * u, d: 1.8, sx: -40, sy: 10}),
};
const VIEWS = SCRIPT.filter((e) => e.cam).map((e) => [e.t, e.cam]);
// a camera move: every parameter eased, the azimuth the short way round, backing off through a wide swing
export const camAt = (t, list = VIEWS, track = trackAt) => {
  let i = 0;
  while (i < list.length - 1 && list[i + 1][0] <= t) i++;
  const view = (k, tt) => VIEW_FN[list[k][1]](tt - list[k][0], track(tt));
  const cur = view(i, t);
  // a tracking switch while isolated glides the camera to the new cylinder
  const tr = SCRIPT.find((e) => e.track && e.t <= t && t < e.t + 2);
  let o = cur;
  if (tr && list[i][1] === 'cyl') {
    const prev = VIEW_FN.cyl(t - list[i][0], trackAt(tr.t - 0.01));
    const k = ss(tr.t, tr.t + 1.8, t);
    o = blend(prev, cur, k, false);
  }
  if (i === 0) return o;
  const k = ss(list[i][0], list[i][0] + 1.6, t);
  if (k >= 1) return o;
  return blend(view(i - 1, t), o, k);
};
function blend(a, b, k, backOff = true) {
  const o = {};
  for (const key in b) o[key] = key === 'az' ? a.az + ((((b.az - a.az + 540) % 360) - 180) * k) : a[key] + (b[key] - a[key]) * k;
  const swing = Math.min(1, Math.abs(((b.az - a.az + 540) % 360) - 180) / 90);
  if (backOff) o.d += Math.max(0, Math.max(a.d, b.d) + 0.6 * swing - o.d) * Math.sin(Math.PI * k);
  return o;
}
export const camName = (t) => {
  const v = VIEWS.filter(([t0]) => t0 <= t).pop()[1];
  return v === 'open' ? 'hero' : v;
};
export const firingNow = (th) => {
  let best = 1;
  let bp = 1e9;
  for (const c of ORDER) {
    const p = psi(c, th);
    if (p < bp) {
      bp = p;
      best = c;
    }
  }
  return best;
};

// ------------------------------------------------------------ the operator's pointer
const REST = [[820, 470], [700, 430], [900, 520], [760, 380]];
export function makeCursor(ui) {
  const EV = SCRIPT.filter((e) => e.t > 0).map((e) => {
    if (e.rpm) return {t: e.t, spot: ui.speedKnob(e.rpm[0] / RPM_MAX), to: ui.speedKnob(e.rpm[1] / RPM_MAX), drag: e.rpm[2]};
    if (e.track) return {t: e.t, spot: ui.foSpot(ORDER.indexOf(e.track)), drag: 0};
    return {t: e.t, spot: ui.spots[e.cam], drag: 0};
  });
  const WP = [[0, 980, 640]];
  EV.forEach((e, i) => {
    WP.push([e.t - 0.16, ...e.spot]);
    if (e.drag) {
      for (let k = 0; k <= 12; k++) {
        const tt = e.t + (k / 12) * e.drag;
        WP.push([tt + 0.02, ...ui.speedKnob(rpmAt(tt) / RPM_MAX)]);
      }
      WP.push([e.t + e.drag + 0.3, ...e.to]);
    } else WP.push([e.t + 0.24, ...e.spot]);
    WP.push([e.t + e.drag + 1.6, ...REST[i % REST.length]]);
  });
  WP.sort((a, b) => a[0] - b[0]);
  return (t) => {
    let i = 0;
    while (i < WP.length - 2 && t > WP[i + 1][0]) i++;
    const [t0, x0, y0] = WP[i];
    const [t1, x1, y1] = WP[i + 1];
    const lin = t1 - t0 < 0.2;
    const k = lin ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : ss(t0, t1, t);
    const len = Math.hypot(x1 - x0, y1 - y0);
    const bow = lin ? 0 : Math.sin(Math.PI * k) * Math.min(24, len * 0.07);
    let press = false;
    let ring = 0;
    let rx = 0;
    let ry = 0;
    for (const e of EV) {
      if (t >= e.t - 0.07 && t < e.t + Math.max(0.09, e.drag + 0.1)) press = true;
      if (t >= e.t && t < e.t + 0.55) {
        ring = 1 - (t - e.t) / 0.55;
        [rx, ry] = e.spot;
      }
    }
    return {x: x0 + (x1 - x0) * k + (len ? (-(y1 - y0) / len) * bow : 0), y: y0 + (y1 - y0) * k + (len ? ((x1 - x0) / len) * bow : 0), press, ring, ringX: rx, ringY: ry};
  };
}

// ------------------------------------------------------------ the overlay's data for a frame
export function makeConsoleFrame(ui, st) {
  const P = (x, y, z) => st.project(mm(x), mm(y), mm(z));
  const Lb = [Math.SQRT1_2, Math.SQRT1_2];
  // live: the page's own state (overlays, cutaway, playback, pause, solo); absent in the film
  return (t, {th, cam, cyl, rpm, cursor, since, live}) => {
    const solo = live ? live.solo : cam === 'cyl';
    const cy = CYL[cyl - 1];
    const vis = ss(1.1, 1.8, since) * (live && !live.overlays.callouts ? 0 : 1);
    const callouts = [];
    if (!solo && (cam === 'hero' || cam === 'side')) {
      callouts.push(
        {name: 'Camshaft', sub: 'HALF CRANK SPEED', at: P((E.deck + 112) * Lb[0], (E.deck + 112) * Lb[1], -60), label: [962, 196], w: 120, o: vis},
        {name: 'Flywheel', sub: 'STORES THE PULSES', at: P(120, -40, -290), label: [1168, 572], w: 124, o: vis},
        {name: 'Helical timing gears', sub: '24 : 72', at: P(150 * Lb[0], 150 * Lb[1], 268), label: [306, 704], w: 128, o: vis},
      );
    }
    if (solo) {
      const [x, y] = bankXY(cy.bank, 0, E.deck + 112);
      callouts.push({name: 'Camshaft', sub: 'HALF CRANK SPEED', at: P(x, y, cy.z), label: [720, 226], w: 120, o: vis});
    }
    // leaders from the panels into the engine: the rev counter to the crank, the indicator panel to the tracked bore
    const [bx, by] = bankXY(cy.bank, 0, E.deck - 40);
    const lead = live && !live.overlays.callouts ? 0 : 1;
    callouts.push({name: '', sub: '', at: solo ? P(0, 0, cy.z) : P(0, 0, 312), label: [282, 152], w: 0, o: lead});
    callouts.push({name: '', sub: '', at: P(bx, by, cy.z), label: [1318 - 0.01, 501], w: 0, o: 0.9 * lead * (live && !live.overlays.charts ? 0 : 1)});
    const ps = psi(cyl, th);
    ui.update({
      rpm,
      rpmK: rpm / RPM_MAX,
      playText: live ? live.playText : '1 / 20 SPEED',
      playK: live ? live.playK : 0.33,
      paused: live ? live.paused : false,
      theta: th,
      psi: ps,
      cyl,
      pin: cy.pin,
      firing: firingNow(th),
      cam,
      cut: live ? live.cut : 'glass',
      overlays: live ? live.overlays : {callouts: 1, charge: 1, arrows: 1, isolate: solo ? 1 : 0, refract: 1, charts: 1},
      callouts,
      cursor,
    });
  };
}

export function makeFilm() {
  return async ({canvas, capture}) => {
    const st = createV8Stage({canvas});
    const resize = () => st.resize(1600, 900, capture ? devicePixelRatio || 1 : Math.min(devicePixelRatio || 1, 2));
    resize();
    const ui = createConsole(document.getElementById('app'));
    const cursorAt = makeCursor(ui);
    const frame = makeConsoleFrame(ui, st);
    window.V8 = st;
    return {
      resize,
      render(t) {
        const th = thetaAt(t);
        const cam = camName(t);
        const cyl = trackAt(t);
        const c = camAt(t);
        const t0 = VIEWS.filter(([a]) => a <= t).pop()[0];
        // isolate once the camera has closed in, and keep it until the next move has pulled back a little
        const vi = VIEWS.findIndex(([a]) => a === t0);
        const soloOn = (cam === 'cyl' && t - t0 > 1.25) || (vi > 0 && VIEWS[vi - 1][1] === 'cyl' && t - t0 < 0.6);
        st.frame(th, {...c, az: c.az * D2R, el: c.el * D2R}, {solo: soloOn ? cyl - 1 : -1});
        frame(t, {th, cam, cyl, rpm: rpmAt(t), cursor: cursorAt(t), since: t - t0});
        return {};
      },
    };
  };
}
void CAMS;

export const run = () => runSim(DEFAULTS, makeFilm());
