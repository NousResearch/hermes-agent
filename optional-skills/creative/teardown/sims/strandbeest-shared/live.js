// SB-10 as a page you drive: the film's console (console.js) over the stage (stage.js), run in real time. The view
// buttons and keys 1-5, ISOLATE LEG (I), TRACE FEET (T), PAUSE (space), the wind slider, an orbit you drag and a zoom
// on the wheel. Left alone it plays the film's script as an auto tour with the operator's cursor, looped; a click,
// drag, wheel or key hands control to the visitor, and 25 s of quiet hands it back. Wind drives the crank here too:
// theta is integrated frame by frame, and the console reads its last seconds from a short history.
import {warmComposer} from '../kit/warm.js';
import {legAt, legZ, phaseOf, S} from './beast.js';
import {createConsole, VIEW_KEYS, WIND_MAX} from './console.js';
import {DURATION, HERO, isoAt, makeConsoleFrame, makeCursor, rpsAt, scriptView, VIEW_FN, windAt} from './film.js';
import {footLift} from './jansen.js';
import {createBeestStage} from './stage.js';

const W = 1600;
const H = 900;
const IDLE = 25;
const D2R = Math.PI / 180;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smoother = (x) => {
  const u = clamp(x, 0, 1);
  return u * u * u * (u * (u * 6 - 15) + 10);
};
const ss = (a, b, t) => smoother((t - a) / (b - a));
const wrapDeg = (d) => ((((d + 180) % 360) + 360) % 360) - 180;

const CSS = (top) => `
html,body{background:#e9e2d6;overflow:hidden}
#app{position:absolute!important;inset:auto!important;left:0;top:0;width:${W}px;height:${H}px;transform-origin:0 0}
#gl{touch-action:none;cursor:grab}
#gl.drag{cursor:grabbing}
.vb{pointer-events:auto;cursor:pointer}
.vb:hover{filter:brightness(.94)}
.windhit{position:absolute;pointer-events:auto;cursor:ew-resize}
.tourchip{position:absolute;left:800px;top:48px;transform:translateX(-50%);display:flex;align-items:center;gap:7px;height:21px;padding:0 11px;border:1px solid rgba(110,98,78,.55);background:rgba(243,238,227,.92);color:#3a342b;font:700 8.5px Inter,sans-serif;letter-spacing:1.4px;white-space:nowrap;cursor:pointer;pointer-events:auto}
.tourchip i{display:block;width:6px;height:6px;border-radius:50%;background:#d0592a}
.tourchip.on i{animation:blink 1.2s steps(2) infinite}
@keyframes blink{50%{opacity:.15}}
.turn{position:fixed;left:0;right:0;bottom:18px;text-align:center;font:600 12px/1.4 'JetBrains Mono',monospace;letter-spacing:1px;color:#8a7f6c;text-transform:uppercase;display:none}
@media (orientation:portrait) and (max-width:820px){.turn{display:block}}
`;

export function runLive() {
  const top = Number(document.body.dataset.top || 0);
  const style = document.createElement('style');
  style.textContent = CSS(top);
  document.head.appendChild(style);
  const app = document.getElementById('app');
  const canvas = document.createElement('canvas');
  canvas.id = 'gl';
  app.appendChild(canvas);

  const st = createBeestStage({canvas, palette: 'dune', ink: 0.7});
  st.usePrintRing(640);
  const ui = createConsole(app);
  const tourCursorAt = makeCursor(ui);

  // ------------------------------------------------------------ quality tiers
  // the film's look, then lighter shadows and MSAA, then the outline without AO, PCF shadows and no MSAA; integrated
  // GPUs settle within a couple of seconds, phones start on the last step
  const TIERS = [
    {samples: 8, shadow: 4096, blur: 20, radius: 7, ao: 16, pcf: false, grains: 9000, pr: 2},
    {samples: 4, shadow: 2048, blur: 10, radius: 5, ao: 8, pcf: false, grains: 6000, pr: 1.5},
    {samples: 0, shadow: 1024, blur: 6, radius: 2, ao: 0, pcf: true, grains: 3000, pr: 1.1},
  ];
  const light = matchMedia('(pointer: coarse)').matches || Math.min(innerWidth, innerHeight) < 700;
  let tier = light ? 2 : 0;
  let fitReady = false;
  const applyTier = () => {
    const {pr, ...q} = TIERS[tier];
    st.setQuality(q);
    if (fitReady) fit();
  };
  applyTier();

  // ------------------------------------------------------------ fit the 1600 x 900 console into the window
  let scale = 1;
  const fit = () => {
    const h = innerHeight - top;
    scale = Math.min(innerWidth / W, h / H);
    app.style.transform = `translate(${(innerWidth - W * scale) / 2}px, ${top + (h - H * scale) / 2}px) scale(${scale})`;
    st.resize(W, H, clamp((devicePixelRatio || 1) * scale, 0.75, TIERS[tier].pr));
  };
  addEventListener('resize', fit);
  fit();
  fitReady = true;
  const toStage = (cx, cy) => {
    const r = app.getBoundingClientRect();
    return [(cx - r.left) / scale, (cy - r.top) / scale];
  };

  const tourChip = document.createElement('div');
  tourChip.className = 'tourchip';
  ui.root.appendChild(tourChip);
  const turnNote = document.createElement('div');
  turnNote.className = 'turn';
  turnNote.textContent = 'Turn your phone sideways for the full console';
  document.body.appendChild(turnNote);
  // the wind slider's hit area over its track
  const [kx0, ky] = ui.knobAt(0);
  const [kx1] = ui.knobAt(WIND_MAX);
  const windHit = document.createElement('div');
  windHit.className = 'windhit';
  Object.assign(windHit.style, {left: `${kx0 - 12}px`, top: `${ky - 13}px`, width: `${kx1 - kx0 + 24}px`, height: '26px'});
  ui.root.appendChild(windHit);

  // ------------------------------------------------------------ state
  let T = 0; // wall clock
  let simT = 0; // stops while paused
  let theta = 0;
  let flow = 0;
  let dist = 0;
  let paused = false;
  let wind = 6;
  let windTarget = 6;
  let iso = 0;
  let isoTarget = 0;
  let isoOnAt = Infinity;
  const tr = {on: false, theta0: 0, done: null, fade: 0};
  let view = 'cine';
  let viewAt0 = 0;
  const off = {az: 0, el: 0, k: 1};
  const cam = {...VIEW_FN.cine(0, 0)};
  let ring = {t: -9, x: 0, y: 0};
  // the last 14 s of the crank and the wind, for the console's histories
  const hist = [];
  const histAt = (t, key) => {
    if (!hist.length) return key === 'th' ? theta : wind;
    if (t <= hist[0].t) return key === 'th' ? hist[0].th - rpsAt(hist[0].w) * Math.PI * 2 * (hist[0].t - t) : hist[0].w;
    const last = hist[hist.length - 1];
    if (t >= last.t) return last[key];
    let lo = 0;
    let hi = hist.length - 1;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (hist[m].t <= t) lo = m;
      else hi = m;
    }
    const a = hist[lo];
    const b = hist[hi];
    return a[key] + (b[key] - a[key]) * ((t - a.t) / (b.t - a.t || 1));
  };
  const frameConsole = makeConsoleFrame(ui, st, {thetaAt: (t) => histAt(t, 'th'), windAt: (t) => histAt(t, 'w')});
  const pulse = (key) => {
    const r = ui.btn[key].getBoundingClientRect();
    const [x, y] = toStage(r.left + r.width / 2, r.top + r.height / 2);
    ring = {t: T, x, y};
  };

  const setView = (v) => {
    if (v === view) return;
    view = v;
    viewAt0 = T;
  };
  const setIso = (on) => {
    isoTarget = on ? 1 : 0;
    if (on) {
      isoOnAt = simT;
      setView('leg');
    } else if (view === 'leg') setView('front');
  };
  const setTrace = (on) => {
    if (on === tr.on) return;
    tr.on = on;
    if (on) Object.assign(tr, {theta0: theta, done: null});
  };
  const setPaused = (p) => {
    paused = p;
    ui.btn.pause.textContent = p ? '▶ RESUME' : '❚❚ PAUSE';
  };

  // ------------------------------------------------------------ the tour: the film's script, looped
  const AP = {on: true, t0: 0, loop: 0, prevIso: 0};
  let lastInput = 0;
  const startTour = () => {
    AP.on = true;
    AP.t0 = T;
    AP.loop = 0;
    AP.prevIso = 0;
    if (paused) setPaused(false);
    setTrace(false);
    Object.assign(off, {az: 0, el: 0, k: 1});
  };
  const takeOver = () => {
    lastInput = T;
    if (!AP.on) return;
    AP.on = false;
    windTarget = wind;
    isoTarget = iso > 0.5 ? 1 : 0;
  };
  tourChip.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    lastInput = T;
    if (AP.on) takeOver();
    else startTour();
  });
  addEventListener('pointerdown', takeOver);
  for (const k of VIEW_KEYS)
    ui.btn[k].addEventListener('click', () => {
      takeOver();
      pulse(k);
      if (k !== 'leg' && isoTarget) isoTarget = 0;
      if (k === 'leg') setIso(true);
      else setView(k);
      Object.assign(off, {az: 0, el: 0, k: 1});
    });
  ui.btn.iso.addEventListener('click', () => {
    takeOver();
    pulse('iso');
    setIso(!isoTarget);
  });
  ui.btn.trace.addEventListener('click', () => {
    takeOver();
    pulse('trace');
    setTrace(!tr.on);
  });
  ui.btn.pause.addEventListener('click', () => {
    takeOver();
    pulse('pause');
    setPaused(!paused);
  });
  {
    const onMove = (e) => {
      const [x] = toStage(e.clientX, e.clientY);
      windTarget = clamp(((x - kx0) / (kx1 - kx0)) * WIND_MAX, 0, WIND_MAX);
      lastInput = T;
    };
    windHit.addEventListener('pointerdown', (e) => {
      takeOver();
      windHit.setPointerCapture(e.pointerId);
      const up = () => {
        removeEventListener('pointermove', onMove);
        removeEventListener('pointerup', up);
      };
      addEventListener('pointermove', onMove);
      addEventListener('pointerup', up);
      onMove(e);
    });
  }
  addEventListener('keydown', (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const key = e.key.toLowerCase();
    const v = {1: 'cine', 2: 'leg', 3: 'front', 4: 'side', 5: 'top'}[key];
    if (v) ui.btn[v].click();
    else if (key === 'i') ui.btn.iso.click();
    else if (key === 't') ui.btn.trace.click();
    else if (key === ' ') ui.btn.pause.click();
    else if (key === 'arrowright' || key === 'arrowleft') {
      takeOver();
      windTarget = clamp(windTarget + (key === 'arrowright' ? 1 : -1), 0, WIND_MAX);
    } else return;
    e.preventDefault();
  });
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => {
    drag = {x: e.clientX, y: e.clientY};
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add('drag');
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = (e.clientX - drag.x) / scale;
    const dy = (e.clientY - drag.y) / scale;
    drag = {x: e.clientX, y: e.clientY};
    off.az -= dx * 0.3;
    off.el = clamp(off.el + dy * 0.2, -10, 70);
    lastInput = T;
  });
  const endDrag = () => {
    drag = null;
    canvas.classList.remove('drag');
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      takeOver();
      off.k = clamp(off.k * Math.exp(e.deltaY * 0.0012), 0.45, 2.2);
    },
    {passive: false},
  );

  // ------------------------------------------------------------ footprints as the feet land
  const wasDown = new Array(20).fill(true);
  const layPrints = (bx) => {
    for (let r = 0; r < 20; r++) {
      const s = r % 10;
      const side = r < 10 ? 0 : 1;
      const ps = theta + phaseOf(s);
      const down = footLift(side ? Math.PI - ps : ps) < 0.5;
      if (down && !wasDown[r]) st.addPrint(bx + legAt(s, side, theta).F[0] * S, legZ(s, side) * S, simT, (((r * 7919) % 100) / 100 - 0.5) * 0.25);
      wasDown[r] = down;
    }
  };

  // ------------------------------------------------------------ the frame
  startTour();
  let last = performance.now();
  const perf = {n: 0, sum: 0, slow: 0, age: 0};
  let odd = false;
  const frame = (now) => {
    const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
    last = now;
    T += dt;
    // quality steps down after a 0.5 s window slower than 45 fps (two in a row once the page has settled)
    if (tier < TIERS.length - 1) {
      perf.age += dt;
      if (perf.age > 0.4) {
        perf.sum += dt;
        perf.n++;
        if (perf.sum >= 0.5 && perf.n >= 6) {
          perf.slow = perf.sum / perf.n > 0.022 ? perf.slow + 1 : 0;
          perf.sum = 0;
          perf.n = 0;
          if (perf.slow >= (perf.age < 6 ? 1 : 2)) {
            tier++;
            applyTier();
            Object.assign(perf, {age: 0, slow: 0});
          }
        }
      }
    }

    // the operator: the tour's script, or the visitor's last choices
    let cyc = 0;
    let since = T - viewAt0;
    let cursor;
    if (AP.on) {
      const c = T - AP.t0;
      const loop = Math.floor(c / DURATION);
      if (loop > AP.loop) {
        AP.loop = loop;
        AP.prevIso = 0;
      }
      cyc = c - loop * DURATION;
      const [t0, v] = scriptView(cyc);
      view = v;
      since = cyc - t0;
      // keep the view's clock running so a takeover continues the move instead of jumping
      viewAt0 = T - since;
      windTarget = windAt(cyc);
      const isoNow = isoAt(cyc);
      if (isoNow > 0.01 && AP.prevIso <= 0.01) isoOnAt = simT;
      AP.prevIso = isoNow;
      isoTarget = isoNow;
      cursor = tourCursorAt(cyc);
      // from the second loop on, the pointer glides from where the last one left it
      if (AP.loop > 0 && cyc < 0.8) {
        const end = tourCursorAt(DURATION - 0.01);
        const k = ss(0, 0.8, cyc);
        cursor = {...cursor, x: end.x + (cursor.x - end.x) * k, y: end.y + (cursor.y - end.y) * k};
      }
    } else {
      if (T - lastInput > IDLE) startTour();
      const rk = Math.max(0, 1 - (T - ring.t) / 0.55);
      cursor = {x: -99, y: -99, press: false, drag: false, ring: rk, ringX: ring.x, ringY: ring.y};
    }
    wind += (windTarget - wind) * (1 - Math.exp(-dt / 0.35));
    iso += (isoTarget - iso) * (1 - Math.exp(-dt / 0.2));

    // the crank: wind sets its rate, a dead calm lets it coast to a stop
    if (!paused) {
      simT += dt;
      const rps = rpsAt(wind) * ss(0.3, 1.5, wind);
      const th0 = theta;
      theta += rps * Math.PI * 2 * dt;
      flow += wind * dt * 0.35;
      dist += Math.abs(st.bodyPos(theta).x - st.bodyPos(th0).x);
    }
    if (!paused || !hist.length) hist.push({t: simT, th: theta, w: wind});
    while (hist.length > 2 && hist[0].t < simT - 14) hist.shift();
    const b = st.bodyPos(theta);
    if (!paused) layPrints(b.x);

    // the foot traces: the hero's loop draws itself from the click, the rest fade in when it closes
    let heroTrace = 0;
    let trace = 0;
    tr.fade += ((tr.on ? 1 : 0) - tr.fade) * (1 - Math.exp(-dt / 0.25));
    if (tr.on || tr.fade > 0.01) {
      heroTrace = Math.min(1, (theta - tr.theta0) / (Math.PI * 2));
      if (heroTrace >= 1 && tr.done === null) tr.done = simT;
      trace = tr.done === null ? 0 : ss(tr.done, tr.done + 0.9, simT);
    }

    // the camera eases toward the view's own move plus the visitor's orbit; a wide swing backs off on the way
    const tgt = VIEW_FN[view](since, b.x);
    tgt.az += off.az;
    tgt.el = clamp(tgt.el + off.el, 2, 82);
    tgt.d *= off.k;
    const k = 1 - Math.exp(-dt / 0.55);
    const dAz = wrapDeg(tgt.az - cam.az);
    cam.az += dAz * k;
    for (const key of ['tx', 'ty', 'tz', 'el', 'd', 'sx', 'sy']) cam[key] += (tgt[key] - cam[key]) * k;
    const swing = Math.min(1, Math.abs(dAz) / 90);
    st.frame(theta, simT, {...cam, el: (cam.el + 8 * swing) * D2R, az: cam.az * D2R, d: cam.d + 2.5 * swing}, {wind, flow, iso, hero: HERO, trace, heroTrace, traceFade: tr.on ? 1 : tr.fade});

    // the console: every frame on a fast GPU, every other frame (the pointer every frame) on a slow one
    odd = !odd;
    if (tier > 0 && odd) ui.pointer(cursor);
    else frameConsole(simT, {view, since, iso, cursor, labelStart: isoOnAt, trace: tr.on ? 1 : 0, dist});

    if (tourChip.classList.contains('on') !== AP.on || !tourChip.firstChild) {
      tourChip.classList.toggle('on', AP.on);
      tourChip.innerHTML = AP.on ? '<i></i>AUTO TOUR · CLICK TO TAKE OVER' : '▶ RESUME TOUR';
    }
    // the site page swaps its poster for the canvas on this
    if (!window.SIM_SHOWN) {
      window.SIM_SHOWN = true;
      dispatchEvent(new Event('simframe'));
    }
    requestAnimationFrame(frame);
  };
  // every shader compiles at once off the main thread before the first frame (kit/warm.js)
  warmComposer(st.renderer, st.composer, st.scene, st.camera).then(() => requestAnimationFrame(frame));
  window.LIVE = {st, cam, startTour, takeOver, get tier() {
    return tier;
  }, get T() {
    return T;
  }};
}
