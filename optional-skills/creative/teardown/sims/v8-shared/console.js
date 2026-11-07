// The four-stroke explorer's overlay: title and the rev counter top left, the slider-crank and cross-plane panels,
// the transport bottom left, the charge legend top centre, the cylinder state, valve timing and indicator diagram on
// the right, camera / cutaway / overlay toggles bottom right, the firing order bottom centre, and callouts with thin
// leaders into the engine. Fixed 1600 x 900 CSS layout; update(s) redraws every live part from the frame's state.
import {E, ORDER, PIN_PHASE, pressure, STROKES, volume, liftEx, liftIn, pistonFrac} from './kin.js';

export const C = {ink: '#16181b', mute: '#8b8f95', faint: '#c9ccd0', line: '#e1e3e6', panel: 'rgba(248,248,249,.94)', red: '#d6182b', blue: '#3d8fe0', grey: '#5d6670', orange: '#f05a23', exh: '#8b8f95'};
export const CAMS = ['hero', 'front', 'side', 'valley', 'cyl'];
export const CUTS = ['glass', 'section', 'solid'];
export const OVERLAYS = ['callouts', 'charge', 'arrows', 'isolate', 'refract', 'charts'];
const CSS = `
#app{font-family:Inter,system-ui,sans-serif;color:${C.ink};-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
.ui{position:absolute;inset:0;pointer-events:none}
.pn{position:absolute;background:${C.panel};border:1px solid ${C.line};box-shadow:0 1px 2px rgba(20,22,26,.04)}
.cp{font-size:9.6px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;white-space:nowrap}
.mu{color:${C.mute}}
.tb{position:absolute;height:19px;padding:0 7px;display:flex;align-items:center;justify-content:center;font-size:8.8px;font-weight:600;letter-spacing:.9px;text-transform:uppercase;border:1px solid ${C.line};background:#fff;color:${C.ink};white-space:nowrap}
.call{position:absolute;white-space:nowrap}
.call b{display:block;font-size:11.5px;font-weight:600;letter-spacing:.1px}
.call span{display:block;font-size:8.6px;font-weight:500;letter-spacing:1.4px;color:${C.mute};text-transform:uppercase;margin-top:2px}
.cur{position:absolute;left:0;top:0;z-index:50;filter:drop-shadow(0 1px 2px rgba(0,0,0,.3))}
.ring{position:absolute;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;border:1.5px solid ${C.red};opacity:0;z-index:49}
`;
const el = (h) => {
  const d = document.createElement('div');
  d.innerHTML = h.trim();
  return d.firstElementChild;
};
export const RPM_MAX = 8000;

export function createConsole(app) {
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);
  const ui = el(`<div class="ui"></div>`);
  app.appendChild(ui);
  const add = (h) => {
    const n = el(h);
    ui.appendChild(n);
    return n;
  };

  // ------------------------------------------------------------ title and rev counter
  add(`<div style="position:absolute;left:35px;top:33px;width:43px;height:3px;background:${C.red}"></div>`);
  add(`<div style="position:absolute;left:35px;top:47px;font-size:24px;font-weight:700;letter-spacing:-.2px">V8 <span style="color:${C.red}">·</span> Four-Stroke</div>`);
  add(`<div class="cp" style="position:absolute;left:35px;top:81px;font-weight:500;letter-spacing:1.7px;font-size:9.4px">90° CROSS-PLANE &nbsp;•&nbsp; FIRING ORDER 1-8-4-3-6-5-7-2 &nbsp;•&nbsp; 5.8 L</div>`);
  const rpmBig = add(`<div style="position:absolute;left:33px;top:103px;font-size:56px;font-weight:800;letter-spacing:-3.2px;color:${C.red};line-height:1"></div>`);
  add(`<div style="position:absolute;left:255px;top:141px;font-size:11px;font-weight:500">8000</div>`);
  add(`<div style="position:absolute;left:35px;top:166px;width:247px;height:6px;border:1px solid ${C.faint};border-top:none"></div>`);
  add(`<div style="position:absolute;left:35px;top:174px;width:247px;height:9px;background:#e6e7e9"></div>`);
  const rpmFill = add(`<div style="position:absolute;left:35px;top:174px;height:9px;background:${C.red}"></div>`);
  add(`<div class="cp mu" style="position:absolute;left:35px;top:188px;font-size:9px;font-weight:500">0</div>`);
  add(`<div class="cp" style="position:absolute;left:212px;top:188px;font-size:9px;font-weight:500;color:${C.ink}">REV / MIN</div>`);
  add(`<div style="position:absolute;left:35px;top:206px;width:247px;height:1px;background:${C.faint}"></div>`);
  add(`<div style="position:absolute;left:35px;top:213px;font-size:12.5px;font-weight:500">Crankshaft</div>`);

  // ------------------------------------------------------------ slider-crank
  add(`<div class="pn" style="left:35px;top:274px;width:247px;height:208px"></div>`);
  add(`<div class="cp" style="position:absolute;left:46px;top:286px;font-weight:700">SLIDER-CRANK</div>`);
  const scAng = add(`<div class="cp mu" style="position:absolute;right:${1600 - 271}px;top:286px;font-weight:500"></div>`);
  const sc = add(`<svg style="position:absolute;left:46px;top:303px" width="226" height="130"></svg>`);
  add(`<div style="position:absolute;left:46px;top:440px;width:226px;height:1px;background:${C.line}"></div>`);
  add(`<div style="position:absolute;left:46px;top:447px;font-size:9.5px;color:${C.mute};font-family:'JetBrains Mono',monospace">s = r·cos φ + √(L² − r² sin²φ)</div>`);
  add(`<div style="position:absolute;left:46px;top:461px;font-size:9.5px;color:${C.mute}">r <b style="color:${C.ink}">4.4</b> cm &nbsp;·&nbsp; L <b style="color:${C.ink}">14.5</b> cm</div>`);

  // ------------------------------------------------------------ cross-plane crank
  add(`<div class="pn" style="left:35px;top:490px;width:247px;height:192px"></div>`);
  add(`<div class="cp" style="position:absolute;left:46px;top:502px;font-weight:700">CROSS-PLANE CRANK</div>`);
  add(`<div class="cp mu" style="position:absolute;right:${1600 - 271}px;top:502px;font-weight:500">FRONT VIEW</div>`);
  add(`<div class="cp mu" style="position:absolute;left:46px;top:516px;font-size:8.6px;font-weight:500">R BANK</div>`);
  add(`<div class="cp mu" style="position:absolute;right:${1600 - 271}px;top:516px;font-size:8.6px;font-weight:500">L BANK</div>`);
  const cp = add(`<svg style="position:absolute;left:46px;top:526px" width="226" height="130"></svg>`);
  add(`<div style="position:absolute;left:46px;top:654px;width:226px;height:1px;background:${C.line}"></div>`);
  add(`<div style="position:absolute;left:46px;top:661px;font-size:9.5px;color:${C.mute}">Journals 90° apart &nbsp;·&nbsp; banks at ±45°</div>`);

  // ------------------------------------------------------------ transport
  add(`<div class="pn" style="left:35px;top:774px;width:259px;height:111px"></div>`);
  const btn = {};
  btn.pause = add(`<div class="tb" style="left:45px;top:782px;width:99px;height:26px;background:${C.red};color:#fff;border-color:${C.red};font-size:9.6px">❚❚ &nbsp;PAUSE</div>`);
  btn.back = add(`<div class="tb" style="left:149px;top:782px;width:66px;height:26px;font-size:9.6px">−15°</div>`);
  btn.fwd = add(`<div class="tb" style="left:219px;top:782px;width:66px;height:26px;font-size:9.6px">+15°</div>`);
  add(`<div class="cp" style="position:absolute;left:45px;top:818px;font-weight:500">ENGINE SPEED</div>`);
  const speedVal = add(`<div class="cp" style="position:absolute;right:${1600 - 284}px;top:818px;font-weight:700"></div>`);
  const SL = {x: 46, w: 238};
  add(`<div style="position:absolute;left:${SL.x}px;top:836px;width:${SL.w}px;height:2px;background:#e1e3e6"></div>`);
  const speedKnob = add(`<div style="position:absolute;top:831px;width:9px;height:11px;margin-left:-4.5px;background:${C.red}"></div>`);
  add(`<div class="cp" style="position:absolute;left:45px;top:851px;font-weight:500">PLAYBACK</div>`);
  const playVal = add(`<div class="cp" style="position:absolute;right:${1600 - 284}px;top:851px;font-weight:700"></div>`);
  add(`<div style="position:absolute;left:${SL.x}px;top:869px;width:${SL.w}px;height:2px;background:#e1e3e6"></div>`);
  const playKnob = add(`<div style="position:absolute;top:864px;width:9px;height:11px;margin-left:-4.5px;background:${C.red}"></div>`);

  // ------------------------------------------------------------ legend
  add(`<div class="cp" style="position:absolute;left:598px;top:36px;display:flex;gap:16px;font-weight:500;font-size:9.6px">
    ${[['AIR & FUEL', C.blue], ['COMPRESSED', C.grey], ['COMBUSTION', C.orange], ['EXHAUST', C.exh]].map(([n, c]) => `<span><i style="display:inline-block;width:15px;height:2.4px;background:${c};vertical-align:middle;margin-right:6px"></i>${n}</span>`).join('')}</div>`);

  // ------------------------------------------------------------ cylinder state
  add(`<div class="pn" style="left:1318px;top:36px;width:245px;height:209px"></div>`);
  const cylName = add(`<div class="cp" style="position:absolute;left:1329px;top:47px;font-weight:500"></div>`);
  const cylCrank = add(`<div class="cp mu" style="position:absolute;right:${1600 - 1552}px;top:47px;font-weight:500"></div>`);
  const strokeBig = add(`<div style="position:absolute;left:1329px;top:61px;font-size:24px;font-weight:700"></div>`);
  const strokeList = add(`<div style="position:absolute;left:1329px;top:97px"></div>`);
  add(`<div style="position:absolute;left:1329px;top:165px;width:224px;height:1px;background:${C.line}"></div>`);
  const barBig = add(`<div style="position:absolute;left:1328px;top:173px;font-size:36px;font-weight:800;letter-spacing:-1.6px;color:${C.red};line-height:1"></div>`);
  const barUnit = add(`<div style="position:absolute;top:197px;font-size:10.5px;font-weight:500;letter-spacing:1px;line-height:1.35">BAR<br><span style="color:${C.mute};letter-spacing:1.5px">CHAMBER PRESSURE</span></div>`);
  add(`<div style="position:absolute;left:1329px;top:228px;width:224px;height:5px;background:#e6e7e9"></div>`);
  const barFill = add(`<div style="position:absolute;left:1329px;top:228px;height:5px;background:${C.red}"></div>`);

  // ------------------------------------------------------------ valve timing
  add(`<div class="pn" data-g="charts" style="left:1318px;top:252px;width:245px;height:184px"></div>`);
  add(`<div class="cp" data-g="charts" style="position:absolute;left:1329px;top:263px;font-weight:700">VALVE TIMING</div>`);
  add(`<div class="cp mu" data-g="charts" style="position:absolute;right:${1600 - 1552}px;top:263px;font-weight:500">DRAG TO SCRUB</div>`);
  const vtLeg = add(`<div class="cp" data-g="charts" style="position:absolute;left:1329px;top:280px;font-weight:500;font-size:9px"></div>`);
  const vt = add(`<svg data-g="charts" style="position:absolute;left:1326px;top:296px" width="232" height="135"></svg>`);

  // ------------------------------------------------------------ indicator diagram
  add(`<div class="pn" data-g="charts" style="left:1318px;top:443px;width:245px;height:150px"></div>`);
  add(`<div class="cp" data-g="charts" style="position:absolute;left:1329px;top:455px;font-weight:700">INDICATOR DIAGRAM</div>`);
  const idCyl = add(`<div class="cp mu" data-g="charts" style="position:absolute;right:${1600 - 1552}px;top:455px;font-weight:500"></div>`);
  const id = add(`<svg data-g="charts" style="position:absolute;left:1326px;top:470px" width="232" height="120"></svg>`);

  // ------------------------------------------------------------ camera, cutaway, overlays
  add(`<div class="pn" style="left:1296px;top:720px;width:268px;height:165px"></div>`);
  const group = (title, keys, y, labels) => {
    add(`<div class="cp mu" style="position:absolute;left:1305px;top:${y}px;font-weight:500">${title}</div>`);
    let x = 1305;
    let yy = y + 15;
    keys.forEach((k) => {
      const w = Math.round(labels[k].length * 6.2 + 14);
      if (x + w > 1556) {
        x = 1305;
        yy += 23;
      }
      btn[k] = add(`<div class="tb" style="left:${x}px;top:${yy}px;width:${w}px">${labels[k]}</div>`);
      x += w + 4;
    });
  };
  const L = Object.fromEntries([...CAMS, ...CUTS, ...OVERLAYS].map((k) => [k, k.toUpperCase()]));
  group('CAMERA', CAMS, 727, L);
  group('CUTAWAY', CUTS, 771, L);
  group('OVERLAYS', OVERLAYS, 815, L);

  // ------------------------------------------------------------ firing order
  add(`<div class="cp mu" style="position:absolute;left:0;right:0;top:795px;text-align:center;font-weight:500"><span style="color:${C.ink}">FIRING ORDER</span> &nbsp;— &nbsp;CLICK TO TRACK</div>`);
  const fo = add(`<svg style="position:absolute;left:640px;top:815px" width="320" height="34"></svg>`);
  add(`<div class="cp mu" style="position:absolute;left:0;right:0;top:852px;text-align:center;font-weight:500">ONE POWER STROKE EVERY 90° OF CRANK ROTATION</div>`);
  add(`<div class="cp" style="position:absolute;left:0;right:0;top:874px;text-align:center;font-weight:500;color:${C.faint}">DRAG TO ORBIT &nbsp;·&nbsp; SCROLL TO ZOOM &nbsp;·&nbsp; H HIDES THE OVERLAY</div>`);

  // ------------------------------------------------------------ callouts, the pointer
  const callSvg = add(`<svg data-callouts style="position:absolute;inset:0;overflow:visible" width="1600" height="900"></svg>`);
  const callG = add(`<div data-callouts style="position:absolute;inset:0"></div>`);
  const ring = add(`<div class="ring"></div>`);
  const cur = add(`<div class="cur"><svg width="22" height="22" viewBox="0 0 34 34"><path d="M4 3 L30 13.5 L17.5 17.5 L13.5 30 Z" fill="#111" stroke="#fff" stroke-width="2.2" stroke-linejoin="round"/></svg></div>`);
  const centre = (n) => {
    const r = n.getBoundingClientRect();
    return [r.left + r.width / 2, r.top + r.height / 2];
  };
  const foSpot = (i) => [640 + 20 + i * 40, 815 + 17];
  const spots = Object.fromEntries(Object.entries(btn).map(([k, n]) => [k, centre(n)]));
  const knobX = (v) => SL.x + v * SL.w;

  const pointer = (c) => {
    cur.style.display = c.x < 0 ? 'none' : 'block';
    cur.style.transform = `translate(${(c.x - 3).toFixed(2)}px, ${(c.y - 2).toFixed(2)}px) scale(${c.press ? 0.88 : 1})`;
    ring.style.left = `${c.ringX}px`;
    ring.style.top = `${c.ringY}px`;
    ring.style.opacity = String(c.ring * 0.85);
    ring.style.transform = `scale(${(0.35 + 0.9 * (1 - c.ring)).toFixed(3)})`;
  };

  return {
    spots,
    foSpot,
    speedKnob: (v) => [knobX(v), 836],
    playKnob: (v) => [knobX(v), 869],
    btn,
    root: ui,
    update(s) {
      rpmBig.textContent = Math.round(s.rpm);
      rpmFill.style.width = `${((s.rpm / RPM_MAX) * 247).toFixed(1)}px`;
      speedVal.textContent = `${Math.round(s.rpm)} RPM`;
      speedKnob.style.left = `${knobX(s.rpmK).toFixed(1)}px`;
      playVal.textContent = s.playText;
      playKnob.style.left = `${knobX(s.playK).toFixed(1)}px`;
      btn.pause.innerHTML = s.paused ? '▶ &nbsp;PLAY' : '❚❚ &nbsp;PAUSE';

      // slider-crank: the tracked cylinder's crank and piston
      const ph = s.psi;
      const ca = ((ph % 360) + 360) % 360;
      scAng.innerHTML = `<b style="color:${C.ink}">${Math.round(ca)}°</b> AFTER TDC`;
      {
        const cx = 40;
        const cy = 62;
        const r = 18;
        const Lr = 60;
        const a = (ca * Math.PI) / 180;
        const px = cx + r * Math.cos(a);
        const py = cy - r * Math.sin(a);
        const sx = cx + r * Math.cos(a) + Math.sqrt(Lr * Lr - (r * Math.sin(a)) ** 2);
        sc.innerHTML = `<circle cx="${cx}" cy="${cy}" r="${r + 18}" fill="none" stroke="${C.line}"/><circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${C.faint}" stroke-dasharray="2 2"/>
          <line x1="${cx}" y1="${cy}" x2="${px.toFixed(1)}" y2="${py.toFixed(1)}" stroke="${C.red}" stroke-width="2"/>
          <line x1="${px.toFixed(1)}" y1="${py.toFixed(1)}" x2="${sx.toFixed(1)}" y2="${cy}" stroke="${C.ink}" stroke-width="1.2"/>
          <rect x="${(sx - 10).toFixed(1)}" y="${cy - 30}" width="20" height="60" fill="#fff" stroke="${C.ink}" stroke-width="1.6"/>
          <line x1="${cx + Lr - r + 10}" x2="226" y1="${cy - 32}" y2="${cy - 32}" stroke="${C.ink}" stroke-width="1"/><line x1="${cx + Lr - r + 10}" x2="226" y1="${cy + 32}" y2="${cy + 32}" stroke="${C.ink}" stroke-width="1"/>
          <circle cx="${cx}" cy="${cy}" r="3.5" fill="#fff" stroke="${C.ink}" stroke-width="1.6"/><circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="3.5" fill="#fff" stroke="${C.ink}" stroke-width="1.6"/><circle cx="${sx.toFixed(1)}" cy="${cy}" r="3" fill="#fff" stroke="${C.ink}" stroke-width="1.4"/>
          <text x="${(cx + px) / 2 + 4}" y="${(cy + py) / 2 - 3}" font-size="9" fill="${C.ink}">r</text><text x="${(px + sx) / 2}" y="${cy - 6}" font-size="9" fill="${C.ink}">L</text>
          <line x1="${cx + Lr - r}" x2="${cx + Lr + r}" y1="112" y2="112" stroke="${C.ink}" stroke-width=".8"/><line x1="${cx + Lr - r}" x2="${cx + Lr - r}" y1="106" y2="116" stroke="${C.ink}" stroke-width=".8"/><line x1="${cx + Lr + r}" x2="${cx + Lr + r}" y1="106" y2="116" stroke="${C.ink}" stroke-width=".8"/>
          <text x="${cx + Lr + r + 6}" y="126" font-size="9.4" letter-spacing="1.4" fill="${C.mute}">STROKE 89 MM</text>`;
      }
      // cross-plane: four throws at their real angles, the tracked pin lit
      {
        const cx = 113;
        const cy = 70;
        let h = `<circle cx="${cx}" cy="${cy}" r="40" fill="none" stroke="${C.line}"/>`;
        for (const sgn of [-1, 1]) {
          const a = (-90 + sgn * 45) * (Math.PI / 180);
          const bx = cx + Math.cos(a) * 70;
          const by = cy + Math.sin(a) * 70;
          h += `<line x1="${cx}" y1="${cy}" x2="${bx}" y2="${by}" stroke="${C.faint}" stroke-width=".8" stroke-dasharray="3 3"/><g transform="translate(${bx} ${by}) rotate(${sgn * 45})"><rect x="-10" y="-12" width="20" height="16" fill="#fff" stroke="${C.ink}" stroke-width="1.4"/><line x1="-6" x2="6" y1="-1" y2="-1" stroke="${C.ink}" stroke-width="1.2"/></g>`;
        }
        const names = ['1·2', '3·4', '5·6', '7·8'];
        PIN_PHASE.forEach((ph0, p) => {
          const a = ((s.theta + ph0) * Math.PI) / 180;
          const x = cx + Math.sin(a) * 40;
          const y = cy - Math.cos(a) * 40;
          const lit = p === s.pin;
          h += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${C.ink}" stroke-width=".9"/><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="9.5" fill="${lit ? C.red : '#fff'}" stroke="${lit ? C.red : C.ink}" stroke-width="1.3"/><text x="${x.toFixed(1)}" y="${(y + 3).toFixed(1)}" font-size="7.6" text-anchor="middle" fill="${lit ? '#fff' : C.ink}" font-weight="600">${names[p]}</text>`;
        });
        h += `<circle cx="${cx}" cy="${cy}" r="6" fill="#fff" stroke="${C.ink}" stroke-width="1.3"/><line x1="${cx - 4}" x2="${cx + 4}" y1="${cy - 4}" y2="${cy + 4}" stroke="${C.ink}"/><line x1="${cx - 4}" x2="${cx + 4}" y1="${cy + 4}" y2="${cy - 4}" stroke="${C.ink}"/><text x="${cx - 12}" y="${cy - 26}" font-size="8" fill="${C.ink}">90°</text>`;
        cp.innerHTML = h;
      }

      // cylinder state
      const stroke = STROKES[Math.floor(((ph % 720) + 720) % 720 / 180)];
      cylName.innerHTML = `CYLINDER <b>${s.cyl}</b>`;
      cylCrank.innerHTML = `<b style="color:${C.ink}">${Math.round(((ph % 720) + 720) % 720)}°</b> CRANK`;
      strokeBig.textContent = stroke.charAt(0) + stroke.slice(1).toLowerCase();
      strokeBig.style.color = stroke === 'POWER' ? C.red : C.ink;
      strokeList.innerHTML = ['INTAKE', 'COMPRESSION', 'POWER', 'EXHAUST']
        .map((n) => `<div class="cp" style="height:16.5px;display:flex;align-items:center;gap:10px;font-weight:500;color:${n === stroke ? C.ink : C.mute}"><i style="display:block;width:31px;height:2px;background:${n === stroke ? C.red : C.line}"></i>${n}</div>`)
        .join('');
      const bar = pressure(ph);
      barBig.textContent = bar < 10 ? bar.toFixed(1) : Math.round(bar);
      barUnit.style.left = `${1334 + barBig.getBoundingClientRect().width + 8}px`;
      barFill.style.width = `${Math.min(224, (bar / 60) * 224).toFixed(1)}px`;

      // valve timing over 720°, the cursor at the tracked cylinder's phase
      {
        const W = 232;
        const X = (d) => 16 + (d / 720) * (W - 24);
        const Y = (v) => 108 - v * 92;
        let h = `<text x="2" y="14" font-size="6.5" fill="${C.mute}">%</text>`;
        for (const [v, l] of [[0, '0'], [0.5, '50'], [1, '100']]) h += `<text x="12" y="${Y(v) + 2}" font-size="6.5" text-anchor="end" fill="${C.mute}">${l}</text>`;
        ['POWER', 'EXHAUST', 'INTAKE', 'COMPR'].forEach((n, i) => (h += `<text x="${X(i * 180 + 90)}" y="10" font-size="6.2" letter-spacing=".8" text-anchor="middle" fill="${C.mute}">${n}</text>`));
        const seg = Math.floor(((ph % 720) + 720) % 720 / 180);
        h += `<rect x="${X(seg * 180)}" y="${Y(1)}" width="${X(180) - X(0)}" height="92" fill="${seg === 0 ? 'rgba(214,24,43,.1)' : 'rgba(20,22,26,.05)'}"/>`;
        for (const d of [180, 360, 540]) h += `<line x1="${X(d)}" x2="${X(d)}" y1="${Y(1)}" y2="${Y(0)}" stroke="${C.line}" stroke-width=".8"/>`;
        h += `<line x1="${X(0)}" x2="${X(720)}" y1="${Y(0)}" y2="${Y(0)}" stroke="${C.ink}" stroke-width=".8"/>`;
        const curve = (fn, col, w) => {
          let p = '';
          for (let d = 0; d <= 720; d += 4) p += `${d ? 'L' : 'M'}${X(d).toFixed(1)} ${Y(fn(d)).toFixed(1)}`;
          return `<path d="${p}" fill="none" stroke="${col}" stroke-width="${w}"/>`;
        };
        h += curve((d) => 1 - pistonFrac(d), C.grey, 1.3) + curve(liftEx, C.orange, 1.5) + curve(liftIn, C.blue, 1.5);
        const cxp = X(((ph % 720) + 720) % 720);
        h += `<line x1="${cxp}" x2="${cxp}" y1="${Y(1) - 4}" y2="${Y(0)}" stroke="${C.red}" stroke-width="1.4"/>`;
        for (const d of [0, 180, 360, 540, 720]) h += `<text x="${X(d)}" y="${Y(0) + 10}" font-size="6.5" text-anchor="middle" fill="${C.mute}">${d}</text>`;
        h += `<text x="${X(720)}" y="${Y(0) + 20}" font-size="6.2" letter-spacing="1" text-anchor="end" fill="${C.mute}">CRANK ANGLE AFTER IGNITION</text>`;
        vt.innerHTML = h;
        vtLeg.innerHTML = `<span style="color:${C.grey}">—</span> PISTON <b>${Math.round((1 - pistonFrac(ph)) * 100)}</b> % &nbsp; <span style="color:${C.blue}">—</span> IN <b>${liftIn(ph).toFixed(2)}</b> &nbsp; <span style="color:${C.orange}">—</span> EX <b>${liftEx(ph).toFixed(2)}</b>`;
      }
      // indicator diagram: pressure against volume over the cycle, the dot at the tracked cylinder's phase
      {
        idCyl.innerHTML = `CYL <b style="color:${C.ink}">${s.cyl}</b>`;
        const W = 232;
        const X = (v) => 18 + ((v - 1) / (E.cr - 1)) * (W - 30);
        const Y = (p) => 98 - Math.min(1, p / 55) * 82;
        let loop = '';
        for (let d = 0; d <= 720; d += 3) loop += `${d ? 'L' : 'M'}${X(volume(d)).toFixed(1)} ${Y(pressure(d)).toFixed(1)}`;
        let h = `<text x="2" y="12" font-size="6.5" fill="${C.mute}">bar</text><text x="12" y="${Y(50) + 2}" font-size="6.5" text-anchor="end" fill="${C.mute}">50</text><text x="12" y="${Y(25) + 2}" font-size="6.5" text-anchor="end" fill="${C.mute}">25</text><text x="12" y="${Y(0) + 2}" font-size="6.5" text-anchor="end" fill="${C.mute}">0</text>`;
        h += `<line x1="${X(1)}" x2="${X(E.cr)}" y1="${Y(0)}" y2="${Y(0)}" stroke="${C.ink}" stroke-width=".8"/><line x1="${X(1)}" x2="${X(1)}" y1="${Y(55)}" y2="${Y(0)}" stroke="${C.ink}" stroke-width=".8"/>`;
        h += `<path d="${loop} Z" fill="rgba(214,24,43,.12)" stroke="${C.ink}" stroke-width="1.3"/>`;
        h += `<text x="${X(1.3)}" y="${Y(52)}" font-size="6.2" letter-spacing=".8" fill="${C.mute}">IGNITION</text><text x="${X(4.2)}" y="${Y(14)}" font-size="6.2" letter-spacing=".8" fill="${C.mute}">EXPANSION</text><text x="${X(6)}" y="${Y(3.5)}" font-size="6.2" letter-spacing=".8" fill="${C.mute}">COMPRESSION</text>`;
        h += `<circle cx="${X(volume(ph)).toFixed(1)}" cy="${Y(pressure(ph)).toFixed(1)}" r="3" fill="${C.red}"/>`;
        h += `<text x="${X(1)}" y="${Y(0) + 10}" font-size="6.2" fill="${C.mute}">TDC</text><text x="${X(E.cr)}" y="${Y(0) + 10}" font-size="6.2" text-anchor="end" fill="${C.mute}">BDC</text><text x="${X(E.cr)}" y="${Y(0) + 19}" font-size="6" letter-spacing=".8" text-anchor="end" fill="${C.mute}">VOLUME ×VC · 1 → 10.2</text>`;
        id.innerHTML = h;
      }

      // firing order: the tracked cylinder ringed, the one firing now lit
      {
        let h = '';
        ORDER.forEach((c, i) => {
          const x = 20 + i * 40;
          if (i) h += `<line x1="${x - 28}" x2="${x - 12}" y1="17" y2="17" stroke="${C.faint}"/>`;
          const firing = c === s.firing;
          const tracked = c === s.cyl;
          h += `<circle cx="${x}" cy="17" r="11.5" fill="${firing ? 'rgba(214,24,43,.55)' : '#fff'}" stroke="${tracked ? C.red : C.faint}" stroke-width="${tracked ? 1.8 : 1}"/>`;
          if (tracked) h += `<circle cx="${x}" cy="17" r="15" fill="none" stroke="${C.red}" stroke-width=".8" opacity=".55"/>`;
          h += `<text x="${x}" y="20.5" font-size="10" text-anchor="middle" font-weight="600" fill="${firing ? '#fff' : C.ink}">${c}</text>`;
        });
        fo.innerHTML = h;
      }

      // toggles
      for (const k of CAMS) {
        const on = s.cam === k;
        btn[k].style.background = on ? C.ink : '#fff';
        btn[k].style.color = on ? '#fff' : C.ink;
        btn[k].style.borderColor = on ? C.ink : C.line;
      }
      for (const k of CUTS) {
        const on = s.cut === k;
        btn[k].style.background = on ? C.ink : '#fff';
        btn[k].style.color = on ? '#fff' : C.ink;
        btn[k].style.borderColor = on ? C.ink : C.line;
      }
      for (const k of OVERLAYS) {
        const on = !!s.overlays[k];
        btn[k].style.background = on ? C.red : '#fff';
        btn[k].style.color = on ? '#fff' : C.ink;
        btn[k].style.borderColor = on ? C.red : C.line;
      }

      // callouts: a label, a thin leader to its point in the engine
      let lh = '';
      let ch = '';
      for (const c of s.callouts) {
        if (c.o < 0.02 || !c.at) continue;
        const [ax, ay] = c.at;
        const [lx, ly] = c.label;
        lh += `<polyline points="${ax.toFixed(1)},${ay.toFixed(1)} ${lx.toFixed(1)},${(ly + 16).toFixed(1)} ${(lx + c.w).toFixed(1)},${(ly + 16).toFixed(1)}" fill="none" stroke="${C.mute}" stroke-width=".8" opacity="${c.o.toFixed(2)}"/>`;
        ch += `<div class="call" style="left:${lx.toFixed(1)}px;top:${(ly - 10).toFixed(1)}px;opacity:${c.o.toFixed(2)}"><b>${c.name}</b><span>${c.sub}</span></div>`;
      }
      callSvg.innerHTML = lh;
      callG.innerHTML = ch;
      pointer(s.cursor);
    },
    pointer,
  };
}
