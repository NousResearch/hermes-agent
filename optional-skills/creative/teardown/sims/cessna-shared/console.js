// The Skymaster gear console, laid out at 1600 x 900: the orange 337 badge and the title,
// the mechanism state with its live schematic, the flight panel, the system overview, the sequence switch, the views,
// the display toggles, the gear operation timeline, floor notes and the labels with leaders into the mechanism.
// update(s) redraws every live part from the frame's state.
export const C = {ink: '#2a2521', mute: '#9b8f80', faint: '#d9cdbb', line: '#eadfcd', panel: 'rgba(252,249,244,.95)', or: '#e3562a', green: '#2fa36b', blue: '#3c79a8'};
export const VIEWS = ['hero', 'front', 'plan', 'top', 'gearbox', 'sidetube'];
export const VIEW_LABEL = {hero: 'HERO', front: 'FRONT', plan: 'PLAN', top: 'TOP ¾', gearbox: 'GEARBOX', sidetube: 'SIDE TUBE'};
export const DISPLAY = ['labels', 'section', 'xray', 'exploded', 'fuselage', 'orbit'];
export const DISPLAY_LABEL = {labels: 'LABELS', section: 'SECTION', xray: 'X-RAY PLATE', exploded: 'EXPLODED', fuselage: 'FUSELAGE', orbit: 'ORBIT'};
const CSS = `
#app{font-family:Inter,system-ui,sans-serif;color:${C.ink};-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
.ui{position:absolute;inset:0;pointer-events:none}
.mo{font-family:'JetBrains Mono',ui-monospace,monospace}
.cp{position:absolute;background:${C.panel};border:1px solid ${C.line};border-radius:6px;box-shadow:0 2px 10px rgba(120,90,50,.08)}
.ph{position:absolute;display:flex;align-items:center;gap:9px;font:700 10.5px 'JetBrains Mono',monospace;letter-spacing:1.8px}
.ph i{display:block;width:3px;height:13px;background:${C.or}}
.pn{position:absolute;font:500 9.5px 'JetBrains Mono',monospace;color:${C.mute};letter-spacing:1px}
.sm{font:500 9px 'JetBrains Mono',monospace;letter-spacing:1.6px;color:${C.mute};text-transform:uppercase;white-space:nowrap}
.bt{position:absolute;height:21px;padding:0 10px;display:flex;align-items:center;justify-content:center;border:1px solid ${C.faint};border-radius:4px;background:#fffdf9;font:700 9.4px 'JetBrains Mono',monospace;letter-spacing:1.3px;color:${C.ink};white-space:nowrap}
.call{position:absolute;white-space:nowrap}
.call b{display:flex;align-items:center;gap:6px;font:700 9.4px 'JetBrains Mono',monospace;letter-spacing:1.3px;color:${C.ink}}
.call b i{display:block;width:6px;height:6px;background:${C.or}}
.call span{display:block;font:500 7.8px 'JetBrains Mono',monospace;letter-spacing:1.2px;color:${C.mute};margin-top:2px}
.cur{position:absolute;left:0;top:0;z-index:50;filter:drop-shadow(0 1px 2px rgba(0,0,0,.3))}
.ring{position:absolute;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;border:1.5px solid ${C.or};opacity:0;z-index:49}
`;
const el = (h) => {
  const d = document.createElement('div');
  d.innerHTML = h.trim();
  return d.firstElementChild;
};

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
  const panel = (x, y, w, h, title, n) => {
    add(`<div class="cp" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px"></div>`);
    add(`<div class="ph" style="left:${x + 14}px;top:${y + 13}px"><i></i>${title}</div>`);
    add(`<div class="pn" style="right:${1600 - x - w + 14}px;top:${y + 15}px">${n}</div>`);
  };

  // ------------------------------------------------------------ header
  add(`<div style="position:absolute;left:0;top:0;width:160px;height:89px;background:${C.or}"></div>`);
  add(`<svg style="position:absolute;left:26px;top:30px" width="46" height="28"><g fill="#fff"><path d="M6 0h7L7 28H0z"/><path d="M19 0h7l-6 28h-7z"/><path d="M32 0h7l-6 28h-7z"/></g><circle cx="43" cy="25" r="3" fill="#fff"/></svg>`);
  add(`<div class="mo" style="position:absolute;left:85px;top:35px;color:#fff;font-size:16px;font-weight:600;letter-spacing:4px">337</div>`);
  add(`<div class="sm" style="position:absolute;left:182px;top:22px;letter-spacing:2.4px">GENERAL AVIATION</div>`);
  add(`<div style="position:absolute;left:181px;top:37px;font-size:33px;font-weight:700;letter-spacing:-.6px;line-height:1">Cessna 337 Skymaster <span style="color:${C.or};font-weight:400">/</span></div>`);
  add(`<div class="mo" style="position:absolute;left:182px;top:77px;font-size:11.2px;letter-spacing:2.2px;color:#6d6357">MAIN LANDING GEAR / RETRACTION MECHANISM</div>`);
  add(`<div class="sm" style="position:absolute;left:1087px;top:49px;letter-spacing:2.6px;font-size:9.6px">TWIN ENGINE · CENTRELINE THRUST</div>`);
  add(`<div class="mo" style="position:absolute;left:1352px;top:47px;color:${C.or};font-size:12px;letter-spacing:-1px">//////</div>`);
  add(`<div style="position:absolute;left:1418px;top:34px;color:${C.or};font-size:39px;font-weight:700;letter-spacing:-.8px;line-height:1">337</div>`);
  add(`<div style="position:absolute;left:1500px;top:33px;width:1px;height:44px;background:${C.faint}"></div>`);
  add(`<div class="sm" style="position:absolute;left:1516px;top:34px;line-height:14px;letter-spacing:2.2px">PUSH<br>PULL<br>PERFORM</div>`);
  {
    let h = '';
    for (let r = 0; r < 3; r++) for (let c = 0; c < 12; c++) h += `<path d="M${c * 25.5 + 4} ${r * 25.5}v8M${c * 25.5} ${r * 25.5 + 4}h8" stroke="${C.or}" stroke-width="1.1" opacity=".75"/>`;
    add(`<svg style="position:absolute;left:808px;top:80px" width="300" height="64">${h}</svg>`);
  }

  // ------------------------------------------------------------ 01 mechanism state
  panel(28, 126, 282, 254, 'MECHANISM STATE', '01');
  const chip = add(`<div style="position:absolute;left:42px;top:166px;width:13px;height:24px;border-radius:2px"></div>`);
  const stTitle = add(`<div style="position:absolute;left:66px;top:164px;font-size:15px;font-weight:600"></div>`);
  const stSub = add(`<div class="sm" style="position:absolute;left:67px;top:183px"></div>`);
  add(`<div style="position:absolute;left:42px;top:202px;width:254px;height:1px;background:${C.line}"></div>`);
  const sch = add(`<svg style="position:absolute;left:42px;top:214px" width="254" height="156"></svg>`);

  // ------------------------------------------------------------ 02 flight
  panel(28, 394, 282, 232, 'FLIGHT', '02');
  const fl = add(`<svg style="position:absolute;left:42px;top:430px" width="254" height="70"></svg>`);
  const bars = ['HEIGHT', 'GEAR', 'STROKE'].map((n, i) => {
    const y = 508 + i * 23;
    add(`<div class="sm" style="position:absolute;left:42px;top:${y}px">${n}</div>`);
    add(`<div style="position:absolute;left:97px;top:${y + 4}px;width:134px;height:4px;background:#efe6d8;border-radius:2px"></div>`);
    const f = add(`<div style="position:absolute;left:97px;top:${y + 4}px;height:4px;background:${C.or};border-radius:2px"></div>`);
    const v = add(`<div class="mo" style="position:absolute;right:${1600 - 296}px;top:${y - 2}px;font-size:11px;font-weight:600"></div>`);
    return {f, v};
  });
  add(`<div style="position:absolute;left:42px;top:578px;width:254px;height:1px;background:${C.line}"></div>`);
  add(`<div class="sm" style="position:absolute;left:42px;top:596px">WHEEL</div>`);
  const wheelIcon = add(`<svg style="position:absolute;left:274px;top:590px" width="22" height="22"><g data-r><circle cx="11" cy="11" r="9" fill="none" stroke="${C.blue}" stroke-width="1.5"/><path d="M11 11V2M11 11l7.8 4.5M11 11l-7.8 4.5" stroke="${C.blue}" stroke-width="1.4"/><circle cx="11" cy="11" r="2.4" fill="${C.blue}"/></g></svg>`);

  // ------------------------------------------------------------ 03 system overview
  panel(1313, 126, 258, 210, 'SYSTEM OVERVIEW', '03');
  add(`<div class="mo" style="position:absolute;left:1327px;top:163px;font-size:10.5px;font-weight:700;letter-spacing:1.6px">MAIN LANDING GEAR</div>`);
  add(`<div class="mo" style="position:absolute;left:1327px;top:183px;width:232px;font-size:9.4px;line-height:1.75;color:#5f564b">One hydraulic actuator, a single rack and pinion, and a universal joint at each end of the output shaft. The side tubes spin; they never swing.</div>`);
  add(`<div style="position:absolute;left:1327px;top:276px;width:232px;height:1px;background:${C.line}"></div>`);
  [['1', 'ACTUATOR'], ['2', 'LEGS'], ['180°', 'SWEEP']].forEach(([v, n], i) => {
    const x = 1327 + i * 85;
    if (i) add(`<div style="position:absolute;left:${x - 8}px;top:286px;width:1px;height:38px;background:${C.line}"></div>`);
    add(`<div style="position:absolute;left:${x}px;top:288px;font-size:20px;font-weight:700">${v}</div>`);
    add(`<div class="sm" style="position:absolute;left:${x}px;top:313px">${n}</div>`);
  });

  // ------------------------------------------------------------ 04 sequence, 05 views, 06 display
  const btn = {};
  panel(1313, 353, 258, 82, 'SEQUENCE', '04');
  add(`<div style="position:absolute;left:1327px;top:390px;width:230px;height:30px;border-radius:5px;background:#efe6d8"></div>`);
  btn.takeoff = add(`<div class="bt" style="left:1330px;top:393px;width:112px;height:24px;border:none;background:transparent">TAKEOFF</div>`);
  btn.landing = add(`<div class="bt" style="left:1442px;top:393px;width:112px;height:24px;border:none;background:transparent">LANDING</div>`);
  const btnRow = (keys, labels, x0, y, wmax = 230) => {
    let x = x0;
    let yy = y;
    for (const k of keys) {
      const w = Math.round(labels[k].length * 7.1 + 22);
      if (x + w > x0 + wmax) {
        x = x0;
        yy += 31;
      }
      btn[k] = add(`<div class="bt" style="left:${x}px;top:${yy}px;width:${w}px;height:23px">${labels[k]}</div>`);
      x += w + 6;
    }
  };
  panel(1313, 450, 258, 107, 'VIEWS', '05');
  btnRow(VIEWS, VIEW_LABEL, 1327, 487);
  panel(1313, 572, 258, 138, 'DISPLAY', '06');
  btnRow(DISPLAY, DISPLAY_LABEL, 1327, 609);

  // ------------------------------------------------------------ 07 gear operation
  panel(430, 752, 740, 108, 'GEAR OPERATION', '07');
  btn.play = add(`<div class="bt" style="left:445px;top:795px;width:44px;height:44px;border-radius:50%;border:none;background:${C.or};color:#fff;font-size:15px"></div>`);
  const opFrom = add(`<div class="sm" style="position:absolute;left:506px;top:791px"></div>`);
  const opPhase = add(`<div class="mo" style="position:absolute;left:506px;width:500px;top:790px;text-align:center;font-size:9.6px;font-weight:700;letter-spacing:1.6px;color:${C.or}"></div>`);
  const opTo = add(`<div class="sm" style="position:absolute;right:${1600 - 1010}px;top:791px"></div>`);
  const TL = {x: 508, w: 504};
  add(`<div style="position:absolute;left:${TL.x}px;top:820px;width:${TL.w}px;height:3px;background:#efe6d8;border-radius:2px"></div>`);
  const tlFill = add(`<div style="position:absolute;left:${TL.x}px;top:820px;height:3px;background:${C.or};border-radius:2px"></div>`);
  const tlKnob = add(`<div style="position:absolute;top:814px;width:15px;height:15px;margin-left:-7.5px;border-radius:50%;background:#fff;border:2px solid ${C.or};box-sizing:border-box"></div>`);
  const tlTime = add(`<div class="mo" style="position:absolute;left:506px;top:834px;font-size:9.6px;color:${C.mute};letter-spacing:1px"></div>`);
  add(`<div class="sm" style="position:absolute;left:1029px;top:805px">SPEED</div>`);
  const SP = {x: 1031, w: 92};
  add(`<div style="position:absolute;left:${SP.x}px;top:823px;width:${SP.w}px;height:3px;background:#efe6d8;border-radius:2px"></div>`);
  const spFill = add(`<div style="position:absolute;left:${SP.x}px;top:823px;height:3px;background:${C.or};border-radius:2px"></div>`);
  const spKnob = add(`<div style="position:absolute;top:817px;width:15px;height:15px;margin-left:-7.5px;border-radius:50%;background:#fff;border:2px solid ${C.or};box-sizing:border-box"></div>`);
  const spVal = add(`<div class="mo" style="position:absolute;left:1128px;top:818px;font-size:10px;font-weight:700"></div>`);

  // ------------------------------------------------------------ footer and floor notes
  add(`<div class="sm" style="position:absolute;left:27px;top:882px;letter-spacing:2.2px">MECHANISM STUDY &nbsp;·&nbsp; PROCEDURAL MODEL &nbsp;·&nbsp; SINGLE DEGREE OF FREEDOM</div>`);
  add(`<div class="sm" style="position:absolute;right:28px;top:882px;letter-spacing:2.2px">BUILT FROM THE TRACED PLATE <span style="color:${C.or}">//////</span></div>`);
  const notes = add(`<div style="position:absolute;inset:0"></div>`);
  notes.innerHTML = `<div class="sm" style="position:absolute;left:367px;top:658px;line-height:16px;letter-spacing:2px">RIGHT MAIN GEAR<br>SHOWN (TYPICAL)</div>
    <svg style="position:absolute;left:752px;top:658px" width="48" height="10"><path d="M0 5h40" stroke="${C.mute}" stroke-width="1.2"/><path d="M40 1l7 4-7 4z" fill="${C.mute}"/></svg>
    <div class="sm" style="position:absolute;left:768px;top:677px">FWD</div>
    <div class="sm" style="position:absolute;left:912px;top:682px;line-height:16px;letter-spacing:2px"><span style="color:${C.mute}">//////</span><br>MAIN LANDING GEAR<br>REV B</div>
    <div style="position:absolute;left:1046px;top:690px;font-size:28px;font-weight:700;color:rgba(155,143,128,.25)">337</div>
    <svg style="position:absolute;left:398px;top:162px" width="90" height="80"><path d="M0 80V0h80" fill="none" stroke="${C.or}" stroke-width="1" stroke-dasharray="3 3" opacity=".6"/></svg>
    <svg style="position:absolute;left:1088px;top:606px" width="80" height="80"><path d="M80 0v76H0" fill="none" stroke="${C.or}" stroke-width="1" stroke-dasharray="3 3" opacity=".6"/></svg>`;

  // ------------------------------------------------------------ labels, the pointer
  const callSvg = add(`<svg data-callouts style="position:absolute;inset:0;overflow:visible" width="1600" height="900"></svg>`);
  const callG = add(`<div data-callouts style="position:absolute;inset:0"></div>`);
  const ring = add(`<div class="ring"></div>`);
  const cur = add(`<div class="cur"><svg width="22" height="22" viewBox="0 0 34 34"><path d="M4 3 L30 13.5 L17.5 17.5 L13.5 30 Z" fill="#111" stroke="#fff" stroke-width="2.2" stroke-linejoin="round"/></svg></div>`);
  const centre = (n) => {
    const r = n.getBoundingClientRect();
    return [r.left + r.width / 2, r.top + r.height / 2];
  };
  const spots = Object.fromEntries(Object.entries(btn).map(([k, n]) => [k, centre(n)]));
  const pointer = (c) => {
    cur.style.display = c.x < 0 ? 'none' : 'block';
    cur.style.transform = `translate(${(c.x - 3).toFixed(2)}px, ${(c.y - 2).toFixed(2)}px) scale(${c.press ? 0.88 : 1})`;
    ring.style.left = `${c.ringX}px`;
    ring.style.top = `${c.ringY}px`;
    ring.style.opacity = String(c.ring * 0.85);
    ring.style.transform = `scale(${(0.35 + 0.9 * (1 - c.ring)).toFixed(3)})`;
  };
  const fmt = (s) => `00:${String(Math.floor(s)).padStart(2, '0')}`;

  return {
    spots,
    btn,
    root: ui,
    timeline: (k) => [TL.x + k * TL.w, 821],
    speedKnob: (k) => [SP.x + k * SP.w, 824],
    TL,
    SP,
    update(s) {
      // state chip and the schematic: actuator, rack, pinion, shaft, the two legs from the front
      const [title, sub, kind] = s.state;
      chip.style.background = kind === 'ok' ? C.green : C.or;
      stTitle.textContent = title;
      stSub.textContent = sub;
      {
        const d = s.stroke;
        const rx = 92 - d * 52;
        const th = s.stroke * Math.PI;
        let h = `<text x="2" y="10" font-size="7.4" letter-spacing="1" fill="${C.mute}" font-family="JetBrains Mono">ACTUATOR</text>`;
        h += `<rect x="2" y="18" width="${rx - 8}" height="17" rx="8" fill="#fbf8f2" stroke="${C.ink}" stroke-width="1.2"/>`;
        h += `<line x1="${rx - 8}" x2="${rx + 2}" y1="26.5" y2="26.5" stroke="${C.ink}" stroke-width="1.6"/><rect x="${rx}" y="14" width="7" height="27" fill="${C.or}"/>`;
        h += `<line x1="0" x2="250" y1="36.5" y2="36.5" stroke="${C.ink}" stroke-width="1"/>`;
        h += `<rect x="${rx - 70}" y="40" width="96" height="7" fill="#fbf8f2" stroke="${C.ink}" stroke-width="1"/>`;
        for (let i = 0; i < 16; i++) h += `<line x1="${rx - 68 + i * 6}" x2="${rx - 68 + i * 6}" y1="47" y2="51" stroke="${C.ink}" stroke-width=".9"/>`;
        h += `<text x="24" y="72" font-size="7.4" letter-spacing="1" fill="${C.mute}" font-family="JetBrains Mono">RACK</text><text x="134" y="76" font-size="7.4" letter-spacing="1" fill="${C.mute}" font-family="JetBrains Mono">PINION</text>`;
        h += `<g transform="translate(108 70) rotate(${((th * 180) / Math.PI).toFixed(1)})"><circle r="17" fill="none" stroke="${C.ink}" stroke-width="1.1"/>${Array.from({length: 16}, (_, i) => `<line x1="0" y1="-17" x2="0" y2="-20.5" stroke="${C.ink}" stroke-width="1" transform="rotate(${i * 22.5})"/>`).join('')}<path d="M-12 -12L12 12M12 -12L-12 12" stroke="${C.ink}" stroke-width="1"/></g>`;
        h += `<text x="96" y="104" font-size="7.4" letter-spacing="1" fill="${C.mute}" font-family="JetBrains Mono">SHAFT</text><text x="220" y="100" font-size="7.4" letter-spacing="1" fill="${C.mute}" font-family="JetBrains Mono">LEG</text>`;
        h += `<line x1="108" x2="108" y1="88" y2="150" stroke="${C.mute}" stroke-width=".8" stroke-dasharray="3 3"/>`;
        // the legs from the front: a stub out, then the leg down, folding up as the tube turns
        const leg = (sx) => {
          const x0 = 108 + sx * 46;
          const y0 = 122;
          const xa = x0 + sx * 44;
          const a = th;
          const lx = xa + sx * 6 * Math.cos(a);
          const ly = y0 + 34 * Math.cos(a);
          const lx2 = xa - sx * 30 * Math.sin(a) * 0.9;
          const X = lx * (1 - Math.sin(a)) + lx2 * Math.sin(a);
          return `<path d="M${x0} ${y0}H${xa}L${X.toFixed(1)} ${ly.toFixed(1)}" fill="none" stroke="${C.or}" stroke-width="2.6" stroke-linejoin="round"/><circle cx="${X.toFixed(1)}" cy="${ly.toFixed(1)}" r="5.5" fill="${C.or}"/><circle cx="${x0}" cy="${y0}" r="4" fill="${C.or}"/>`;
        };
        h += `<line x1="62" x2="154" y1="122" y2="122" stroke="${C.ink}" stroke-width="1.4"/>` + leg(1) + leg(-1);
        sch.innerHTML = h;
      }
      // flight: the climb or the descent, the marker where the aircraft is
      {
        const W = 254;
        const X = (u) => 20 + u * (W - 30);
        const Y = (h) => 58 - (h / 1.7) * 46;
        let p = '';
        for (let i = 0; i <= 40; i++) {
          const u = i / 40;
          const h = s.seq === 'takeoff' ? 1.7 * Math.pow(Math.max(0, (u - 0.38) / 0.62), 1.6) : 1.7 * Math.pow(Math.max(0, 1 - u / 0.74), 1.6);
          p += `${i ? 'L' : 'M'}${X(u).toFixed(1)} ${Y(h).toFixed(1)}`;
        }
        const u = s.prog;
        const mh = s.height;
        const dir = s.seq === 'takeoff' ? 1 : -1;
        const ang = (Math.atan2(-(s.climb ?? 0) * 18, 1) * 180) / Math.PI;
        fl.innerHTML = `<line x1="0" x2="${W}" y1="58" y2="58" stroke="${C.ink}" stroke-width="1"/><path d="${p}" fill="none" stroke="${C.or}" stroke-width="1.1" stroke-dasharray="3 3"/><g transform="translate(${X(u).toFixed(1)} ${Y(mh).toFixed(1)}) scale(${dir} 1) rotate(${ang.toFixed(1)})"><path d="M-10 -6L8 0L-10 6L-6 0z" fill="${C.or}"/></g>`;
        const vals = [[s.height / 1.7, `${s.height.toFixed(2)} m`], [s.stroke, `${Math.round(s.stroke * 180)}°`], [s.stroke, `${Math.round(s.stroke * 201)} mm`]];
        bars.forEach((b, i) => {
          b.f.style.width = `${(Math.min(1, vals[i][0]) * 134).toFixed(1)}px`;
          b.v.textContent = vals[i][1];
        });
        wheelIcon.querySelector('[data-r]').setAttribute('transform', `rotate(${((s.roll * 180) / Math.PI) % 360} 11 11)`);
      }
      // sequence, views, display
      const onBtn = (k, on) => {
        btn[k].style.background = on ? C.or : '#fffdf9';
        btn[k].style.color = on ? '#fff' : C.ink;
        btn[k].style.borderColor = on ? C.or : C.faint;
      };
      for (const k of ['takeoff', 'landing']) {
        const on = s.seq === k;
        btn[k].style.background = on ? C.or : 'transparent';
        btn[k].style.color = on ? '#fff' : '#8d8274';
        btn[k].style.borderRadius = '4px';
      }
      for (const k of VIEWS) onBtn(k, s.view === k);
      for (const k of DISPLAY) onBtn(k, !!s.display[k]);
      // the operation timeline
      btn.play.innerHTML = s.paused ? '▶' : '❚❚';
      opFrom.textContent = s.from;
      opTo.textContent = s.to;
      opPhase.textContent = s.phase;
      const kx = TL.x + s.prog * TL.w;
      tlFill.style.width = `${(s.prog * TL.w).toFixed(1)}px`;
      tlKnob.style.left = `${kx.toFixed(1)}px`;
      tlTime.textContent = `${fmt(s.tau)} / ${fmt(s.dur)}`;
      const sk = SP.x + s.speedK * SP.w;
      spFill.style.width = `${(s.speedK * SP.w).toFixed(1)}px`;
      spKnob.style.left = `${sk.toFixed(1)}px`;
      spVal.textContent = `${s.speed.toFixed(1)}×`;
      // labels: an orange square, the name, the note, a thin leader to the part
      let lh = '';
      let ch = '';
      for (const c of s.callouts) {
        if (c.o < 0.02 || !c.at) continue;
        const [ax, ay] = c.at;
        const [lx, ly] = c.label;
        const right = lx > ax;
        const ex = right ? lx - 6 : lx + c.w + 6;
        lh += `<polyline points="${ax.toFixed(1)},${ay.toFixed(1)} ${ex.toFixed(1)},${(ly + 6).toFixed(1)}" fill="none" stroke="${C.or}" stroke-width=".9" opacity="${c.o.toFixed(2)}"/><circle cx="${ax.toFixed(1)}" cy="${ay.toFixed(1)}" r="2.4" fill="${C.or}" opacity="${c.o.toFixed(2)}"/>`;
        ch += `<div class="call" style="left:${lx.toFixed(1)}px;top:${(ly - 1).toFixed(1)}px;opacity:${c.o.toFixed(2)}"><b><i></i>${c.name}</b><span>${c.sub}</span></div>`;
      }
      callSvg.innerHTML = lh;
      callG.innerHTML = ch;
      pointer(s.cursor);
    },
    pointer,
  };
}
