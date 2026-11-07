import {clamp01, easeOutExpo, fmt, hash, smooth} from './util.js';

/**
 * The director edits a sim to the music: hard cuts between wide and tight
 * shots on bar lines, punch-ins on kicks, a shake and flash on the drop,
 * glitches on stage changes, and big readable story captions on the beat.
 *
 * Beats come from cfg.beats (sims/beats.py output, injected by capture.mjs).
 * Without music it runs on a synthetic grid (cfg.director.bpm) so live pages
 * still feel edited.
 *
 * Shots crop the 3D render (crisp re-render of a sub-rectangle) when the sim
 * uses kit/stage3d; otherwise they scale the target element with CSS.
 * Sims can return `foci: [{x, y}]` (screen px) in their render info: tight
 * shots frame those points.
 *
 * Story: cfg.story = [{at | bar | drop, text, sub, kind: 'hero'|'num', hold,
 * from, to, prefix, suffix, tag}]. `drop` is in bars relative to the drop,
 * `bar` counts bars from the clip start, `hold` is in beats. *word* gets the
 * accent box.
 */
export const DIRECTOR = {
  enabled: true,
  bpm: 128,
  drop: null,
  hero: null,
  // cut every N beats before / after the drop
  cutBeats: [4, 2],
  shots: [
    {z: 1},
    {z: 1.75, f: 0},
    {z: 1.3, f: 1},
    {z: 2.2, f: 2},
    {z: 1.12, f: 0},
    {z: 1.6, f: 1},
  ],
  punch: [0.03, 0.07],
  shake: 1,
  flash: 1,
  glitch: 1,
  target: '#gl',
};

const before = (arr, t) => {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] <= t) lo = m + 1;
    else hi = m;
  }
  return lo - 1;
};

export function createDirector(app, cfg) {
  const d = {...DIRECTOR, ...(cfg.director || {})};
  const D = cfg.duration;
  const music = cfg.beats && cfg.beats.beats && cfg.beats.beats.length > 4;
  const spb = music ? 60 / cfg.beats.bpm : 60 / d.bpm;
  const beats = music ? cfg.beats.beats : Array.from({length: Math.ceil(D / spb) + 8}, (_, i) => i * spb);
  // pad the grid past both ends so lookups never fall off
  while (beats[0] > 0) beats.unshift(beats[0] - spb);
  while (beats[beats.length - 1] < D + 4 * spb) beats.push(beats[beats.length - 1] + spb);
  let downs = music && cfg.beats.downbeats.length ? cfg.beats.downbeats.slice() : beats.filter((_, i) => i % 4 === 0);
  while (downs[0] > 0) downs.unshift(downs[0] - 4 * spb);
  while (downs[downs.length - 1] < D + 8 * spb) downs.push(downs[downs.length - 1] + 4 * spb);
  const kicks = music ? cfg.beats.kicks : beats.map((b) => [b, 0.8]);
  const kickT = kicks.map((k) => k[0]);
  const energy = music ? cfg.beats.barEnergy : null;
  const drop = (music && cfg.beats.drop) ?? d.drop ?? Math.round((0.4 * D) / (4 * spb)) * 4 * spb;
  const dropBar = before(downs, drop + 0.01);

  const barTime = (b) => downs[Math.max(0, Math.min(downs.length - 1, b))];

  // ---- time warp: the sim's hero moment lands on the drop
  const warp = (t) => {
    if (d.hero == null || !d.enabled) return t;
    const h = d.hero;
    if (t <= drop) return (t / drop) * h;
    return h + ((t - drop) / (D - drop)) * (D - h);
  };

  const beat = (t) => {
    const sh = shotAt(t);
    const bi = before(beats, t);
    const b0 = beats[Math.max(0, bi)];
    const b1 = beats[bi + 1] ?? b0 + spb;
    const ri = before(downs, t);
    const r0 = downs[Math.max(0, ri)];
    const r1 = downs[ri + 1] ?? r0 + 4 * spb;
    const ki = before(kickT, t);
    const kick = ki >= 0 ? kicks[ki][1] * Math.exp(-(t - kicks[ki][0]) / 0.11) : 0;
    const down = Math.exp(-(t - r0) / 0.18);
    return {
      music,
      bpm: 60 / spb,
      beat: bi,
      beatPhase: clamp01((t - b0) / (b1 - b0)),
      bar: ri,
      barPhase: clamp01((t - r0) / (r1 - r0)),
      kick,
      down,
      energy: energy ? energy[Math.max(0, Math.min(energy.length - 1, ri))] ?? 1 : 1,
      drop,
      sinceDrop: t - drop,
      isDrop: t >= drop,
      // current shot: sims may change camera angle per shot index
      shot: sh.k,
      shotZ: sh.shot.z || 1,
    };
  };

  // ---- story timing
  const story = (cfg.story || []).map((s, i) => {
    const start = s.at ?? (s.bar != null ? barTime(s.bar) : s.drop != null ? barTime(dropBar + s.drop) : 0);
    const b0 = before(beats, start + 0.02);
    const hold = s.hold ?? 8;
    const end = s.end ?? beats[Math.min(beats.length - 1, b0 + hold)] ?? start + hold * spb;
    return {...s, i, start, end, b0};
  });

  // ---- DOM layers
  app.insertAdjacentHTML(
    'beforeend',
    `<canvas class="dir-glitch"></canvas><div class="dir-flash"></div><div class="dir-story"></div>`,
  );
  const gcv = app.querySelector('.dir-glitch');
  const gctx = gcv.getContext('2d');
  const flash = app.querySelector('.dir-flash');
  const storyEl = app.querySelector('.dir-story');

  let foci = [];
  let crop = {z: 1, x0: 0, y0: 0};
  let built = -1;
  let parts = null;

  const stageCuts = (cfg.stages || []).map((s) => s.at).filter((a) => a > 0.2);

  const dropBeat = before(beats, drop + 0.01);
  function shotAt(t) {
    const bi = before(beats, t);
    const isDrop = t >= drop;
    const every = isDrop ? d.cutBeats[1] : d.cutBeats[0];
    // the first beats of the drop always open wide
    if (isDrop && bi - dropBeat < every) return {shot: d.shots[0], k: 0};
    const rel = isDrop ? bi - dropBeat : bi - before(beats, 0.01);
    const k = Math.floor(rel / every) + (isDrop ? 1000 : 0);
    return {shot: d.shots[((k % d.shots.length) + d.shots.length) % d.shots.length], k};
  }

  const target = () => document.querySelector(d.target);

  return {
    warp,
    beat,
    /** Before the sim renders: set this frame's crop. */
    pre(t) {
      if (!d.enabled) return;
      const {shot} = shotAt(t);
      const b = beat(t);
      const punch = 1 + (b.isDrop ? d.punch[1] : d.punch[0]) * b.kick;
      const drift = 1 + 0.035 * b.barPhase;
      let z = (shot.z || 1) * punch * (shot.z > 1 ? drift : 1);
      const vw = innerWidth;
      const vh = innerHeight;
      let f = {x: vw / 2, y: vh / 2};
      if (shot.z > 1 && foci.length) f = foci[(shot.f ?? 0) % foci.length];
      else if (window.__STAGE) {
        const vp = window.__STAGE.hudViewport();
        f = {x: vw / 2, y: (vp.top + vh - vp.bottom) / 2};
      }
      // drop shake (decays over ~0.6 s)
      const sk = d.shake * (b.sinceDrop >= 0 ? Math.exp(-b.sinceDrop / 0.25) : 0);
      const sx = sk * (hash(t * 91.7) - 0.5) * 26;
      const sy = sk * (hash(t * 53.3 + 7) - 0.5) * 26;
      const st = window.__STAGE;
      if (st) {
        crop = st.setCrop(z, f.x + sx, f.y + sy);
      } else {
        const el = target();
        if (el) {
          el.style.transformOrigin = `${f.x}px ${f.y}px`;
          el.style.transform = `translate(${sx}px, ${sy}px) scale(${z})`;
        }
        crop = {z: 1, x0: 0, y0: 0};
      }
    },
    /** After the sim renders: remember its foci, draw overlays and the story. */
    post(t, info = {}) {
      if (!d.enabled) return;
      if (info.foci && info.foci.length) {
        // convert from this frame's cropped screen back to base screen coords
        foci = info.foci.map((p) => ({x: crop.x0 + p.x / crop.z, y: crop.y0 + p.y / crop.z}));
      }
      const b = beat(t);
      const vw = innerWidth;
      const vh = innerHeight;

      // flash: accent bloom on downbeats after the drop, white hit on the drop
      const dropHit = b.sinceDrop >= 0 ? Math.exp(-b.sinceDrop / 0.22) : 0;
      const onBar = b.isDrop ? 0.16 * b.down * b.energy : 0;
      flash.style.opacity = String(d.flash * Math.min(0.85, dropHit * 0.7 + onBar));

      // glitch: 3 frames around each stage change and the drop
      const gl = [...stageCuts, b.drop].some((c) => t >= c - 0.03 && t < c + 0.1);
      if (gcv.width !== vw || gcv.height !== vh) {
        gcv.width = vw;
        gcv.height = vh;
      }
      gctx.clearRect(0, 0, vw, vh);
      const src = document.getElementById('gl');
      if (gl && d.glitch && src && src.width) {
        const n = 7;
        for (let i = 0; i < n; i++) {
          const y = Math.floor(hash(t * 13 + i) * vh);
          const h = 6 + Math.floor(hash(t * 7 + i * 3) * 60);
          const dx = (hash(t * 31 + i * 5) - 0.5) * 120;
          const sx = src.width / vw;
          gctx.globalAlpha = 0.9;
          gctx.drawImage(src, 0, y * sx, src.width, h * sx, dx, y, vw, h);
        }
        gctx.globalAlpha = 0.25;
        gctx.fillStyle = cfg.accent;
        gctx.fillRect(0, Math.floor(hash(t * 3.3) * vh), vw, 3 + hash(t) * 10);
        gctx.globalAlpha = 1;
      }

      // ---- story
      const cur = story.find((s) => t >= s.start && t < s.end);
      if (!cur) {
        storyEl.style.opacity = '0';
        return;
      }
      if (built !== cur.i) {
        built = cur.i;
        const words = (cur.text || '').split(/\s+/).filter(Boolean);
        storyEl.innerHTML = `<div class="cap ${cur.kind || 'hero'}">
          ${cur.tag ? `<div class="cap-tag">${cur.tag}</div>` : ''}
          <div class="cap-main">${
            cur.kind === 'num'
              ? `<span class="w"><span class="n"></span></span>`
              : (() => {
                  // *several words* → each word gets the accent box
                  let on = false;
                  return words
                    .map((w) => {
                      const open = w.startsWith('*');
                      const close = w.endsWith('*') && (w.length > 1 || on);
                      if (open) on = true;
                      const hl = on;
                      if (close) on = false;
                      return `<span class="w${hl ? ' hl' : ''}">${w.replace(/^\*|\*$/g, '')}</span>`;
                    })
                    .join(' ');
                })()
          }</div>
          ${cur.sub ? `<div class="cap-sub"></div>` : ''}
        </div>`;
        parts = {
          words: [...storyEl.querySelectorAll('.w')],
          num: storyEl.querySelector('.n'),
          sub: storyEl.querySelector('.cap-sub'),
          main: storyEl.querySelector('.cap-main'),
        };
      }
      const vp = window.__HUDVP ? window.__HUDVP() : {top: 0, bottom: 0};
      storyEl.style.top = `${vp.top}px`;
      storyEl.style.bottom = `${vp.bottom}px`;
      storyEl.style.opacity = '1';
      const ex = clamp01((t - (cur.end - 0.18)) / 0.18);
      // words slam in on successive beats (half-beats if the text is long)
      const step = parts.words.length > 4 ? 0.5 : 1;
      let lastIn = cur.start;
      parts.words.forEach((w, i) => {
        const bi = cur.b0 + i * step;
        const wt = Math.floor(bi) === bi ? beats[bi] : (beats[Math.floor(bi)] + beats[Math.ceil(bi)]) / 2;
        const k = cur.start <= 0.001 ? 1 : clamp01((t - wt) / 0.12);
        lastIn = Math.max(lastIn, wt);
        const s = 1 + 0.35 * (1 - easeOutExpo(k)) + 0.03 * b.kick;
        w.style.opacity = String(k > 0 ? 1 : 0);
        w.style.transform = `scale(${s}) translateY(${ex * -0.25}em)`;
        w.style.filter = k < 1 ? `blur(${(1 - k) * 8}px)` : 'none';
      });
      if (cur.kind === 'num' && parts.num) {
        const roll = cur.roll ?? 6;
        const k = easeOutExpo(clamp01((t - cur.start) / (roll * spb)));
        const v = cur.from + (cur.to - cur.from) * k;
        parts.num.textContent = `${cur.prefix ?? ''}${fmt(v, cur.decimals ?? 0)}${cur.suffix ?? ''}`;
        lastIn = cur.start + roll * spb * 0.6;
      }
      if (parts.sub) {
        const n = Math.floor(Math.max(0, t - lastIn - 0.15) / 0.022);
        parts.sub.textContent = cur.sub.slice(0, n);
      }
      parts.main.style.opacity = String(1 - ex);
      if (parts.sub) parts.sub.style.opacity = String(1 - ex);
      storyEl.firstElementChild.style.transform = `translateX(${ex * (hash(t * 41) - 0.5) * 40}px)`;
    },
  };
}
