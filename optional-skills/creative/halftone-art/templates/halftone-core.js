/* ============================================================================
   halftone-core.js — tone-accurate AM halftone engine (v2)
   No dependencies. Any modern browser. MIT.

   Model, same as a printing press:
     1. SEPARATE -> continuous-tone ink coverage per pixel, per ink plate
     2. SCREEN   -> average coverage over a rotated cell
     3. SOLVE    -> dot geometry whose AREA equals coverage x cell²
     4. OVERPRINT-> paint plates at their own angles with multiply blending
   ========================================================================== */

const D2R = Math.PI / 180;
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/* deterministic hash — reproducible renders -------------------------------- */
function hash(x, y, seed) {
  let h = x * 374761393 + y * 668265263 + seed * 144665;
  h = (h ^ (h >>> 13)) * 1274126177;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/* --- tone transfer --------------------------------------------------------- */
export function shapeTone(t, o = {}) {
  const { contrast = 1, brightness = 0, gamma = 1, levels = 0 } = o;
  let v = clamp01(t);
  if (contrast !== 1) v = clamp01((v - 0.5) * contrast + 0.5);
  if (brightness) v = clamp01(v + brightness);
  if (gamma !== 1) v = Math.pow(v, gamma);
  if (levels > 1) v = Math.round(v * (levels - 1)) / (levels - 1);
  return v;
}

/* --- named plate curves (real second-separation behaviour) ----------------- */
export const CURVES = {
  full: (t) => t,
  shadows: (t) => t * t * t,
  midtone: (t) => 4 * t * (1 - t),
  highlights: (t) => Math.pow(1 - t, 2),
  liner: (t) => Math.pow(t, 0.6),
};

/* --- draw a source into a scratch canvas at tone-map resolution ------------ */
function drawScaled(source, mw, mh, o, keepAlpha = false) {
  const { fit = 'cover', focal = [0.5, 0.5] } = o;
  const c = document.createElement('canvas');
  c.width = mw;
  c.height = mh;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  if (!keepAlpha) {
    // opaque paper for tone sampling: a transparent pixel must read as bare stock
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, mw, mh);
  }
  const sw = source.width || source.naturalWidth || mw;
  const sh = source.height || source.naturalHeight || mh;
  if (fit === 'stretch') {
    ctx.drawImage(source, 0, 0, mw, mh);
  } else {
    const zoom = o.zoom || 1;
    const k = (fit === 'contain' ? Math.min(mw / sw, mh / sh) : Math.max(mw / sw, mh / sh)) * zoom;
    ctx.drawImage(source, (mw - sw * k) * focal[0], (mh - sh * k) * focal[1], sw * k, sh * k);
  }
  return ctx.getImageData(0, 0, mw, mh).data;
}

const LUMA = ({ r, g, b }) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/* --- 1/2. one plate: coverage map at half resolution ----------------------- */
/**
 * @param kind 'luma' | 'r' | 'g' | 'b' | 'max' | 'min'
 * @param sourceKind 'image' | 'fn'
 */
export function plateCoverage(source, w, h, o = {}, kind = 'luma', fn = null) {
  const density = o.density || 2;
  const mw = Math.max(1, Math.round(w / density));
  const mh = Math.max(1, Math.round(h / density));
  const tone = new Float32Array(mw * mh);
  if (sourceKindIsFn(source)) {
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        tone[y * mw + x] = clamp01(source((x + 0.5) / mw, (y + 0.5) / mh, x, y));
      }
    }
    return { tone, mw, mh };
  }
  const px = drawScaled(source, mw, mh, o);
  const invert = !!o.invert;
  const hist = o.autoLevels ? new Uint32Array(256) : null;
  const lum = new Float32Array(tone.length);
  for (let i = 0, p = 0; i < tone.length; i++, p += 4) {
    const r = px[p] / 255;
    const g = px[p + 1] / 255;
    const b = px[p + 2] / 255;
    let v;
    if (kind === 'r') v = r;
    else if (kind === 'g') v = g;
    else if (kind === 'b') v = b;
    else if (kind === 'max') v = Math.max(r, g, b);
    else if (kind === 'min') v = Math.min(r, g, b);
    else v = LUMA({ r, g, b });
    lum[i] = v;
    if (hist) hist[Math.min(255, Math.max(0, (v * 255) | 0))]++;
  }

  /* autoLevels: stretch the actual tonal range of the source to 0..1 before
     conversion. Scans and phone photos rarely use the full range; without this
     every midtone collapses into one dot size and the print looks like fog. */
  let lo = 0;
  let hi = 1;
  if (hist) {
    const total = tone.length;
    const cut = Math.max(1, Math.round(total * (o.clipPercent ?? 0.004)));
    let acc = 0;
    for (let i = 0; i < 256; i++) { acc += hist[i]; if (acc >= cut) { lo = i / 255; break; } }
    acc = 0;
    for (let i = 255; i >= 0; i--) { acc += hist[i]; if (acc >= cut) { hi = i / 255; break; } }
    if (o.levelsIn) [lo, hi] = o.levelsIn;
    if (hi - lo < 0.02) { lo = 0; hi = 1; }
  } else if (o.levelsIn) {
    [lo, hi] = o.levelsIn;
  }

  const span = hi - lo || 1;
  for (let i = 0; i < tone.length; i++) {
    let v = clamp01((lum[i] - lo) / span);
    tone[i] = invert ? v : 1 - v; // ink coverage
  }
  return { tone, mw, mh };
}
function sourceKindIsFn(s) {
  return typeof s === 'function';
}

/* --- CMYK separations in one pass ----------------------------------------- */
export function cmykCoverage(source, w, h, o = {}) {
  const density = o.density || 2;
  const mw = Math.max(1, Math.round(w / density));
  const mh = Math.max(1, Math.round(h / density));
  const n = mw * mh;
  const C = new Float32Array(n);
  const M = new Float32Array(n);
  const Y = new Float32Array(n);
  const K = new Float32Array(n);
  if (sourceKindIsFn(source)) {
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        const [c, m, yy, k] = source((x + 0.5) / mw, (y + 0.5) / mh); // fn returns [c,m,y,k]
        const i = y * mw + x;
        C[i] = clamp01(c); M[i] = clamp01(m); Y[i] = clamp01(yy); K[i] = clamp01(k);
      }
    }
    return [{ tone: C, mw, mh }, { tone: M, mw, mh }, { tone: Y, mw, mh }, { tone: K, mw, mh }];
  }
  const px = drawScaled(source, mw, mh, o);
  let lo = 0;
  let hi = 1;
  if (o.autoLevels || o.levelsIn) {
    if (o.levelsIn) {
      [lo, hi] = o.levelsIn;
    } else {
      const hist = new Uint32Array(256);
      for (let p = 0; p < px.length; p += 4) {
        const m = Math.max(px[p], px[p + 1], px[p + 2]);
        hist[m]++;
      }
      const cut = Math.max(1, Math.round(n * (o.clipPercent ?? 0.004)));
      let acc = 0;
      for (let i = 0; i < 256; i++) { acc += hist[i]; if (acc >= cut) { lo = i / 255; break; } }
      acc = 0;
      for (let i = 255; i >= 0; i--) { acc += hist[i]; if (acc >= cut) { hi = i / 255; break; } }
      if (hi - lo < 0.02) { lo = 0; hi = 1; }
    }
  }
  const span = hi - lo || 1;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = clamp01((px[p] / 255 - lo) / span);
    const g = clamp01((px[p + 1] / 255 - lo) / span);
    const b = clamp01((px[p + 2] / 255 - lo) / span);
    const k = 1 - Math.max(r, g, b);
    C[i] = k >= 1 ? 0 : clamp01((1 - r - k) / (1 - k));
    M[i] = k >= 1 ? 0 : clamp01((1 - g - k) / (1 - k));
    Y[i] = k >= 1 ? 0 : clamp01((1 - b - k) / (1 - k));
    K[i] = clamp01(k);
  }
  return [{ tone: C, mw, mh }, { tone: M, mw, mh }, { tone: Y, mw, mh }, { tone: K, mw, mh }];
}

/* --- 2. rotated-grid cell average ----------------------------------------- */
/* Bilinear sampling + taps rotated with the screen angle. Nearest-neighbour
   taps on an axis-aligned footprint beat against the dot grid and produce
   visible low-frequency moiré — this is the single biggest quality factor. */
function bilerp(tone, mw, mh, x, y) {
  if (x < 0) x = 0; else if (x > mw - 1.001) x = mw - 1.001;
  if (y < 0) y = 0; else if (y > mh - 1.001) y = mh - 1.001;
  const x0 = x | 0;
  const y0 = y | 0;
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * mw + x0;
  const a = tone[i];
  const b = tone[i + 1] || a;
  const c = tone[i + mw] || a;
  const d = tone[i + mw + 1] || c;
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

function cellTone(tone, mw, mh, cx, cy, cellPx, angleRad, taps = 4) {
  const reach = cellPx * 0.47;
  const ca = Math.cos(angleRad);
  const sa = Math.sin(angleRad);
  let sum = 0;
  let n = 0;
  for (let a = 0; a < taps; a++) {
    const u = ((a / (taps - 1)) * 2 - 1) * reach;
    for (let b = 0; b < taps; b++) {
      const v = ((b / (taps - 1)) * 2 - 1) * reach;
      sum += bilerp(tone, mw, mh, cx + u * ca - v * sa, cy + u * sa + v * ca);
      n++;
    }
  }
  return sum / n;
}

/* --- 3. dot geometry whose area encodes coverage -------------------------- */
const SOLID_AT = 0.952;   // above this coverage the cell floods (no pinholes)
const VANISH_BELOW = 0.012;

function dotPath(ctx, shape, x, y, cell, t, gain) {
  const area = t * cell * cell * gain;
  if (t <= VANISH_BELOW) return null;
  if (area >= cell * cell * SOLID_AT) { // solid ink — kill the lattice artifacts
    ctx.beginPath();
    ctx.rect(x - cell * 0.5, y - cell * 0.5, cell, cell);
    return 'fill';
  }
  ctx.beginPath();
  switch (shape) {
    case 'square': {
      const s = Math.sqrt(area);
      ctx.rect(x - s / 2, y - s / 2, s, s);
      return 'fill';
    }
    case 'diamond': {
      const d = Math.sqrt(2 * area);
      ctx.moveTo(x, y - d / 2);
      ctx.lineTo(x + d / 2, y);
      ctx.lineTo(x, y + d / 2);
      ctx.lineTo(x - d / 2, y);
      ctx.closePath();
      return 'fill';
    }
    case 'line': {
      const th = Math.max(0.35, area / (cell * 1.12));
      ctx.rect(x - cell * 0.56, y - th / 2, cell * 1.12, th);
      return 'fill';
    }
    case 'cross': {
      const arm = cell * 0.8;
      const th = Math.max(0.35, area / (arm * 1.7));
      ctx.rect(x - arm, y - th / 2, arm * 2, th);
      ctx.rect(x - th / 2, y - arm, th, arm * 2);
      return 'fill';
    }
    case 'ellipse': {
      const r = Math.sqrt(area / Math.PI);
      ctx.ellipse(x, y, r * 1.4, r * 0.68, 0, 0, Math.PI * 2);
      return 'fill';
    }
    case 'ring': {
      const r = Math.sqrt(area / Math.PI) * 0.9;
      ctx.lineWidth = Math.max(0.5, cell * 0.08);
      ctx.arc(x, y, r, 0, Math.PI * 2);
      return 'stroke';
    }
    default: {
      const r = Math.sqrt(area / Math.PI);
      ctx.arc(x, y, r, 0, Math.PI * 2);
      return 'fill';
    }
  }
}

/* --- 4. paint one plate --------------------------------------------------- */
function paintPlate(ctx, w, h, tone, mw, mh, o) {
  const { cell, angle, shape, gain, grain, seed, offset, toneMapScale } = o;
  const rad = angle * D2R;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const cx = w / 2;
  const cy = h / 2;
  const reach = Math.ceil(Math.hypot(w, h) / 2 / cell) + 1;
  const scale = toneMapScale || 1;

  let drawn = 0;
  for (let j = -reach; j <= reach; j++) {
    for (let i = -reach; i <= reach; i++) {
      const gx = i * cell;
      const gy = j * cell;
      let x = cx + gx * cos - gy * sin;
      let y = cy + gx * sin + gy * cos;
      if (offset) { x += offset[0]; y += offset[1]; }
      if (x < -cell || y < -cell || x > w + cell || y > h + cell) continue;

      let t = cellTone(tone, mw, mh, x * scale, y * scale, cell * scale, rad);
      if (grain) t = clamp01(t + (hash(i, j, seed) - 0.5) * grain * 0.28);
      const mode = dotPath(ctx, shape, x, y, cell, t, gain);
      if (!mode) continue;
      if (mode === 'stroke') ctx.stroke(); else ctx.fill();
      drawn++;
    }
  }
  return drawn;
}


/* --- subject mask (cutouts / stickers) ------------------------------------ */
/* No ML and no background service: either the source already carries alpha, or
   the background is flooded inward from the frame edges in RGB space. That is
   enough for studio photos, product shots and cut-outs — which is exactly what
   stickers are made from. */

function parseColour(col) {
  const c = parseColour._c || (parseColour._c = document.createElement('canvas'));
  c.width = c.height = 1;
  const x = c.getContext('2d');
  x.clearRect(0, 0, 1, 1);
  x.fillStyle = col;
  x.fillRect(0, 0, 1, 1);
  const d = x.getImageData(0, 0, 1, 1).data;
  return [d[0], d[1], d[2]];
}

/* Backdrop colour from the four corner patches, per-channel median.
   Averaging the whole border fails as soon as the subject touches an edge —
   the mean lands between two colours and matches neither. Corners are the one
   region that is almost always backdrop in a portrait or product shot. */
function keyColour(px, mw, mh, explicit) {
  if (explicit) return parseColour(explicit);
  const patch = Math.max(2, Math.round(Math.min(mw, mh) * 0.07));
  const s = [[], [], []];
  const addPatch = (x0, y0) => {
    for (let y = y0; y < Math.min(y0 + patch, mh); y++) {
      for (let x = x0; x < Math.min(x0 + patch, mw); x++) {
        const p = (y * mw + x) * 4;
        s[0].push(px[p]); s[1].push(px[p + 1]); s[2].push(px[p + 2]);
      }
    }
  };
  addPatch(0, 0);
  addPatch(mw - patch, 0);
  addPatch(0, mh - patch);
  addPatch(mw - patch, mh - patch);
  const med = (a) => { a.sort((x, y) => x - y); return a[a.length >> 1]; };
  return [med(s[0]), med(s[1]), med(s[2])];
}

/**
 * 1 = subject (kept), 0 = background (dropped).
 * @returns {{mask: Uint8Array, mw: number, mh: number}}
 */
export function buildSubjectMask(source, w, h, o = {}) {
  const density = o.density || 2;
  const mw = Math.max(1, Math.round(w / density));
  const mh = Math.max(1, Math.round(h / density));
  const px = drawScaled(source, mw, mh, o, true); // keep alpha for detection
  const n = mw * mh;
  const mask = new Uint8Array(n);

  // a source that already has transparency needs no guessing
  let hasAlpha = false;
  for (let i = 3; i < px.length; i += 4) {
    if (px[i] < 250) { hasAlpha = true; break; }
  }
  if (hasAlpha || o.cutoutMode === 'alpha') {
    const cut = o.cutoutAlpha ?? 128;
    for (let i = 0; i < n; i++) mask[i] = px[i * 4 + 3] > cut ? 1 : 0;
    return { mask, mw, mh };
  }

  // flood the backdrop inward from every edge pixel, using the corner colour
  const [er, eg, eb] = keyColour(px, mw, mh, o.cutoutKey);
  const tol = (o.cutoutTolerance ?? 0.22) * 441.67;
  const tol2 = tol * tol;
  const isBg = (i) => {
    const p = i * 4;
    const dr = px[p] - er;
    const dg = px[p + 1] - eg;
    const db = px[p + 2] - eb;
    return dr * dr + dg * dg + db * db <= tol2;
  };
  const bg = new Uint8Array(n);
  const stack = [];
  const push = (i) => {
    if (!bg[i] && isBg(i)) { bg[i] = 1; stack.push(i); }
  };
  for (let x = 0; x < mw; x++) { push(x); push((mh - 1) * mw + x); }
  for (let y = 0; y < mh; y++) { push(y * mw); push(y * mw + mw - 1); }
  while (stack.length) {
    const i = stack.pop();
    const x = i % mw;
    const y = (i / mw) | 0;
    if (x > 0) push(i - 1);
    if (x < mw - 1) push(i + 1);
    if (y > 0) push(i - mw);
    if (y < mh - 1) push(i + mw);
  }
  for (let i = 0; i < n; i++) mask[i] = bg[i] ? 0 : 1;

  // cleanup: closing seals leaks through fur/fabric, then optionally drop specks
  let out = closeMask(mask, mw, mh, o.cutoutClose ?? 2);
  if (o.cutoutKeepLargest) out = largestBlob(out, mw, mh);
  return { mask: out, mw, mh };
}

/* Two-pass chamfer distance transform — grows the silhouette for the die-cut
   rim without the blockiness of iterative dilation. */
function dilateMask(mask, mw, mh, rad) {
  if (rad <= 0) return mask;
  const INF = 1e9;
  const d = new Float32Array(mw * mh);
  for (let i = 0; i < d.length; i++) d[i] = mask[i] ? 0 : INF;
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      const i = y * mw + x;
      let v = d[i];
      if (x > 0) v = Math.min(v, d[i - 1] + 1);
      if (y > 0) v = Math.min(v, d[i - mw] + 1);
      if (x > 0 && y > 0) v = Math.min(v, d[i - mw - 1] + 1.4142);
      if (x < mw - 1 && y > 0) v = Math.min(v, d[i - mw + 1] + 1.4142);
      d[i] = v;
    }
  }
  for (let y = mh - 1; y >= 0; y--) {
    for (let x = mw - 1; x >= 0; x--) {
      const i = y * mw + x;
      let v = d[i];
      if (x < mw - 1) v = Math.min(v, d[i + 1] + 1);
      if (y < mh - 1) v = Math.min(v, d[i + mw] + 1);
      if (x < mw - 1 && y < mh - 1) v = Math.min(v, d[i + mw + 1] + 1.4142);
      if (x > 0 && y < mh - 1) v = Math.min(v, d[i + mw - 1] + 1.4142);
      d[i] = v;
    }
  }
  const out = new Uint8Array(mw * mh);
  for (let i = 0; i < out.length; i++) out[i] = d[i] <= rad ? 1 : 0;
  return out;
}

/* Morphological closing: grow, then shrink. Closes the thin channels a colour
   flood leaks through when fur or fabric is close to the backdrop colour. */
function closeMask(mask, mw, mh, r) {
  if (r <= 0) return mask;
  const grown = dilateMask(mask, mw, mh, r);
  const inv = new Uint8Array(grown.length);
  for (let i = 0; i < grown.length; i++) inv[i] = grown[i] ? 0 : 1;
  const grownBg = dilateMask(inv, mw, mh, r);
  const out = new Uint8Array(grown.length);
  for (let i = 0; i < out.length; i++) out[i] = grownBg[i] ? 0 : 1;
  return out;
}

/* Fill enclosed background pockets (eyes, handle holes) so the die-cut rim can
   be computed from a solid silhouette, then subtracted again afterwards. */
function fillHoles(mask, mw, mh) {
  const n = mw * mh;
  const outside = new Uint8Array(n);
  const stack = [];
  const push = (i) => { if (!outside[i] && !mask[i]) { outside[i] = 1; stack.push(i); } };
  for (let x = 0; x < mw; x++) { push(x); push((mh - 1) * mw + x); }
  for (let y = 0; y < mh; y++) { push(y * mw); push(y * mw + mw - 1); }
  while (stack.length) {
    const i = stack.pop();
    const x = i % mw;
    const y = (i / mw) | 0;
    if (x > 0) push(i - 1);
    if (x < mw - 1) push(i + 1);
    if (y > 0) push(i - mw);
    if (y < mh - 1) push(i + mw);
  }
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = mask[i] || !outside[i] ? 1 : 0;
  return out;
}

/* Keep only the largest connected blob — removes the specks a flood leaves
   behind in textured backdrops. */
function largestBlob(mask, mw, mh) {
  const seen = new Uint8Array(mask.length);
  let best = null;
  let bestSize = 0;
  const stack = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    const blob = [];
    stack.length = 0;
    stack.push(start);
    seen[start] = 1;
    while (stack.length) {
      const i = stack.pop();
      blob.push(i);
      const x = i % mw;
      const y = (i / mw) | 0;
      const neighbours = [
        x > 0 ? i - 1 : -1,
        x < mw - 1 ? i + 1 : -1,
        y > 0 ? i - mw : -1,
        y < mh - 1 ? i + mw : -1,
      ];
      for (const j of neighbours) {
        if (j >= 0 && mask[j] && !seen[j]) { seen[j] = 1; stack.push(j); }
      }
    }
    if (blob.length > bestSize) { bestSize = blob.length; best = blob; }
  }
  const out = new Uint8Array(mask.length);
  if (best) for (const i of best) out[i] = 1;
  return out;
}

/* Mask -> smooth canvas at output size. Upscaling with smoothing gives the
   sub-pixel edge that a hard 1-bit mask would otherwise alias away. */
function maskToCanvas(mask, mw, mh, colour, w, h, grow = 0) {
  const m = grow > 0 ? dilateMask(mask, mw, mh, grow) : mask;
  const src = document.createElement('canvas');
  src.width = mw;
  src.height = mh;
  const sx = src.getContext('2d');
  const img = sx.createImageData(mw, mh);
  const rgb = parseColour(colour);
  for (let i = 0; i < m.length; i++) {
    const p = i * 4;
    img.data[p] = rgb[0];
    img.data[p + 1] = rgb[1];
    img.data[p + 2] = rgb[2];
    img.data[p + 3] = m[i] ? 255 : 0;
  }
  sx.putImageData(img, 0, 0);

  const out = document.createElement('canvas');
  out.width = w;
  out.height = h;
  const ox = out.getContext('2d');
  ox.imageSmoothingEnabled = true;
  ox.imageSmoothingQuality = 'high';
  ox.drawImage(src, 0, 0, w, h);
  return out;
}

/* "clear vinyl" mode: drop the flat paper colour, keep the dots and their
   antialiased edges (which are blends, not bars of paper). */
function knockOutColour(ctx, w, h, colour, tol = 12) {
  const img = ctx.getImageData(0, 0, w, h);
  const [r, g, b] = parseColour(colour);
  const t2 = tol * tol * 3;
  const d = img.data;
  for (let p = 0; p < d.length; p += 4) {
    if (d[p + 3] === 0) continue;
    const dr = d[p] - r;
    const dg = d[p + 1] - g;
    const db = d[p + 2] - b;
    if (dr * dr + dg * dg + db * db <= t2) d[p + 3] = 0;
  }
  ctx.putImageData(img, 0, 0);
}

/* --- public: defaults ----------------------------------------------------- */
export const DEFAULTS = {
  cell: 8,
  angle: 45,
  shape: 'round',
  ink: '#141414',
  paper: '#f4f0e6',
  gain: 1,
  contrast: 1,
  brightness: 0,
  gamma: 1,
  levels: 0,
  invert: false,
  grain: 0,
  seed: 1,
  density: 2,
  fit: 'cover',
  focal: [0.5, 0.5],
  zoom: 1,
  autoLevels: false,   // stretch source tones to full range
  levelsIn: null,      // manual [blackPoint, whitePoint] 0..1
  mode: 'mono',        // mono | duotone | cmyk
  inks: null,          // duotone [a,b] / cmyk [c,m,y,k]
  plateAngles: null,   // custom angles per plate
  plateCurves: null,   // 'full'|'shadows'|'midtone'|'highlights'|'liner' per plate
  misregister: 0,      // px shift between plates
  /* --- cutout / sticker --- */
  cutout: false,               // transparent background + subject silhouette
  cutoutMode: 'auto',          // auto | flood | alpha
  cutoutTolerance: 0.22,       // 0..1 colour distance that counts as backdrop
  cutoutAlpha: 128,            // alpha threshold when the source carries alpha
  cutoutFill: 'paper',         // paper (die-cut stock) | none (clear vinyl)
  cutoutOutline: 0,            // px of paper rim around the silhouette
  cutoutOutlineColor: null,    // defaults to paper
  cutoutKeyTolerance: 12,      // 0-255 for cutoutFill: 'none' paper removal
  cutoutClose: 2,              // morphological closing radius (mask px) — seals leaks
  cutoutKeepLargest: false,    // drop detached specks, keep one blob
};

const INK_CMYK = ['#00aeef', '#ec008c', '#ffe600', '#1a1a1a'];
const INK_ROSETTE = [15, 75, 0, 45]; // conventional screen angles

/* --- paint paper + plates into any 2D context ----------------------------- */
function paintHalftone(ctx, w, h, source, o, dpr) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.fillStyle = o.paper;
  ctx.fillRect(0, 0, w, h);

  const cell = Math.max(2, o.cell * dpr);
  const mis = o.misregister * dpr;
  const base = { cell, shape: o.shape, gain: o.gain, grain: o.grain, seed: o.seed };

  // device px -> tone-map px
  const toneMapScale = 1 / (o.density || 2);

  const paintScaled = (plateIdx, coverage, ink, angle, curve) => {
    const { tone, mw, mh } = coverage;
    const fn = CURVES[curve] || CURVES.full;
    for (let i = 0; i < tone.length; i++) tone[i] = shapeTone(fn(tone[i]), o);
    ctx.globalCompositeOperation = plateIdx === 0 ? 'source-over' : 'multiply';
    ctx.fillStyle = ink;
    ctx.strokeStyle = ink;
    const offset = mis && plateIdx > 0
      ? [(hash(plateIdx, 1, 7) - 0.5) * mis * 2, (hash(plateIdx, 2, 7) - 0.5) * mis * 2]
      : null;
    paintPlate(ctx, w, h, tone, mw, mh, { ...base, angle, offset, toneMapScale });
  };

  if (o.mode === 'mono') {
    const cov = plateCoverage(source, w, h, o, o.plate || 'luma');
    paintScaled(0, cov, o.ink, o.angle, 'full');
  } else if (o.mode === 'duotone') {
    const inks = o.inks || ['#0f4bd8', '#ff5f8f'];
    const curves = o.plateCurves || ['full', 'shadows'];
    const angles = o.plateAngles || [o.angle, o.angle + 30];
    const covA = plateCoverage(source, w, h, { ...o, invert: false }, 'luma');
    const covB = plateCoverage(source, w, h, { ...o, invert: !!o.invertB }, 'luma');
    paintScaled(0, covA, inks[0], angles[0], curves[0]);
    paintScaled(1, covB, inks[1] || inks[0], angles[1], curves[1] || 'shadows');
  } else {
    const inks = o.inks || INK_CMYK;
    const angles = o.plateAngles || INK_ROSETTE;
    const curves = o.plateCurves || ['full', 'full', 'full', 'full'];
    const maps = cmykCoverage(source, w, h, o);
    maps.forEach((m, i) => paintScaled(i, m, inks[i], angles[i], curves[i]));
  }

  ctx.globalCompositeOperation = 'source-over';
}

/* --- public: render ------------------------------------------------------- */
export function renderHalftone(canvas, source, userOpts = {}) {
  const o = { ...DEFAULTS, ...userOpts };
  const dpr = o.pixelRatio || Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  // detached/off-screen canvases report a 0-size rect — fall back to their own
  // pixel size so exports and contact sheets render 1:1 instead of zoomed.
  const cssW = o.width || rect.width || canvas.clientWidth || canvas.width || 300;
  const cssH = o.height || rect.height || canvas.clientHeight || canvas.height || 200;
  const w = Math.max(1, Math.round(cssW * dpr));
  const h = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;

  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);

  if (!o.cutout) {
    paintHalftone(ctx, w, h, source, o, dpr);
    return canvas;
  }

  /* ---- cutout / sticker path --------------------------------------------
     The halftone is painted on its own layer first, so multiply overprint
     still resolves against real paper. The layer is then clipped to the
     subject silhouette. The die-cut rim is that same silhouette grown by
     cutoutOutline and painted behind the artwork; everything outside stays
     transparent, so the result drops onto any background. */
  const layer = document.createElement('canvas');
  layer.width = w;
  layer.height = h;
  const lctx = layer.getContext('2d');
  paintHalftone(lctx, w, h, source, o, dpr);

  if (o.cutoutFill === 'none') knockOutColour(lctx, w, h, o.paper, o.cutoutKeyTolerance ?? 12);

  const { mask, mw, mh } = buildSubjectMask(source, w, h, o);

  lctx.globalCompositeOperation = 'destination-in';
  lctx.drawImage(maskToCanvas(mask, mw, mh, '#000', w, h), 0, 0);
  lctx.globalCompositeOperation = 'source-over';

  const rim = (o.cutoutOutline || 0) * dpr;
  if (rim > 0) {
    const growPx = Math.max(1, Math.round(rim / (w / mw)));
    // solid silhouette for the rim, so the bleed bridges small gaps…
    const solid = fillHoles(mask, mw, mh);
    ctx.drawImage(
      maskToCanvas(solid, mw, mh, o.cutoutOutlineColor || o.paper, w, h, growPx),
      0, 0,
    );
    // …then re-punch the enclosed cut-outs (eyes, handles) so they stay see-through
    const holes = new Uint8Array(solid.length);
    let hasHoles = false;
    for (let i = 0; i < solid.length; i++) {
      if (solid[i] && !mask[i]) { holes[i] = 1; hasHoles = true; }
    }
    if (hasHoles) {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.drawImage(maskToCanvas(holes, mw, mh, '#000', w, h, 0), 0, 0);
      ctx.globalCompositeOperation = 'source-over';
    }
  }
  ctx.drawImage(layer, 0, 0);
  return canvas;
}

/* --- procedural field (hero backgrounds, type reveals, animated) ---------- */
export function renderHalftoneField(canvas, fn, userOpts = {}) {
  const o = { ...DEFAULTS, ...userOpts };
  const dpr = o.pixelRatio || Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round((o.width || rect.width || canvas.width || 300) * dpr));
  const h = Math.max(1, Math.round((o.height || rect.height || canvas.height || 200) * dpr));
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = o.paper;
  ctx.fillRect(0, 0, w, h);
  const cell = Math.max(2, o.cell * dpr);
  const rad = o.angle * D2R;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const cx = w / 2;
  const cy = h / 2;
  const reach = Math.ceil(Math.hypot(w, h) / 2 / cell) + 1;
  ctx.fillStyle = ctx.strokeStyle = o.ink;

  for (let j = -reach; j <= reach; j++) {
    for (let i = -reach; i <= reach; i++) {
      const x = cx + i * cell * cos - j * cell * sin;
      const y = cy + i * cell * sin + j * cell * cos;
      if (x < -cell || y < -cell || x > w + cell || y > h + cell) continue;
      let t = shapeTone(clamp01(fn(x / w, y / h, i, j)), o);
      if (o.grain) t = clamp01(t + (hash(i, j, o.seed) - 0.5) * o.grain * 0.28);
      const mode = dotPath(ctx, o.shape, x, y, cell, t, o.gain);
      if (!mode) continue;
      if (mode === 'stroke') ctx.stroke(); else ctx.fill();
    }
  }
  return canvas;
}

/* --- export helper: PNG at N× for downloads / print ---------------------- */
export function halftoneToDataURL(source, opts = {}) {
  const canvas = document.createElement('canvas');
  const w = opts.exportWidth || 1200;
  const ar = (source.height || source.naturalHeight) / (source.width || source.naturalWidth) || 1;
  renderHalftone(canvas, source, { ...opts, width: w, height: Math.round(w * ar), pixelRatio: 1 });
  return canvas.toDataURL('image/png');
}

/* --- presets ------------------------------------------------------------- */
export const PRESETS = {
  newsprint: { cell: 5, angle: 45, shape: 'round', ink: '#171717', paper: '#f2eee2', contrast: 1.12, gain: 0.94, grain: 0.35, mode: 'mono' },
  'riso-duo': { cell: 9, angle: 27, shape: 'round', ink: '#0f4bd8', paper: '#fdf3e3', mode: 'duotone', inks: ['#0f4bd8', '#ff5f8f'], plateAngles: [27, 62], plateCurves: ['full', 'shadows'], misregister: 2.4, contrast: 1.08 },
  pop: { cell: 14, angle: 15, shape: 'round', ink: '#e8175d', paper: '#ffe95c', contrast: 1.35, levels: 4, gain: 1.08 },
  'cmyk-proof': { cell: 7, angle: 0, shape: 'round', paper: '#f8f4ec', mode: 'cmyk', misregister: 0.6 },
  'acid-line': { cell: 10, angle: 38, shape: 'line', ink: '#111111', paper: '#c8ff3d', contrast: 1.2 },
  'quiet-dots': { cell: 12, angle: 45, shape: 'round', ink: '#1d1d1d', paper: '#ffffff', gain: 0.82, contrast: 0.92 },
  'screenprint': { cell: 11, angle: 22, shape: 'round', ink: '#12100e', paper: '#e9dfcc', levels: 3, contrast: 1.4, gain: 1.05 },
};

export default { renderHalftone, renderHalftoneField, plateCoverage, cmykCoverage, buildSubjectMask, halftoneToDataURL, shapeTone, PRESETS, CURVES, DEFAULTS };
