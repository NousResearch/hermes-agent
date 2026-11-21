// Small pure helpers shared by every sim. Everything animated must be a pure
// function of `t` (seconds) so live playback and frame capture match.

export const clamp01 = (x) => Math.min(1, Math.max(0, x));
export const lerp = (a, b, k) => a + (b - a) * k;
export const smooth = (x) => {
  const k = clamp01(x);
  return k * k * (3 - 2 * k);
};
export const easeOutCubic = (x) => 1 - Math.pow(1 - clamp01(x), 3);
export const easeInOutCubic = (x) => {
  const k = clamp01(x);
  return k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
};
export const easeOutExpo = (x) => (x >= 1 ? 1 : 1 - Math.pow(2, -10 * clamp01(x)));
export const easeOutBack = (x, s = 1.7) => {
  const k = clamp01(x) - 1;
  return 1 + (s + 1) * k * k * k + s * k * k;
};
/** 0..1 progress of t between t0 and t1, eased. */
export const prog = (t, t0, t1, ease = easeInOutCubic) => ease((t - t0) / (t1 - t0));

/** Seeded PRNG (mulberry32). Same seed → same sequence. */
export const rng = (seed) => {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let x = s;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
};

/** Stateless hash noise in [0,1). */
export const hash = (n) => {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/** Random unit vector from a PRNG. */
export const randDir = (r) => {
  const u = r() * 2 - 1;
  const a = r() * Math.PI * 2;
  const s = Math.sqrt(1 - u * u);
  return [s * Math.cos(a), u, s * Math.sin(a)];
};

/**
 * Keyframed value: keys = [[t, value], ...] with numbers or arrays of numbers.
 * Eased between keys, held outside.
 */
export const keyed = (keys, t, ease = easeInOutCubic) => {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 0; i < keys.length - 1; i++) {
    const [t0, v0] = keys[i];
    const [t1, v1] = keys[i + 1];
    if (t <= t1) {
      const k = ease((t - t0) / (t1 - t0));
      return Array.isArray(v0) ? v0.map((v, j) => lerp(v, v1[j], k)) : lerp(v0, v1, k);
    }
  }
  return keys[keys.length - 1][1];
};

export const fmt = (n, d = 0) =>
  Number(n).toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d});
export const pad = (n, w) => String(Math.max(0, Math.floor(n))).padStart(w, '0');

/** Deep merge plain objects; arrays and scalars from `b` replace `a`. */
export const merge = (a, b) => {
  if (!b || typeof b !== 'object' || Array.isArray(b)) return b === undefined ? a : b;
  const out = {...a};
  for (const k of Object.keys(b)) {
    out[k] =
      a && typeof a[k] === 'object' && !Array.isArray(a[k]) && a[k] !== null ? merge(a[k], b[k]) : b[k];
  }
  return out;
};
