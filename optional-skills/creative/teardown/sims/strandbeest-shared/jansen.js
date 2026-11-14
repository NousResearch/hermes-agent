// Theo Jansen's linkage, solved exactly. Eleven "holy numbers" (the link lengths Jansen evolved in 1990) plus the
// crank: turn the crank and every joint follows by circle intersection, no IK, no springs. The foot traces the
// classic path: a flat bottom for ~155° of crank, then a 22-unit lift back. Units are Jansen's own (≈ cm).
export const HOLY = {a: 38, b: 41.5, c: 39.3, d: 40.1, e: 55.8, f: 39.4, g: 36.7, h: 65.7, i: 49, j: 50, k: 61.9, l: 7.8, m: 15};
const {a, b, c, d, e, f, g, h, i, j, k, l, m} = HOLY;
export const PIVOT = [-a, -l];

// the two points at distance r0 from p0 and r1 from p1; pick chooses one
const cc = (p0, r0, p1, r1, pick) => {
  const dx = p1[0] - p0[0];
  const dy = p1[1] - p0[1];
  const D = Math.hypot(dx, dy);
  const A = (r0 * r0 - r1 * r1 + D * D) / (2 * D);
  const H = Math.sqrt(Math.max(0, r0 * r0 - A * A));
  const mx = p0[0] + (A * dx) / D;
  const my = p0[1] + (A * dy) / D;
  return pick([mx + (H * dy) / D, my - (H * dx) / D], [mx - (H * dy) / D, my + (H * dx) / D]);
};
const hiY = (s, t) => (s[1] > t[1] ? s : t);
const loY = (s, t) => (s[1] < t[1] ? s : t);
const loX = (s, t) => (s[0] < t[0] ? s : t);

/** Every joint of one leg for crank angle th (radians), crank axle at the origin, foot toward -x. */
export function solveLeg(th) {
  const A = [m * Math.cos(th), m * Math.sin(th)];
  const B = cc(A, j, PIVOT, b, hiY);
  const C = cc(A, k, PIVOT, c, loY);
  const D = cc(B, e, PIVOT, d, loX);
  const E = cc(D, f, C, g, loX);
  const F = cc(E, h, C, i, loY);
  return {O: [0, 0], P: PIVOT, A, B, C, D, E, F};
}
/** The mirrored leg on the same crank pin: foot toward +x, pushing the same way. */
export function solveMirror(th) {
  const s = solveLeg(Math.PI - th);
  const out = {};
  for (const key in s) out[key] = [-s[key][0], s[key][1]];
  return out;
}
// the links of one leg as joint pairs, with the holy number each one carries
export const LINKS = [
  ['A', 'B', 'j'],
  ['A', 'C', 'k'],
  ['P', 'B', 'b'],
  ['P', 'C', 'c'],
  ['P', 'D', 'd'],
  ['B', 'D', 'e'],
  ['D', 'E', 'f'],
  ['C', 'E', 'g'],
  ['E', 'F', 'h'],
  ['C', 'F', 'i'],
];
export const JOINTS = ['B', 'C', 'D', 'E', 'P'];

// the foot path sampled once: its floor, its stride, and the stance velocity used to move the body without slip
const N = 720;
export const PATH = Array.from({length: N}, (_, n) => solveLeg((n / N) * Math.PI * 2).F);
export const FOOT_MIN = Math.min(...PATH.map((p) => p[1]));
export const FOOT_MAX = Math.max(...PATH.map((p) => p[1]));
/** How far above its lowest point the foot is at crank angle th (0 = planted). */
export const footLift = (th) => {
  const u = ((((th / (Math.PI * 2)) % 1) + 1) % 1) * N;
  const n = Math.floor(u);
  const p0 = PATH[n % N][1];
  const p1 = PATH[(n + 1) % N][1];
  return p0 + (p1 - p0) * (u - n) - FOOT_MIN;
};
