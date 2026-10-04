export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const mix = (a, b, t) => a + (b - a) * t;
export const ease = (dt, rate) => 1 - Math.exp(-dt * rate);   // fração de aproximação independente do fps

// mulberry32: aleatório com semente fixa, então a nuvem de partículas é a mesma a cada carga
let rngState = 0x9e3779b9;
export function rnd() {
  rngState = (rngState + 0x6d2b79f5) | 0;
  let t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
export function gauss() { let u = 0; while (u === 0) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * rnd()); }

export const $ = (id) => document.getElementById(id);

let toastTimer = 0;
export function toast(msg, ms = 4200) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('on'), ms);
}
