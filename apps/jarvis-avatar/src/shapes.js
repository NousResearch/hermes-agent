/* Formas que as partículas podem assumir. Alvos em px da imagem; z = frente/trás (só afeta o brilho). */
import { IMG, FONT } from './config.js';
import { rnd, gauss } from './util.js';

export function buildShapes(N, homeA) {
  const SHAPES = { head: new Float32Array(N * 3), sphere: new Float32Array(N * 3), galaxy: new Float32Array(N * 3), text: null };
  const ct2 = Math.cos(0.5), st2 = Math.sin(0.5);       // inclinação do disco da galáxia
  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    SHAPES.head[i3] = homeA[i * 2]; SHAPES.head[i3 + 1] = homeA[i * 2 + 1];
    const phi = rnd() * 6.283185307, ct = 1 - 2 * rnd(), st = Math.sqrt(1 - ct * ct);
    const r = rnd() < 0.72 ? 270 + gauss() * 4 : 270 * Math.cbrt(rnd());
    SHAPES.sphere[i3] = IMG.axisX + st * Math.cos(phi) * r;
    SHAPES.sphere[i3 + 1] = 420 - ct * r;
    SHAPES.sphere[i3 + 2] = (st * Math.sin(phi) * r) / 270;
    let gx, gy, gz;
    if (rnd() < 0.12) { gx = gauss() * 55; gz = gauss() * 55; gy = gauss() * 25; }
    else {
      const arm = (rnd() * 3) | 0, gr = Math.pow(rnd(), 0.6) * 600 + 12;
      const th = (gr / 300) * 2.1 + arm * 2.0944 + gauss() * (0.3 / (0.35 + gr / 500));
      gx = Math.cos(th) * gr; gz = Math.sin(th) * gr; gy = gauss() * 12 * (1.4 - gr / 1700);
    }
    SHAPES.galaxy[i3] = IMG.axisX + gx;
    SHAPES.galaxy[i3 + 1] = 450 + gy * ct2 - gz * st2;
    SHAPES.galaxy[i3 + 2] = (gy * st2 + gz * ct2) / 640;
  }
  return SHAPES;
}

// Desenha o texto num canvas e sorteia, pra cada partícula, um pixel aceso dele. null se nada acendeu.
export function buildText(text, N) {
  const W = 1400, H = 440;
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const c = cv.getContext('2d', { willReadFrequently: true });
  let size = 300;
  c.font = `700 ${size}px ${FONT}`;
  size = Math.max(40, Math.min(300, (size * W * 0.9) / (c.measureText(text).width || 1)));
  c.font = `700 ${size}px ${FONT}`;
  c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillStyle = '#fff';
  c.fillText(text, W / 2, H / 2);
  const data = c.getImageData(0, 0, W, H).data;
  const pts = [];
  for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) if (data[(y * W + x) * 4 + 3] > 140) pts.push(x, y);
  const count = pts.length / 2;
  if (!count) return null;
  const out = new Float32Array(N * 3), s = 1300 / W;
  for (let i = 0; i < N; i++) {
    const j = ((rnd() * count) | 0) * 2;
    out[i * 3] = IMG.axisX + (pts[j] - W / 2 + rnd() * 2) * s;
    out[i * 3 + 1] = 430 + (pts[j + 1] - H / 2 + rnd() * 2) * s;
    out[i * 3 + 2] = 0.3;
  }
  return out;
}
