/* Busto 3D do manequim: sólidos feitos de fatias elípticas (uma por linha da imagem), a nuvem de partículas na
   superfície deles e a malha escura que tampa o que fica atrás. Coordenadas em px da imagem; z = pra frente.
     cabeça : sólido rígido (gira inteiro); a mandíbula afina e avança, então o queixo fica na frente do pescoço
     corpo  : pescoço + tronco, atrás da cabeça; torce (1 no alto do pescoço, TWIST nos ombros)
     orelhas: placas presas na lateral do crânio
   Cada ponto leva (x, y) com giro 0, z, normal (nx, nz) e o quanto acompanha o giro; o shader só gira. */
import { IMG, HEAD3D } from './config.js';
import { clamp, rnd } from './util.js';

const H = HEAD3D;
const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };

// Catmull-Rom sobre [y, valor] (fora da faixa: 0)
function curve(P, y) {
  const last = P.length - 1;
  if (y < P[0][0] || y > P[last][0]) return 0;
  let k = 0;
  while (k < last - 1 && y >= P[k + 1][0]) k++;
  const p0 = P[Math.max(0, k - 1)][1], p1 = P[k][1], p2 = P[k + 1][1], p3 = P[Math.min(last, k + 2)][1];
  const t = clamp((y - P[k][0]) / (P[k + 1][0] - P[k][0]), 0, 1), t2 = t * t, t3 = t2 * t;
  return Math.max(0, 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (3 * p1 - p0 - 3 * p2 + p3) * t3));
}

// fatia de cada sólido na altura y: meia-largura w, meia-profundidade d, centro em z, peso do giro
export function headSlice(y) {
  const w = curve(H.HEAD, y), j = smooth(H.JAW[0], H.JAW[1], y);
  return { w, d: w * (H.DEPTH - H.JAW_FLAT * j), z: H.CHIN_FWD * j, k: 1 };
}
export function bodySlice(y) {
  const w = curve(H.BODY, y);
  return { w, d: Math.min(H.NECK_DEPTH * w, H.TORSO_DEPTH), z: H.BODY_Z, k: 1 + (H.TWIST - 1) * smooth(H.TWIST_Y[0], H.TWIST_Y[1], y) };
}
const HEAD_Y = [H.HEAD[0][0], H.HEAD[H.HEAD.length - 1][0]], BODY_Y = [H.BODY[0][0], IMG.h];

// ponto da fatia no ângulo f (0 = de frente): posição (x relativo ao eixo, z) e normal (nx, nz)
function onSlice(S, f) {
  const sf = Math.sin(f), cf = Math.cos(f), nl = Math.hypot(S.d * sf, S.w * cf) || 1;
  return { x: S.w * sf, z: S.z + S.d * cf, nx: (S.d * sf) / nl, nz: (S.w * cf) / nl };
}

// rosto: relevo (px, pra fora) no ponto da frente a u px do eixo, na altura y
function relief(u, y) {
  let h = 0;
  for (const F of H.FACE) {
    for (const sx of F.pair ? [-1, 1] : [1]) h += F.h * Math.exp(-(((u - sx * F.x) / F.rx) ** 2) - (((y - F.y) / F.ry) ** 2));
  }
  const N = H.NOSE, t = clamp((y - N.top) / (N.tip - N.top), 0, 1);
  const nw = N.w * (0.7 + 0.6 * t), under = y > N.tip ? Math.exp(-(((y - N.tip) / 7) ** 2)) : 1;
  return h + N.h * t * t * (3 - 2 * t) * under * Math.exp(-((u / nw) ** 2)) * (y > N.top ? 1 : 0);
}

// Partículas: aura da foto (em volta da figura) + superfície dos sólidos + orelhas.
// Cada uma: x, y, r, g, b, relevo da foto, borda, aura (0/255), z, nx, nz, peso do giro
export const STRIDE = 12;
export function buildParticles(em, inf, keep) {
  const pts = [];
  // 1) foto: só a aura (a figura agora é 3D)
  for (let y = 0, px = 0; y < IMG.h; y++) {
    for (let x = 0; x < IMG.w; x++, px += 4) {
      if (inf[px + 2] <= 200 || rnd() > keep) continue;
      pts.push(x + 0.5, y + 0.5, em[px], em[px + 1], em[px + 2], inf[px], inf[px + 1], 255, -400, 0, 1, y < 700 ? 1 : H.TWIST);
    }
  }
  const dens = H.DENS * keep, AX = IMG.axisX, [mx, my] = IMG.mouth;
  const L = [-0.28, -0.5, 0.82];                       // luz do rosto: do alto, um pouco da esquerda
  // 2) superfície: amostragem uniforme por área (arco da fatia x inclinação entre linhas)
  const surface = (sliceAt, y0, y1, isHead) => {
    for (let y = Math.max(0, Math.floor(y0)); y < Math.min(IMG.h, y1); y++) {
      const S = sliceAt(y + 0.5), Sn = sliceAt(y + 1.5);
      if (S.w < 1) continue;
      const tilt = Math.hypot(1, Sn.w - S.w, Sn.z - S.z);
      const span = !isHead && y > 720 ? Math.PI / 2 + 0.4 : Math.PI;   // o verso do tronco nunca aparece
      const ds = (f) => Math.hypot(S.w * Math.cos(f), S.d * Math.sin(f));
      let per = 0;
      for (let k = 0; k < 96; k++) per += ds(-span + 2 * span * (k + 0.5) / 96);
      per *= 2 * span / 96;
      // tronco: menos partículas (ele é escuro e grande); o brilho de cada uma compensa
      const sparse = isHead ? 1 : 1 - 0.6 * smooth(660, 760, y);
      const want = per * tilt * dens * sparse, n = Math.floor(want) + (rnd() < want % 1 ? 1 : 0);
      const dsMax = Math.max(S.w, S.d);
      for (let j = 0; j < n; j++) {
        let f;
        do f = -span + 2 * span * rnd(); while (rnd() * dsMax > ds(f));   // uniforme em comprimento de arco
        const yy = y + rnd(), q = onSlice(S, f);
        // rosto: relevo só na frente da cabeça. A luz dele vira brilho da própria partícula (gira junto): luz suave de
        // cima + oclusão (fundo escurece, saliência clareia), que dá leitura de rosto mesmo de frente
        const front = isHead ? smooth(0.1, 0.6, Math.cos(f)) * (1 - smooth(640, 662, yy)) : 0;
        let rel = 0, shade = 1;
        if (front > 0) {
          const u = q.x, h0 = relief(u, yy), hx1 = relief(u + 2, yy), hx0 = relief(u - 2, yy), hy1 = relief(u, yy + 2), hy0 = relief(u, yy - 2);
          const gx = (hx1 - hx0) / 4 * front, gy = (hy1 - hy0) / 4 * front, lap = (hx1 + hx0 + hy1 + hy0 - 4 * h0) / 4 * front;
          const dir = (-gx * L[0] - gy * L[1] + L[2]) / Math.hypot(gx, gy, 1) / L[2];
          rel = h0 * front;
          shade = clamp(Math.max(dir, 0) ** 1.8 * Math.exp((h0 > 0 ? 0.016 : 0.085) * rel) * clamp(Math.exp(-7 * lap), 0.45, 2.2), 0.15, 2.8);
        }
        const x = AX + q.x + rel * q.nx, z = q.z + rel * q.nz;
        // malha sutil na cabeça (meridianos a cada 22,5°, paralelos a cada 30 px): gira junto e mostra o volume; some
        // no rosto (as feições é que contam). O tronco escurece pra baixo: o olhar fica no rosto
        const mer = Math.abs(Math.sin(f * 8)) * S.w / 8, lat = Math.abs(((yy - IMG.headTop) % 30) - 15) - 13.5;
        const line = isHead ? Math.max(Math.exp(-mer * mer / 2), lat > 0 ? Math.exp(-((1.5 - lat) ** 2) / 0.6) : 0) * (1 - 0.85 * front) : 0;
        const fade = 1 - 0.45 * smooth(680, IMG.h, yy);
        const l = (0.55 + 0.45 * (1 - 0.6 * front) * Math.pow(rnd(), 1.5)) * (1 + 2.2 * line) * shade * fade / Math.sqrt(sparse);
        const spark = rnd() < 0.025 * fade * (1 - front) ? 1 : 0;
        let r = Math.min(255, 40 * l + 160 * spark), g = Math.min(255, 125 * l + 110 * spark), b = 255 * Math.min(1, l + 0.2);
        const gold = isHead && Math.cos(f) > 0 ? Math.exp(-(((x - mx) / 85) ** 2) - (((yy - my) / 60) ** 2)) : 0;   // boca
        r += (255 - r) * gold; g += (175 - g) * gold; b += (55 - b) * gold;
        pts.push(x, yy, r, g, b, 255 * Math.max(0, Math.cos(f)), 0, 0, z, q.nx, q.nz, S.k);
      }
    }
  };
  surface(headSlice, HEAD_Y[0], HEAD_Y[1], true);
  surface(bodySlice, BODY_Y[0], BODY_Y[1], false);
  // 3) orelhas: placas ovais na lateral do crânio, um pouco atrás do centro e estendidas pra trás; borda mais clara
  {
    const E = H.EAR;
    for (let y = Math.ceil(E.cy - E.ry); y < E.cy + E.ry; y++) {
      const v = (y + 0.5 - E.cy) / E.ry, U = E.out * Math.sqrt(Math.max(0, 1 - v * v)), S = headSlice(y + 0.5);
      for (const sg of [-1, 1]) {
        const want = U * 2.2 * dens * 1.6;              // a placa tem ~2,2x a largura vista de frente
        for (let j = 0, n = Math.floor(want) + (rnd() < want % 1 ? 1 : 0); j < n; j++) {
          const u = U * Math.sqrt(rnd()), edge = Math.exp(-((U - u) ** 2) / 8), l = 0.45 + 0.35 * rnd() + 0.9 * edge;
          pts.push(AX + sg * (S.w + u), y + rnd(), Math.min(255, 45 * l), Math.min(255, 135 * l), 255, 120, 0, 0,
            -20 - 1.3 * u, sg * 0.8, 0.6, 1);
        }
      }
    }
  }
  return pts;
}

// Malha escura dos sólidos (anéis por linha): tampa o que fica atrás (nuca, orelha do outro lado, pescoço atrás do
// queixo) pelo teste de profundidade. Mesmos atributos das partículas: (x, y) com giro 0 e (z, nx, nz, peso).
export function buildMesh() {
  const home = [], p3 = [], idx = [], SEG = 96;
  const ring = (sliceAt, y0, y1, step) => {
    const rows = [];
    for (let y = y0; y < y1; y += step) rows.push(y);
    rows.push(y1);
    const base = home.length / 2;
    for (const y of rows) {
      const S = sliceAt(y);
      for (let k = 0; k <= SEG; k++) {
        const q = onSlice(S, -Math.PI + 2 * Math.PI * k / SEG);
        home.push(IMG.axisX + q.x, y); p3.push(q.z, q.nx, q.nz, S.k);
      }
    }
    for (let r = 0; r < rows.length - 1; r++) {
      for (let k = 0; k < SEG; k++) {
        const a = base + r * (SEG + 1) + k, b = a + SEG + 1;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
  };
  ring(headSlice, HEAD_Y[0], HEAD_Y[1], 3);
  ring(bodySlice, BODY_Y[0], BODY_Y[1], 4);
  return { home: new Float32Array(home), p3: new Float32Array(p3), idx: new Uint16Array(idx) };
}
