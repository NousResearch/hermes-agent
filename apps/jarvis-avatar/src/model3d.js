/* Busto 3D por vistas. A cabeça (com o pescoço) é um sólido de fatias elípticas, uma por linha da imagem, medido em
   duas vistas: a largura vem da frente (alfa do corpo da foto) e a profundidade e o centro vêm do perfil (vista de 90°).
   Cada vista desenhada (a foto de frente e as vistas giradas em assets/img/vistas) vira um conjunto de partículas
   preso à superfície no ângulo daquela vista: no próprio ângulo o conjunto é a imagem exata; entre duas vistas, os
   dois conjuntos giram juntos pela superfície e se fundem (pesos no main.js). O lado esquerdo é o espelho.
   Dos ombros pra baixo nada gira (as vistas foram compostas com os mesmos ombros da frente).
   Coordenadas em px da imagem; z = pra frente. */
import { IMG, HEAD3D } from './config.js';
import { clamp, rnd } from './util.js';

const H = HEAD3D;
const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
const AX = IMG.axisX;

// quanto cada linha acompanha o giro: a cabeça e o pescoço inteiros, sumindo na base do pescoço
export const turnOf = (y) => 1 - smooth(H.TURN_Y[0], H.TURN_Y[1], y);

// Sólido por linha: W (meia-largura de frente), D (meia-profundidade), Z (centro em z), K (peso do giro).
// alfa: pixels do corpo-alfa da foto · prof: [esq, dir] da silhueta por linha na vista de 90°.
export function buildSolid(alfa, prof) {
  const W = new Float32Array(IMG.h), D = new Float32Array(IMG.h), Z = new Float32Array(IMG.h), K = new Float32Array(IMG.h);
  const edge = (y, dir) => {                    // última coluna opaca saindo do eixo (tolera buracos de até 6 px)
    let last = -1, run = 0;
    for (let d = 0; d < 700; d++) {
      const x = Math.round(AX) + dir * d;
      if (x < 0 || x >= IMG.w) break;
      if (alfa[(y * IMG.w + x) * 4] >= 128) { last = d; run = 0; } else if (last >= 0 && ++run > 6) break;
    }
    return last;
  };
  for (let y = 0; y < IMG.h; y++) {
    const l = edge(y, -1), r = edge(y, 1);
    W[y] = l >= 0 && r >= 0 ? 0.5 * (l + r) + H.HALO : 0;
    const p = prof[y];
    if (p && p[0] >= 0 && turnOf(y) > 0) { Z[y] = 0.5 * (p[0] + p[1]) - AX - H.PIVOT; D[y] = 0.5 * (p[1] - p[0]); }
    K[y] = turnOf(y);
  }
  // de frente as orelhas saem do crânio: nessas linhas a largura é a do crânio (curva suave entre acima e abaixo)
  const [e0, e1] = H.EARS, n = e1 - e0;
  const d0 = (W[e0] - W[e0 - 10]) / 10 * n, d1 = (W[e1 + 10] - W[e1]) / 10 * n, v0 = W[e0], v1 = W[e1];
  for (let y = e0 + 1; y < e1; y++) {
    const t = (y - e0) / n, t2 = t * t, t3 = t2 * t;
    W[y] = (2 * t3 - 3 * t2 + 1) * v0 + (t3 - 2 * t2 + t) * d0 + (-2 * t3 + 3 * t2) * v1 + (t3 - t2) * d1;
  }
  // abaixo da cabeça o perfil não importa (não gira): profundidade rasa, centro no eixo
  for (let y = 0; y < IMG.h; y++) if (!D[y]) { D[y] = Math.min(W[y], 140); Z[y] = -H.PIVOT; }
  for (const A of [W, D, Z]) {                  // alisa na vertical: sem degraus de uma borda medida linha a linha
    for (let pass = 0; pass < 3; pass++) {
      const src = A.slice();
      for (let y = 0; y < IMG.h; y++) {
        let acc = 0, c = 0;
        for (let k = Math.max(0, y - 6); k <= Math.min(IMG.h - 1, y + 6); k++) { acc += src[k]; c++; }
        A[y] = acc / c;
      }
    }
  }
  return { W, D, Z, K };
}

// fatia girada (giro a já multiplicado por K): centro na tela, meia-largura projetada e o ângulo phi que a projeta
// no ponto X (ponto que encara a câmera); null fora da silhueta
function unproject(S, y, X, a) {
  const w = S.W[y], d = S.D[y], z = S.Z[y], c = Math.cos(a), s = Math.sin(a);
  const xc = AX + (z + H.PIVOT) * s, ry = Math.hypot(w * c, d * s);
  return { xc, ry, phi: Math.abs(X - xc) < ry ? Math.asin(clamp((X - xc) / ry, -1, 1)) - Math.atan2(d * s, w * c) : null };
}

// Partículas. Cada uma: x, y, r, g, b, papel, relevo, borda, aura, z|Z, nx|W, nz|D, peso do giro, conjunto.
// Papel 0: ponto da superfície (x, y canônicos com giro 0; z e normal). Papel 4: brilho fora da silhueta (halo,
// orelha que sai do crânio): anda colado à borda girada (p3 = Z, W, D da linha). Conjunto -1: parado (ombros, aura);
// 0: a foto de frente; 1..: as vistas giradas.
export const STRIDE = 14;
export function buildParticles(em, inf, keep, solid, views) {
  const pts = [], S = solid;
  const place = (X, y, rgb, extra, set, a) => {
    const yi = Math.min(IMG.h - 1, Math.max(0, Math.floor(y))), w = S.W[yi];
    if (w < 1) return false;
    const u = unproject(S, yi, X, a * S.K[yi]);
    if (u.phi !== null) {
      const sf = Math.sin(u.phi), cf = Math.cos(u.phi), d = S.D[yi], nl = Math.hypot(d * sf, w * cf) || 1;
      pts.push(AX + w * sf, y, rgb[0], rgb[1], rgb[2], 0, extra[0], extra[1], 0, S.Z[yi] + d * cf, (d * sf) / nl, (w * cf) / nl, S.K[yi], set);
    } else {
      if (Math.abs(X - u.xc) > u.ry + H.HALO_MAX) return false;
      // canônico: o ponto que, girado do mesmo jeito, cai exatamente em X (o desvio da borda escala com a largura)
      pts.push(AX + (X - u.xc) * w / u.ry, y, rgb[0], rgb[1], rgb[2], 4, extra[0], extra[1], 0, S.Z[yi], w, S.D[yi], S.K[yi], set);
    }
    return true;
  };
  // foto de frente: a cabeça vai pro conjunto 0; ombros e aura ficam parados
  for (let y = 0, px = 0; y < IMG.h; y++) {
    const yc = y + 0.5, turning = S.K[y] > 0;
    for (let x = 0; x < IMG.w; x++, px += 4) {
      const t = inf[px + 2];
      if (!t || rnd() > keep) continue;
      const rgb = [em[px], em[px + 1], em[px + 2]], extra = [inf[px], inf[px + 1]];
      if (t > 200) pts.push(x + 0.5, yc, rgb[0], rgb[1], rgb[2], 0, extra[0], extra[1], 255, -400, 0, 1, S.K[y], -1);
      // ombros e o que não cabe no sólido: parados, na superfície da malha (senão ficam atrás dela)
      else if (!place(x + 0.5, yc, rgb, extra, turning ? 0 : -1, 0)) pts.push(x + 0.5, yc, rgb[0], rgb[1], rgb[2], 0, extra[0], extra[1], 0, 1000, 0, 1, 0, -1);
    }
  }
  // vistas giradas: todo pixel aceso da cabeça e do pescoço (o resto da imagem é igual à frente)
  views.forEach((v, i) => {
    const a = v.yaw, px = v.pixels;
    for (let y = 0; y < IMG.h; y++) {
      if (S.K[y] <= 0) continue;
      for (let x = 0; x < IMG.w; x++) {
        const k = (y * IMG.w + x) * 4, r = px[k], g = px[k + 1], b = px[k + 2];
        if (r + g + b < H.LUM || rnd() > keep) continue;
        place(x + 0.5, y + 0.5, [r, g, b], [255 * Math.max(0, 1 - y / 900), (r + g + b) > 600 ? 200 : 0], i + 1, a);
      }
    }
  });
  return pts;
}

// Malha escura do sólido (anéis por linha): tampa o fundo e o que fica atrás (nuca, orelha do outro lado) pelo teste
// de profundidade. Mesmos atributos das partículas: (x, y) com giro 0 e (z, nx, nz, peso do giro).
export function buildMesh(solid) {
  const S = solid, home = [], p3 = [], idx = [], SEG = 96, rows = [];
  let y0 = 0;
  while (y0 < IMG.h && S.W[y0] < 1) y0++;
  for (let y = y0; y < IMG.h - 1; y += 3) rows.push(y);
  rows.push(IMG.h - 1);
  for (const y of rows) {
    // um pouco menor que o sólido: só precisa tampar o fundo e o verso; o brilho da borda das vistas fica por fora
    const w = Math.max(0, S.W[y] - H.HALO - H.MESH_IN), d = Math.max(0, S.D[y] - H.MESH_IN);
    for (let k = 0; k <= SEG; k++) {
      const f = -Math.PI + 2 * Math.PI * k / SEG, sf = Math.sin(f), cf = Math.cos(f), nl = Math.hypot(d * sf, w * cf) || 1;
      home.push(AX + w * sf, y + 0.5); p3.push(S.Z[y] + d * cf, (d * sf) / nl, (w * cf) / nl, S.K[y]);
    }
  }
  for (let r = 0; r < rows.length - 1; r++) {
    for (let k = 0; k < SEG; k++) {
      const a = r * (SEG + 1) + k, b = a + SEG + 1;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return { home: new Float32Array(home), p3: new Float32Array(p3), idx: new Uint16Array(idx) };
}
