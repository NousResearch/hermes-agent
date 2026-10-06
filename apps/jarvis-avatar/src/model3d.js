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

// Partículas: as da própria foto (info.png marca, emissao.png dá a cor), vestidas nos sólidos. Cada partícula da
// figura vira um ponto da superfície no ângulo que a projeta exatamente onde ela está na foto, então de frente a cena
// é a foto. A foto é comprimida perto da borda: girando, essa faixa (|ângulo| > BAND) esticaria em listras, então ela
// apaga quando vira pra câmera e no lugar entram partículas extras com o desenho do miolo da MESMA linha continuado em
// volta (lateral e nuca), que só aparecem quando aquele lado vira. No alto do crânio a foto é só o contorno (escuro
// por dentro): ali a frente ganha o mesmo desenho enquanto ele gira. Papel (aCol.a): 0 miolo, 1 faixa, 2 extra, 3 topo,
// 4 contorno do topo, 5 calota da foto (some girando).
// Cada uma: x, y, r, g, b, papel, relevo da foto, borda, aura (0/255), z, nx, nz, peso do giro
export const STRIDE = 13;
const BAND = 0.85;                                   // rad: a partir daqui a foto já está comprimida demais
export function buildParticles(em, inf, keep) {
  const pts = [], AX = IMG.axisX, E = H.EAR;
  const put = (x, y, px, role, S, f) => {
    const q = onSlice(S, f);
    let r = em[px], g = em[px + 1], b = em[px + 2];
    if (role === 2 && r > b * 0.8) {                  // o brilho dourado da boca fica só na frente: na lateral vira azul
      const l = 0.3 * r + 0.5 * g + 0.2 * b;
      r = l * 0.35; g = l * 0.75; b = Math.min(255, l * 1.3);
    }
    pts.push(x, y, r, g, b, role, inf[px], inf[px + 1], 0, q.z, q.nx, q.nz, S.k);
  };
  // miolo de cada linha (pra continuar o desenho em volta): índices de pixel com |sin f| < sin(BAND)
  const coreH = new Array(IMG.h), coreB = new Array(IMG.h);
  for (let y = 0, px = 0; y < IMG.h; y++) {
    const yc = y + 0.5, Sh = headSlice(yc), Sb = bodySlice(yc);
    coreH[y] = []; coreB[y] = [];
    for (let x = 0; x < IMG.w; x++, px += 4) {
      const t = inf[px + 2];
      if (!t || rnd() > keep) continue;
      const xc = x + 0.5, dx = xc - AX, ad = Math.abs(dx);
      const S = t <= 200 && ad < Sh.w ? Sh : t <= 200 && ad < Sb.w ? Sb : null;
      const crown = 1 - smooth(165, 215, yc);           // calota: troca inteira entre a foto (parado) e o desenho em volta (girando)
      if (t <= 200 && inf[px + 1] >= 90 && yc < 175 && ad < Sh.w + 14) {
        // contorno do alto do crânio: é silhueta (a calota olha pra cima), não superfície. Fica colado à silhueta girada:
        // o shader o põe no centro girado da fatia + dx vezes a largura projetada / largura (z, nx, nz = centro, w, d)
        pts.push(xc, yc, em[px], em[px + 1], em[px + 2], 4, inf[px], inf[px + 1], 0, Sh.z, Math.max(Sh.w, 1), Sh.d, Sh.k);
      } else if (S) {
        const f = Math.asin(clamp(dx / S.w, -1, 1)), band = Math.abs(f) > BAND;
        put(xc, yc, px, S === Sh && rnd() < crown ? 5 : band ? 1 : 0, S, f);
        if (!band && inf[px + 1] < 90) (S === Sh ? coreH : coreB)[y].push(px, f);
      } else if (t <= 200 && Math.abs(yc - E.cy) < E.ry + 10 && ad < Sh.w + E.out + 20) {   // orelha: placa pra trás
        const u = ad - Sh.w;
        pts.push(xc, yc, em[px], em[px + 1], em[px + 2], 0, inf[px], inf[px + 1], 0, -20 - 1.3 * u, Math.sign(dx) * 0.8, 0.6, 1);
      } else {                                                    // aura e o resto: plano, atrás do busto
        pts.push(xc, yc, em[px], em[px + 1], em[px + 2], 0, inf[px], inf[px + 1], 255, -400, 0, 1, yc < 700 ? 1 : H.TWIST);
      }
    }
  }
  // lateral e nuca: o miolo da linha ladrilhado por comprimento de arco, de BAND até a nuca, dos dois lados. Uma
  // passada pra cabeça, outra pro pescoço (o tronco quase não gira). Linha sem miolo (o alto do crânio é só contorno):
  // pega o da linha mais perto que tem.
  const STEPS = 96;
  const wrap = (core, sliceAt, y0, y1, behindHead) => {
    for (let y = y0; y < y1; y++) {
      const yc = y + 0.5, T = sliceAt(yc), hw = behindHead ? headSlice(yc).w : 0;
      if (T.w < 2) continue;
      // pescoço atrás do queixo: a foto mostra só um pedaço da linha; o desenho vem da primeira linha abaixo do queixo
      // calota: na foto ela é quase só escuro por dentro; o desenho vem das linhas da testa (alturas alternadas, sem listras)
      let list = hw >= 2 || (!behindHead && y < 215) ? [] : core[y];
      if (!behindHead && y < 215) for (let k = 0; !list.length && k < 60; k++) list = core[215 + ((y * 7 + k) % 50)];
      if (hw >= 2) {                                   // linhas diferentes pra alturas diferentes (sem listras verticais)
        let y0 = y;
        while (y0 < IMG.h && headSlice(y0 + 0.5).w >= 2) y0++;
        for (let k = 0; !list.length && k < 60; k++) list = core[Math.min(719, y0 + ((y * 7 + k) % 50))] || [];
      }
      for (let k = 1; !list.length && k < 260; k++) list = core[Math.min(IMG.h - 1, y + k)].length ? core[Math.min(IMG.h - 1, y + k)] : core[Math.max(0, y - k)];
      if (!list.length) continue;
      // tudo em comprimento de arco: a densidade da foto é por px de tela, que no miolo inclinado vale 1/cos px de arco
      const dsAt = (f) => Math.hypot(T.w * Math.cos(f), T.d * Math.sin(f));
      const arcOf = (f) => { let a = 0; const n = 24; for (let k = 0; k < n; k++) a += dsAt(-BAND + (f + BAND) * (k + 0.5) / n); return a * (f + BAND) / n; };
      const tile = arcOf(BAND);                                          // comprimento do miolo
      const phis = new Float32Array(STEPS + 1), arcs = new Float32Array(STEPS + 1);
      for (let k = 0; k <= STEPS; k++) {
        phis[k] = BAND + (Math.PI - BAND) * k / STEPS;
        if (k) { const f = 0.5 * (phis[k] + phis[k - 1]); arcs[k] = arcs[k - 1] + Math.hypot(T.w * Math.cos(f), T.d * Math.sin(f)) * (phis[k] - phis[k - 1]); }
      }
      const L = arcs[STEPS];
      const phiAt = (a) => { let k = 1; while (k < STEPS && arcs[k] < a) k++; return phis[k - 1] + (phis[k] - phis[k - 1]) * (a - arcs[k - 1]) / (arcs[k] - arcs[k - 1]); };
      // no alto do crânio a superfície deita (uma linha da tela cobre muito mais área): mais cópias, em alturas sorteadas
      const nrep = Math.min(2, Math.max(1, Math.round(Math.hypot(1, (sliceAt(yc + 1).w - sliceAt(yc - 1).w) / 2))));
      for (let i = 0; i < list.length; i += 2) {
        const px = list[i], fo = list[i + 1], o = arcOf(fo);              // posição no miolo (arco), da borda esquerda
        // a foto tem 1 partícula por px de tela; no arco, cos(fo)/fator do miolo: descarta o excesso
        if (rnd() > Math.cos(fo) * T.w / dsAt(fo)) continue;
        for (const side of [-1, 1]) {
          for (let a = side < 0 ? tile - o : o; a < L; a += tile) {       // espelhado à esquerda: os lados se encontram na nuca
            const f = side * phiAt(a);
            const role = !behindHead && rnd() < 1 - smooth(165, 215, yc) ? 3 : 2;   // calota: aparece só girando
            for (let r = 0; r < nrep; r++) put(AX + T.w * Math.sin(f), nrep > 1 ? y + rnd() : yc, px, role, T, f);
          }
        }
        // pescoço: a frente que a foto não mostra (está atrás do queixo). Sempre visível; parada, a cabeça a tampa
        const xf = T.w * Math.sin(fo);
        if (!behindHead && rnd() < 1 - smooth(165, 215, yc)) put(AX + xf, yc, px, 3, T, Math.asin(clamp(xf / T.w, -1, 1)));
        if (hw > 0 && Math.abs(xf) < hw) put(AX + xf, yc, px, 0, T, Math.asin(clamp(xf / T.w, -1, 1)));
      }
    }
  };
  wrap(coreH, headSlice, 0, HEAD_Y[1]);
  wrap(coreB, bodySlice, BODY_Y[0], 720, true);
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
