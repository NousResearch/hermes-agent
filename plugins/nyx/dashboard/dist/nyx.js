// Nyx — avatar de partículas do Hermes (módulo ES carregado pelo painel do dashboard).
//
// Em repouso só o busto: ~1,2 milhão de grãos de luz em fios finos, contas e névoa, como na foto de referência.
// Os eventos do Hermes (barramento do plugin) dirigem o resto:
//   pensando        → a galáxia nasce do crânio pra fora (3 braços em espiral, disco inclinado)
//   ferramenta      → um cometa sai do disco e orbita a cabeça enquanto a ferramenta roda
//   ferramenta_fim  → o cometa volta pro disco e se apaga (âmbar se deu erro)
//   subagente       → uma galáxia-satélite menor, inclinada ao contrário
//   falando         → a galáxia recolhe pro crânio e o dourado do rosto pulsa
//   pronto          → volta ao repouso
// A forma (HEAD/BODY) foi medida na foto de referência: [y, meia-largura, meia-profundidade, centro z]
// em px da imagem; 1 unidade da cena = 100 px.
import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { EffectComposer } from './vendor/EffectComposer.js';
import { RenderPass } from './vendor/RenderPass.js';
import { UnrealBloomPass } from './vendor/UnrealBloomPass.js';

/* ---------- silhueta ---------- */
const HEAD = [[141,0,12,30],[143,6,37,30],[147,28,69,30],[151,45,86,30],[156,61,99,30],[160,75,111,30],[166,86,121,30],[172,96,130,30],[178,105,137,30],[184,114,145,30],[190,119,151,30],[196,126,157,30],[202,132,163,30],[208,137,168,30],[214,142,173,30],[220,147,176,30],[226,151,180,30],[232,155,183,30],[238,160,186,30],[244,163,189,30],[250,165,191,30],[256,170,194,29],[262,172,196,29],[268,174,198,29],[274,176,199,29],[280,178,201,29],[286,178,203,29],[292,180,204,29],[298,181,204,29],[304,182,205,29],[310,182,206,29],[316,183,206,29],[322,184,206,29],[328,184,206,29],[334,184,207,29],[340,184,207,28],[346,183,206,27],[352,183,206,26],[364,183,204,24],[376,181,201,24],[388,178,199,25],[400,175,197,27],[412,172,195,30],[424,171,195,35],[436,170,192,40],[448,168,190,46],[460,166,187,52],[472,164,184,57],[484,162,181,60],[496,160,180,64],[508,156,181,66],[520,152,182,67],[532,149,179,68],[544,145,172,72],[558,134,157,80],[572,127,139,91],[584,118,121,101],[592.5,108,107,109],[602,98,91,118],[609.5,88,77,126],[615.5,78,64,133],[623,68,50,140],[630.5,58,36,148],[635.5,48,24,154],[640.5,38,15,158],[644.5,28,10,160],[646,18,9,161],[648,9,8,161],[649,2,8,162]];
const BODY = [[520,116,112,-2],[560,118,112,0],[600,128,112,-6],[610,128,112,-8],[620,128,113,-9],[630,128,114,-10],[640,128,114,-12],[650,129,116,-13],[660,130,118,-15],[670,133,120,-16],[680,138,123,-17],[690,144,125,-19],[700,152,128,-20],[710,162,132,-22],[720,176,136,-24],[730,187,140,-26],[740,197,145,-28],[750,214,150,-29],[760,238,155,-30],[770,259,160,-31],[780,276,165,-32],[790,300,170,-33],[795,305,172,-34],[800,309,175,-34],[805,318,178,-34],[810,350,180,-34],[815,386,182,-34],[820,414,184,-35],[825,437,186,-36],[830,456,188,-36],[840,477,192,-36],[850,498,195,-36],[860,513,198,-36],[870,526,200,-36],[880,537,203,-36],[890,546,204,-36],[900,556,206,-36],[910,562,207,-36],[920,571,208,-36],[930,576,209,-36],[940,581,210,-36]];
const U = 0.01, Y0 = 380;
// câmera: o mesmo enquadramento da foto (1 px da foto = 1 px numa tela de 941 de altura). Lente fechada e
// câmera longe: a foto é quase sem perspectiva, e perto o tronco 'subia' e abria os ombros antes da hora
const FOV = 18, DIST = 29.8, ALVO_Y = -0.9;
const CABECA = new THREE.Vector3(0, -(410 - Y0) * U, 0.3);   // centro do crânio na cena

function row(T, y) {                          // Catmull-Rom entre as linhas medidas
  if (y <= T[0][0]) return [T[0][1], T[0][2], T[0][3]];
  const n = T.length - 1;
  if (y >= T[n][0]) return [T[n][1], T[n][2], T[n][3]];
  let i = 0;
  while (y > T[i + 1][0]) i++;
  const P = (k) => T[Math.max(0, Math.min(n, k))], t = (y - T[i][0]) / (T[i + 1][0] - T[i][0]);
  const a = P(i - 1), b = P(i), c = P(i + 1), d = P(i + 2);
  return [1, 2, 3].map((k) => {
    const m1 = (c[k] - a[k]) / (c[0] - a[0] || 1) * (c[0] - b[0]), m2 = (d[k] - b[k]) / (d[0] - b[0] || 1) * (c[0] - b[0]);
    return (2 * t ** 3 - 3 * t * t + 1) * b[k] + (t ** 3 - 2 * t * t + t) * m1 + (-2 * t ** 3 + 3 * t * t) * c[k] + (t ** 3 - t * t) * m2;
  });
}
function insideHead(x, y, z) {                // px; o pescoço não entra na cabeça
  if (y < 143 || y > 649) return false;
  const r = rowR(HEAD, y);
  return (x / r[0]) ** 2 + ((z - r[2]) / r[1]) ** 2 < 0.97;
}

/* ---------- estado: eventos do Hermes → o que o avatar mostra (função pura, sem WebGL) ---------- */
export function estadoInicial() {
  return { pensando: false, desde: 0, falando: false, subagentes: 0, ferramentas: {} };
}
const REDUTORES = {
  pensando: (s, ev, t) => ({ ...s, pensando: true, desde: s.pensando ? s.desde : t, falando: false }),
  ferramenta: (s, ev, t) => ({
    ...s, pensando: true, desde: s.pensando ? s.desde : t, falando: false,
    ferramentas: { ...s.ferramentas, [ev.id || ev.ferramenta]: { nome: ev.ferramenta || '', t0: t, fim: null, erro: false } },
  }),
  ferramenta_fim: (s, ev, t) => {
    const k = ev.id || ev.ferramenta, f = s.ferramentas[k];
    return f ? { ...s, ferramentas: { ...s.ferramentas, [k]: { ...f, fim: t, erro: !!ev.erro } } } : s;
  },
  falando: (s) => ({ ...s, pensando: false, falando: true }),
  pronto: (s, ev, t) => ({ ...estadoInicial(), ferramentas: encerrar(s.ferramentas, t) }),
  subagente: (s, ev, t) => ({ ...s, subagentes: s.subagentes + 1, pensando: true, desde: s.pensando ? s.desde : t }),
  subagente_fim: (s) => ({ ...s, subagentes: Math.max(0, s.subagentes - 1) }),
};
function encerrar(ferramentas, t) {
  const out = {};
  for (const [k, f] of Object.entries(ferramentas)) out[k] = f.fim === null ? { ...f, fim: t } : f;
  return out;
}
export function reduzir(s, ev, t) {
  const fn = ev && REDUTORES[ev.tipo];
  return fn ? fn(s, ev, t) : s;
}
// alvo da galáxia: pensar por mais de 0,3 s (pensamento curto não pisca), ferramenta rodando ou subagente vivo
export function alvoGalaxia(s, t) {
  if (s.falando) return 0;
  const rodando = Object.values(s.ferramentas).some((f) => f.fim === null);
  return (s.pensando && t - s.desde > 0.3) || rodando || s.subagentes > 0 ? 1 : 0;
}

/* ---------- partículas ---------- */
const DEEP = [0.0, 0.17, 1.0], BLUE = [0.0, 0.3, 1.0], CYAN = [0.12, 0.58, 1.0], WHITE = [0.6, 0.86, 1.0], GOLD = [1.0, 0.62, 0.14];

function criarAleatorio(semente) {
  let s = semente;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  let resto = null;                           // Box-Muller dá dois valores por sorteio: guarda o segundo
  const gauss = () => {
    if (resto !== null) { const g = resto; resto = null; return g; }
    const r = Math.sqrt(-2 * Math.log(rnd() + 1e-9)), a = 6.283185 * rnd();
    resto = r * Math.sin(a);
    return r * Math.cos(a);
  };
  return { rnd, gauss };
}

// row() pré-calculado a cada 1/4 px: com milhões de grãos a interpolação de Catmull-Rom vira o gargalo.
// Devolve sempre o mesmo array [w, d, c]: quem chama copia os valores antes da próxima chamada.
const LUTS = new Map(), PASSO = 4, _r = [0, 0, 0];
function rowR(T, y) {
  let L = LUTS.get(T);
  if (!L) {
    const y0 = T[0][0], n = (T[T.length - 1][0] - y0) * PASSO + 1, w = new Float32Array(n), d = new Float32Array(n), c = new Float32Array(n);
    for (let i = 0; i < n; i++) { const r = row(T, y0 + i / PASSO); w[i] = r[0]; d[i] = r[1]; c[i] = r[2]; }
    L = { y0, n, w, d, c };
    LUTS.set(T, L);
  }
  const f = Math.min(L.n - 1.001, Math.max(0, (y - L.y0) * PASSO)), i = f | 0, t = f - i;
  _r[0] = L.w[i] + (L.w[i + 1] - L.w[i]) * t;
  _r[1] = L.d[i] + (L.d[i + 1] - L.d[i]) * t;
  _r[2] = L.c[i] + (L.c[i + 1] - L.c[i]) * t;
  return _r;
}

// sorteio por área: cada linha pesa o perímetro da fatia vezes a inclinação (o alto da cabeça e o topo dos
// ombros são quase horizontais: 1 px de y ali cobre muito mais superfície)
function sampler(rnd, T, y0, y1) {
  const ys = [], acc = [];
  let s = 0;
  for (let y = y0; y < y1; y++) {
    const [w, d] = rowR(T, y), [wa, da, ca] = rowR(T, y - 1), [wb, db, cb] = rowR(T, y + 1);
    const dw = (wb - wa) / 2, dd = (db - da) / 2, dc = (cb - ca) / 2;
    s += Math.PI * (3 * (w + d) - Math.sqrt((3 * w + d) * (w + 3 * d))) * Math.sqrt(1 + 0.5 * (dw * dw + dd * dd) + dc * dc);
    ys.push(y); acc.push(s);
  }
  return () => {
    const v = rnd() * s;
    let lo = 0, hi = acc.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (acc[m] < v) lo = m + 1; else hi = m; }
    return ys[lo] + rnd();
  };
}

function insideBody(x, y, z) {                // px; a parte de baixo da mandíbula que fica dentro do pescoço
  if (y < 520 || y > 940) return false;
  const r = rowR(BODY, y);
  return (x / r[0]) ** 2 + ((z - r[2]) / r[1]) ** 2 < 0.97;
}

const AX = 832.5;                             // eixo da figura na foto
const QUEIXO = [[832, 649], [850, 646], [870, 640.5], [890, 630.5], [910, 615.5], [930, 602], [950, 584], [966, 558]];
const MARCAS = [                              // [tabela, curva, fios no feixe, brilho do grão]
  [HEAD, QUEIXO, 12, 0.15],                                                              // contorno do queixo
  [BODY, [[957, 600], [956, 700], [957, 780], [961, 800], [973, 820], [995, 840], [1003, 846]], 12, 0.26],  // lado do pescoço → trapézio
  [BODY, [[878, 662], [872, 720], [862, 780], [850, 840], [838, 892]], 5, 0.1],          // esternocleidomastoide até o esterno
  [BODY, [[930, 849], [970, 851], [1003, 846], [1052, 840], [1100, 828], [1150, 820], [1195, 823]], 12, 0.3],  // clavícula
];
const queixoY = (ax) => {                     // y do contorno do queixo na meia-largura |x| (px)
  const x = AX + Math.abs(ax);
  for (let i = 0; i < QUEIXO.length - 1; i++) {
    const [x0, y0] = QUEIXO[i], [x1, y1] = QUEIXO[i + 1];
    if (x <= x1) return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  }
  return QUEIXO[QUEIXO.length - 1][1];
};
const suave = (a, b, v) => { const t = Math.min(1, Math.max(0, (v - a) / (b - a))); return t * t * (3 - 2 * t); };
// sombras da foto na frente do pescoço: a faixa logo abaixo do queixo e o V entre os dois músculos
function sombraPescoco(x, y, z) {
  if (z < 0) return 1;
  let s = 1;
  const dq = y - queixoY(x);
  if (Math.abs(x) < 130 && dq > 0 && dq < 50) s *= 0.15 + 0.85 * suave(4, 50, dq);
  if (y > 650 && y < 905) {
    const meia = 40 - 34 * (y - 650) / 255;      // o V: ~40 px de meia-largura sob o queixo, fecha no esterno
    s *= 0.22 + 0.78 * suave(meia * 0.55, meia * 1.2, Math.abs(x));
  }
  return s;
}

// Orelha, medida na foto linha a linha. De frente ela é um anel fino: a hélice clara por fora, a concha
// escura no meio e, por dentro, a borda da cabeça (que atrás da orelha é ~15 px mais estreita que acima).
// ORELHA_EXT: até onde a hélice chega de frente, [y, px a partir do eixo].
const ORELHA = { cy: 436, meiaAltura: 66, meiaFundo: 44, cz: 4 };
const ORELHA_EXT = [[366, 184], [370, 193], [378, 201], [386, 203], [394, 204], [402, 203], [410, 202], [418, 200], [426, 199], [434, 199], [442, 196], [450, 194], [458, 191], [466, 189], [474, 185], [482, 181], [490, 176], [498, 170], [506, 160]];
const extOrelha = (y) => {
  const E = ORELHA_EXT;
  if (y <= E[0][0]) return E[0][1];
  for (let i = 0; i < E.length - 1; i++) if (y <= E[i + 1][0]) return E[i][1] + (E[i + 1][1] - E[i][1]) * (y - E[i][0]) / (E[i + 1][0] - E[i][0]);
  return E[E.length - 1][1];
};
const cranio = (y, z) => { const [w, d, c] = rowR(HEAD, y), zz = (z - c) / d; return w * Math.sqrt(Math.max(0, 1 - zz * zz)); };
// t = 0 aponta pra nuca, π/2 pro alto; r = 0 no fundo da concha (no crânio), r = 1 na hélice. A frente fica
// presa no crânio; de cima até o lóbulo a hélice abre até a medida da foto.
function naOrelha(sg, t, r) {
  const O = ORELHA, sn = Math.sin(t), cs = Math.cos(t);
  const fundo = O.meiaFundo * (sn < 0 ? 1 + 0.35 * sn : 1);   // de perfil o lóbulo é mais estreito que o alto
  const hy = O.cy - sn * O.meiaAltura - 5 * cs, hz = O.cz - cs * fundo;
  const abre = Math.max(0, extOrelha(hy) - cranio(hy, hz)) * suave(-0.85, -0.05, cs);
  const ey = O.cy + (hy - O.cy) * r, ez = O.cz + (hz - O.cz) * r;
  return [sg * (cranio(ey, ez) + abre * r ** 1.5), ey, ez];
}

// O busto só com partículas, como na foto, em três camadas:
//  - fios: linhas finas e contínuas (grãos a cada 0,4 px), retas por uns 10 px e então dobram, como trilhas;
//  - contas: pontos claros e nítidos que a maioria dos fios carrega a cada ~5,5 px (as correntinhas da foto);
//  - névoa: grãos fracos espalhados, o azul-marinho entre os fios.
// Nada de contorno desenhado: cada grão guarda a normal da superfície e o shader acende os grãos vistos de
// raspão e apaga os de trás, então a borda e o volume aparecem pela luz. Cabeça e pescoço não se atravessam.
// Tudo vai direto pra arrays tipados (posição, cor em bytes, brilho/semente/tamanho, normal em bytes).
const FIOS = 500000, NEVOA = 700000, PASSO_FIO = 0.4, VAO_CONTA = 5.5;
function gerarBusto(densidade) {
  const { rnd, gauss } = criarAleatorio(7);
  const cap = Math.ceil((FIOS + NEVOA + 160000) * densidade) + 40000;
  const pos = new Float32Array(cap * 3), col = new Uint8Array(cap * 3), inf = new Float32Array(cap * 3), nor = new Int8Array(cap * 3);
  let n = 0;
  const ouro = (x, y, z) => (z > 40 ? 1 : 0) * Math.exp(-((x / 46) ** 2 + ((y - 548) / 50) ** 2));   // frente baixa do rosto
  const put = (x, y, z, nx, ny, nz, b, sz, cor) => {
    if (n >= cap) return;
    const i = n * 3;
    pos[i] = x * U; pos[i + 1] = -(y - Y0) * U; pos[i + 2] = z * U;
    col[i] = cor[0] * 255; col[i + 1] = cor[1] * 255; col[i + 2] = cor[2] * 255;
    inf[i] = b; inf[i + 1] = rnd(); inf[i + 2] = sz;
    nor[i] = nx * 127; nor[i + 1] = ny * 127; nor[i + 2] = nz * 127;
    n++;
  };
  const fadeY = (y) => (y < 932 ? 1 : Math.max(0, 1 - (y - 932) / 9));   // só a borda de baixo se desfaz
  // ponto da casca e a normal de verdade: inclui a inclinação vertical (topo dos ombros, queixo, alto da
  // cabeça), senão uma rampa acende como se estivesse de frente pra câmera. sy e sp: quanto a superfície
  // anda por px de y e por radiano de φ (os fios andam com passo constante na superfície, não no mapa).
  const P = { x: 0, z: 0, dy: 0, nx: 0, ny: 0, nz: 0, sy: 1, sp: 1 };
  const naCasca = (T, y, ph, t) => {
    let r = rowR(T, y + 1); const w1 = r[0], d1 = r[1], c1 = r[2];
    r = rowR(T, y - 1); const w0 = r[0], d0 = r[1], c0 = r[2];
    r = rowR(T, y); const w = r[0], d = r[1], c = r[2];
    const sn = Math.sin(ph), cs = Math.cos(ph);
    const ayx = (w1 - w0) / 2 * sn, ayz = (c1 - c0) / 2 + (d1 - d0) / 2 * cs;   // ∂P/∂y (y da cena = -y da foto)
    const apx = w * cs, apz = -d * sn;                                         // ∂P/∂φ
    let nx = -apz, ny = ayz * apx - ayx * apz, nz = apx;                       // ∂P/∂y × ∂P/∂φ
    const nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;   // sqrt em vez de hypot: aqui roda mais de um milhão de vezes
    nx /= nl; ny /= nl; nz /= nl;
    if (nx * sn * w + nz * cs * d < 0) { nx = -nx; ny = -ny; nz = -nz; }    // pra fora
    P.x = w * sn + nx * t; P.z = c + d * cs + nz * t; P.dy = -ny * t; P.nx = nx; P.ny = ny; P.nz = nz;
    P.sy = Math.sqrt(1 + ayx * ayx + ayz * ayz); P.sp = Math.max(6, Math.sqrt(apx * apx + apz * apz));
    return P;
  };
  // φ com a mesma densidade por área: nos ombros (largos e rasos) φ uniforme amontoaria grãos nas pontas
  const sortearPh = (T, y) => {
    const [w, d] = rowR(T, y), m = Math.max(w, d);
    for (;;) { const ph = rnd() * 6.283185, a = w * Math.cos(ph), b = d * Math.sin(ph); if ((rnd() * m) ** 2 < a * a + b * b) return ph; }
  };
  // ocupação em voxels de 5 px: cada fio nasce onde a vizinhança (~15 px) está mais vazia entre 8 candidatos,
  // então os fios se espalham por igual, sem os bolos e buracos do sorteio puro
  const VX = 5, GX = 252, GY = 196, GZ = 126, ocup = new Uint8Array(GX * GY * GZ);
  const voxel = (x, y, z) => {
    const i = Math.floor(x / VX) + 126, j = Math.floor(y / VX), k = Math.floor(z / VX) + 63;
    return i < 0 || i >= GX || j < 0 || j >= GY || k < 0 || k >= GZ ? -1 : (k * GY + j) * GX + i;
  };
  const vizinhos = (x, y, z) => {
    let soma = 0;
    for (let a = -VX; a <= VX; a += VX) for (let b = -VX; b <= VX; b += VX) for (let c = -VX; c <= VX; c += VX) {
      const v = voxel(x + a, y + b, z + c);
      if (v >= 0) soma += ocup[v];
    }
    return soma;
  };
  const corConta = (g, branco) => (rnd() < g * 0.4 ? GOLD : rnd() < branco ? WHITE : CYAN);
  // um fio: anda na superfície com passo de 0,4 px, quase reto, e de vez em quando dobra seco
  const fios = (T, alvo, fora, sombra, ganho) => {
    const pick = sampler(rnd, T, T === HEAD ? 143 : 520, T === HEAD ? 649 : 940), yMin = T[0][0] + 1, yMax = T[T.length - 1][0];
    for (let feitos = 0; feitos < alvo * densidade;) {
      let y = 0, ph = 0, menor = 1e9, dir = rnd() * 6.283185;
      for (let c = 0; c < 8; c++) {
        const yc = pick(), pc = sortearPh(T, yc), q = naCasca(T, yc, pc, 0), oc = vizinhos(q.x, yc, q.z);
        if (oc < menor) { menor = oc; y = yc; ph = pc; }
      }
      const comContas = rnd() < 0.7, L = comContas ? 18 + 30 * rnd() : 10 + 30 * rnd(), t0 = -1.2 + gauss() * 1.8;
      const bF = (comContas ? 0.07 + 0.07 * rnd() : 0.035 + 0.025 * rnd()) * ganho, r0 = rnd(), corF = r0 < 0.45 ? BLUE : r0 < 0.9 ? CYAN : DEEP;
      let conta = comContas ? rnd() * VAO_CONTA : Infinity;
      for (let s = 0; s < L; s += PASSO_FIO, feitos++) {
        if (y < yMin || y > yMax) break;
        const p = naCasca(T, y, ph, t0 + gauss() * 0.12);
        const sy = p.sy, sp = p.sp;
        if (!fora(p.x, y, p.z) && rnd() < fadeY(y)) {
          const v = voxel(p.x, y, p.z);
          if (v >= 0 && ocup[v] < 255) ocup[v]++;
          const g = ouro(p.x, y, p.z), sb = sombra(p.x, y, p.z);
          put(p.x, y + p.dy, p.z, p.nx, p.ny, p.nz, bF * sb, 0.7 + 0.2 * rnd(), rnd() < g * 0.6 ? GOLD : corF);
          if (s >= conta) {
            conta += VAO_CONTA * (0.85 + 0.3 * rnd());
            put(p.x, y + p.dy, p.z, p.nx, p.ny, p.nz, (1.0 + 0.6 * rnd()) * sb * Math.sqrt(ganho), 0.85 + 0.35 * rnd(), corConta(g, T === HEAD ? 0.6 : 0.4));   // no corpo da foto as contas são mais azuis
          }
        }
        dir += gauss() * 0.03 + (rnd() < 0.036 ? (rnd() < 0.5 ? -1 : 1) * (0.6 + 1.0 * rnd()) : 0);
        y += Math.sin(dir) * PASSO_FIO / sy;
        ph += Math.cos(dir) * PASSO_FIO / sp;
      }
    }
  };
  const nevoa = (T, alvo, fora, sombra, ganho) => {
    const pick = sampler(rnd, T, T === HEAD ? 143 : 520, T === HEAD ? 649 : 940);
    for (let i = 0; i < alvo * densidade; i++) {
      const y = pick(), p = naCasca(T, y, sortearPh(T, y), -1.2 + gauss() * 1.8);
      if (fora(p.x, y, p.z) || rnd() > fadeY(y)) continue;
      const g = ouro(p.x, y, p.z), dourado = rnd() < g;   // o brilho dourado da boca é névoa
      put(p.x, y + p.dy, p.z, p.nx, p.ny, p.nz, (0.02 + 0.02 * rnd()) * (dourado ? 4 : 1) * sombra(p.x, y, p.z) * ganho, 1.0 + 0.5 * rnd(), dourado ? GOLD : rnd() < 0.5 ? DEEP : BLUE);
    }
  };
  const semSombra = () => 1;
  fios(HEAD, FIOS * 0.42, insideBody, semSombra, 1);
  fios(BODY, FIOS * 0.58, insideHead, sombraPescoco, 1.6);      // na foto pescoço e ombros brilham mais que o rosto
  nevoa(HEAD, NEVOA * 0.42, insideBody, semSombra, 1);
  nevoa(BODY, NEVOA * 0.58, insideHead, sombraPescoco, 1.4);
  // marcas da foto, medidas como cristas de brilho (lado direito, px da foto; o esquerdo é o espelho).
  // Cada uma é um feixe de fios que segue a curva com desvio e ondulação próprios, assentado na
  // superfície: acende e apaga com a mesma luz de raspão do resto do corpo.
  for (const [T, curva, nFios, br] of MARCAS) {
    for (const sg of [-1, 1]) {
      for (let m = 0; m < nFios; m++) {
        const o = gauss() * 1.8, fase = rnd() * 6.283185, fr = 0.04 + 0.1 * rnd(), amp = 0.3 + 0.9 * rnd(), t0 = -0.6 + gauss() * 0.5;
        const bM = br * (0.7 + 0.6 * rnd());
        let conta = m % 2 === 0 ? rnd() * VAO_CONTA : Infinity, s = 0;
        const passo = PASSO_FIO / Math.sqrt(densidade);
        for (let i = 0; i < curva.length - 1; i++) {
          const [x0, y0] = curva[i], [x1, y1] = curva[i + 1], len = Math.hypot(x1 - x0, y1 - y0);
          const ux = -(y1 - y0) / len, uy = (x1 - x0) / len;
          for (let a = 0; a < len; a += passo, s += passo) {
            const lat = o + amp * Math.sin(s * fr + fase), t = a / len;
            const xp = (x0 + (x1 - x0) * t + ux * lat - AX) * sg, y = y0 + (y1 - y0) * t + uy * lat;
            const w = rowR(T, y)[0], ph = Math.asin(Math.max(-1, Math.min(1, xp / Math.max(1, w))));
            const p = naCasca(T, y, ph, t0 + gauss() * 0.15);
            put(p.x, y + p.dy, p.z, p.nx, p.ny, p.nz, bM, 0.7 + 0.2 * rnd(), rnd() < 0.7 ? CYAN : BLUE);
            if (s >= conta) {
              conta += VAO_CONTA * (0.85 + 0.3 * rnd());
              put(p.x, y + p.dy, p.z, p.nx, p.ny, p.nz, 1.0 + 0.5 * rnd(), 0.85 + 0.3 * rnd(), rnd() < 0.4 ? WHITE : CYAN);
            }
          }
        }
      }
    }
  }
  // orelhas: hélice (feixe de fios claros com contas), antélice mais fraca e concha escura. Sem normal:
  // a orelha é fina e se vê das duas faces, então o brilho dela não depende do ângulo
  for (const sg of [-1, 1]) {
    const linha = (r, ta, tb, b, comContas) => {
      const dr = gauss() * 0.06, dx = gauss() * 1.8;
      let conta = comContas ? rnd() * VAO_CONTA : Infinity;
      for (let t = ta, s = 0; t < tb; ) {
        const [x, y, z] = naOrelha(sg, t, r + dr);
        put(x + sg * dx, y, z, 0, 0, 0, b * (0.8 + 0.4 * rnd()), 0.7 + 0.2 * rnd(), rnd() < 0.4 ? WHITE : CYAN);
        if (s >= conta) { conta += VAO_CONTA * (0.85 + 0.3 * rnd()); put(x + sg * dx, y, z, 0, 0, 0, 0.9 + 0.5 * rnd(), 0.85 + 0.3 * rnd(), rnd() < 0.7 ? WHITE : CYAN); }
        const raio = Math.max(8, Math.hypot(ORELHA.meiaAltura * Math.cos(t), ORELHA.meiaFundo * Math.sin(t)) * r);   // aproximado: só regula o passo
        const dt = PASSO_FIO / raio / Math.sqrt(densidade);
        t += dt; s += PASSO_FIO / Math.sqrt(densidade);
      }
    };
    for (let m = 0; m < 16; m++) linha(0.95, -0.7 * Math.PI, 0.75 * Math.PI, 0.42, m < 8);   // hélice
    for (let m = 0; m < 4; m++) linha(0.6, -0.25 * Math.PI, 0.65 * Math.PI, 0.15, m < 2);    // antélice
    for (let i = 0; i < 6000 * densidade; i++) {                                                 // concha
      const [x, y, z] = naOrelha(sg, rnd() * 6.283185, Math.sqrt(rnd()) * 0.9);
      put(x, y, z, 0, 0, 0, 0.02 + 0.02 * rnd(), 1.0 + 0.5 * rnd(), DEEP);
    }
  }
  return { n, pos, col, inf, nor };
}

// braços: (ângulo inicial, raio px, altura px) — a receita da forma "galáxia" do avatar
function gerarBracos(n, r0, r1, semente) {
  const { rnd, gauss } = criarAleatorio(semente);
  const pos = [], col = [], inf = [];
  for (let i = 0; i < n; i++) {
    const arm = (rnd() * 3) | 0, gr = r0 + (r1 - r0) * rnd() ** 0.75;
    const t = gr / 300 * 2.1 + arm * 2.0944 + gauss() * (0.3 / (0.35 + gr / 500));
    pos.push(t, gr, gauss() * 10 * (1.3 - gr / r1));
    const r = rnd(), c = r < 0.5 ? BLUE : r < 0.88 ? CYAN : r < 0.97 ? WHITE : GOLD;
    col.push(...c);
    const edge = 1 - ((gr - r0) / (r1 - r0)) ** 1.5;
    inf.push((0.35 + 0.65 * rnd() ** 2) * edge, rnd(), 0.6 + 0.9 * rnd() ** 3);
  }
  return { pos, col, inf };
}

// cometas: SLOTS cometas × CAUDA grãos; (slot, atraso na cauda, semente)
const SLOTS = 6, CAUDA = 320;
function gerarCometas() {
  const { rnd } = criarAleatorio(99);
  const pos = [], col = [], inf = [];
  for (let s = 0; s < SLOTS; s++) {
    for (let i = 0; i < CAUDA; i++) {
      const lag = (i / CAUDA) ** 2.2;               // grãos concentrados na cabeça do cometa
      pos.push(s, lag, rnd());
      col.push(1, 1, 1);
      inf.push(2.2 * (1 - lag) ** 2, rnd(), 2.0 - lag * 1.3);
    }
  }
  return { pos, col, inf };
}
// cor do cometa por ferramenta (tabela; o resto fica ciano). Índices em CORES_COMETA.
const CORES_COMETA = [[0.45, 0.85, 1.0], [1.0, 0.7, 0.2], [0.55, 0.65, 1.0], [0.8, 0.55, 1.0], [0.95, 0.97, 1.0], [1.0, 0.5, 0.1]];
const COR_POR_FERRAMENTA = {
  terminal: 1, execute_code: 1, process: 1,
  read_file: 2, write_file: 2, patch: 2, search_files: 2,
  browser_navigate: 3, browser_click: 3, browser_type: 3, browser_snapshot: 3,
  delegate_task: 4, memory: 4, skill_view: 4,
};
const COR_ERRO = 5;
const corDaFerramenta = (nome) => COR_POR_FERRAMENTA[nome] ?? (nome.startsWith('browser_') ? 3 : 0);

/* ---------- shaders ---------- */
const VS_COMUM = `
  attribute vec3 color, aInf;                 // cor · brilho, semente, tamanho
  uniform float uTime, uSize, uC, uScale, uPensa, uFala, uExpo;
  varying vec3 vCol;
  vec3 fimDoPonto(vec3 p, float sd, float b, vec3 cor) {
    p += vec3(sin(uTime * (0.4 + sd * 0.5) + sd * 40.0), cos(uTime * (0.33 + sd * 0.4) + sd * 27.0), sin(uTime * 0.37 + sd * 19.0)) * 0.012;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float k = smoothstep(-2.3, 2.1, mv.z - uC);                       // 0 = fundo, 1 = mais perto
    float rit = 0.8 + sd * 2.5 + uPensa * 1.5;                          // pensando: cintila mais rápido
    float tw = 1.0 + (0.9 + uPensa * 0.6) * pow(0.5 + 0.5 * sin(uTime * rit + sd * 50.0), 12.0);
    float ouro = step(0.9, cor.r) * step(cor.b, 0.3);
    float boca = 1.0 + ouro * uFala * (0.7 + 0.6 * sin(uTime * 9.0 + sd * 3.0));   // falando: o dourado pulsa
    vCol = cor * b * mix(0.28, 1.15, k) * tw * boca * 0.42 * uExpo;
    gl_Position = projectionMatrix * mv;
    float persp = uScale / -mv.z;
    gl_PointSize = uSize * aInf.z * mix(0.7, 1.2, k) * persp * (1.0 + 0.4 * (tw - 1.0));
    return p;
  }`;
const VS_BUSTO = VS_COMUM + `
  attribute vec3 aNor;
  void main() {
    // grão visto de frente apaga, de raspão acende: a borda e o volume saem da luz, não de um contorno
    float luz = 0.8, branco = 0.0;
    if (dot(aNor, aNor) > 0.25) {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vec3 nv = normalize(normalMatrix * aNor);
      float fd = dot(nv, normalize(-mv.xyz));
      float de = pow(abs(fd), 0.55);
      luz = mix(2.0, 0.22, de);                // luz de recorte: só o raspão acende de verdade
      luz = mix(luz, mix(1.4, 0.6, de), step(0.5, aInf.x));    // contas: menos contraste (a foto tem pontos claros no meio do rosto)
      luz *= 0.85 + 0.45 * max(0.0, nv.y);     // luz de cima: alto da cabeça, testa e topo dos ombros mais claros, como na foto
      if (fd < 0.0) luz *= 0.12;              // o lado de trás quase some: lê como superfície, não como nuvem
      float ouroG = step(0.9, color.r) * step(color.b, 0.3) * step(0.0, fd);
      luz = mix(luz, 1.1, ouroG);              // o dourado é luz própria do rosto: não apaga de frente
      branco = smoothstep(0.5, 0.05, fd) * step(0.0, fd) * (1.0 - ouroG) * 0.8;     // a borda da foto é quase branca
    }
    fimDoPonto(position, aInf.y, aInf.x * luz, mix(color, vec3(0.72, 0.9, 1.0), branco));
  }`;
const VS_BRACOS = VS_COMUM + `
  uniform float uGal, uTilt, uR0, uR1, uGiro;
  uniform vec3 uCentro;
  void main() {
    // nasce do crânio pra fora e recolhe de fora pra dentro: cada grão tem seu próprio limiar pelo raio
    float u = (position.y - uR0) / (uR1 - uR0);
    float g = clamp(uGal * 1.6 - u * 0.6, 0.0, 1.0);
    g = g * g * (3.0 - 2.0 * g);
    float r = mix(uR0 * 0.7, position.y, g);
    float th = position.x + uTime * uGiro + 0.08 * sin(uTime * 0.3 + position.y * 0.02);
    vec3 d = vec3(cos(th) * r, position.z, sin(th) * r);
    float ct = cos(uTilt), st = sin(uTilt);
    vec3 p = vec3(d.x, d.y * ct - d.z * st, d.y * st + d.z * ct) * 0.01 + uCentro;
    fimDoPonto(p, aInf.y, aInf.x * g, color);
    if (g <= 0.001) gl_PointSize = 0.0;
  }`;
const VS_COMETAS = VS_COMUM + `
  uniform vec4 uCometa[${SLOTS}];             // início, fim (ou -1), cor, inclinação
  uniform vec3 uCores[${CORES_COMETA.length}];
  uniform vec3 uCentro;
  void main() {
    int s = int(position.x + 0.5);
    vec4 c = vec4(0.0);
    for (int i = 0; i < ${SLOTS}; i++) if (i == s) c = uCometa[i];
    float lag = position.y, sd = position.z;
    float tt = uTime - c.x - lag * 0.3;
    float ativo = step(0.0, c.x) * step(0.0, tt);
    float fim = c.y < 0.0 ? 0.0 : clamp((uTime - c.y - lag * 0.3) / 0.8, 0.0, 1.0);
    float entra = smoothstep(0.0, 0.9, tt);
    float r = mix(5.2, 2.55, entra) + fim * 2.6;                     // do disco pra perto da cabeça e de volta
    float th = c.w * 6.2831 + tt * 1.9;
    vec3 d = vec3(cos(th) * r, (sd - 0.5) * 0.08 * (0.3 + lag), sin(th) * r);
    float inc = 0.35 + 0.5 * fract(c.w * 7.0);
    float ct = cos(inc), st = sin(inc);
    vec3 p = vec3(d.x, d.y * ct - d.z * st, d.y * st + d.z * ct) + uCentro;
    fimDoPonto(p, sd, aInf.x * ativo * (1.0 - fim) * 1.8, color);
    vCol *= uCores[int(c.z + 0.5)];            // color é branco nos cometas: a cor vem da ferramenta
    if (ativo * (1.0 - fim) <= 0.001) gl_PointSize = 0.0;
  }`;
const FS = `
  varying vec3 vCol;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = length(c) * 2.0;                             // ponto de luz redondo: miolo nítido, borda curta
    if (r >= 1.0) discard;
    float a = smoothstep(1.0, 0.35, r);
    gl_FragColor = vec4(vCol * a * a * 1.5, 1.0);
  }`;

function pontos(dados, material) {
  const g = new THREE.BufferGeometry();
  if (dados.n !== undefined) {                // busto: arrays tipados, cor e normal em bytes normalizados
    const k = dados.n * 3;
    g.setAttribute('position', new THREE.BufferAttribute(dados.pos.subarray(0, k), 3));
    g.setAttribute('color', new THREE.BufferAttribute(dados.col.subarray(0, k), 3, true));
    g.setAttribute('aInf', new THREE.BufferAttribute(dados.inf.subarray(0, k), 3));
    g.setAttribute('aNor', new THREE.BufferAttribute(dados.nor.subarray(0, k), 3, true));
  } else {
    g.setAttribute('position', new THREE.Float32BufferAttribute(dados.pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(dados.col, 3));
    g.setAttribute('aInf', new THREE.Float32BufferAttribute(dados.inf, 3));
  }
  const p = new THREE.Points(g, material);
  p.frustumCulled = false;                    // braços e cometas guardam parâmetros em position, não xyz
  return p;
}

/* ---------- cena ---------- */
export function montar(el, { densidade = 1 } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  const pr = Math.min(2, window.devicePixelRatio || 1);
  renderer.setPixelRatio(pr);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x010309);
  const group = new THREE.Group();
  scene.add(group);
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 200);
  camera.position.set(0, ALVO_Y, DIST);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, ALVO_Y, 0);
  controls.minDistance = 14;
  controls.maxDistance = 70;

  const uniforms = {
    uTime: { value: 0 }, uSize: { value: 2.6 * pr }, uC: { value: 0 }, uScale: { value: DIST },
    uPensa: { value: 0 }, uFala: { value: 0 }, uExpo: { value: 1 },
  };
  const material = (vs, extra = {}) => new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    uniforms: { ...uniforms, ...extra }, vertexShader: vs, fragmentShader: FS,
  });
  group.add(pontos(gerarBusto(densidade), material(VS_BUSTO)));

  const centro = { value: CABECA.clone() };
  const galaxia = material(VS_BRACOS, {
    uGal: { value: 0 }, uTilt: { value: 0.42 }, uR0: { value: 235 }, uR1: { value: 680 }, uGiro: { value: 0.12 }, uCentro: centro,
  });
  group.add(pontos(gerarBracos(Math.round(16000 * densidade), 235, 680, 11), galaxia));
  const satelite = material(VS_BRACOS, {
    uGal: { value: 0 }, uTilt: { value: -0.75 }, uR0: { value: 250 }, uR1: { value: 430 }, uGiro: { value: -0.2 }, uCentro: centro,
  });
  group.add(pontos(gerarBracos(Math.round(5000 * densidade), 250, 430, 23), satelite));
  const cometa = Array.from({ length: SLOTS }, () => new THREE.Vector4(-1, -1, 0, 0));
  const cometas = material(VS_COMETAS, {
    uCometa: { value: cometa }, uCores: { value: CORES_COMETA.map((c) => new THREE.Vector3(...c)) }, uCentro: centro,
  });
  group.add(pontos(gerarCometas(), cometas));

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.38, 0.05, 0.3);
  composer.addPass(bloom);
  const ajustar = () => {
    const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    // tela estreita (celular em pé): afasta a câmera pra caber cabeça e ombros, mantendo a direção do giro
    const dist = DIST * Math.max(1, 1.35 / camera.aspect);   // em tela estreita afasta até caber os ombros
    camera.position.sub(controls.target).setLength(dist).add(controls.target);
    // a figura ocupa ~metade da altura; com a contagem de grãos fixa, a sobreposição cresce com 1/altura²
    uniforms.uExpo.value = Math.min(1.2, Math.max(0.2, (h * DIST / dist / 820) ** 2));
  };
  const ro = new ResizeObserver(ajustar);
  ro.observe(el);
  ajustar();

  // estado vivo: o redutor decide o alvo; molas suavizam (nada salta)
  const t0 = performance.now(), agora = () => (performance.now() - t0) / 1000;
  let estado = estadoInicial(), gal = 0, sat = 0, pensa = 0, fala = 0, subiuEm = -1;
  const slotDe = {};                          // id da ferramenta → slot do cometa
  function ocuparSlot(id, t) {
    let livre = cometa.findIndex((c) => c.x < 0 || (c.y >= 0 && t - c.y > 1.2));
    if (livre < 0) livre = cometa.reduce((m, c, i) => (c.x < cometa[m].x ? i : m), 0);
    for (const k of Object.keys(slotDe)) if (slotDe[k] === livre) delete slotDe[k];
    slotDe[id] = livre;
    return livre;
  }
  function evento(ev) {
    const t = agora(), antes = estado;
    estado = reduzir(estado, ev, t);
    const id = ev && (ev.id || ev.ferramenta);
    if (ev && ev.tipo === 'ferramenta' && id) {
      const s = ocuparSlot(id, t);
      cometa[s].set(t, -1, corDaFerramenta(ev.ferramenta || ''), Math.random());
    }
    for (const [k, f] of Object.entries(estado.ferramentas)) {   // fins (inclusive os que o "pronto" fechou)
      const s = slotDe[k];
      if (s !== undefined && f.fim !== null && !(antes.ferramentas[k] && antes.ferramentas[k].fim !== null)) {
        cometa[s].y = f.fim;
        if (f.erro) cometa[s].z = COR_ERRO;
      }
    }
    if (ev && ev.tipo === 'pronto') estado = { ...estado, ferramentas: {} };
  }

  let quadro = 0, vivo = true, ultimo = 0;
  const alvoCamera = new THREE.Vector3();
  function loop() {
    if (!vivo) return;
    quadro = requestAnimationFrame(loop);
    const t = agora(), dt = Math.min(0.1, t - ultimo);
    ultimo = t;
    let alvo = alvoGalaxia(estado, t);
    if (alvo > 0 && subiuEm < 0) subiuEm = t;
    if (alvo === 0 && subiuEm >= 0 && t - subiuEm < 1.2 && !estado.falando) alvo = 1;   // segura no mínimo 1,2 s
    if (alvo === 0) subiuEm = -1;
    gal += (alvo - gal) * Math.min(1, dt * (alvo > gal ? 1.4 : 0.9));
    sat += ((estado.subagentes > 0 ? 1 : 0) - sat) * Math.min(1, dt * 1.2);
    pensa += ((estado.pensando ? 1 : 0) - pensa) * Math.min(1, dt * 2);
    fala += ((estado.falando ? 1 : 0) - fala) * Math.min(1, dt * 3);
    galaxia.uniforms.uGal.value = gal;
    satelite.uniforms.uGal.value = sat;
    uniforms.uTime.value = t;
    uniforms.uPensa.value = pensa;
    uniforms.uFala.value = fala;
    group.rotation.y = 0.1 * Math.sin(t * 0.21);                 // respiração do corpo
    controls.update();
    uniforms.uC.value = alvoCamera.copy(controls.target).applyMatrix4(camera.matrixWorldInverse).z;
    composer.render();
  }
  loop();

  return {
    evento,
    camera, controls,                         // pra integrações e testes (enquadrar, girar a vista)
    get estado() { return estado; },
    destruir() {
      vivo = false;
      cancelAnimationFrame(quadro);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
      composer.dispose?.();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
