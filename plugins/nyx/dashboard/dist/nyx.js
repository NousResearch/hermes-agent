// Nyx — avatar de partículas do Hermes (módulo ES carregado pelo painel do dashboard).
//
// Em repouso só o busto: ~3 milhões de pontos de luz em pontilhismo, opaco, contorno só por pontos.
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
import { criarPreenchimento } from './preenchimento.js';

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

// row() pré-calculado a cada 1/4 px: com milhões de pontos a interpolação de Catmull-Rom vira o gargalo.
// A tabela é medida, então tem degraus de ~1 px a cada linha; a média móvel de 10 px (uma linha da tabela)
// tira esse ruído, que no pontilhismo vira listras (a densidade de pontos segue a inclinação da superfície).
// Devolve sempre o mesmo array [w, d, c]: quem chama copia os valores antes da próxima chamada.
const LUTS = new Map(), PASSO = 4, _r = [0, 0, 0];
function rowR(T, y) {
  let L = LUTS.get(T);
  if (!L) {
    const y0 = T[0][0], n = (T[T.length - 1][0] - y0) * PASSO + 1, w = new Float32Array(n), d = new Float32Array(n), c = new Float32Array(n);
    const bw = new Float32Array(n), bd = new Float32Array(n), bc = new Float32Array(n);
    for (let i = 0; i < n; i++) { const r = row(T, y0 + i / PASSO); bw[i] = r[0]; bd[i] = r[1]; bc[i] = r[2]; }
    for (let i = 0; i < n; i++) {
      const R = Math.min(20, i, n - 1 - i);      // janela simétrica que encolhe nas pontas: o alto da cabeça e o queixo ficam no lugar
      let sw = 0, sd = 0, sc = 0;
      for (let j = i - R; j <= i + R; j++) { sw += bw[j]; sd += bd[j]; sc += bc[j]; }
      w[i] = sw / (2 * R + 1); d[i] = sd / (2 * R + 1); c[i] = sc / (2 * R + 1);
    }
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
// ponto da casca e a normal de verdade: inclui a inclinação vertical (topo dos ombros, queixo, alto da
// cabeça), senão uma rampa acende como se estivesse de frente pra câmera. sy e sp: quanto a superfície
// anda por px de y e por radiano de φ (as correntinhas andam com passo constante na superfície).
// Devolve sempre o mesmo objeto: quem chama copia os valores antes da próxima chamada.
const P = { x: 0, z: 0, dy: 0, nx: 0, ny: 0, nz: 0, sy: 1, sp: 1 };
function naCasca(T, y, ph, t) {
  let r = rowR(T, y + 1); const w1 = r[0], d1 = r[1], c1 = r[2];
  r = rowR(T, y - 1); const w0 = r[0], d0 = r[1], c0 = r[2];
  r = rowR(T, y); const w = r[0], d = r[1], c = r[2];
  const sn = Math.sin(ph), cs = Math.cos(ph);
  const ayx = (w1 - w0) / 2 * sn, ayz = (c1 - c0) / 2 + (d1 - d0) / 2 * cs;   // ∂P/∂y (y da cena = -y da foto)
  const apx = w * cs, apz = -d * sn;                                         // ∂P/∂φ
  let nx = -apz, ny = ayz * apx - ayx * apz, nz = apx;                       // ∂P/∂y × ∂P/∂φ
  const nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;   // sqrt em vez de hypot: aqui roda milhões de vezes
  nx /= nl; ny /= nl; nz /= nl;
  if (nx * sn * w + nz * cs * d < 0) { nx = -nx; ny = -ny; nz = -nz; }    // pra fora
  P.x = w * sn + nx * t; P.z = c + d * cs + nz * t; P.dy = -ny * t; P.nx = nx; P.ny = ny; P.nz = nz;
  P.sy = Math.sqrt(1 + ayx * ayx + ayz * ayz); P.sp = Math.max(6, Math.sqrt(apx * apx + apz * apz));
  return P;
}
// φ com a mesma densidade por área: nos ombros (largos e rasos) φ uniforme amontoaria pontos nas pontas
function sortearPh(rnd, T, y) {
  const [w, d] = rowR(T, y), m = Math.max(w, d);
  for (;;) { const ph = rnd() * 6.283185, a = w * Math.cos(ph), b = d * Math.sin(ph); if ((rnd() * m) ** 2 < a * a + b * b) return ph; }
}
const corDoPonto = (rnd) => { const r = rnd(); return r < 0.12 ? DEEP : r < 0.67 ? BLUE : r < 0.95 ? CYAN : WHITE; };

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
const MARCAS = [                              // [tabela, curva, pontos a mais por px de curva]
  [HEAD, QUEIXO, 14],                                                                    // contorno do queixo
  [BODY, [[957, 600], [956, 700], [957, 780], [961, 800], [973, 820], [995, 840], [1003, 846]], 16],   // lado do pescoço → trapézio
  [BODY, [[878, 662], [872, 720], [862, 780], [850, 840], [838, 892]], 8],               // esternocleidomastoide até o esterno
  [BODY, [[930, 849], [970, 851], [1003, 846], [1052, 840], [1100, 828], [1150, 820], [1195, 823]], 16],  // clavícula
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
  if (Math.abs(x) < 130) {
    const dq = y - queixoY(x);
    if (dq > 0 && dq < 50) s *= 0.15 + 0.85 * suave(4, 50, dq);
  }
  if (y > 650 && y < 880) {
    const meia = 40 - 28 * (y - 650) / 230;      // o V: ~40 px de meia-largura sob o queixo, fecha no esterno
    const forca = 0.55 * (1 - suave(780, 880, y));   // e some aos poucos, senão vira um risco escuro no peito
    s *= 1 - forca * (1 - suave(meia * 0.55, meia * 1.2, Math.abs(x)));
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
// presa no crânio; de cima até o lóbulo a hélice abre até a medida da foto. O perfil em r dá o relevo: a
// concha fica rente ao crânio, a borda sobe rápido (a hélice salta) e a antélice é uma crista no meio.
function naOrelha(sg, t, r) {
  const O = ORELHA, sn = Math.sin(t), cs = Math.cos(t);
  const fundo = O.meiaFundo * (sn < 0 ? 1 + 0.35 * sn : 1);   // de perfil o lóbulo é mais estreito que o alto
  const hy = O.cy - sn * O.meiaAltura - 5 * cs, hz = O.cz - cs * fundo;
  const abre = Math.max(0, extOrelha(hy) - cranio(hy, hz)) * suave(-0.85, -0.05, cs);
  const ey = O.cy + (hy - O.cy) * r, ez = O.cz + (hz - O.cz) * r;
  const antelice = 5 * Math.exp(-(((r - 0.68) / 0.08) ** 2)) * suave(-0.5, 0.2, cs);
  return [sg * (cranio(ey, ez) + abre * (0.15 * r + 0.85 * r ** 3) + antelice), ey, ez];
}

// O busto é pontilhismo: milhões de pontos na superfície, espalhados por igual (faixa a faixa de y, em
// intervalos de arco de mesma área, com sorteio dentro de cada intervalo), e o shader decide quais aparecem
// pela luz do lugar (VS_BUSTO). O busto é opaco: uma máscara de profundidade (gerarMalha) esconde o que
// está atrás da pele. Por cima, correntinhas de
// contas (os riscos pontilhados da foto) e as marcas do pescoço como faixas um pouco mais densas.
// Cada ponto guarda a normal da superfície; cabeça e pescoço não se atravessam.
// Tudo vai direto pra arrays tipados (posição, cor em bytes, brilho/semente/tamanho, normal em bytes).
const PONTOS = 3000000, CONTAS = 45000, VAO_CONTA = 5.5;
function gerarBusto(densidade) {
  const { rnd, gauss } = criarAleatorio(7);
  const cap = Math.ceil((PONTOS + CONTAS + 120000) * densidade) + 60000;
  // 3 milhões de pontos: tudo que dá vai em bytes (cor, normal e [brilho/2, semente, tamanho/1,5], que o
  // shader desfaz com uInf); só a posição fica em float. ~21 bytes por ponto
  const pos = new Float32Array(cap * 3), col = new Uint8Array(cap * 3), inf = new Uint8Array(cap * 3), nor = new Int8Array(cap * 3);
  let n = 0;
  const ouro = (x, y, z) => (z > 40 && y > 400 && y < 700 ? Math.exp(-((x / 46) ** 2 + ((y - 548) / 50) ** 2)) : 0);   // frente baixa do rosto
  const put = (x, y, z, nx, ny, nz, b, sz, cor) => {
    if (n >= cap) return;
    const i = n * 3;
    pos[i] = x * U; pos[i + 1] = -(y - Y0) * U; pos[i + 2] = z * U;
    col[i] = cor[0] * 255; col[i + 1] = cor[1] * 255; col[i + 2] = cor[2] * 255;
    inf[i] = Math.min(255, b * 127.5); inf[i + 1] = rnd() * 255; inf[i + 2] = Math.min(255, sz * 170);
    nor[i] = nx * 127; nor[i + 1] = ny * 127; nor[i + 2] = nz * 127;
    n++;
  };
  const fadeY = (y) => (y < 932 ? 1 : Math.max(0, 1 - (y - 932) / 9));   // só a borda de baixo se desfaz
  const corPonto = () => corDoPonto(rnd);
  // o fundo: faixas de 0,5 px de y; em cada uma, o perímetro em k intervalos de mesmo arco, um ponto sorteado
  // dentro de cada (e a origem do arco gira de faixa pra faixa), então não há bolos nem buracos
  const pontilhar = (T, total, fora, sombra, ganho) => {
    const ya = T === HEAD ? 143 : 520, yb = T === HEAD ? 649 : 940, H = 0.5, M = 96;
    const areas = [];
    let soma = 0;
    for (let y = ya; y < yb; y += H) {
      const [w, d] = rowR(T, y + H / 2), [wa, da, ca] = rowR(T, y - 0.5), [wb, db, cb] = rowR(T, y + H + 0.5);
      const dw = (wb - wa) / (H + 1), dd = (db - da) / (H + 1), dc = (cb - ca) / (H + 1);
      const a = Math.PI * (3 * (w + d) - Math.sqrt((3 * w + d) * (w + 3 * d))) * Math.sqrt(1 + 0.5 * (dw * dw + dd * dd) + dc * dc) * H;
      areas.push(a); soma += a;
    }
    const arco = new Float64Array(M + 1);
    let sobra = 0;
    for (let f = 0; f < areas.length; f++) {
      const y0 = ya + f * H, quer = areas[f] / soma * total * densidade + sobra, k0 = Math.floor(quer);
      sobra = quer - k0;
      if (!k0) continue;
      const [w, d] = rowR(T, y0 + H / 2);
      for (let j = 1; j <= M; j++) {
        const ph = (j - 0.5) / M * 6.283185, a = w * Math.cos(ph), b = d * Math.sin(ph);
        arco[j] = arco[j - 1] + Math.sqrt(a * a + b * b) * 6.283185 / M;
      }
      const per = arco[M], giro = rnd(), yc = y0 + H / 2;
      let r = rowR(T, yc + 1); const w1 = r[0], d1 = r[1], c1 = r[2];
      r = rowR(T, yc - 1); const w0 = r[0], d0 = r[1], c0 = r[2];
      r = rowR(T, yc); const c = r[2];
      const dw = (w1 - w0) / 2, dd = (d1 - d0) / 2, dc = (c1 - c0) / 2;
      // quanto a superfície estica por px de y depende de φ: onde os ombros abrem, estica nos lados e não no
      // meio do peito. Sorteia a mais e aceita cada ponto pelo esticão local, senão o peito ganha uma faixa clara
      const estMedia = Math.sqrt(1 + 0.5 * (dw * dw + dd * dd) + dc * dc), estMax = Math.sqrt(1 + dw * dw + (Math.abs(dc) + Math.abs(dd)) ** 2);
      const k = Math.round(k0 * estMax / estMedia);
      for (let i = 0, j = 0; i < k; i++) {
        let u = (i + rnd()) / k + giro;
        if (u >= 1) u -= 1;
        const sArc = u * per;
        while (j > 0 && arco[j] > sArc) j--;
        while (j < M - 1 && arco[j + 1] < sArc) j++;
        const ph = (j + (sArc - arco[j]) / (arco[j + 1] - arco[j] || 1)) * 6.283185 / M, y = y0 + rnd() * H;
        // a casca e a normal, como em naCasca, mas com a linha da faixa já calculada (aqui roda milhões de vezes)
        const sn = Math.sin(ph), cs = Math.cos(ph), ayz = dc + dd * cs, apx = w * cs, apz = -d * sn, ayx = dw * sn;
        if (rnd() * estMax > Math.sqrt(1 + ayx * ayx + ayz * ayz)) continue;
        let nx = -apz, ny = ayz * apx - dw * sn * apz, nz = apx;
        const nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1, t = -0.5 + (rnd() - rnd()) * 1.4;
        nx /= nl; ny /= nl; nz /= nl;
        if (nx * sn * w + nz * cs * d < 0) { nx = -nx; ny = -ny; nz = -nz; }
        const px = w * sn + nx * t, pz = c + d * cs + nz * t;
        if (fora(px, y, pz) || (y > 932 && rnd() > fadeY(y)) || rnd() > sombra(px, y, pz)) continue;
        const g = ouro(px, y, pz), dourado = g > 0 && rnd() < g;
        put(px, y - ny * t, pz, nx, ny, nz, (0.7 + 0.3 * rnd()) * (dourado ? 0.8 : ganho), 0.45 + 0.15 * rnd(), dourado ? GOLD : corPonto());   // tamanho < 0,75: o shader lê ≥ 0,75 como conta
      }
    }
    return total * densidade / soma;
  };
  // ocupação em voxels de 5 px: cada correntinha nasce onde a vizinhança (~15 px) está mais vazia entre 8
  // candidatos, então elas se espalham por igual
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
  // correntinha: contas a cada ~5,5 px por um caminho quase reto que de vez em quando dobra seco (só as
  // contas: não há linha entre elas)
  const correntes = (T, total, fora, sombra, ganho) => {
    const pick = sampler(rnd, T, T === HEAD ? 143 : 520, T === HEAD ? 649 : 940), yMin = T[0][0] + 1, yMax = T[T.length - 1][0], PASSO = 0.5;
    for (let feitas = 0; feitas < total * densidade;) {
      let y = 0, ph = 0, menor = 1e9, dir = rnd() * 6.283185;
      for (let c = 0; c < 8; c++) {
        const yc = pick(), pc = sortearPh(rnd, T, yc), q = naCasca(T, yc, pc, 0), oc = vizinhos(q.x, yc, q.z);
        if (oc < menor) { menor = oc; y = yc; ph = pc; }
      }
      const L = 18 + 30 * rnd(), t0 = -0.5 + gauss() * 0.8;
      let conta = rnd() * VAO_CONTA, feitasAqui = 0;
      for (let s = 0; s < L; s += PASSO) {
        if (y < yMin || y > yMax) break;
        const p = naCasca(T, y, ph, t0);
        const sy = p.sy, sp = p.sp;
        if (!fora(p.x, y, p.z)) {
          const v = voxel(p.x, y, p.z);
          if (v >= 0 && ocup[v] < 255) ocup[v]++;
          if (s >= conta && rnd() < fadeY(y) * sombra(p.x, y, p.z)) {
            conta += VAO_CONTA * (0.85 + 0.3 * rnd());
            const g = ouro(p.x, y, p.z);
            put(p.x, y + p.dy, p.z, p.nx, p.ny, p.nz, (0.9 + 0.3 * rnd()) * ganho, 0.8 + 0.15 * rnd(), rnd() < g * 0.4 ? GOLD : rnd() < 0.5 ? WHITE : CYAN);
            feitasAqui++;
          }
        }
        dir += gauss() * 0.04 + (rnd() < 0.045 ? (rnd() < 0.5 ? -1 : 1) * (0.6 + 1.0 * rnd()) : 0);
        y += Math.sin(dir) * PASSO / sy;
        ph += Math.cos(dir) * PASSO / sp;
      }
      feitas += Math.max(1, feitasAqui);
    }
  };
  const semSombra = () => 1;
  const porPx2 = pontilhar(HEAD, PONTOS * 0.4, insideBody, semSombra, 1);
  pontilhar(BODY, PONTOS * 0.6, insideHead, sombraPescoco, 1.3);   // na foto pescoço e ombros brilham mais que o rosto
  correntes(HEAD, CONTAS * 0.42, insideBody, semSombra, 1);
  correntes(BODY, CONTAS * 0.58, insideHead, sombraPescoco, 1.1);
  // marcas da foto, medidas como cristas de brilho (lado direito, px da foto; o esquerdo é o espelho): uma
  // faixa de ~5 px com mais pontos ao longo da curva, que passam pelo mesmo filtro de luz do resto: a marca
  // aparece só como um adensamento discreto, não como linha
  for (const [T, curva, porPx] of MARCAS) {
    for (const sg of [-1, 1]) {
      for (let i = 0; i < curva.length - 1; i++) {
        const [x0, y0] = curva[i], [x1, y1] = curva[i + 1], len = Math.hypot(x1 - x0, y1 - y0);
        const ux = -(y1 - y0) / len, uy = (x1 - x0) / len;
        for (let k = 0; k < len * porPx * densidade; k++) {
          const t = rnd(), o = gauss() * 2.2;
          const xp = (x0 + (x1 - x0) * t + ux * o - AX) * sg, y = y0 + (y1 - y0) * t + uy * o;
          const w = rowR(T, y)[0], ph = Math.asin(Math.max(-1, Math.min(1, xp / Math.max(1, w))));
          const p = naCasca(T, y, ph, -0.5 + gauss() * 0.6);
          put(p.x, y + p.dy, p.z, p.nx, p.ny, p.nz, 0.7 + 0.3 * rnd(), 0.45 + 0.15 * rnd(), rnd() < 0.7 ? CYAN : BLUE);
        }
      }
    }
  }
  // orelhas: superfície de verdade, com a mesma densidade de pontos da pele e normais reais, então o mesmo
  // filtro de luz vale pra elas (a orelha do lado de lá some atrás da cabeça, o contorno sai do acúmulo).
  // As duas faces da placa (de fora e a que olha pro crânio) e, na borda, a hélice como um tubinho de ~3,5 px.
  const derivadas = (sg, t, r) => {
    const p = naOrelha(sg, t, r), a = naOrelha(sg, t + 1e-3, r), b = naOrelha(sg, t, r + 1e-3);
    const pt = [(a[0] - p[0]) / 1e-3, (a[1] - p[1]) / 1e-3, (a[2] - p[2]) / 1e-3], pr = [(b[0] - p[0]) / 1e-3, (b[1] - p[1]) / 1e-3, (b[2] - p[2]) / 1e-3];
    let nx = pt[1] * pr[2] - pt[2] * pr[1], ny = pt[2] * pr[0] - pt[0] * pr[2], nz = pt[0] * pr[1] - pt[1] * pr[0];
    const area = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    nx /= area; ny /= area; nz /= area;
    if (nx * sg < 0) { nx = -nx; ny = -ny; nz = -nz; }   // face de fora: aponta pra longe da cabeça
    return { p, pt, pr, nx, ny, nz, area };              // normal em px da foto (y pra baixo)
  };
  const pontoOrelha = (x, y, z, nx, nyFoto, nz) => put(x, y, z, nx, -nyFoto, nz, 1.1 + 0.4 * rnd(), 0.45 + 0.15 * rnd(), corPonto());
  const densOrelha = porPx2 * 1.8;   // a orelha é pequena e quase sempre vista de raspão: mais pontos que a pele
  for (const sg of [-1, 1]) {
    // área da placa (r de 0,05 a 0,92) por amostragem, pra saber quantos pontos cabem e aceitar por área
    let somaA = 0, maxA = 0;
    for (let i = 0; i < 4000; i++) { const d = derivadas(sg, (rnd() * 2 - 1) * Math.PI, 0.05 + 0.87 * rnd()); somaA += d.area; maxA = Math.max(maxA, d.area); }
    const areaPlaca = somaA / 4000 * 2 * Math.PI * 0.87;
    for (const face of [1, -1]) {
      for (let k = 0, quer = Math.round(areaPlaca * densOrelha); k < quer;) {
        const r = 0.05 + 0.87 * rnd(), d = derivadas(sg, (rnd() * 2 - 1) * Math.PI, r);
        if (rnd() * maxA > d.area) continue;
        k++;
        // concha e escafa (o sulco atrás da hélice) ficam na sombra; a antélice entre elas, na luz. De frente o
        // que se vê é a parte que abre (r > 0,7): a hélice clara por fora e o sulco escuro, o "C" da foto
        const luzOrelha = (0.3 + 0.7 * suave(0.45, 0.65, r)) * (1 - 0.65 * suave(0.74, 0.8, r) * (1 - suave(0.88, 0.93, r)));
        if (rnd() > luzOrelha) continue;
        pontoOrelha(d.p[0] + d.nx * face, d.p[1] + d.ny * face, d.p[2] + d.nz * face, d.nx * face, d.ny * face, d.nz * face);
      }
    }
    // hélice: tubo em volta da borda (r ≈ 0,96), da frente de cima até o lóbulo
    const ta = -0.7 * Math.PI, tb = 0.75 * Math.PI, R = 3.5, densHelice = densOrelha * 1.3;
    let compr = 0;
    for (let i = 0; i < 200; i++) { const d = derivadas(sg, ta + (tb - ta) * (i + 0.5) / 200, 0.96); compr += Math.hypot(d.pt[0], d.pt[1], d.pt[2]) * (tb - ta) / 200; }
    for (let k = 0, quer = Math.round(compr * 2 * Math.PI * R * densHelice); k < quer; k++) {
      const d = derivadas(sg, ta + (tb - ta) * rnd(), 0.96), al = rnd() * 6.283185;
      const rl = Math.hypot(d.pr[0], d.pr[1], d.pr[2]) || 1, rx = d.pr[0] / rl, ry = d.pr[1] / rl, rz = d.pr[2] / rl;   // pra fora, no plano da orelha
      const nx = Math.cos(al) * d.nx + Math.sin(al) * rx, ny = Math.cos(al) * d.ny + Math.sin(al) * ry, nz = Math.cos(al) * d.nz + Math.sin(al) * rz;
      pontoOrelha(d.p[0] + nx * R, d.p[1] + ny * R, d.p[2] + nz * R, nx, ny, nz);
    }
  }
  return { n, pos, col, inf, nor };
}

// A forma do busto como malha: cabeça, corpo e as placas das orelhas.
//  - máscara (modo partículas): recuada 2,5 px pra dentro da pele e sem cor; só grava a profundidade, então os
//    pontos atrás dela (o outro lado da cabeça, a orelha de lá) não são desenhados e o rosto fica opaco. Os
//    pontos da pele ficam todos na frente dela (o mais fundo está a 1,9 px da superfície);
//  - guia (modo preenchimento): na pele, com normais, o tubo da hélice e o código da parte (aParte), que o
//    passo de preenchimento lê pra pintar os pontos.
// aParte: 1 cabeça · 0,875 corpo · 0,75 hélice · 0,02 + 0,48·r placa da orelha (r = 0 no crânio, 0,96 na borda)
const PARTE = { cabeca: 1, corpo: 0.875, helice: 0.75 };
function normalOrelha(sg, t, r) {             // px da foto (y pra baixo), apontando pra fora da cabeça
  const p = naOrelha(sg, t, r), a = naOrelha(sg, t + 1e-3, r), b = naOrelha(sg, t, r + 1e-3);
  const pt = [a[0] - p[0], a[1] - p[1], a[2] - p[2]], pr = [b[0] - p[0], b[1] - p[1], b[2] - p[2]];
  let nx = pt[1] * pr[2] - pt[2] * pr[1], ny = pt[2] * pr[0] - pt[0] * pr[2], nz = pt[0] * pr[1] - pt[1] * pr[0];
  const nl = Math.hypot(nx, ny, nz);
  if (nl < 1e-9) return [sg, 0, 0];
  if (nx * sg < 0) { nx = -nx; ny = -ny; nz = -nz; }
  return [nx / nl, ny / nl, nz / nl];
}
function gerarMalha({ recuo = 2.5, guia = false } = {}) {
  const pos = [], nor = [], parte = [], idx = [];
  const NS = guia ? 160 : 96, PY = guia ? 1.5 : 2;
  const emCena = (x, y, z, nx = 0, ny = 0, nz = 0, pt = 0) => {   // normal em px da foto: y da cena é -y
    pos.push(x * U, -(y - Y0) * U, z * U); nor.push(nx, -ny, nz); parte.push(pt);
  };
  const casca = (T, ya, yb, pt) => {
    const base = pos.length / 3;
    const linhas = Math.ceil((yb - ya) / PY) + 1;      // espaçadas por igual, a última exatamente em yb (a ponta do queixo)
    for (let i = 0; i < linhas; i++) {
      const y = ya + (yb - ya) * i / (linhas - 1);
      let r = rowR(T, y); const w = r[0], d = r[1], c = r[2];
      r = rowR(T, y - 0.5); const wa = r[0], da = r[1], ca = r[2];
      r = rowR(T, y + 0.5); const wb = r[0], db = r[1], cb = r[2];
      for (let j = 0; j < NS; j++) {
        const ph = j / NS * 6.283185, sn = Math.sin(ph), cs = Math.cos(ph);
        const dx = (wb - wa) * sn, dz = (cb - ca) + (db - da) * cs;      // ∂P/∂y (px da foto, y pra baixo)
        // normal pra fora = ∂P/∂φ × ∂P/∂y
        let nx = d * sn, ny = -d * sn * dx - w * cs * dz, nz = w * cs;
        const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (nl < 1e-6) { nx = 0; ny = -1; nz = 0; } else { nx /= nl; ny /= nl; nz /= nl; }
        emCena(w * sn - nx * recuo, y - ny * recuo, c + d * cs - nz * recuo, nx, ny, nz, pt);
      }
    }
    for (let i = 0; i < linhas - 1; i++) for (let j = 0; j < NS; j++) {
      const a = base + i * NS + j, b = base + i * NS + (j + 1) % NS;
      idx.push(a, a + NS, b, b, a + NS, b + NS);
    }
  };
  casca(HEAD, 141, 649, PARTE.cabeca);
  casca(BODY, 520, 940, PARTE.corpo);
  // orelhas: a própria placa, até o meio do tubo da hélice (cada face da orelha esconde a de trás)
  const NT = 64, NR = guia ? 16 : 10;
  for (const sg of [-1, 1]) {
    let base = pos.length / 3;
    for (let i = 0; i <= NR; i++) for (let j = 0; j < NT; j++) {
      const t = (j / NT * 2 - 1) * Math.PI, r = i / NR * 0.96, [x, y, z] = naOrelha(sg, t, r);
      const [nx, ny, nz] = guia ? normalOrelha(sg, t, r) : [0, 0, 0];
      emCena(x, y, z, nx, ny, nz, 0.02 + 0.48 * r);
    }
    for (let i = 0; i < NR; i++) for (let j = 0; j < NT; j++) {
      const a = base + i * NT + j, b = base + i * NT + (j + 1) % NT;
      idx.push(a, a + NT, b, b, a + NT, b + NT);
    }
    if (!guia) continue;
    // hélice: tubo de 3,5 px em volta da borda, da frente de cima até o lóbulo
    const ta = -0.7 * Math.PI, tb = 0.75 * Math.PI, R = 3.5, NA = 96, NC = 12;
    base = pos.length / 3;
    for (let i = 0; i <= NA; i++) {
      const t = ta + (tb - ta) * i / NA, c = naOrelha(sg, t, 0.96), e = naOrelha(sg, t, 0.96 + 1e-3);
      const n = normalOrelha(sg, t, 0.96);
      let rx = e[0] - c[0], ry = e[1] - c[1], rz = e[2] - c[2];
      const rl = Math.hypot(rx, ry, rz) || 1;
      rx /= rl; ry /= rl; rz /= rl;
      for (let k = 0; k < NC; k++) {
        const al = k / NC * 6.283185, ca = Math.cos(al), sa = Math.sin(al);
        const nx = ca * n[0] + sa * rx, ny = ca * n[1] + sa * ry, nz = ca * n[2] + sa * rz;
        emCena(c[0] + nx * R, c[1] + ny * R, c[2] + nz * R, nx, ny, nz, PARTE.helice);
      }
    }
    for (let i = 0; i < NA; i++) for (let k = 0; k < NC; k++) {
      const a = base + i * NC + k, b = base + i * NC + (k + 1) % NC;
      idx.push(a, a + NC, b, b, a + NC, b + NC);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  if (guia) {
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    g.setAttribute('aParte', new THREE.Float32BufferAttribute(parte, 1));
  }
  g.setIndex(idx);
  return g;
}

// Franja (modo preenchimento): partículas que saem da pele e rareiam pra fora (~3 px em média, até 14), mais
// fracas quanto mais longe. O shader só mostra as que estão no contorno e faz cada uma subir devagar e sumir:
// é o limite da nuvem de pontos, em vez de um corte seco. Mesmo formato de dados do busto de partículas.
const FRANJA = 380000;
function gerarFranja(densidade) {
  const { rnd } = criarAleatorio(29);
  const cap = Math.ceil(FRANJA * densidade);
  const pos = new Float32Array(cap * 3), col = new Uint8Array(cap * 3), inf = new Uint8Array(cap * 3), nor = new Int8Array(cap * 3);
  const pickH = sampler(rnd, HEAD, 143, 649), pickB = sampler(rnd, BODY, 520, 925);
  let n = 0;
  for (let tentativa = 0; n < cap && tentativa < cap * 4; tentativa++) {
    const cabeca = rnd() < 0.4, T = cabeca ? HEAD : BODY, y = (cabeca ? pickH : pickB)();
    const t = Math.min(14, -Math.log(1 - rnd()) * 3), p = naCasca(T, y, sortearPh(rnd, T, y), t), yp = y + p.dy;
    if ((cabeca ? insideBody : insideHead)(p.x, yp, p.z)) continue;
    const i = n * 3, cor = corDoPonto(rnd), b = 0.75 * Math.exp(-t / 6) * (cabeca ? 1 : 1.3);
    pos[i] = p.x * U; pos[i + 1] = -(yp - Y0) * U; pos[i + 2] = p.z * U;
    col[i] = cor[0] * 255; col[i + 1] = cor[1] * 255; col[i + 2] = cor[2] * 255;
    inf[i] = Math.min(255, b * 127.5); inf[i + 1] = rnd() * 255; inf[i + 2] = Math.min(255, (0.38 + 0.06 * rnd()) * 170);
    nor[i] = p.nx * 127; nor[i + 1] = p.ny * 127; nor[i + 2] = p.nz * 127;   // naCasca já dá a normal com o y da cena
    n++;
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
  uniform vec3 uInf;                          // escala de aInf: o busto guarda em bytes (0..1), o resto em float
  #define INF (aInf * uInf)
  varying vec3 vCol;
  // ponto de tamanho 0 vira ponto de 1 px (o WebGL arredonda pro mínimo): pra sumir de verdade, sai da tela
  void esconder() { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0; vCol = vec3(0.0); }
  // pulsação leve de cada ponto, no ritmo dele (período de ~2,4 a 5 s): brilho ±30% (±45% pensando), tamanho ±12%.
  // Pensar aumenta a amplitude, não a velocidade: mudar a velocidade com o tempo correndo faria a fase saltar
  float pulso(float sd) { return 1.0 + (0.3 + 0.15 * uPensa) * sin(uTime * (1.25 + 1.35 * fract(sd * 7.31)) + sd * 83.0); }
  vec3 fimDoPonto(vec3 p, float sd, float b, vec3 cor) {
    p += vec3(sin(uTime * (0.4 + sd * 0.5) + sd * 40.0), cos(uTime * (0.33 + sd * 0.4) + sd * 27.0), sin(uTime * 0.37 + sd * 19.0)) * 0.012;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float k = smoothstep(-2.3, 2.1, mv.z - uC);                       // 0 = fundo, 1 = mais perto
    float rit = 0.8 + sd * 2.5 + uPensa * 1.5;                          // pensando: cintila mais rápido
    float tw = 1.0 + (0.9 + uPensa * 0.6) * pow(0.5 + 0.5 * sin(uTime * rit + sd * 50.0), 12.0);
    float ouro = step(0.9, cor.r) * step(cor.b, 0.3);
    float boca = 1.0 + ouro * uFala * (0.7 + 0.6 * sin(uTime * 9.0 + sd * 3.0));   // falando: o dourado pulsa
    vCol = cor * b * mix(0.28, 1.15, k) * tw * boca * pulso(sd) * 0.42 * uExpo;
    gl_Position = projectionMatrix * mv;
    float persp = uScale / -mv.z;
    gl_PointSize = uSize * INF.z * mix(0.7, 1.2, k) * persp * (1.0 + 0.4 * (tw - 1.0)) * (1.0 + 0.4 * (pulso(sd) - 1.0));
    return p;
  }`;
const VS_BUSTO = VS_COMUM + `
  attribute vec3 aNor;
  void main() {
    // pontilhismo: cada ponto tem um limiar próprio (tirado da semente) e só aparece se a fração visível
    // dali passar dele; os que aparecem têm quase o mesmo brilho. A fração muda pouco com a luz (de frente
    // ~16%, de raspão ~31%): o busto fica cheio de pontos e o contorno sai do acúmulo deles, não de uma
    // linha clara. Pontos sem normal aparecem sempre.
    float vis = 1.0, raspao = 1.0, frente = 1.0;
    if (dot(aNor, aNor) > 0.25) {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vec3 nv = normalize(normalMatrix * aNor);
      float fd = dot(nv, normalize(-mv.xyz));
      float luz = mix(2.0, 0.22, pow(abs(fd), 0.55));   // luz de recorte
      luz *= 0.85 + 0.45 * max(0.0, nv.y);            // luz de cima: alto da cabeça, testa e topo dos ombros
      if (fd < 0.0) luz *= 0.12;
      frente = step(0.0, fd);
      float l = clamp(luz / 2.0, 0.0, 1.0);
      // bem de raspão a superfície se empilha em poucos px: menos pontos ali, e menos atrás do centro
      vis = (0.15 + 0.16 * l) * mix(0.3, 1.0, smoothstep(0.0, 0.25, abs(fd))) * mix(0.35, 1.0, smoothstep(-2.3, 2.1, mv.z - uC));
      float ouroG = step(0.9, color.r) * step(color.b, 0.3) * step(0.0, fd);
      vis = mix(vis, max(vis, 0.3), ouroG);           // o dourado é luz própria do rosto
      raspao = mix(0.35, 1.0, smoothstep(0.0, 0.5, abs(fd)));   // de raspão cada ponto brilha menos
    }
    vis = mix(vis, 0.25 + 0.75 * vis, step(0.75, INF.z) * frente);   // contas e marcas (pontos maiores) aparecem mais, só do lado de cá
    fimDoPonto(position, INF.y, INF.x * (0.7 + 0.5 * vis) * raspao, color);
    if (fract(INF.y * 91.7) > vis) esconder();
  }`;
const VS_FRANJA = VS_COMUM + `
  attribute vec3 aNor;
  void main() {
    // cada grão sobe devagar 3 px pra fora da pele e some (uma volta a cada ~16 s, cada um na sua fase);
    // só aparece onde a pele está de lado pra câmera: no contorno, onde vira o limite da nuvem de pontos
    float sd = INF.y, fase = fract(uTime * 0.06 + sd * 7.0);
    vec3 p = position + aNor * (fase * 0.03);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float fd = dot(normalize(normalMatrix * aNor), normalize(-mv.xyz));
    float vis = 0.4 * (1.0 - smoothstep(0.05, 0.5, abs(fd)));
    fimDoPonto(p, sd, INF.x * sin(3.14159 * fase), color);
    if (fract(sd * 91.7) > vis) esconder();
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
    fimDoPonto(p, INF.y, INF.x * g, color);
    if (g <= 0.001) esconder();
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
    fimDoPonto(p, sd, INF.x * ativo * (1.0 - fim) * 1.8, color);
    vCol *= uCores[int(c.z + 0.5)];            // color é branco nos cometas: a cor vem da ferramenta
    if (ativo * (1.0 - fim) <= 0.001) esconder();
  }`;
const FS = `
  varying vec3 vCol;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = length(c) * 2.0;                             // ponto redondo e nítido, como de caneta
    if (r >= 1.0) discard;
    float a = smoothstep(1.0, 0.55, r);
    gl_FragColor = vec4(vCol * a * 1.4, 1.0);
  }`;

function pontos(dados, material) {
  const g = new THREE.BufferGeometry();
  if (dados.n !== undefined) {                // busto: arrays tipados; cor, brilho/semente/tamanho e normal em bytes normalizados
    const k = dados.n * 3;
    g.setAttribute('position', new THREE.BufferAttribute(dados.pos.subarray(0, k), 3));
    g.setAttribute('color', new THREE.BufferAttribute(dados.col.subarray(0, k), 3, true));
    g.setAttribute('aInf', new THREE.BufferAttribute(dados.inf.subarray(0, k), 3, true));
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
// modo: 'particulas' (o busto de ~3 milhões de partículas) ou 'preenchimento' (pintado por shader, preenchimento.js)
export function montar(el, { densidade = 1, modo = 'particulas' } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  const pr = Math.min(2, window.devicePixelRatio || 1);
  renderer.setPixelRatio(pr);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);   // preto puro, como a foto
  const group = new THREE.Group();
  scene.add(group);
  const camera = new THREE.PerspectiveCamera(FOV, 1, 4, 200);   // near longe do zero: a máscara precisa de precisão de profundidade
  camera.position.set(0, ALVO_Y, DIST);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, ALVO_Y, 0);
  controls.minDistance = 14;
  controls.maxDistance = 70;

  const uniforms = {
    uTime: { value: 0 }, uSize: { value: 2.6 * pr }, uC: { value: 0 }, uScale: { value: DIST },
    uPensa: { value: 0 }, uFala: { value: 0 }, uExpo: { value: 1 }, uInf: { value: new THREE.Vector3(1, 1, 1) },
  };
  const material = (vs, extra = {}) => new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: true, blending: THREE.AdditiveBlending,   // a máscara esconde o que está atrás do busto
    uniforms: { ...uniforms, ...extra }, vertexShader: vs, fragmentShader: FS,
  });
  const preencher = modo === 'preenchimento';
  const malha = gerarMalha(preencher ? { recuo: 0, guia: true } : {});
  const oclusor = new THREE.Mesh(malha, new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide }));
  oclusor.renderOrder = -1;                   // grava a profundidade antes de qualquer ponto
  group.add(oclusor);
  const bytes = { uInf: { value: new THREE.Vector3(2, 1, 1.5) } };   // busto e franja guardam brilho/semente/tamanho em bytes
  let preenchimento = null;
  if (preencher) {
    // peso de cada marca no pico: porPx pontos por px de curva espalhados com σ = 2,2 px, sobre ~2,15 pontos por px²,
    // pela metade: com o mesmo peso do modo partículas os pontos do preenchimento viram linha
    const marcas = MARCAS.map(([T, curva, porPx]) => [T === HEAD ? PARTE.cabeca : PARTE.corpo, curva, 0.5 * porPx / (2.2 * Math.sqrt(2 * Math.PI)) / 2.15]);
    preenchimento = criarPreenchimento({
      renderer, camera, malha, uniforms, forma: { U, Y0, AX, paleta: { DEEP, BLUE, CYAN, WHITE, GOLD }, queixo: QUEIXO, marcas },
    });
    scene.add(preenchimento.quadro);
    group.add(pontos(gerarFranja(densidade), material(VS_FRANJA, bytes)));
  } else {
    group.add(pontos(gerarBusto(densidade), material(VS_BUSTO, bytes)));
  }

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
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.12, 0.05, 0.45);
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
    preenchimento?.ajustar();
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
    preenchimento?.desenharGuia(group);
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
      preenchimento?.descartar();
      bloom.dispose();
      composer.dispose?.();
      renderer.dispose();
      renderer.forceContextLoss();             // libera o contexto já: trocar de modo monta um renderer novo
      renderer.domElement.remove();
    },
  };
}
