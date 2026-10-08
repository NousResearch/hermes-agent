// Nyx — avatar de partículas do Hermes (módulo ES carregado pelo painel do dashboard).
//
// Em repouso só o busto (casca + miolo de pontos de luz, no estilo do globo/galáxia do avatar).
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
const HEAD = [[134,0,12,30],[136,10,37,30],[142,45,69,30],[148,62,86,30],[154,78,99,30],[160,90,111,30],[166,100,121,30],[172,109,130,30],[178,116,137,30],[184,124,145,30],[190,128,151,30],[196,135,157,30],[202,140,163,30],[208,144,168,30],[214,149,173,30],[220,154,176,30],[226,157,180,30],[232,160,183,30],[238,164,186,30],[244,166,189,30],[250,168,191,30],[256,170,194,29],[262,172,196,29],[268,174,198,29],[274,176,199,29],[280,178,201,29],[286,178,203,29],[292,180,204,29],[298,181,204,29],[304,182,205,29],[310,182,206,29],[316,183,206,29],[322,184,206,29],[328,184,206,29],[334,184,207,29],[340,184,207,28],[346,183,206,27],[352,183,206,26],[364,184,204,24],[376,185,201,24],[388,186,199,25],[400,188,197,27],[412,187,195,30],[424,186,195,35],[436,184,192,40],[448,182,190,46],[460,180,187,52],[472,177,184,57],[484,173,181,60],[496,170,180,64],[508,156,181,66],[520,152,182,67],[532,149,179,68],[544,145,172,72],[556,141,161,78],[568,136,148,85],[578,131,135,93],[590,125,118,103],[600,120,102,112],[607,116,89,119],[615,108,75,127],[624,98,59,136],[633,88,43,144],[641,70,28,152],[650,48,11,160],[655,24,9,161],[658,3,7,162]];
const BODY = [[520,116,112,-2],[560,118,112,0],[600,128,112,-6],[610,128,112,-8],[620,128,113,-9],[630,128,114,-10],[640,128,114,-12],[650,129,116,-13],[660,130,118,-15],[670,133,120,-16],[680,138,123,-17],[690,144,125,-19],[700,152,128,-20],[710,162,132,-22],[720,172,136,-24],[730,186,140,-26],[740,202,145,-28],[750,220,150,-29],[760,241,155,-30],[770,262,160,-31],[780,284,165,-32],[790,309,170,-33],[795,322,172,-34],[800,338,175,-34],[805,354,178,-34],[810,380,180,-34],[815,406,182,-34],[820,429,184,-35],[825,446,186,-36],[830,459,188,-36],[840,480,192,-36],[850,498,195,-36],[860,513,198,-36],[870,526,200,-36],[880,537,203,-36],[890,546,204,-36],[900,556,206,-36],[910,562,207,-36],[920,571,208,-36],[930,576,209,-36],[940,581,210,-36]];
const U = 0.01, Y0 = 380;
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
  if (y < 136 || y > 658) return false;
  const r = row(HEAD, y);
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
const BLUE = [0.08, 0.42, 1.0], CYAN = [0.45, 0.85, 1.0], WHITE = [0.85, 0.95, 1.0], GOLD = [1.0, 0.62, 0.14];

function criarAleatorio(semente) {
  let s = semente;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(6.283185 * rnd());
  return { rnd, gauss };
}

// sorteio por área: cada linha pesa o perímetro da fatia (casca) ou a área (volume)
function sampler(rnd, T, y0, y1, vol) {
  const ys = [], acc = [];
  let s = 0;
  for (let y = y0; y < y1; y++) {
    const [w, d] = row(T, y);
    s += vol ? Math.PI * w * d : Math.PI * (3 * (w + d) - Math.sqrt((3 * w + d) * (w + 3 * d)));
    ys.push(y); acc.push(s);
  }
  return () => {
    const v = rnd() * s;
    let lo = 0, hi = acc.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (acc[m] < v) lo = m + 1; else hi = m; }
    return ys[lo] + rnd();
  };
}

function gerarBusto(densidade) {
  const { rnd, gauss } = criarAleatorio(7);
  const pos = [], col = [], inf = [];
  const put = (x, y, z, b, sz) => {
    const g = (z > 40 ? 1 : 0) * Math.exp(-((x / 60) ** 2 + ((y - 552) / 62) ** 2));   // dourado: frente baixa do rosto
    const r = rnd(), c = rnd() < g ? GOLD : r < 0.62 ? BLUE : r < 0.93 ? CYAN : WHITE;
    pos.push(x * U, -(y - Y0) * U, z * U);
    col.push(...c);
    inf.push(b, rnd(), sz);
  };
  const fadeY = (y) => (y < 860 ? 1 : Math.max(0, 1 - (y - 860) / 80));   // a base se desfaz
  const casca = (T, y0, y1, n, skip) => {
    const pick = sampler(rnd, T, y0, y1, false);
    for (let i = 0; i < n * densidade; i++) {
      const y = pick(), [w, d, c] = row(T, y), ph = rnd() * 6.283185, t = gauss() * 4;
      const nx = d * Math.sin(ph), nz = w * Math.cos(ph), nl = Math.hypot(nx, nz) || 1;
      const x = w * Math.sin(ph) + nx / nl * t, z = c + d * Math.cos(ph) + nz / nl * t;
      if ((skip && skip(x, y, z)) || rnd() > fadeY(y)) continue;
      put(x, y, z, 0.55 + 0.45 * rnd(), 0.7 + 0.6 * rnd() ** 2);
    }
  };
  const miolo = (T, y0, y1, n, skip) => {
    const pick = sampler(rnd, T, y0, y1, true);
    for (let i = 0; i < n * densidade; i++) {
      const y = pick(), [w, d, c] = row(T, y), ph = rnd() * 6.283185, r = Math.sqrt(rnd());
      const x = w * r * Math.sin(ph), z = c + d * r * Math.cos(ph);
      if ((skip && skip(x, y, z)) || rnd() > fadeY(y)) continue;
      put(x, y, z, 0.25 + 0.3 * rnd(), 0.55 + 0.4 * rnd());
    }
  };
  casca(HEAD, 136, 658, 42000);
  casca(BODY, 520, 940, 52000, insideHead);
  miolo(HEAD, 136, 658, 9000);
  miolo(BODY, 520, 940, 9000, insideHead);
  for (const sg of [-1, 1]) {                 // orelhas: presas no crânio, borda de trás aberta pra fora
    for (let i = 0; i < 3500 * densidade; i++) {
      const t = (rnd() * 2 - 1) * Math.PI, r = Math.sqrt(rnd());
      const ey = 442 - Math.sin(t) * 62 * r - 6 * Math.cos(t) * r, ez = 2 - Math.cos(t) * 46 * r;
      const h = row(HEAD, ey), zz = (ez - h[2]) / h[1], xs = h[0] * Math.sqrt(Math.max(0, 1 - zz * zz));
      const out = 2 + 28 * r * (0.2 + 0.8 * Math.max(0, Math.cos(t)) ** 1.5) * (ey < 482 ? 1 : Math.max(0.2, 1 - (ey - 482) / 30));
      put(sg * (xs - 2 + out), ey, ez, r > 0.85 ? 0.9 : 0.5, 0.8);
    }
  }
  {                                           // poeira rala colada no corpo
    const ph = sampler(rnd, HEAD, 140, 640, false), pb = sampler(rnd, BODY, 640, 900, false);
    for (let i = 0; i < 3000 * densidade; i++) {
      const head = rnd() < 0.55, T = head ? HEAD : BODY, y = head ? ph() : pb(), [w, d, c] = row(T, y);
      const a = rnd() * 6.283185, out = 6 + 70 * rnd() ** 2.5;
      const nx = d * Math.sin(a), nz = w * Math.cos(a), nl = Math.hypot(nx, nz) || 1;
      const top = head && y < 260 ? -Math.abs(gauss()) * 30 * (260 - y) / 120 : 0;
      put(w * Math.sin(a) + nx / nl * out, y + gauss() * 12 + top, c + d * Math.cos(a) + nz / nl * out, 0.3 + 0.6 * rnd() ** 3, 0.5 + 0.8 * rnd());
    }
  }
  return { pos, col, inf };
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
  vec3 fimDoPonto(vec3 p, float sd, float b) {
    p += vec3(sin(uTime * (0.4 + sd * 0.5) + sd * 40.0), cos(uTime * (0.33 + sd * 0.4) + sd * 27.0), sin(uTime * 0.37 + sd * 19.0)) * 0.012;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float k = smoothstep(-2.3, 2.1, mv.z - uC);                       // 0 = fundo, 1 = mais perto
    float rit = 0.8 + sd * 2.5 + uPensa * 1.5;                          // pensando: cintila mais rápido
    float tw = 1.0 + (0.9 + uPensa * 0.6) * pow(0.5 + 0.5 * sin(uTime * rit + sd * 50.0), 12.0);
    float ouro = step(0.9, color.r) * step(color.b, 0.3);
    float boca = 1.0 + ouro * uFala * (0.7 + 0.6 * sin(uTime * 9.0 + sd * 3.0));   // falando: o dourado pulsa
    vCol = color * b * mix(0.28, 1.15, k) * tw * boca * 0.45 * uExpo;
    gl_Position = projectionMatrix * mv;
    float persp = uScale / -mv.z;
    gl_PointSize = uSize * aInf.z * mix(0.7, 1.2, k) * persp * (1.0 + 0.4 * (tw - 1.0));
    return p;
  }`;
const VS_BUSTO = VS_COMUM + `
  void main() { fimDoPonto(position, aInf.y, aInf.x); }`;
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
    fimDoPonto(p, aInf.y, aInf.x * g);
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
    fimDoPonto(p, sd, aInf.x * ativo * (1.0 - fim) * 1.8);
    vCol *= uCores[int(c.z + 0.5)];            // color é branco nos cometas: a cor vem da ferramenta
    if (ativo * (1.0 - fim) <= 0.001) gl_PointSize = 0.0;
  }`;
const FS = `
  varying vec3 vCol;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float rd = clamp(1.0 - dot(c, c) * 4.0, 0.0, 1.0);   // ponto de luz redondo
    if (rd <= 0.001) discard;
    gl_FragColor = vec4(vCol * rd * rd * 1.6, 1.0);
  }`;

function pontos(dados, material) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(dados.pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(dados.col, 3));
  g.setAttribute('aInf', new THREE.Float32BufferAttribute(dados.inf, 3));
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
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.set(0, -1.6, 19);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, -1.9, 0);
  controls.minDistance = 8;
  controls.maxDistance = 40;

  const uniforms = {
    uTime: { value: 0 }, uSize: { value: 2.6 * pr }, uC: { value: 0 }, uScale: { value: 19 },
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
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.55, 0.35, 0.2);
  composer.addPass(bloom);
  const ajustar = () => {
    const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    // tela estreita (celular em pé): afasta a câmera pra caber cabeça e ombros, mantendo a direção do giro
    const dist = 19 * Math.max(1, 0.9 / camera.aspect);
    camera.position.sub(controls.target).setLength(dist).add(controls.target);
    // a figura ocupa ~metade da altura; com a contagem de grãos fixa, a sobreposição cresce com 1/altura²
    uniforms.uExpo.value = Math.min(1.2, Math.max(0.2, (h * 19 / dist / 940) ** 2));
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
    uniforms.uC.value = alvoCamera.set(0, -1.9, 0).applyMatrix4(camera.matrixWorldInverse).z;
    composer.render();
  }
  loop();

  return {
    evento,
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
