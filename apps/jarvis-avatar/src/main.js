/*
  Jarvis — a cena é a própria imagem de referência, separada em camadas:
    fundo      → a original onde ela é visível; atrás da figura, a reconstrução
    corpo      → base escura da figura (com transparência) + brilho suave do contorno
    partículas → ~220 mil pixels brilhantes da figura e da aura, cada um com a cor exata da foto
  Em repouso a soma das camadas reproduz a foto. Mouse, ondas, voz e estados deslocam
  partículas e corpo pela MESMA função na GPU, então tudo se move junto.

  API pública (window.JARVIS, disponível após o evento `jarvis:ready` no window) — ver README.md.
  Parâmetros de URL: ?estatico · ?sofundo · ?fps · ?q=0.4..1
  Atalhos: H esconde a interface · F tela cheia · M microfone · P pensar · espaço fala · 1–4 formas · D mostra FPS
*/
import { CFG, IMG, NECK_Y, HEAD3D, ASSETS, PARAMS, STILL, SO_FUNDO, REDUCED, COARSE, MOBILE, FONT } from './config.js';
import { clamp, mix, ease, rnd, $, toast } from './util.js';
import { QUAD_VS, BG_FS, BODY_FS, PART_VS, PART_FS } from './shaders.js';
import { createAudio } from './audio.js';
import { createPose } from './pose.js';
import { buildShapes, buildText } from './shapes.js';

const canvas = $('gl');
const bloomCv = $('bloom');
const ui = {
  state: $('state'), fps: $('fps'), say: $('say'),
  speak: $('speak'), mic: $('mic'), think: $('think'),
  shapes: Array.from(document.querySelectorAll('[data-shape]')),
};
if (PARAMS.has('fps')) document.body.classList.add('debug');
if (COARSE) $('hint').textContent = 'Arraste o dedo pela figura e toque para soltar uma onda';

function fatal(err) {
  console.error(err);
  const box = $('nogl');
  box.querySelector('p').textContent = 'Não foi possível iniciar a cena 3D: ' + (err && err.message ? err.message : String(err));
  box.hidden = false;
  $('dock').hidden = true;
  $('loading').hidden = true;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error(`a camada "${src.startsWith('data:') ? 'embutida' : src}" não carregou`));
    im.src = src;
  });
}

function start(imgs) {
  const [imFundo, imMascaras, imCorpoRGB, imCorpoA, imEmissao, imInfo] = imgs;
  const gl = canvas.getContext('webgl', { antialias: false, alpha: false, depth: false, stencil: false, powerPreference: 'high-performance' });
  if (!gl) throw new Error('este navegador não liberou WebGL (ative a aceleração de hardware ou use Chrome, Edge ou Safari)');
  const bctx = bloomCv.getContext('2d');

  /* ---------- programas ---------- */
  const ATTR = { aPos: 0, aHome: 0, aCol: 1, aInfo: 2, aFrom: 3, aTo: 4, aSil: 5, aPhi: 6 };
  function makeProgram(name, vsrc, fsrc) {
    const prog = gl.createProgram();
    for (const [src, type] of [[vsrc, gl.VERTEX_SHADER], [fsrc, gl.FRAGMENT_SHADER]]) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(`shader "${name}" não compilou: ${gl.getShaderInfoLog(sh)}`);
      gl.attachShader(prog, sh);
    }
    for (const k in ATTR) gl.bindAttribLocation(prog, ATTR[k], k);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`programa "${name}" não linkou: ${gl.getProgramInfoLog(prog)}`);
    const u = {};
    const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(prog, i);
      u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(prog, info.name);
    }
    return { prog, u };
  }
  const P = {
    bg: makeProgram('fundo', QUAD_VS, BG_FS),
    body: makeProgram('corpo', QUAD_VS, BODY_FS),
    part: makeProgram('partículas', PART_VS, PART_FS),
  };

  /* ---------- texturas (sem conversão de cor nem pré-multiplicação: valores exatos da foto) ---------- */
  function texture(img) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  const T = { bg: texture(imFundo), masks: texture(imMascaras), bodyRGB: texture(imCorpoRGB), bodyA: texture(imCorpoA) };

  /* ---------- partículas: pixels marcados em info.png, com a cor de emissao.png ---------- */
  function pixels(img) {
    const cv = document.createElement('canvas');
    cv.width = IMG.w; cv.height = IMG.h;
    const c = cv.getContext('2d', { willReadFrequently: true });
    c.drawImage(img, 0, 0);
    return c.getImageData(0, 0, IMG.w, IMG.h).data;
  }
  const em = pixels(imEmissao), inf = pixels(imInfo);

  // silhueta da cabeça 3D por linha (bordas esquerda e direita, px): o perfil desenhado do manequim (HEAD3D.PROFILE)
  // centrado no eixo. Quantizada a 1/32 px: o corpo lê a mesma tabela numa textura (16 bits por borda) e as
  // partículas num atributo, com valores idênticos.
  const sil = new Float32Array(IMG.h * 2);
  {
    const P = HEAD3D.PROFILE, last = P.length - 1;
    for (let y = 0; y < IMG.h; y++) {
      let w = 0;
      if (y >= P[0][0]) {
        let k = 0;
        while (k < last - 1 && y >= P[k + 1][0]) k++;
        const p0 = P[Math.max(0, k - 1)][1], p1 = P[k][1], p2 = P[k + 1][1], p3 = P[Math.min(last, k + 2)][1];
        const t = clamp((y - P[k][0]) / (P[k + 1][0] - P[k][0]), 0, 1), t2 = t * t, t3 = t2 * t;
        w = Math.max(0, 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (3 * p1 - p0 - 3 * p2 + p3) * t3));
      }
      sil[y * 2] = Math.round((IMG.axisX + 0.5 - w) * 32) / 32; sil[y * 2 + 1] = Math.round((IMG.axisX + 0.5 + w) * 32) / 32;
    }
  }
  const silBytes = new Uint8Array(IMG.h * 4);
  for (let i = 0; i < IMG.h * 2; i++) { const v = Math.round(sil[i] * 32); silBytes[i * 2] = v >> 8; silBytes[i * 2 + 1] = v & 255; }
  T.sil = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, T.sil);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, IMG.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, silBytes);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  let total = 0;
  for (let i = 2; i < inf.length; i += 4) if (inf[i] > 0) total++;
  if (!total) throw new Error('as camadas de partículas vieram vazias');
  const cap = CFG.maxParticles || (MOBILE ? 90000 : 400000);
  const keep = Math.min(1, cap / total);                 // celular: amostra e compensa o brilho (uGain)
  // A cabeça e o pescoço são 3D: uma nuvem de partículas na superfície de um manequim moldado pela silhueta da foto
  // (cada linha é uma fatia elíptica, sliceOf no shader). A foto continua valendo pro que não gira: ombros, peito e
  // aura. Entre o pescoço e os ombros as duas se trocam aos poucos (headZone, igual no shader do corpo).
  const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
  const headZone = (y) => 1 - smooth(NECK_Y - 80, NECK_Y - 10, y);
  const sliceAt = (y) => {
    const m = 0.5 * (sil[y * 2] + sil[y * 2 + 1]), w = Math.max(0.5 * (sil[y * 2 + 1] - sil[y * 2]), 0.001);
    return { m, w, d: HEAD3D.K * w };
  };
  const pts = [];          // x, y, cor r g b, relevo, borda, aura, phi
  // 1) foto: tudo fora da cabeça (a aura em volta dela inclusive)
  for (let y = 0, px = 0; y < IMG.h; y++) {
    const S = sliceAt(y), hz = headZone(y + 0.5);
    for (let x = 0; x < IMG.w; x++, px += 4) {
      const t = inf[px + 2];
      if (!t || rnd() > keep) continue;
      const aura = t > 200;
      if (!aura && hz > 0 && Math.abs(x + 0.5 - S.m) < S.w + 45 && rnd() < hz) continue;   // cabeça: vira 3D
      pts.push(x + 0.5, y + 0.5, em[px], em[px + 1], em[px + 2], inf[px], inf[px + 1], aura ? 255 : 0, 9);
    }
  }
  // 2) superfície do manequim: amostragem uniforme por área (arco da fatia x inclinação entre linhas)
  {
    const dens = HEAD3D.DENS * keep, STEPS = 128;
    const [mx, my] = IMG.mouth;
    for (let y = Math.max(0, IMG.headTop - 4); y < NECK_Y - 10; y++) {
      const hz = headZone(y + 0.5);
      const S = sliceAt(y), Sn = sliceAt(Math.min(IMG.h - 1, y + 1));
      if (S.w < 2 || hz <= 0) continue;
      const tilt = Math.hypot(1, Sn.w - S.w);
      const ds = (f) => Math.hypot(S.w * Math.cos(f), S.d * Math.sin(f));
      let per = 0;
      for (let k = 0; k < STEPS; k++) per += ds(-Math.PI + 2 * Math.PI * (k + 0.5) / STEPS);
      per *= 2 * Math.PI / STEPS;
      const want = per * tilt * dens * hz;
      const n = Math.floor(want) + (rnd() < want % 1 ? 1 : 0);
      const dsMax = Math.max(S.w, S.d);
      for (let j = 0; j < n; j++) {
        let f;
        do f = -Math.PI + 2 * Math.PI * rnd(); while (rnd() * dsMax > ds(f));    // uniforme em comprimento de arco
        const x = S.m + S.w * Math.sin(f), yy = y + rnd();
        // azul do holograma, com variação de brilho e algumas faíscas; dourado na boca, só na frente
        // malha de holograma: meridianos a cada 22,5° e paralelos a cada 30 px, finos. Giram com a cabeça e deixam o
        // volume legível (numa nuvem uniforme só a silhueta mostra o giro)
        const mer = Math.abs(Math.sin(f * 8)) * S.w / 8, lat = Math.abs(((yy - IMG.headTop) % 30) - 15) - 13.5;
        const line = Math.max(Math.exp(-mer * mer / 2), lat > 0 ? Math.exp(-((1.5 - lat) ** 2) / 0.6) : 0);
        const l = (0.55 + 0.45 * Math.pow(rnd(), 1.5)) * (1 + 2.2 * line), spark = rnd() < 0.025 ? 1 : 0;
        let r = Math.min(255, 40 * l + 160 * spark), g = Math.min(255, 125 * l + 110 * spark), bl = 255 * Math.min(1, l + 0.2);
        const gold = Math.cos(f) > 0 ? Math.exp(-((x - mx) ** 2) / (85 * 85) - ((yy - my) ** 2) / (60 * 60)) : 0;
        r = r + (255 - r) * gold; g = g + (175 - g) * gold; bl = bl + (55 - bl) * gold;
        pts.push(x, yy, r, g, bl, 255 * Math.max(0, Math.cos(f)), 0, 0, f);
      }
    }
  }
  // 3) orelhas: placas ovais presas na lateral do crânio (HEAD3D.EAR), estendidas pra trás no shader. A borda da
  // orelha é mais clara (é ela que desenha a orelha de frente).
  {
    const E = HEAD3D.EAR, dens = HEAD3D.DENS * keep * 1.6;
    for (let y = Math.ceil(E.cy - E.ry); y < E.cy + E.ry; y++) {
      const v = (y + 0.5 - E.cy) / E.ry, U = E.out * Math.sqrt(Math.max(0, 1 - v * v));
      const S = sliceAt(y);
      for (const sg of [-1, 1]) {
        const want = U * 2.2 * dens;                   // a placa tem ~2,2x a largura vista de frente (vai pra trás)
        for (let j = 0, n = Math.floor(want) + (rnd() < want % 1 ? 1 : 0); j < n; j++) {
          const u = U * Math.sqrt(rnd()), edge = Math.exp(-((U - u) ** 2) / 8);
          const l = 0.45 + 0.35 * rnd() + 0.9 * edge;
          pts.push(S.m + sg * (S.w + u), y + rnd(), Math.min(255, 45 * l), Math.min(255, 135 * l), 255, 120, 0, 0, 10 + u);
        }
      }
    }
  }
  const K9 = 9;
  let N = pts.length / K9;
  const homeT = new Float32Array(N * 2), silT = new Float32Array(N * 2), colT = new Uint8Array(N * 4), infoT = new Uint8Array(N * 4);
  const phiT = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const o = i * K9, y = Math.min(IMG.h - 1, Math.floor(pts[o + 1]));
    homeT[i * 2] = pts[o]; homeT[i * 2 + 1] = pts[o + 1];
    silT[i * 2] = sil[y * 2]; silT[i * 2 + 1] = sil[y * 2 + 1];
    colT[i * 4] = pts[o + 2]; colT[i * 4 + 1] = pts[o + 3]; colT[i * 4 + 2] = pts[o + 4];
    infoT[i * 4] = pts[o + 5]; infoT[i * 4 + 1] = pts[o + 6]; infoT[i * 4 + 2] = pts[o + 7]; infoT[i * 4 + 3] = (rnd() * 255) | 0;
    phiT[i] = pts[o + 8];
  }
  const gain = 1 / keep;
  const SHAPES = buildShapes(N, homeT);

  /* ---------- buffers ---------- */
  function buffer(data) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }
  const B = {
    tri: buffer(new Float32Array([-1, -1, 3, -1, -1, 3])),
    home: buffer(homeT), sil: buffer(silT), col: buffer(colT), info: buffer(infoT), phi: buffer(phiT),
    shape: { head: buffer(SHAPES.head), sphere: buffer(SHAPES.sphere), galaxy: buffer(SHAPES.galaxy), text: null },
  };

  /* ---------- estado ---------- */
  const clock = { time: 0 };
  let scale = 1, offX = 0, offY = 0, dpr = 1;
  let quality = clamp(parseFloat(PARAMS.get('q')) || 1, 0.4, 1);
  const ADAPT = !PARAMS.has('q');
  let shape = 'head', fromShape = STILL || REDUCED ? 'head' : 'galaxy', toShape = 'head';
  let morphT0 = STILL || REDUCED ? -10 : 0.3;            // intro: a galáxia vira o Jarvis
  const mouse = { x: 0, y: 0, sx: 0, sy: 0, ix: -9999, iy: -9999, fx: 0, fy: 0, seen: false, active: false };
  const trail = new Float32Array(24);                 // 8 pontos (x, y, força); o 0 é a ponta que segue o cursor
  const TRAIL_GAP = 18;                                // px da imagem entre pontos do rastro
  const waves = [];
  const wavesU = new Float32Array(16);
  function addWave(x, y, power) {
    if (waves.length >= 4) waves.shift();
    waves.push({ x, y, t0: clock.time, p: power });
  }

  // ataque forte da voz solta uma onda a partir da boca
  const audio = createAudio({ clock, micButton: ui.mic, onOnset: () => { if (shape === 'head') addWave(IMG.mouth[0], IMG.mouth[1], 0.3); } });
  const { A, speech } = audio;

  /* ---------- estados ---------- */
  const api = { state: null };
  let thinkUntil = 0, curState = '';
  const STATE_TEXT = { idle: 'está pronto', listening: 'está ouvindo', thinking: 'está pensando', speaking: 'está falando' };
  const ST = { listen: 0, think: 0, speak: 0, head: STILL || REDUCED ? 1 : 0 };
  function effectiveState() {
    if (speech.active || api.state === 'speaking') return 'speaking';
    if (clock.time < thinkUntil || api.state === 'thinking') return 'thinking';
    if (A.mic || api.state === 'listening') return 'listening';
    return 'idle';
  }
  function updateStates(dt) {
    const s = effectiveState();
    if (s !== curState) {
      curState = s;
      document.body.dataset.state = s;
      ui.state.textContent = STATE_TEXT[s];
      ui.think.setAttribute('aria-pressed', String(s === 'thinking'));
    }
    const k = ease(dt, 3.5);
    ST.listen += ((s === 'listening' ? 1 : 0) - ST.listen) * k;
    ST.think += ((s === 'thinking' ? 1 : 0) - ST.think) * k;
    ST.speak += ((s === 'speaking' ? 1 : 0) - ST.speak) * k;
    // o corpo aparece quando as partículas chegam na cabeça e some nas outras formas
    const headTarget = toShape === 'head' && clock.time > morphT0 + 0.9 ? 1 : 0;
    ST.head += (headTarget - ST.head) * ease(dt, headTarget ? 1.6 : 4);
    if (STILL) ST.head = 1;
  }
  const toggleThink = () => { thinkUntil = clock.time < thinkUntil ? 0 : clock.time + 4.5; };

  const head = createPose({ clock, mouse, ST, audio });

  /* ---------- formas ---------- */
  let shapeToken = 0;
  async function setShape(name) {
    if (!Object.prototype.hasOwnProperty.call(SHAPES, name)) return;
    const token = ++shapeToken;
    if (name === 'text') {
      const word = (ui.say.value.trim() || 'Jarvis').slice(0, 24);
      try { await document.fonts.load(`700 120px ${FONT}`); }
      catch (err) { console.warn('Fonte não carregou; usando a fonte padrão do sistema.', err); }
      if (token !== shapeToken) return;
      const pts = buildText(word, N);
      if (!pts) { toast('Não consegui desenhar esse texto com partículas.'); return; }
      SHAPES.text = pts;
      if (!B.shape.text) B.shape.text = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, B.shape.text);
      gl.bufferData(gl.ARRAY_BUFFER, pts, gl.STATIC_DRAW);
    }
    fromShape = toShape === name && name !== 'text' ? fromShape : toShape;   // mesma forma de novo: não reanima
    toShape = name;
    shape = name;
    morphT0 = clock.time;
    ui.shapes.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.shape === name)));
  }

  /* ---------- enquadramento: a foto inteira em telas largas; em telas estreitas, a figura e o horizonte ---------- */
  function layout() {
    const Wd = canvas.width, Hd = canvas.height, a = Wd / Hd;
    const half = a >= 1 ? mix(0.42, 0.5, clamp((a - 1) / 0.78, 0, 1)) : mix(0.2, 0.42, clamp((a - 0.5) / 0.5, 0, 1));
    const sx0 = 0.5 - half, sx1 = 0.5 + half, sy0 = 0.08, sy1 = 1.0;
    const cover = Math.max(Wd / IMG.w, Hd / IMG.h);
    const fit = Math.min(Wd / ((sx1 - sx0) * IMG.w), Hd / ((sy1 - sy0) * IMG.h));
    scale = Math.min(cover, fit);
    offX = Wd / 2 - 0.5 * IMG.w * scale;
    const ih = IMG.h * scale;
    offY = ih <= Hd ? Hd - ih : clamp(Hd / 2 - ((sy0 + sy1) / 2) * IMG.h * scale, Hd - ih, 0);
  }
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, CFG.maxDpr) * quality;
    canvas.width = Math.max(1, Math.round(window.innerWidth * dpr));
    canvas.height = Math.max(1, Math.round(window.innerHeight * dpr));
    bloomCv.width = Math.max(1, Math.round(window.innerWidth / 5));
    bloomCv.height = Math.max(1, Math.round(window.innerHeight / 5));
    bctx.imageSmoothingEnabled = true;
    bctx.imageSmoothingQuality = 'high';
    layout();
  }

  /* ---------- desenho ---------- */
  function quad() {
    for (let loc = 1; loc < 7; loc++) gl.disableVertexAttribArray(loc);
    gl.bindBuffer(gl.ARRAY_BUFFER, B.tri);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  function shared(p) {
    const u = p.u;
    gl.uniform2f(u.uRes, canvas.width, canvas.height);
    gl.uniform2f(u.uOffset, offX, offY);
    gl.uniform2f(u.uImg, IMG.w, IMG.h);
    gl.uniform1f(u.uScale, scale);
    gl.uniform2f(u.uPar, STILL ? 0 : mouse.sx, STILL ? 0 : mouse.sy);
  }
  function fieldUniforms(p, on) {
    gl.uniform3fv(p.u.uTrail, trail);
    gl.uniform4fv(p.u.uWaves, wavesU);
    gl.uniform1f(p.u.uMouseR, CFG.mouseRadius);
    gl.uniform1f(p.u.uFieldOn, on ? 1 : 0);
  }
  const attr = (loc, buf, size, type, norm, stride) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, type, norm, stride, 0);
  };
  function render(fieldOn) {
    const time = clock.time;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.disable(gl.BLEND);
    let p = P.bg;
    gl.useProgram(p.prog);
    shared(p);
    gl.uniform1f(p.u.uTime, time % 1000);
    gl.uniform1f(p.u.uStill, STILL ? 1 : 0);
    gl.uniform1f(p.u.uLevel, A.level);
    gl.uniform1f(p.u.uHorizon, IMG.horizon);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.bg); gl.uniform1i(p.u.uBg, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, T.masks); gl.uniform1i(p.u.uMasks, 1);
    quad();

    if (!SO_FUNDO) {
      gl.enable(gl.BLEND);
      if (ST.head > 0.002) {
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // corpo pré-multiplicado: tampa o fundo e soma o brilho do contorno
        p = P.body;
        gl.useProgram(p.prog);
        shared(p);
        fieldUniforms(p, fieldOn);
        head.uniforms(gl, p);
        gl.uniform1f(p.u.uVis, ST.head);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.bodyRGB); gl.uniform1i(p.u.uBodyRGB, 0);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, T.bodyA); gl.uniform1i(p.u.uBodyA, 1);
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, T.sil); gl.uniform1i(p.u.uSil, 2);
        quad();
      }

      gl.blendFunc(gl.ONE, gl.ONE);                     // partículas somam luz
      p = P.part;
      gl.useProgram(p.prog);
      shared(p);
      fieldUniforms(p, fieldOn);
      head.uniforms(gl, p);
      const u = p.u;
      gl.uniform1f(u.uTime, time);
      gl.uniform1f(u.uStill, STILL ? 1 : 0);
      gl.uniform1f(u.uMorphT0, morphT0);
      gl.uniform1f(u.uLevel, A.level);
      gl.uniform1f(u.uThink, ST.think);
      gl.uniform1f(u.uListen, ST.listen);
      gl.uniform1f(u.uGain, gain);
      gl.uniform1f(u.uHeadness, ST.head);
      gl.uniform1f(u.uAxisX, IMG.axisX);
      gl.uniform1f(u.uHeadCY, IMG.headCY);
      gl.uniform1f(u.uChinY, IMG.chinY);
      gl.uniform2f(u.uMouth, IMG.mouth[0], IMG.mouth[1]);
      gl.uniform1fv(u.uBands, A.bands);
      attr(0, B.home, 2, gl.FLOAT, false, 0);
      attr(1, B.col, 3, gl.UNSIGNED_BYTE, true, 4);
      attr(2, B.info, 4, gl.UNSIGNED_BYTE, true, 0);
      attr(3, B.shape[fromShape] || B.shape.head, 3, gl.FLOAT, false, 0);
      attr(4, B.shape[toShape] || B.shape.head, 3, gl.FLOAT, false, 0);
      attr(5, B.sil, 2, gl.FLOAT, false, 0);
      attr(6, B.phi, 1, gl.FLOAT, false, 0);
      gl.drawArrays(gl.POINTS, 0, N);
    }

    const glowAmt = CFG.bloom && !STILL && !SO_FUNDO ? clamp(A.level * 0.9 + ST.think * 0.3 + ST.speak * 0.2, 0, 0.85) : 0;
    bloomCv.style.opacity = glowAmt.toFixed(3);
    if (glowAmt > 0.01) {                               // brilho extra só quando ele fala ou pensa
      bctx.globalCompositeOperation = 'copy';
      bctx.drawImage(canvas, 0, 0, bloomCv.width, bloomCv.height);
      bctx.globalCompositeOperation = 'multiply';
      bctx.drawImage(bloomCv, 0, 0);
      bctx.globalCompositeOperation = 'source-over';
    }
  }

  /* ---------- entrada ---------- */
  function pointer(e) {
    const kx = canvas.width / window.innerWidth, ky = canvas.height / window.innerHeight;
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = (e.clientY / window.innerHeight) * 2 - 1;
    mouse.ix = (e.clientX * kx - offX) / scale;
    mouse.iy = (e.clientY * ky - offY) / scale;
    mouse.active = true;
  }
  const release = () => { mouse.active = false; };
  window.addEventListener('pointermove', pointer, { passive: true });
  window.addEventListener('pointerdown', (e) => {
    if (e.target.closest && e.target.closest('#dock')) return;
    pointer(e);
    addWave(mouse.ix, mouse.iy, 1);
  });
  window.addEventListener('pointerup', (e) => { if (e.pointerType !== 'mouse') release(); });
  window.addEventListener('pointercancel', release);   // toque interrompido pelo sistema: sem isso a cabeça fica presa olhando pro dedo
  document.documentElement.addEventListener('mouseleave', release);
  window.addEventListener('blur', release);
  window.addEventListener('resize', resize);

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      const req = document.documentElement.requestFullscreen;
      if (!req) { toast('Tela cheia não é suportada neste navegador.'); return; }
      req.call(document.documentElement).catch(() => toast('O navegador bloqueou a tela cheia aqui.'));
    } else if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }
  function sayAndForm() {
    if (shape === 'text') setShape('text');
    audio.speak(ui.say.value);
  }
  const KEYS = {
    h: () => document.body.classList.toggle('hide-ui'),
    f: toggleFullscreen,
    m: audio.toggleMic,
    p: toggleThink,
    d: () => document.body.classList.toggle('debug'),
    ' ': () => audio.speak(ui.say.value),
    1: () => setShape('head'), 2: () => setShape('sphere'), 3: () => setShape('galaxy'), 4: () => setShape('text'),
  };
  ui.speak.addEventListener('click', sayAndForm);
  ui.mic.addEventListener('click', audio.toggleMic);
  ui.think.addEventListener('click', toggleThink);
  ui.shapes.forEach((b) => b.addEventListener('click', () => setShape(b.dataset.shape)));
  window.addEventListener('keydown', (e) => {
    if (e.target === ui.say) {
      if (e.key === 'Enter') { e.preventDefault(); sayAndForm(); }
      else if (e.key === 'Escape') ui.say.blur();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if ((k === ' ' || k === 'enter') && e.target.closest && e.target.closest('button')) return;
    if (!KEYS[k]) return;
    if (k === ' ') e.preventDefault();
    KEYS[k]();
  });

  // A GPU pode derrubar o contexto (driver reiniciou, aba em segundo plano no celular). Os buffers e texturas
  // se perdem; recarregar é o caminho mais curto e confiável pra voltar inteiro.
  let lost = false;
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    lost = true;
    toast('A placa de vídeo reiniciou o WebGL. Recarregando…', 9000);
  });
  canvas.addEventListener('webglcontextrestored', () => location.reload());

  /* ---------- API pública ---------- */
  window.JARVIS = Object.freeze({
    setState(s) { api.state = ['idle', 'listening', 'thinking', 'speaking'].includes(s) ? s : null; },
    setLevel(v) { A.manual = clamp(Number(v) || 0, 0, 1); },
    attachAudio: audio.attachAudio,
    speak: audio.speak,
    think(ms = 4500) { const v = Number(ms); thinkUntil = clock.time + (Number.isFinite(v) ? Math.max(0, v) : 4500) / 1000; },
    shape: setShape,
    wave(x = IMG.axisX, y = IMG.headCY, power = 1) { addWave(Number(x) || 0, Number(y) || 0, Number(power) || 1); },
    get particles() { return N; },
    get time() { return clock.time; },
  });

  /* ---------- loop ---------- */
  let last = performance.now(), fpsEMA = 60, fpsShownAt = 0, lastAdapt = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    if (lost) return;
    const raw = Math.max(0.0001, (now - last) / 1000);
    last = now;
    const dt = Math.min(raw, 1 / 30);
    clock.time += dt;
    const time = clock.time;
    fpsEMA = mix(fpsEMA, 1 / raw, 0.05);
    if (ADAPT && time > 5 && time - lastAdapt > 3 && fpsEMA < 40 && quality > 0.55) {   // GPU fraca: baixa a resolução
      quality = Math.max(0.55, quality * 0.82);
      lastAdapt = time;
      resize();
    }
    if (document.body.classList.contains('debug') && now - fpsShownAt > 500) {
      fpsShownAt = now;
      ui.fps.textContent = `${Math.round(fpsEMA)} fps, ${N.toLocaleString('pt-BR')} partículas`;
    }
    const tx = mouse.active ? mouse.x : 0, ty = mouse.active ? mouse.y : 0;
    mouse.sx += (tx - mouse.sx) * ease(dt, 3);
    mouse.sy += (ty - mouse.sy) * ease(dt, 3);
    // rastro do mouse, amanteigado: o cursor é suavizado, a ponta do rastro anda junto a cada frame e um ponto
    // novo só nasce quando a ponta se afasta TRAIL_GAP. Ele nasce com força 0 e cresce em rampa, então a soma
    // das forças nunca salta; os pontos antigos se apagam devagar e as partículas voltam sozinhas.
    if (mouse.active && !STILL) {
      if (!mouse.seen) { mouse.fx = mouse.ix; mouse.fy = mouse.iy; trail[0] = mouse.fx; trail[1] = mouse.fy; mouse.seen = true; }
      const k = ease(dt, 12);
      mouse.fx += (mouse.ix - mouse.fx) * k;
      mouse.fy += (mouse.iy - mouse.fy) * k;
      if (Math.hypot(mouse.fx - trail[0], mouse.fy - trail[1]) > TRAIL_GAP) {
        trail.copyWithin(3, 0, 21);
        trail[2] = 0;
      }
      trail[0] = mouse.fx; trail[1] = mouse.fy;
      trail[2] += (1 - trail[2]) * ease(dt, 6);
    } else {
      mouse.seen = false;
      trail[2] *= Math.exp(-dt * 2.5);
    }
    let trailOn = trail[2] > 0.01;
    for (let i = 1; i < 8; i++) { trail[i * 3 + 2] *= Math.exp(-dt * 3.5); if (trail[i * 3 + 2] > 0.01) trailOn = true; }
    for (let i = waves.length - 1; i >= 0; i--) if (time - waves[i].t0 > 2.6) waves.splice(i, 1);
    wavesU.fill(0);
    waves.forEach((w, i) => { wavesU[i * 4] = w.x; wavesU[i * 4 + 1] = w.y; wavesU[i * 4 + 2] = time - w.t0; wavesU[i * 4 + 3] = w.p; });
    if (time > morphT0 + 2.2 && fromShape !== toShape) fromShape = toShape;
    audio.update(dt);
    updateStates(dt);
    head.update(dt);
    render(trailOn || waves.length > 0);
  }
  resize();
  document.body.classList.add('ready');
  window.dispatchEvent(new CustomEvent('jarvis:ready', { detail: window.JARVIS }));
  requestAnimationFrame(frame);
}

if (SO_FUNDO) {                                          // sem humanoide, os controles e a dica dele não fazem sentido
  document.body.classList.add('hide-ui');
  $('hint').remove();
}
Promise.all(ASSETS.map(loadImage)).then(start).catch(fatal);
