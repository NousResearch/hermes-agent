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
import { CFG, IMG, ASSETS, PARAMS, STILL, SO_FUNDO, REDUCED, COARSE, MOBILE, FONT } from './config.js';
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
  const ATTR = { aPos: 0, aHome: 0, aCol: 1, aInfo: 2, aFrom: 3, aTo: 4 };
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
  let total = 0;
  for (let i = 2; i < inf.length; i += 4) if (inf[i] > 0) total++;
  if (!total) throw new Error('as camadas de partículas vieram vazias');
  const cap = CFG.maxParticles || (MOBILE ? 90000 : 400000);
  const keep = Math.min(1, cap / total);                 // celular: amostra e compensa o brilho (uGain)
  const homeA = new Float32Array(total * 2), colA = new Uint8Array(total * 4), infoA = new Uint8Array(total * 4);
  let N = 0;
  for (let y = 0, px = 0; y < IMG.h; y++) {
    for (let x = 0; x < IMG.w; x++, px += 4) {
      const t = inf[px + 2];
      if (!t) continue;
      if (rnd() > keep) continue;
      homeA[N * 2] = x + 0.5; homeA[N * 2 + 1] = y + 0.5;
      colA[N * 4] = em[px]; colA[N * 4 + 1] = em[px + 1]; colA[N * 4 + 2] = em[px + 2];
      infoA[N * 4] = inf[px]; infoA[N * 4 + 1] = inf[px + 1]; infoA[N * 4 + 2] = t > 200 ? 255 : 0; infoA[N * 4 + 3] = (rnd() * 255) | 0;
      N++;
    }
  }
  const gain = 1 / keep;
  const SHAPES = buildShapes(N, homeA);

  /* ---------- buffers ---------- */
  function buffer(data) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }
  const B = {
    tri: buffer(new Float32Array([-1, -1, 3, -1, -1, 3])),
    home: buffer(homeA.subarray(0, N * 2)), col: buffer(colA.subarray(0, N * 4)), info: buffer(infoA.subarray(0, N * 4)),
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
    for (let loc = 1; loc < 5; loc++) gl.disableVertexAttribArray(loc);
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
      gl.uniform1f(u.uBack, 0);
      gl.drawArrays(gl.POINTS, 0, N);
      if (Math.abs(head.pose.yaw.x) > 0.03 && ST.head > 0.5) {  // verso da cabeça: preenche o lado que aparece ao girar
        gl.uniform1f(u.uBack, 1);
        gl.drawArrays(gl.POINTS, 0, N);
      }
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
