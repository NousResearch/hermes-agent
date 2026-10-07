/* Shaders GLSL (WebGL 1). Coordenadas em pixels da imagem de referência (IMG). */
import { IMG, HEAD_RX, HEAD_RY, NECK_Y, HEAD3D } from './config.js';

// Deslocamento (dx, dy em px da imagem; dz = "pra frente") causado pelo rastro do mouse e pelas ondas.
// É a mesma função nas partículas e no corpo: os dois se movem juntos.
export const FIELD = `
uniform vec3 uTrail[8];
uniform vec4 uWaves[4];
uniform float uMouseR, uFieldOn;
vec3 field(vec2 p) {
  vec3 acc = vec3(0.0);
  if (uFieldOn < 0.5) return acc;
  float r2 = uMouseR * uMouseR;
  for (int i = 0; i < 8; i++) {
    vec3 t = uTrail[i];
    vec2 d = p - t.xy;
    float L2 = dot(d, d);
    float f = t.z * exp(-L2 / r2);
    acc += vec3(d * inversesqrt(L2 + 1.0) * f * 14.0, f * 5.0);
  }
  for (int i = 0; i < 4; i++) {
    vec4 w = uWaves[i];
    vec2 d = p - w.xy;
    float L = length(d) + 0.001;
    float off = (L - w.z * 720.0) / 75.0;
    float band = exp(-off * off) * w.w * exp(-w.z * 1.6);
    acc += vec3(d / L * band * 42.0, band * 18.0);
  }
  return acc;
}`;

// Pose da figura (px da imagem): aceno, inclinação, respiração e avanço. São 2,5D (o crânio do aceno é uma elipse com
// relevo analítico), com amplitudes pequenas de propósito. O giro (yaw) é 3D de verdade: ver "Cabeça 3D" abaixo.
export const POSE = `
uniform vec4 uPose;    // x: giro (rad) · y: aceno (rad, + = olhar pra baixo) · z: inclinação (rad) · w: respiração (-1..1)
uniform vec4 uPoseLag; // a mesma pose, atrasada por uma mola mole: a aura vem atrás da cabeça e balança
uniform vec3 uLean, uLeanLag;   // x, y: a cabeça vai na direção do mouse (px) · z: chega perto da tela (escala)
uniform vec2 uYaw;     // giro (rad, + = pra direita de quem olha) e sua cópia atrasada (aura); o giro 3D está mais abaixo
const float YAW_PIV = ${HEAD3D.PIVOT.toFixed(1)};                  // px: eixo do giro, atrás do centro da cabeça
// cx: quanto o giro levou a cabeça pro lado (px). A pose é aplicada DEPOIS do giro, então o crânio do aceno
// acompanha a cabeça girada; centrado na foto, a lateral girada saía da elipse e dobrava (faixa cinza no topo).
vec2 poseOf(vec2 p, vec4 P, vec3 L, float cx) {
  float head = 1.0 - smoothstep(${IMG.chinY.toFixed(1)}, ${NECK_Y.toFixed(1)}, p.y);   // 1 na cabeça, some ao longo do pescoço
  vec2 d = p - vec2(${IMG.axisX.toFixed(1)}, ${NECK_Y.toFixed(1)});
  float r = P.z * head;
  vec2 roll = vec2(cos(r) * d.x - sin(r) * d.y, sin(r) * d.x + cos(r) * d.y) - d;
  vec2 e = (p - vec2(${IMG.axisX.toFixed(1)} + cx, ${IMG.headCY.toFixed(1)})) / vec2(${HEAD_RX.toFixed(1)}, ${HEAD_RY.toFixed(1)});
  float z = sqrt(max(0.0, 1.0 - dot(e, e)));            // relevo: 1 no centro do rosto, 0 na borda
  // giro de verdade: cada ponto roda sobre uma esfera. O miolo do rosto anda, a silhueta quase não sai do
  // lugar, o lado que se afasta comprime e o que vem pra frente abre. Primeiro giro (x), depois aceno (y).
  float ex = e.x * cos(P.x) + z * sin(P.x);
  float z1 = z * cos(P.x) - e.x * sin(P.x);
  float ey = e.y * cos(P.y) + z1 * sin(P.y);
  vec2 turn = (vec2(ex, ey) - e) * vec2(${HEAD_RX.toFixed(1)}, ${HEAD_RY.toFixed(1)}) * head;
  // o pivô do giro é o pescoço, atrás do rosto: a cabeça inteira (borda, orelhas, silhueta) vira junto,
  // o miolo um pouco mais. Sem isso o rosto parecia deslizar dentro de uma máscara parada.
  turn += vec2(sin(P.x) * ${HEAD_RX.toFixed(1)} * 0.38, sin(P.y) * ${HEAD_RY.toFixed(1)} * 0.25) * head;
  float chest = smoothstep(${IMG.chinY.toFixed(1)}, ${IMG.h.toFixed(1)}, p.y);
  vec2 breath = P.w * vec2((p.x - ${IMG.axisX.toFixed(1)}) * 0.004 * chest, -(1.5 + 2.5 * chest));
  // avançar e se inclinar: o pescoço inteiro dobra (peso longo, sem quina) e os ombros acompanham um pouco
  float neck = 1.0 - smoothstep(${(IMG.chinY - 40).toFixed(1)}, ${(NECK_Y + 90).toFixed(1)}, p.y);
  vec2 lean = (L.xy + (p - vec2(${IMG.axisX.toFixed(1)}, ${IMG.headCY.toFixed(1)})) * L.z) * neck + L.xy * 0.12 * chest;
  return roll + turn + breath + lean;
}
vec2 pose(vec2 p) { return poseOf(p, uPose, uLean, YAW_PIV * sin(uYaw.x)); }
// Busto 3D (model3d.js): cada ponto chega com (x, y) de giro 0, z (pra frente), normal (nx, nz) e o quanto acompanha
// o giro (k: 1 na cabeça, o pescoço torcendo, pouco nos ombros). O giro é uma rotação 3D em torno de um eixo vertical
// YAW_PIV px atrás do centro da cabeça; projeção ortográfica. Partículas e malha escura usam a MESMA conta.
const float AX = ${IMG.axisX.toFixed(1)};
// (x na tela, z, normal z na tela): z serve pro teste de profundidade, a normal pra luz (1 = encara a câmera)
vec3 place3D(vec2 home, vec4 p3, float yaw) {
  float a = yaw * p3.w, c = cos(a), sn = sin(a);
  float rx = home.x - AX, rz = p3.x + YAW_PIV;
  return vec3(AX + rx * c + rz * sn, -rx * sn + rz * c - YAW_PIV, -p3.y * sn + p3.z * c);
}
// profundidade pro WebGL (perto = menor)
float depthOf(float z) { return clamp(-z / 1500.0, -0.999, 0.999); }
float turnShade(vec2 p) {                        // quanto o ponto passou a olhar pra câmera (+) ou pra longe (-)
  vec2 e = (p - vec2(${IMG.axisX.toFixed(1)}, ${IMG.headCY.toFixed(1)})) / vec2(${HEAD_RX.toFixed(1)}, ${HEAD_RY.toFixed(1)});
  float z = sqrt(max(0.0, 1.0 - dot(e, e)));
  float z1 = z * cos(uPose.x) - e.x * sin(uPose.x);
  float z2 = z1 * cos(uPose.y) - e.y * sin(uPose.y);
  return (z2 - z) * (1.0 - smoothstep(${IMG.chinY.toFixed(1)}, ${NECK_Y.toFixed(1)}, p.y));
}`;

export const QUAD_VS = `
attribute vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

export const BG_FS = `
precision highp float;
uniform vec2 uRes, uOffset, uImg, uPar;
uniform float uScale, uTime, uStill, uLevel, uHorizon;
uniform sampler2D uBg, uMasks;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  vec2 sp = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 p = (sp - uOffset) / uScale;
  float near = smoothstep(uHorizon, uImg.y, p.y);
  p += uPar * (1.5 + 5.0 * near);                // paralaxe: a água (perto) anda mais que o céu
  float live = 1.0 - uStill;
  vec2 uv = p / uImg;
  float fade = 1.0;                              // fora da foto: espelha a borda e escurece
  if (uv.y < 0.0) { fade *= exp(uv.y * 5.0); uv.y = min(-uv.y, 0.98); }
  if (uv.y > 1.0) { fade *= exp((1.0 - uv.y) * 4.0); uv.y = max(2.0 - uv.y, 0.02); }
  if (uv.x < 0.0) { fade *= exp(uv.x * 5.0); uv.x = min(-uv.x, 0.98); }
  if (uv.x > 1.0) { fade *= exp((1.0 - uv.x) * 5.0); uv.x = max(2.0 - uv.x, 0.02); }
  vec3 m = texture2D(uMasks, uv).rgb;            // r: brilhos pontuais · g: água · b: via láctea
  vec2 w = vec2(sin(p.y * 0.35 + uTime * 1.6) * 0.6, sin(p.x * 0.02 + uTime * 1.1) * 0.4) * m.g * live;
  vec2 nf = vec2(sin(uTime * 0.07 + p.y * 0.011), cos(uTime * 0.05 + p.x * 0.009)) * 0.8 * m.b * live;
  vec3 c = texture2D(uBg, uv + (w + nf) / uImg).rgb;
  float h = hash(floor(p));
  c *= 1.0 + m.r * 0.45 * sin(uTime * (1.2 + h * 3.0) + h * 60.0) * live;
  c *= 1.0 + m.g * 0.06 * sin(p.x * 0.05 + p.y * 0.2 - uTime * 2.0) * live;
  c += vec3(0.10, 0.35, 0.90) * uLevel * 0.22 * exp(-abs(p.y - uHorizon) * 0.08);
  c *= fade;
  c += vec3(0.75, 0.85, 1.0) * step(0.9965, hash(floor(sp / 2.5))) * (1.0 - fade) * (0.4 + 0.3 * sin(uTime * 2.0 + h * 40.0) * live);
  gl_FragColor = vec4(c, 1.0);
}`;

// Corpo da foto (a luz da figura: base escura, contorno neon, halo) na malha 3D: cada vértice lê a foto no ponto onde
// estava com giro 0, então parado é a foto exata. Na cabeça essa luz vale só junto com a vista de frente.
export const MESH_VS = `
precision highp float;
attribute vec2 aHome;
attribute vec4 aP3;
uniform vec2 uRes, uOffset, uPar, uImg;
uniform float uScale;
` + FIELD + POSE + `
varying vec2 vUv;
varying float vK;
void main() {
  vec3 q = place3D(aHome, aP3, uYaw.x);
  vec2 p = vec2(q.x, aHome.y);
  p += poseOf(p, uPose, uLean, YAW_PIV * sin(uYaw.x));
  p += field(p).xy + uPar * (4.0 + 3.0 * max(aP3.z, 0.0));   // a mesma paralaxe das partículas (relevo = de frente)
  vec2 s = p * uScale + uOffset;
  gl_Position = vec4(s.x / uRes.x * 2.0 - 1.0, 1.0 - s.y / uRes.y * 2.0, depthOf(q.y), 1.0);
  vUv = aHome / uImg;
  vK = aP3.w;
}`;

export const MESH_FS = `
precision mediump float;
uniform sampler2D uBodyRGB, uBodyA;
uniform float uVis, uFrontW;
varying vec2 vUv;
varying float vK;
void main() {
  // a luz da foto de frente vale na cabeça só enquanto a vista de frente pesa (as outras vistas trazem a própria luz);
  // a cabeça girada continua tampando o fundo
  vec3 rgb = texture2D(uBodyRGB, vUv).rgb * mix(1.0, uFrontW, vK);   // pré-multiplicado
  float a = max(texture2D(uBodyA, vUv).r, 0.8 * vK * (1.0 - uFrontW));
  gl_FragColor = vec4(rgb, a) * uVis;
}`;

// O resto do corpo da foto (halo e brilho que ficam fora da malha: em volta da cabeça, orelhas, ombros): tela cheia,
// desenhado só onde a malha não está (teste de profundidade). Some ao girar: o halo é da silhueta parada.
export const BODY_FS = `
precision highp float;
uniform vec2 uRes, uOffset, uImg, uPar;
uniform float uScale, uVis, uFrontW;
uniform sampler2D uBodyRGB, uBodyA;
` + FIELD + POSE + `
void main() {
  vec2 sp = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 p = (sp - uOffset) / uScale - uPar * 5.5;
  vec3 f = field(p);
  vec2 q = p - f.xy;
  vec2 h = q - pose(q);
  h = q - pose(h);
  h = q - pose(h);
  h = q - pose(h);                                // inverte a pose (ponto fixo, 4 passos)
  vec2 uv = h / uImg;
  float keep = mix(uFrontW, 1.0, smoothstep(${HEAD3D.TURN_Y[0].toFixed(1)}, ${HEAD3D.TURN_Y[1].toFixed(1)}, h.y));   // ombros: sempre
  if (uVis * keep < 0.002 || uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) discard;
  gl_FragColor = vec4(texture2D(uBodyRGB, uv).rgb, texture2D(uBodyA, uv).r) * uVis * keep;
}`;

export const PART_VS = `
precision highp float;
attribute vec2 aHome;
attribute vec4 aCol;              // rgb da foto · a: papel * 42/255 (0 miolo, 1 faixa, 2 lateral/nuca, 3 calota girando, 4 contorno do topo, 5 calota da foto)
attribute vec4 aInfo;            // relevo, borda, aura (0/1), semente
attribute vec3 aFrom;
attribute vec3 aTo;
attribute vec4 aP3;              // z (px, pra frente), normal (nx, nz), quanto acompanha o giro (model3d.js)
uniform vec2 uRes, uOffset, uPar, uMouth;
uniform float uScale, uTime, uStill, uMorphT0, uLevel, uThink, uListen, uGain, uHeadness, uAxisX, uHeadCY, uChinY;
uniform float uSetW, uMirror;   // peso do conjunto (vista) desta chamada · -1: vista espelhada (lado esquerdo)
uniform float uBands[8];
` + FIELD + POSE + `
varying vec3 vCol;
varying float vRound, vMv;
varying vec4 vSpr;               // tamanho do sprite (px) · tamanho do ponto parado (px) · meia-largura e meia-altura do retângulo
void main() {
  float depth = aInfo.x, rim = aInfo.y, aura = aInfo.z, seed = aInfo.w;
  float live = 1.0 - uStill;
  float hd = uHeadness;
  // troca de forma: cada partícula sai no seu tempo e faz uma curva no meio do caminho
  float k = smoothstep(0.0, 1.0, clamp((uTime - uMorphT0 - seed * 0.7) / 1.1, 0.0, 1.0));
  vec3 base = mix(aFrom, aTo, k);
  // vista espelhada: o lado esquerdo usa as vistas da direita refletidas no eixo da figura
  vec2 home = aHome;
  vec4 p3 = aP3;
  float role = floor(aCol.a * ${(255 / 42).toFixed(4)} + 0.5);
  if (uMirror < 0.0) {
    home.x = 2.0 * AX - home.x;
    base.x = 2.0 * AX - base.x;
    if (role < 3.5) p3.y = -p3.y;
  }
  float fly = sin(k * 3.14159265);
  base.xy += fly * vec2(sin(seed * 40.0 + uTime * 0.7), cos(seed * 31.0 + uTime * 0.6)) * 60.0;
  vec2 p = base.xy;

  // brilho dourado da boca: gira ao pensar, se expande com a voz
  float gold = clamp((aCol.r - aCol.b) * 2.5, 0.0, 1.0);
  vec2 fm = p - uMouth;
  float rm = length(fm) + 0.001;
  float ang = uThink * gold * 0.5 * sin(uTime * 1.3 - rm * 0.04) * hd;
  float ca = cos(ang), sa = sin(ang);
  fm = vec2(ca * fm.x - sa * fm.y, sa * fm.x + ca * fm.y);
  p = uMouth + fm + (fm / rm) * gold * uLevel * 16.0 * hd;

  // voz: cada faixa de altura pulsa com uma banda de frequência
  float yN = clamp((p.y - 120.0) / 820.0, 0.0, 0.999);
  float bnd = uBands[int(yN * 8.0)];
  vec2 nrm = normalize(vec2(p.x - uAxisX, p.y < uChinY ? (p.y - uHeadCY) * 0.9 : 0.0) + vec2(0.0001, 0.0));
  float wv = 0.5 + 0.5 * sin(p.y * 0.05 - uTime * 6.0 + seed * 6.28);
  p += nrm * (uLevel * 2.5 + bnd * 6.0) * (0.3 + 0.7 * seed) * (0.4 + 0.6 * wv) * hd;

  // giro 3D: onde o ponto vai parar menos onde estava com giro 0, aplicado sobre base (a troca de forma continua
  // funcionando com base fora de casa). A malha escura usa a mesma conta (place3D) e tampa o que fica atrás.
  // A luz já vem desenhada em cada vista: aqui só a posição e a profundidade mudam.
  float yawS = mix(uYaw.x, uYaw.y, aura) * hd;
  vec2 turned = base.xy;
  float lit = 1.0, zd = 0.0;
  bool solid = aura < 0.5;
  if (solid && role > 3.5) {                        // brilho fora da silhueta (halo, orelha): colado à borda girada
    float a = yawS * p3.w, c = cos(a), sn = sin(a);
    turned.x += (p3.x + YAW_PIV) * sn + (home.x - AX) * (sqrt(p3.y * p3.y * c * c + p3.z * p3.z * sn * sn) / max(p3.y, 1.0) - 1.0);
    zd = 2000.0;
  } else if (solid) {
    vec3 q = place3D(home, p3, yawS);
    turned.x += q.x - home.x;
    zd = q.y + 6.0 * clamp(q.z * 3.0, 0.0, 1.0);    // um pouco à frente da malha, só o que encara a câmera (o verso não vaza)
  } else {
    // aura: um halo em volta da silhueta; acompanha o centro da cabeça girada, sempre atrás do busto
    turned.x += YAW_PIV * sin(yawS * p3.w);
    zd = -400.0;
  }
  lit = mix(1.0, lit, hd);
  vMv = 0.0;
  float stretch = 1.0, wx = 1.0;
  // girando, a grade regular de pixels das vistas vira moiré (ainda mais com a tela menor que a imagem): um
  // deslocamento sub-pixel por partícula quebra o padrão. Parado: zero, a imagem fica exata
  float jit = solid ? smoothstep(0.004, 0.05, abs(yawS)) * hd : 0.0;
  turned += (vec2(fract(seed * 91.7), fract(seed * 57.3)) - 0.5) * 1.2 * jit;
  p += turned - base.xy;
  float sh = YAW_PIV * sin(yawS);                // o crânio do aceno vai com a cabeça girada (igual ao corpo)
  p += mix(poseOf(turned, uPose, uLean, sh), poseOf(turned, uPoseLag, uLeanLag, sh), aura) * hd;   // a pose vem antes do campo, igual ao corpo
  // pulsação leve do contorno externo: uma onda de luz sobe pela borda (o corpo não pulsa)
  float outline = smoothstep(0.35, 0.9, rim) * (1.0 - aura);
  float pulse = 0.5 + 0.5 * sin(uTime * 2.1 + p.y * 0.012 + seed * 0.6);
  p += nrm * outline * pulse * 1.1 * live * hd;
  vec3 f = field(p);
  // aura balançando: cada partícula flutua no seu ritmo, e todas juntas oscilam como alga na corrente,
  // mais longe da cabeça = mais solta
  float loose = 0.6 + 0.4 * clamp(length((base.xy - vec2(uAxisX, uHeadCY)) / vec2(260.0, 320.0)), 0.0, 1.5);
  vec2 drift = vec2(sin(uTime * (0.35 + seed * 0.45) + seed * 20.0) * 5.0, cos(uTime * (0.28 + seed * 0.4) + seed * 13.0) * 3.5);
  drift += vec2(sin(uTime * 0.42 + base.y * 0.006) * 6.0, sin(uTime * 0.31 + base.x * 0.005 + 1.7) * 2.5);
  drift *= aura * loose * live * hd;
  p += f.xy * (1.0 + aura * 0.3) + drift + uPar * (4.0 + depth * 3.0);

  vec2 s = p * uScale + uOffset;
  gl_Position = vec4(s.x / uRes.x * 2.0 - 1.0, 1.0 - s.y / uRes.y * 2.0, mix(-0.999, depthOf(zd), hd), 1.0);   // outras formas: na frente de tudo
  float pop = clamp(f.z / 25.0, 0.0, 1.5);
  float sz = uScale * (1.0 + pop * 0.5 + fly * 1.2 + (1.0 - hd) * 0.6) * (1.0 + uLean.z * 1.2 * hd);   // perto: pontos maiores
  // girando, a partícula sai do centro do pixel: vira um ponto um pouco maior e redondo (sem buracos nem xadrez)
  float mv = solid ? smoothstep(0.004, 0.05, abs(yawS)) : 0.0;
  float s0 = max(1.0, sz * (1.0 + 0.5 * mv));
  vSpr = vec4(s0, s0, 0.5 * sz * wx, 0.5 * sz);
  if (vMv > 0.0) vSpr.x = ceil(2.0 * max(vSpr.z, vSpr.w)) + 1.0;   // cobre todo pixel que o retângulo toca
  gl_PointSize = vSpr.x;

  float b = uGain * mix(2.4, 1.0, hd) * (1.0 + pop * 0.9);
  b *= 1.0 + 0.35 * base.z * (1.0 - hd);
  b *= 1.0 + hd * (gold * (uLevel * 1.6 + uThink * 0.6 * (0.5 + 0.5 * sin(uTime * 3.0))) + rim * uListen * 0.5 + uLevel * 0.25);
  b *= 1.0 + uThink * hd * 0.5 * pow(0.5 + 0.5 * sin(p.y * 0.03 + uTime * 4.0), 6.0);
  b *= 1.0 + live * 0.3 * pow(0.5 + 0.5 * sin(uTime * (0.8 + seed * 2.5) + seed * 50.0), 12.0);
  b *= 1.0 + live * hd * outline * 0.3 * pulse;
  b *= max(0.6, 1.0 + 0.35 * turnShade(base.xy) * hd);         // luz do microgiro
  b *= lit * uSetW;
  if (uScale < 1.0) b *= mix(uScale * uScale, 1.0, vMv);   // tela menor que a foto: conserva o brilho (a cobertura já conserva)
  vCol = aCol.rgb * b;
  vRound = clamp(length(f.xy) / 6.0 + pop + fly + (1.0 - hd) + mv, 0.0, 1.0);
}`;

export const PART_FS = `
precision mediump float;
varying vec3 vCol;
varying float vRound, vMv;
varying vec4 vSpr;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  vec2 o = c * vSpr.x;                                                // px a partir do centro da partícula
  vec2 co = vMv > 0.0 ? o / vSpr.y : c;                               // coordenada no ponto parado
  float sq = 1.0 - smoothstep(0.40, 0.5, max(abs(co.x), abs(co.y)));  // parado: um pixel exato da foto
  vec2 ov = clamp(min(o + 0.5, vSpr.zw) - max(o - 0.5, -vSpr.zw), 0.0, 1.0);
  sq = mix(sq, ov.x * ov.y, vMv);                                     // girando: área do retângulo dentro deste pixel
  float rd = clamp(1.0 - dot(co, co) * 4.0, 0.0, 1.0);                // em movimento: um ponto de luz redondo
  float a = mix(sq, rd * rd * 1.6, vRound);
  if (a <= 0.001) discard;
  gl_FragColor = vec4(vCol * a, 1.0);
}`;
