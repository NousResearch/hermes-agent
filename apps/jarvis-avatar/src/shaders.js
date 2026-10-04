/* Shaders GLSL (WebGL 1). Coordenadas em pixels da imagem de referência (IMG). */
import { IMG, HEAD_RX, HEAD_RY, NECK_Y } from './config.js';

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

// Pose da figura (px da imagem). A foto é frontal, então o "giro" é 2,5D: o crânio vira uma elipse
// com relevo analítico; pontos no meio do rosto andam mais que a borda, o que lê como a cabeça virando.
// Amplitudes pequenas de propósito: giro grande revelaria que a foto é plana.
export const POSE = `
uniform vec4 uPose;    // x: giro (rad) · y: aceno (rad, + = olhar pra baixo) · z: inclinação (rad) · w: respiração (-1..1)
uniform vec4 uPoseLag; // a mesma pose, atrasada por uma mola mole: a aura vem atrás da cabeça e balança
uniform vec3 uLean, uLeanLag;   // x, y: a cabeça vai na direção do mouse (px) · z: chega perto da tela (escala)
uniform vec2 uYaw;     // giro (rad, + = pra direita de quem olha) e sua cópia atrasada (aura); o giro 3D está mais abaixo
const float YAW_PIV = ${(0.45 * HEAD_RX).toFixed(1)};                  // px: eixo do pescoço atrás do centro da cabeça
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
// Giro 3D da cabeça (yaw). Cada linha da foto é um corte do crânio: uma elipse com a largura REAL da silhueta
// naquela linha (lida do alfa do corpo; nas orelhas, a do crânio) e profundidade YAW_K vezes a largura, girando em torno
// do pescoço (YAW_PIV px atrás do centro). Do modelo físico vêm: a silhueta nova (anda e alarga), e o quanto o
// miolo do rosto anda. A luz de borda da foto não é superfície, é silhueta: perto da borda a superfície "desliza"
// até a borda da foto cair exatamente na borda nova. O mapa por linha é contínuo e monotônico, então
// não dobra, não duplica contorno e não abre buraco; corpo (inversa) e partículas (direta) usam a MESMA função.
const float YAW_K = 1.15;                         // profundidade / largura do crânio (mais fundo que largo)
float headW(vec2 p) { return 1.0 - smoothstep(${IMG.chinY.toFixed(1)}, ${NECK_Y.toFixed(1)}, p.y); }
// constantes do giro: (meia-largura nova / antiga, quanto o miolo roda no corte, quanto a cabeça anda em px, cos)
vec4 yawParams(float c, float sn) {
  return vec4(sqrt(c * c + YAW_K * YAW_K * sn * sn), atan(YAW_K * sn, c), YAW_PIV * sn, c);
}
// peso das orelhas na linha y: 1 em toda a orelha (IMG.ears), a rampa fica FORA dela (acima/abaixo só há aura
// colada ao crânio); rampa dentro da orelha cisalhava o topo dela em diagonal
float earW(float y) { return smoothstep(${(IMG.ears[0] - 24).toFixed(1)}, ${IMG.ears[0].toFixed(1)}, y) * (1.0 - smoothstep(${IMG.ears[1].toFixed(1)}, ${(IMG.ears[1] + 24).toFixed(1)}, y)); }
// quanto o miolo da linha gira (1) ou só translada com a silhueta (0). No alto do crânio a linha inteira é a luz
// de borda horizontal da foto (silhueta, não superfície; e a calota olha pra cima, quase não roda): girar
// esticava essa faixa numa tampa cinza pontilhada. Sobe suave até a testa.
float rotW(float y) { return smoothstep(${(IMG.headTop + 20).toFixed(1)}, ${(IMG.headTop + 95).toFixed(1)}, y); }
const float YAW_RIM = 38.0;                        // px: faixa da luz de borda (só translada com a silhueta, não estica)
const float EAR_E1 = 30.0, EAR_E2 = 100.0;         // px além do crânio: fim da orelha · fim da volta à translação
// Fora da silhueta o mapa é linear por trechos, R = L.x * ad + L.y (ad = distância da foto ao meio da linha):
// 0: placa rígida da orelha que se afasta (encolhe por cos) · 1: volta suave à translação · 2: translação da borda.
// e = peso da orelha do lado que se afasta (0 no resto: tudo é translação). Linear por trechos = inversa exata no corpo.
vec2 outLin(float piece, float w, vec4 Y, float e) {
  vec2 t = vec2(1.0, w * (Y.x - 1.0));
  if (piece > 1.5) return t;
  float a1 = w + EAR_E1, a2 = w + EAR_E2;
  vec2 q = vec2(Y.w, 0.0);
  if (piece > 0.5) { float al = (a2 + t.y - a1 * Y.w) / (a2 - a1); q = vec2(al, a1 * Y.w - al * a1); }
  return mix(t, q, e);
}
// x da foto -> x na tela, na linha cuja silhueta é sil = (borda esquerda, borda direita) em px
float yawMapX(float x, vec2 sil, vec4 Y, float hw, vec2 er) {   // er: (earW, rotW) da linha
  float ear = er.x;
  float m = 0.5 * (sil.x + sil.y), w0 = 0.5 * (sil.y - sil.x), w = max(w0, 0.001);
  float B = min(YAW_RIM, 0.35 * w), wi = w - B, Yi = (w * Y.x - B) / wi;
  float d = x - m, ad = abs(d), sg = d < 0.0 ? -1.0 : 1.0, X;
  if (ad < wi) {
    // miolo: giro físico num corte de meia-largura wi que vai parar em w*Y.x - B (a faixa da borda fica de fora)
    float ph = asin(clamp(d / wi, -1.0, 1.0));       // clamp: alguns compiladores avaliam os dois ramos (NaN fora)
    float k = (1.0 - smoothstep(0.0, 1.15, abs(ph))) * er.y; // 1 no miolo (giro físico) -> 0 perto da faixa (sem dobra)
    X = m + Y.z + wi * Yi * sin(ph + Y.y * k);
  } else {
    // faixa da borda e fora da silhueta: acompanha a borda nova sem esticar. Orelha do lado que se afasta: placa
    // rígida (encolhe por cos); o que cai dentro da silhueta nova está atrás do crânio (o corpo nunca o escolhe,
    // as partículas apagam). Passada a orelha, volta à translação: a aura não cisalha.
    float e = ad > w && sg * Y.z > 0.0 ? ear : 0.0;
    vec2 L = outLin(ad < w + EAR_E1 ? 0.0 : (ad < w + EAR_E2 ? 1.0 : 2.0), w, Y, e);
    X = m + Y.z + sg * (L.x * ad + L.y);
  }
  return mix(x, X, hw);
}
// luz do giro: a cena é iluminada por trás (luz de borda). O lado que vem pra frente (a linha estica) passa a
// encarar a câmera e perde brilho; o que vai pra trás (encolhe) fica mais rasante e ganha um pouco. Parado: 1.
float yawLight(float stretch) { return clamp(pow(stretch, -0.45), 0.62, 1.25); }
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

export const BODY_FS = `
precision highp float;
uniform vec2 uRes, uOffset, uImg, uPar;
uniform float uScale, uVis;
uniform sampler2D uBodyRGB, uBodyA;
uniform sampler2D uSil;   // silhueta por linha da foto (1 x ${IMG.h}): RG = borda esquerda, BA = direita (px * 32, 16 bits)
` + FIELD + POSE + `
float silDec(vec2 hl) { return (floor(hl.x * 255.0 + 0.5) * 256.0 + floor(hl.y * 255.0 + 0.5)) / 32.0; }
vec2 silRow(float row) {
  vec4 t = texture2D(uSil, vec2(0.5, (clamp(row, 0.0, ${(IMG.h - 1).toFixed(1)}) + 0.5) / ${IMG.h.toFixed(1)}));
  return vec2(silDec(t.xy), silDec(t.zw));
}
vec2 silAt(float y) {                           // interpolada entre centros de linha (= o valor exato da linha da partícula)
  float t = y - 0.5, r0 = floor(t);
  return mix(silRow(r0), silRow(r0 + 1.0), t - r0);
}
void main() {
  vec2 sp = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 p = (sp - uOffset) / uScale - uPar * 5.5;
  vec3 f = field(p);
  vec2 q = p - f.xy;
  vec2 h = q - pose(q);
  h = q - pose(h);
  h = q - pose(h);
  h = q - pose(h);                                // inverte a pose (ponto fixo, 4 passos)
  float shade = 1.0 + 0.35 * turnShade(h);
  // giro 3D: inversa do mapa das partículas (yawMapX). Fora da silhueta ele é linear por trechos (outLin: inversa
  // exata do trecho certo); dentro, bisseção entre as bordas da linha (o mapa é monotônico, a raiz é única).
  float turnLight = 1.0;
  if (abs(uYaw.x) > 0.0005) {
    float hw = headW(h);
    if (hw > 0.0) {
      vec4 Y = yawParams(cos(uYaw.x), sin(uYaw.x));
      vec2 sil = silAt(h.y);
      float ear = earW(h.y);
      vec2 er = vec2(ear, rotW(h.y));
      float m = 0.5 * (sil.x + sil.y), w0 = 0.5 * (sil.y - sil.x);
      float eL = mix(sil.x, m + Y.z - w0 * Y.x, hw), eR = mix(sil.y, m + Y.z + w0 * Y.x, hw);
      if (h.x <= eL || h.x >= eR) {
        float sg = h.x <= eL ? -1.0 : 1.0;
        float D = sg * (h.x - m - hw * Y.z);            // distância nova ao meio da linha
        float e = sg * Y.z > 0.0 ? ear : 0.0;           // orelha do lado que se afasta: inversa da placa rígida
        float w = max(w0, 0.001);
        // posição nova de ad no trecho L: ad + hw * (R - ad); o trecho é o primeiro cuja ponta passa de D
        vec2 L0 = outLin(0.0, w, Y, e), L1 = outLin(1.0, w, Y, e), L = outLin(2.0, w, Y, e);
        float a1 = w + EAR_E1, a2 = w + EAR_E2;
        if (D < a2 + hw * (L1.x * a2 + L1.y - a2)) L = L1;
        if (D < a1 + hw * (L0.x * a1 + L0.y - a1)) L = L0;
        h.x = m + sg * (D - hw * L.y) / (1.0 + hw * (L.x - 1.0));
      }
      else {
        float lo = sil.x, hi = sil.y;
        for (int i = 0; i < 13; i++) {   // meia-largura <= ~260 px: 13 passos = 0,06 px
          float mid = 0.5 * (lo + hi);
          if (yawMapX(mid, sil, Y, hw, er) < h.x) lo = mid; else hi = mid;
        }
        h.x = 0.5 * (lo + hi);
        turnLight = yawLight(yawMapX(h.x + 0.5, sil, Y, hw, er) - yawMapX(h.x - 0.5, sil, Y, hw, er));
      }
    }
  }
  vec2 uv = h / uImg;
  if (uVis < 0.002 || uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) discard;
  float a = texture2D(uBodyA, uv).r;
  vec3 rgb = texture2D(uBodyRGB, uv).rgb * turnLight;   // já pré-multiplicado
  float vis = uVis * (1.0 - clamp(length(f.xy) / 70.0, 0.0, 0.55));
  gl_FragColor = vec4(rgb * shade, a) * vis;
}`;

export const PART_VS = `
precision highp float;
attribute vec2 aHome;
attribute vec3 aCol;
attribute vec4 aInfo;            // relevo, borda, aura (0/1), semente
attribute vec3 aFrom;
attribute vec3 aTo;
attribute vec2 aSil;             // silhueta (borda esquerda, direita; px) da linha de origem: a mesma que o corpo lê em uSil
uniform vec2 uRes, uOffset, uPar, uMouth;
uniform float uScale, uTime, uStill, uMorphT0, uLevel, uThink, uListen, uGain, uHeadness, uAxisX, uHeadCY, uChinY;
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

  // giro 3D: o mesmo mapa por linha do corpo (yawMapX), com a silhueta da linha de origem da partícula
  float yawS = mix(uYaw.x, uYaw.y, aura) * hd;
  vec2 turned = base.xy;
  float hidden = 0.0;              // 1 = atrás do crânio (orelha que se afasta)
  float stretch = 1.0;             // quanto a linha estica (>1, lado que vem pra frente) ou encolhe (<1) aqui
  vMv = 0.0;
  if (abs(yawS) > 0.0005) {
    vec4 Y = yawParams(cos(yawS), sin(yawS));
    float hw = headW(base.xy), ear = earW(base.y);
    vec2 er = vec2(ear, rotW(base.y));
    turned.x = yawMapX(base.x, aSil, Y, hw, er);
    stretch = yawMapX(base.x + 0.5, aSil, Y, hw, er) - yawMapX(base.x - 0.5, aSil, Y, hw, er);
    // orelha que se afasta: o que caiu dentro da silhueta nova está atrás do crânio
    float m = 0.5 * (aSil.x + aSil.y), sg = base.x < m ? -1.0 : 1.0;
    if (sg * Y.z > 0.0 && abs(base.x - m) > 0.5 * (aSil.y - aSil.x))
      hidden = ear * (1.0 - smoothstep(-0.75, 0.75, sg * (turned.x - m - hw * Y.z) - 0.5 * (aSil.y - aSil.x) * mix(1.0, Y.x, hw)));
    // girando, a partícula cai fora do centro do pixel: em vez do quadrado de pixel (que apaga ou dobra com meio
    // pixel de deslocamento: pontilhado cinza, xadrez), ela vira um retângulo pintado pela área que cobre de cada pixel
    vMv = smoothstep(0.0005, 0.003, abs(yawS));
  }
  // onde a linha estica/encolhe s vezes, há 1/s partícula por pixel: o retângulo tem largura s (luz por área = a da
  // foto, sem buracos nem empilhamento). Abaixo de 0,3 só apaga.
  float wx = clamp(stretch, 0.3, 2.5), dens = stretch / wx;
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
  gl_Position = vec4(s.x / uRes.x * 2.0 - 1.0, 1.0 - s.y / uRes.y * 2.0, 0.0, 1.0);
  float pop = clamp(f.z / 25.0, 0.0, 1.5);
  float sz = uScale * (1.0 + pop * 0.5 + fly * 1.2 + (1.0 - hd) * 0.6) * (1.0 + uLean.z * 1.2 * hd);   // perto: pontos maiores
  float s0 = max(1.0, sz);
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
  b *= dens * yawLight(stretch) * (1.0 - hidden);                                 // luz por área constante (ver wx/dens) + luz do giro
  if (uScale < 1.0) b *= mix(uScale * uScale, 1.0, vMv);   // tela menor que a foto: conserva o brilho (a cobertura já conserva)
  vCol = aCol * b;
  vRound = clamp(length(f.xy) / 6.0 + pop + fly + (1.0 - hd), 0.0, 1.0);
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
