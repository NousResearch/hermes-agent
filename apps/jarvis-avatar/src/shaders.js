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
// Cabeça 3D. Cada linha da foto é uma fatia de um crânio sólido: uma elipse com a largura REAL da silhueta naquela
// linha (lida do alfa do corpo; nas orelhas, a do crânio) e profundidade YAW_K vezes essa largura. O giro (yaw) é
// uma rotação 3D de verdade em torno de um eixo vertical no pescoço, YAW_PIV px atrás do rosto; o pescoço torce
// aos poucos (headW) e os ombros não giram. Projeção ortográfica: giro 0 cai exatamente na foto.
//   miolo  (|dx| < wi): ponto da superfície; a frente vem da foto, a lateral e a nuca de partículas extras (aPhi)
//   faixa  (wi..w): a luz de borda da foto não é superfície, é silhueta: anda colada à borda NOVA, sem esticar
//   fora   (> w): orelha = placa que se estende pra trás (abre ao vir pra frente, some atrás do crânio);
//                 aura = translação junto com a borda
const float YAW_K = ${HEAD3D.K.toFixed(3)};                          // profundidade / largura do crânio (mais fundo que largo)
const float YAW_RIM = ${HEAD3D.RIM.toFixed(1)};                        // px: faixa da luz de borda
const float AX = ${IMG.axisX.toFixed(1)};
float headW(vec2 p) { return 1.0 - smoothstep(${IMG.chinY.toFixed(1)}, ${NECK_Y.toFixed(1)}, p.y); }
// peso das orelhas na linha y (a rampa fica FORA da orelha: dentro, cisalhava o topo dela)
float earW(float y) { return smoothstep(${(IMG.ears[0] - 24).toFixed(1)}, ${IMG.ears[0].toFixed(1)}, y) * (1.0 - smoothstep(${IMG.ears[1].toFixed(1)}, ${(IMG.ears[1] + 24).toFixed(1)}, y)); }
// quanto o miolo da linha é superfície que gira (1) ou luz de borda que só acompanha a silhueta (0). No alto do
// crânio a linha inteira é a luz de borda horizontal da foto: girar como superfície virava uma tampa clara.
float rotW(float y) { return smoothstep(${(IMG.headTop + 20).toFixed(1)}, ${(IMG.headTop + 95).toFixed(1)}, y); }
// fatia da linha: (meio, meia-largura com a faixa, meia-largura do miolo, meia-profundidade)
vec4 sliceOf(vec2 sil) {
  float m = 0.5 * (sil.x + sil.y), w = max(0.5 * (sil.y - sil.x), 0.001);
  float wi = w - min(YAW_RIM, 0.35 * w);
  return vec4(m, w, wi, YAW_K * wi);
}
// fatia girada (c, s = cos, sin do giro): (x do centro na tela, meia-largura projetada do miolo)
vec2 sliceRot(vec4 S, float c, float s) {
  return vec2(AX + (S.x - AX) * c + YAW_PIV * s, sqrt(S.z * S.z * c * c + S.w * S.w * s * s));
}
// ponto da superfície no ângulo phi (0 = de frente) girado: (x na tela, quanto encara a câmera: >0 visível)
vec2 surfRot(vec4 S, vec2 R, float phi, float c, float s) {
  float sp = sin(phi), cp = cos(phi);
  float nz = (S.z * cp * c - S.w * sp * s) / length(vec2(S.w * sp, S.z * cp));
  return vec2(R.x + S.z * sp * c + S.w * cp * s, nz);
}
// quanto a foto estica na tela no ângulo phi da fatia girada. Onde passa de STRETCH a foto (comprimida perto da
// borda) vira listras; ali a lateral extra e o miolo do corpo assumem. Virado pra trás: "infinito".
float frontStretch(vec4 S, float phi, float c, float s) {
  float cp = cos(phi), st = (S.z * cp * c - S.w * sin(phi) * s) / max(S.z * cp, 0.001);
  return cp < 0.05 || st < 0.0 ? 99.0 : st;
}
float takeover(float st) { return smoothstep(${HEAD3D.STRETCH[0].toFixed(2)}, ${HEAD3D.STRETCH[1].toFixed(2)}, st); }
// luz do giro: a cena é iluminada por trás (luz de borda). O que passa a encarar a câmera perde um pouco de brilho;
// o que fica rasante ganha. Parado: 1.
float yawLight(float stretch) { return clamp(pow(max(stretch, 0.01), -0.45), 0.62, 1.25); }
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
  // giro 3D: de cada pixel da tela de volta à foto, na linha dele. Miolo: o raio da câmera acerta a fatia girada
  // em phi = asin(u) - theta (a solução que encara a câmera); a cor vem da foto em m + wi*sin(phi), que na nuca
  // (|phi| > 90°) é o espelho da frente (cabeça lisa). Faixa e fora: a mesma distância até a borda nova.
  float turnLight = 1.0, genK = 0.0;
  vec4 genCol = vec4(0.0);
  if (abs(uYaw.x) > 0.0005) {
    float hw = headW(h);
    if (hw > 0.0) {
      float a = uYaw.x * hw, c = cos(a), sn = sin(a);
      vec4 S = sliceOf(silAt(h.y));
      vec2 R = sliceRot(S, c, sn);
      float D = h.x - R.x, aD = abs(D), sg = D < 0.0 ? -1.0 : 1.0;
      if (aD < R.y) {
        float psi = asin(clamp(D / R.y, -1.0, 1.0));
        float phi = psi - atan(S.w * sn, S.z * c);
        float rw = rotW(h.y);
        // onde a foto esticaria demais (lateral) e na nuca, a base escura não vem da beirada da foto (a luz de borda
        // esticada lavava a cabeça de branco e fazia listras): é a média do miolo da linha, tingida de azul
        float g = takeover(frontStretch(S, phi, c, sn)) * rw;
        float ts = 1.0 - rw;                              // alto do crânio: só acompanha a silhueta
        h.x = S.x + mix(S.z * sin(phi), D * S.z / R.y, ts);
        turnLight = mix(1.0, 0.8, smoothstep(1.3, 2.2, abs(phi)) * rw);   // nuca: um pouco mais escura que a frente
        if (g > 0.0) {
          vec4 acc = vec4(0.0);
          for (int i = 0; i < 4; i++) {
            vec2 uvg = vec2(S.x + S.z * (-0.6 + 0.4 * float(i)), h.y) / uImg;
            acc += vec4(texture2D(uBodyRGB, uvg).rgb, texture2D(uBodyA, uvg).r);
          }
          acc *= 0.25;
          genCol = vec4(vec3(0.35, 0.75, 1.3) * 0.7 * dot(acc.rgb, vec3(0.3, 0.5, 0.2)), acc.a);
          genK = g;
        }
      } else {
        h.x = S.x + sg * (S.z + aD - R.y);              // faixa da luz de borda e fora: colados à borda nova
      }
    }
  }
  vec2 uv = h / uImg;
  if (uVis < 0.002 || uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) discard;
  float a = mix(texture2D(uBodyA, uv).r, genCol.a, genK);
  vec3 rgb = mix(texture2D(uBodyRGB, uv).rgb, genCol.rgb, genK) * turnLight;   // já pré-multiplicado
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
attribute float aPhi;            // > 4: partícula da foto · senão: partícula da lateral/nuca, no ângulo aPhi da fatia
uniform vec2 uRes, uOffset, uPar, uMouth;
uniform float uScale, uTime, uStill, uMorphT0, uLevel, uThink, uListen, uGain, uShapeK, uHeadness, uAxisX, uHeadCY, uChinY;
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

  // giro 3D. Calcula onde o ponto vai parar (X) e onde estava com giro 0 (X0) e aplica a diferença sobre base, então
  // a troca de forma (base fora de casa) continua funcionando. Mesma fatia e mesma conta da inversa do corpo.
  float yawS = mix(uYaw.x, uYaw.y, aura) * hd;
  vec2 turned = base.xy;
  float vis = 1.0;                 // 0 = virado pra trás / atrás do crânio
  float stretch = 1.0;             // quanto a superfície estica (>1, vem pra frente) ou encolhe (<1) na tela
  bool extra = aPhi < 4.0;         // lateral/nuca: não existe na foto
  vMv = 0.0;
  if (abs(yawS) > 0.0005) {
    float hw = headW(aHome), a = yawS * hw, c = cos(a), sn = sin(a);
    vec4 S = sliceOf(aSil);
    vec2 R = sliceRot(S, c, sn);
    float dx = aHome.x - S.x, ad = abs(dx), sg = dx < 0.0 ? -1.0 : 1.0;
    float X0 = aHome.x, X;
    if (extra) {
      vec2 q = surfRot(S, R, aPhi, c, sn);
      X0 = S.x + S.z * sin(aPhi);
      X = q.x;
      // só onde a foto esticaria demais ou não existe, e só girando: parado, a soma continua sendo a foto
      vis = smoothstep(0.0, 0.12, q.y) * takeover(frontStretch(S, aPhi, c, sn)) * smoothstep(0.02, 0.12, abs(a)) * rotW(aHome.y);
      stretch = 1.0;
    } else if (ad < S.z) {
      float phi = asin(clamp(dx / S.z, -1.0, 1.0)), rw = rotW(aHome.y);
      vec2 q = surfRot(S, R, phi, c, sn);
      X = mix(R.x + dx * R.y / S.z, q.x, rw);                        // alto do crânio: só acompanha a silhueta
      stretch = mix(R.y / S.z, (S.z * cos(phi) * c - S.w * sin(phi) * sn) / max(S.z * cos(phi), 0.001), rw);
      vis = mix(1.0, smoothstep(0.0, 0.12, q.y) * (1.0 - takeover(stretch)), rw);   // muito esticado: a lateral extra assume
    } else {
      float aura0 = aHome.x + (R.x - S.x) + sg * (R.y - S.z);      // faixa da luz de borda e aura: coladas à borda nova
      float u = ad - S.y, e = earW(aHome.y) * step(0.0, u) * (1.0 - smoothstep(30.0, 60.0, u));
      // orelha: placa presa na lateral do crânio que se estende pra trás (z = -1,4 u). Vista de lado ela abre;
      // indo pra trás, o que cai dentro da silhueta nova some atrás do crânio.
      float z = -1.4 * u;
      float xe = AX + (aHome.x - AX) * c + (z + YAW_PIV) * sn;
      float ze = -(aHome.x - AX) * sn + (z + YAW_PIV) * c - YAW_PIV;
      float behind = (1.0 - smoothstep(-2.0, 2.0, abs(xe - R.x) - R.y - (S.y - S.z))) * (1.0 - smoothstep(-6.0, 0.0, ze));
      X = mix(aura0, xe, e);
      vis = 1.0 - e * behind;
    }
    turned.x = base.x + (X - X0);
    // girando, a partícula cai fora do centro do pixel: em vez do quadrado de pixel (que apaga ou dobra com meio
    // pixel de deslocamento: pontilhado cinza, xadrez), ela vira um retângulo pintado pela área que cobre de cada pixel
    vMv = smoothstep(0.0005, 0.003, abs(yawS));
  } else if (extra) vis = 0.0;
  if (extra) vis = mix(1.0, vis, hd);                            // nas outras formas (esfera, galáxia) ela aparece
  // onde a superfície estica s vezes, há 1/s partícula por pixel: o retângulo tem largura s (luz por área = a da
  // foto, sem buracos nem empilhamento). Abaixo de 0,3 só apaga.
  float wx = clamp(stretch, 0.3, 2.5), dens = min(stretch / wx, 1.0);
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

  float b = uGain * mix(2.4 * uShapeK, 1.0, hd) * (1.0 + pop * 0.9);
  b *= 1.0 + 0.35 * base.z * (1.0 - hd);
  b *= 1.0 + hd * (gold * (uLevel * 1.6 + uThink * 0.6 * (0.5 + 0.5 * sin(uTime * 3.0))) + rim * uListen * 0.5 + uLevel * 0.25);
  b *= 1.0 + uThink * hd * 0.5 * pow(0.5 + 0.5 * sin(p.y * 0.03 + uTime * 4.0), 6.0);
  b *= 1.0 + live * 0.3 * pow(0.5 + 0.5 * sin(uTime * (0.8 + seed * 2.5) + seed * 50.0), 12.0);
  b *= 1.0 + live * hd * outline * 0.3 * pulse;
  b *= max(0.6, 1.0 + 0.35 * turnShade(base.xy) * hd);         // luz do microgiro
  b *= dens * yawLight(stretch) * vis;                                 // luz por área constante (ver wx/dens) + luz do giro
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
