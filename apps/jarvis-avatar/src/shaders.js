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
vec2 poseOf(vec2 p, vec4 P, vec3 L) {
  float head = 1.0 - smoothstep(${IMG.chinY.toFixed(1)}, ${NECK_Y.toFixed(1)}, p.y);   // 1 na cabeça, some ao longo do pescoço
  vec2 d = p - vec2(${IMG.axisX.toFixed(1)}, ${NECK_Y.toFixed(1)});
  float r = P.z * head;
  vec2 roll = vec2(cos(r) * d.x - sin(r) * d.y, sin(r) * d.x + cos(r) * d.y) - d;
  vec2 e = (p - vec2(${IMG.axisX.toFixed(1)}, ${IMG.headCY.toFixed(1)})) / vec2(${HEAD_RX.toFixed(1)}, ${HEAD_RY.toFixed(1)});
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
vec2 pose(vec2 p) { return poseOf(p, uPose, uLean); }
// Giro 3D da cabeça. A cabeça do Jarvis é lisa (manequim, sem olhos nem nariz), então um elipsoide é quase exato.
// O pivô fica no pescoço, atrás do rosto: além de girar, a cabeça anda um pouco pro lado (YAW_PIVOT).
uniform vec2 uYaw;     // giro (rad, + = pra direita de quem olha) e sua cópia atrasada (aura)
const vec2 HC = vec2(${IMG.axisX.toFixed(1)}, ${IMG.headCY.toFixed(1)});
const vec2 HR = vec2(${HEAD_RX.toFixed(1)}, ${HEAD_RY.toFixed(1)});
const float YAW_PIVOT = 0.22;
float headW(vec2 p) { return 1.0 - smoothstep(${IMG.chinY.toFixed(1)}, ${NECK_Y.toFixed(1)}, p.y); }
float earSquash(float ex, float c) { return mix(c, 1.0, smoothstep(1.1, 1.4, abs(ex))); }   // orelhas: cilindro raso
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
  float shade = 1.0 + 0.35 * turnShade(h);
  // giro 3D, inversa exata: o ponto visível da tela é rodado de volta até a foto de frente.
  // Duas leituras inteiras, misturadas só na COR (misturar posições arrastava o brilho da borda em listras):
  // o rosto girado e a borda parada na silhueta (luz de borda não anda com a superfície).
  float bandK = 0.0, edgeK = 0.0;
  vec2 hEdge = h;
  if (abs(uYaw.x) > 0.0005) {
    float c = cos(uYaw.x), sn = sin(uYaw.x), hw = headW(h);
    vec2 piv = vec2(sn * HR.x * YAW_PIVOT * hw, 0.0);
    vec2 e2 = (h - HC - piv) / HR;
    float r2 = dot(e2, e2);
    vec2 e = vec2(e2.x / earSquash(e2.x, c), e2.y);
    if (r2 < 1.0) e.x = e2.x * c - sqrt(1.0 - r2) * sn;
    edgeK = smoothstep(0.84, 0.98, sqrt(r2)) * hw;
    hEdge = h - piv;                                   // borda: só acompanha o pivô do pescoço
    // a leitura que veio da borda da foto PRA DENTRO (a lateral que se abre) vira base escura; o detalhe ali
    // vem das partículas do verso da cabeça. Na borda que não andou, nada muda.
    bandK = 0.9 * smoothstep(0.78, 0.94, length(e)) * smoothstep(0.04, 0.16, length(e) - length(e2)) * hw;
    h = mix(h, e * HR + HC, hw);
  }
  vec2 uv = h / uImg;
  if (uVis < 0.002 || uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) discard;
  float a = texture2D(uBodyA, uv).r;
  vec3 rgb = texture2D(uBodyRGB, uv).rgb;        // já pré-multiplicado (brilho do contorno tem alpha 0 = aditivo)
  if (bandK > 0.0) rgb = mix(rgb, vec3(0.018, 0.05, 0.1), bandK);
  if (edgeK > 0.0) {
    vec2 ue = hEdge / uImg;
    rgb = mix(rgb, texture2D(uBodyRGB, ue).rgb, edgeK);
    a = mix(a, texture2D(uBodyA, ue).r, edgeK);
  }
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
uniform vec2 uRes, uOffset, uPar, uMouth;
uniform float uScale, uTime, uStill, uMorphT0, uLevel, uThink, uListen, uGain, uHeadness, uAxisX, uHeadCY, uChinY;
uniform float uBands[8];
uniform float uBack;             // 1 no segundo passe: o verso da cabeça (espelho), só aparece quando ela gira
` + FIELD + POSE + `
varying vec3 vCol;
varying float vRound;
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

  // giro 3D: cada partícula é um ponto na superfície da cabeça e roda de verdade
  float yawS = mix(uYaw.x, uYaw.y, aura) * hd;
  float yc = cos(yawS), ysn = sin(yawS), hw = headW(base.xy);
  vec2 e = (base.xy - HC) / HR;
  float r2 = dot(e, e);
  float inside = step(r2, 1.0);
  float zf = sqrt(max(0.0, 1.0 - r2)) * (uBack > 0.5 ? -1.0 : 1.0);
  float rimW = smoothstep(0.35, 0.9, rim) * (1.0 - aura);         // contorno: fica na silhueta, não gira (luz de borda)
  float ex = inside > 0.5 ? e.x * yc + zf * ysn : e.x * earSquash(e.x, yc);
  float z2 = inside > 0.5 ? zf * yc - e.x * ysn : -e.x * ysn;
  ex = mix(ex, e.x, rimW);
  vec2 turned = mix(base.xy, vec2(ex, e.y) * HR + HC, hw) + vec2(ysn * HR.x * YAW_PIVOT * hw, 0.0);
  // o que vai pra trás da cabeça some (em vez de se empilhar na borda)
  // frente: parado = 1 exato (pontos perto da borda já começam quase de lado); verso: só o que virou pra câmera
  float zoff = uBack > 0.5 ? 0.0 : max(0.16 - zf, 0.0);
  float facing = mix(1.0, smoothstep(0.0, 0.16, z2 + zoff), hw * (1.0 - aura) * (1.0 - rimW));
  float backOk = uBack * inside * step(0.5, hw) * (1.0 - aura) * (1.0 - rimW) * smoothstep(0.03, 0.12, abs(yawS));
  // brilho interno do contorno (faixa logo dentro da borda na foto): é luz de borda, então apaga quando o giro
  // o traz pra dentro do rosto (senão vira uma segunda linha clara)
  float nearRim = smoothstep(0.82, 0.96, sqrt(r2)) * inside * (1.0 - rimW) * (1.0 - aura) * hw;
  // lateral da cabeça que se abre ao girar: mantém as partículas, mas com brilho de miolo (não de borda)
  float sideOpen = (1.0 - uBack) * nearRim * smoothstep(0.03, 0.15, abs(yawS)) * smoothstep(0.0, 0.12, z2 - zf);
  // essa faixa se estica ao vir pra frente: as partículas crescem na mesma medida pra não abrir buracos
  float spread = mix(1.0, clamp(z2 / max(zf, 0.1), 1.0, 1.8), (1.0 - uBack) * hw * (1.0 - rimW) * (1.0 - aura) * inside);
  p += turned - base.xy;
  // as partículas vêm da grade de pixels da foto: onde a lateral estica (e no verso) a grade regular vira ondas
  // (moiré). Um embaralhamento leve, proporcional ao esticamento, quebra o padrão. Parado: zero.
  float jit = (max(spread, 1.0 + uBack * 0.8) - 1.0) * 2.2;
  p += (vec2(fract(seed * 91.7), fract(seed * 57.3)) - 0.5) * jit;
  p += mix(pose(turned), poseOf(turned, uPoseLag, uLeanLag), aura) * hd;   // a pose vem antes do campo, igual ao corpo
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
  gl_PointSize = max(1.0, uScale * (1.0 + pop * 0.5 + fly * 1.2 + (1.0 - hd) * 0.6) * (1.0 + uLean.z * 1.2 * hd) * spread);   // perto/esticado: pontos maiores, sem buracos

  float b = uGain * mix(2.4, 1.0, hd) * (1.0 + pop * 0.9);
  b *= 1.0 + 0.35 * base.z * (1.0 - hd);
  b *= 1.0 + hd * (gold * (uLevel * 1.6 + uThink * 0.6 * (0.5 + 0.5 * sin(uTime * 3.0))) + rim * uListen * 0.5 + uLevel * 0.25);
  b *= 1.0 + uThink * hd * 0.5 * pow(0.5 + 0.5 * sin(p.y * 0.03 + uTime * 4.0), 6.0);
  b *= 1.0 + live * 0.3 * pow(0.5 + 0.5 * sin(uTime * (0.8 + seed * 2.5) + seed * 50.0), 12.0);
  b *= 1.0 + live * hd * outline * 0.3 * pulse;
  b *= max(0.6, 1.0 + 0.35 * turnShade(base.xy) * hd);         // luz do microgiro
  b *= facing * (1.0 + 0.35 * (z2 - zf) * hw * (1.0 - rimW));   // o lado que vem pra frente acende (zero de diferença parado)
  b *= 1.0 - 0.55 * sideOpen;                                   // brilho de borda vira brilho de miolo
  b /= spread;                                                   // ponto maior cobre mais área: compensa pra não clarear
  if (uBack > 0.5) {                                            // verso: só partículas do miolo, sem o dourado da boca
    b *= backOk * 0.85;
    vec3 blue = vec3(0.12, 0.42, 0.85) * dot(aCol, vec3(0.3, 0.5, 0.2)) * 1.6;
    vCol = mix(aCol, blue, gold) * b;
    if (backOk < 0.001) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; }
    vRound = 0.0;
    return;
  }
  if (uScale < 1.0) b *= uScale * uScale;       // tela menor que a foto: conserva o brilho
  vCol = aCol * b;
  vRound = clamp(length(f.xy) / 6.0 + pop + fly + (1.0 - hd), 0.0, 1.0);
}`;

export const PART_FS = `
precision mediump float;
varying vec3 vCol;
varying float vRound;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float sq = 1.0 - smoothstep(0.40, 0.5, max(abs(c.x), abs(c.y)));   // parado: um pixel exato da foto
  float rd = clamp(1.0 - dot(c, c) * 4.0, 0.0, 1.0);                  // em movimento: um ponto de luz redondo
  float a = mix(sq, rd * rd * 1.6, vRound);
  if (a <= 0.001) discard;
  gl_FragColor = vec4(vCol * a, 1.0);
}`;
