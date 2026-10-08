// Nyx — modo preenchimento: o busto em pontilhismo pintado por um shader de tela, não por milhões de partículas.
//
// A ideia é a do "neural rendering" (DLSS 5), sem a rede neural: a geometria só dá a estrutura e um passo final
// pinta a aparência. A cada quadro:
//   1. a malha guia (a forma do busto na pele, invisível) é desenhada numa textura: a normal e a parte do corpo
//      de cada pixel, e a profundidade;
//   2. um quadro de tela cheia lê essa textura, reconstrói o ponto da pele de cada pixel e pinta os pontos.
// Cada ponto mora numa célula de uma grade 3D presa à pele (células de CEL px, um ponto sorteado dentro de cada),
// então eles giram junto com o busto e não escorregam. A luz é a mesma do modo partículas: decide quantos pontos
// aparecem (o tom vira densidade), não o brilho de cada um.
import * as THREE from './vendor/three.module.min.js';

const CEL = 0.85;                             // px de pele por célula: ~2 pontos por px², como no modo partículas

const VS_GUIA = `
  attribute float aParte;
  varying vec3 vN, vV;
  varying float vParte;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalMatrix * normal;
    vV = mv.xyz;
    vParte = aParte;
    gl_Position = projectionMatrix * mv;
  }`;
const FS_GUIA = `
  varying vec3 vN, vV;
  varying float vParte;
  void main() {
    vec3 n = normalize(vN);
    if (vParte < 0.6 && dot(n, -vV) < 0.0) n = -n;   // placa da orelha: vale a face que está virada pra câmera
    gl_FragColor = vec4(n * 0.5 + 0.5, vParte);
  }`;

const VS_TELA = `void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const vec3 = (c) => `vec3(${c.map((v) => v.toFixed(3)).join(', ')})`;
const vec2 = (c) => `vec2(${c.map((v) => v.toFixed(2)).join(', ')})`;

// marcas: [parte, curva [[x, y], ...], peso]; queixo: [[x, y], ...] (px da foto, lado direito)
function shaderPreencher({ U, Y0, AX, paleta, queixo, marcas }) {
  const segs = [];
  marcas.forEach(([parte, curva, peso]) => {
    for (let i = 0; i < curva.length - 1; i++) segs.push(`seg(q, ${vec2(curva[i])}, ${vec2(curva[i + 1])}, ${parte.toFixed(3)}, ${peso.toFixed(2)}, parte)`);
  });
  return `
  uniform sampler2D tNormal, tDepth;
  uniform mat4 uInvMVP, uInvProj;
  uniform vec2 uRes;
  uniform float uTime, uC, uPensa, uFala;
  const float U = ${U.toFixed(4)}, Y0 = ${Y0.toFixed(1)}, AX = ${AX.toFixed(2)}, CEL = ${CEL.toFixed(3)}, BRILHO = 1.3;
  const vec3 DEEP = ${vec3(paleta.DEEP)}, BLUE = ${vec3(paleta.BLUE)}, CYAN = ${vec3(paleta.CYAN)}, WHITE = ${vec3(paleta.WHITE)}, GOLD = ${vec3(paleta.GOLD)};

  vec4 hash43(vec3 p) {                        // 4 números em [0, 1) por célula (hash sem seno, estável em float)
    vec4 p4 = fract(vec4(p.xyzx) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
  }
  float suave(float a, float b, float v) { float t = clamp((v - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }

  // y do contorno do queixo na meia-largura |x| (px)
  float queixoY(float ax) {
    float x = AX + abs(ax);
    ${queixo.slice(0, -1).map((a, i) => { const b = queixo[i + 1]; return `if (x <= ${b[0].toFixed(1)}) return ${a[1].toFixed(2)} + ${(b[1] - a[1]).toFixed(2)} * (x - ${a[0].toFixed(1)}) / ${(b[0] - a[0]).toFixed(2)};`; }).join('\n    ')}
    return ${queixo[queixo.length - 1][1].toFixed(2)};
  }
  // sombras da foto na frente do pescoço: a faixa logo abaixo do queixo e o V entre os dois músculos
  float sombraPescoco(vec3 P) {
    if (P.z < 0.0) return 1.0;
    float s = 1.0;
    if (abs(P.x) < 130.0) {
      float dq = P.y - queixoY(P.x);
      if (dq > 0.0 && dq < 50.0) s *= 0.15 + 0.85 * suave(4.0, 50.0, dq);
    }
    if (P.y > 650.0 && P.y < 880.0) {
      float meia = 40.0 - 28.0 * (P.y - 650.0) / 230.0, forca = 0.55 * (1.0 - suave(780.0, 880.0, P.y));
      s *= 1.0 - forca * (1.0 - suave(meia * 0.55, meia * 1.2, abs(P.x)));
    }
    return s;
  }
  // marcas da foto (contorno do queixo, lados do pescoço, clavícula): mais pontos numa faixa de ~5 px em volta
  // da curva, vista de frente (|x|, y); passam pelo mesmo filtro de luz do resto
  float seg(vec2 q, vec2 a, vec2 b, float pt, float peso, float parte) {
    if (abs(parte - pt) > 0.05) return 0.0;
    vec2 ab = b - a;
    float d = length(q - (a + ab * clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0)));
    return peso * exp(-(d * d) / 9.68);            // σ = 2,2 px, como o espalhamento das marcas no modo partículas
  }
  float marcas(vec3 P, float parte) {
    if (P.z < (parte > 0.94 ? 60.0 : -10.0)) return 0.0;
    vec2 q = vec2(AX + abs(P.x), P.y);
    return ${segs.join('\n      + ')};
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    float dep = texture2D(tDepth, uv).r;
    // o ponto da pele em px da foto, reconstruído em todo px (no fundo cai no plano de trás): fwidth precisa de
    // P nos 4 px do quadradinho, então ele vem antes de qualquer retorno
    vec4 ndc = vec4(uv * 2.0 - 1.0, min(dep, 0.999999) * 2.0 - 1.0, 1.0);
    vec4 vp = uInvProj * ndc;
    vp.xyz /= vp.w;
    vec4 op = uInvMVP * ndc;
    op.xyz /= op.w;
    vec3 P = vec3(op.x / U, Y0 - op.y / U, op.z / U);
    float pe = max(length(fwidth(P)), 1e-3);           // px de pele por px de tela (de raspão, muitos)
    if (dep >= 1.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
    vec4 g = texture2D(tNormal, uv);
    vec3 n = normalize(g.xyz * 2.0 - 1.0);
    float parte = g.a;

    // a luz: a mesma do modo partículas (recorte, luz de cima, profundidade)
    float fd = dot(n, normalize(-vp.xyz));
    float luz = mix(2.0, 0.22, pow(abs(fd), 0.55));
    luz *= 0.85 + 0.45 * max(0.0, n.y);
    if (fd < 0.0) luz *= 0.12;
    float l = clamp(luz / 2.0, 0.0, 1.0);
    float k = smoothstep(-2.3, 2.1, vp.z - uC);         // 0 = fundo, 1 = mais perto
    // (sem as compensações de raspão do modo partículas: aqui os pontos não se empilham no contorno, cada px de
    // tela é uma amostra da pele)
    float vis = (0.15 + 0.16 * l) * mix(0.35, 1.0, k);

    bool cabeca = parte > 0.94, corpo = parte > 0.82 && !cabeca, orelha = parte <= 0.82;
    float ganho = corpo ? 1.3 : 1.0, dens = 1.0;
    if (corpo) dens *= sombraPescoco(P);
    if (orelha) { dens *= 1.8; ganho = 1.45; }           // a orelha é pequena e quase sempre vista de raspão
    if (parte <= 0.7) {                                  // placa: concha e escafa na sombra, antélice na luz
      float r = (parte - 0.02) / 0.48;
      dens *= (0.3 + 0.7 * suave(0.45, 0.65, r)) * (1.0 - 0.65 * suave(0.74, 0.8, r) * (1.0 - suave(0.88, 0.93, r)));
    }
    if (!orelha) dens *= 1.0 + marcas(P, parte);
    if (P.y > 932.0) dens *= max(0.0, 1.0 - (P.y - 932.0) / 9.0);   // só a borda de baixo se desfaz
    float ouro = (cabeca && P.z > 40.0 && P.y > 400.0 && P.y < 700.0) ? exp(-((P.x / 46.0) * (P.x / 46.0) + ((P.y - 548.0) / 50.0) * ((P.y - 548.0) / 50.0))) : 0.0;
    float vd = vis * dens;

    // os pontos: cada célula da grade tem um ponto sorteado no miolo dela ([0,25; 0,75] da célula, mais o tremor
    // de 0,1), então um ponto de uma célula a duas de distância fica a pelo menos 1,15 célula de P: tudo que um
    // ponto toca (raio + borda suave) cabe em 1,15 célula e basta olhar as 27 vizinhas. O px de tela é uma
    // amostra da pele no centro dele, então a média sai certa mesmo de raspão
    vec3 base = floor(P / CEL) - 1.0;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 27; i++) {
      vec3 cel = base + vec3(float(i % 3), float((i / 3) % 3), float(i / 9));
      vec4 q = hash43(cel + 71.3);
      bool conta = q.w < 0.025;                          // contas: pontos maiores e mais claros, aparecem mais
      bool dourado = q.y < ouro * (conta ? 0.4 : 1.0);
      float visP = conta && fd > 0.0 ? 0.25 + 0.75 * vd : vd;
      if (dourado) visP = max(visP, 0.3);                // o dourado é luz própria do rosto
      if (fract(q.z * 91.7) > visP) continue;
      vec4 h = hash43(cel);
      float sd = q.x;
      // pulsação leve, a mesma das partículas (nyx.js, pulso): brilho ±30% (±45% pensando), raio ±12%
      float pulso = 1.0 + (0.3 + 0.15 * uPensa) * sin(uTime * (1.25 + 1.35 * fract(sd * 7.31)) + sd * 83.0);
      float r = CEL * (conta ? 0.8 : 0.6 + 0.25 * h.w) * (1.0 + 0.4 * (pulso - 1.0));
      vec3 c = (cel + 0.25 + 0.5 * h.xyz) * CEL
             + 0.1 * CEL * vec3(sin(uTime * (0.4 + sd * 0.5) + sd * 40.0), cos(uTime * (0.33 + sd * 0.4) + sd * 27.0), sin(uTime * 0.37 + sd * 19.0));
      float aa = min(0.35 * min(pe, 1.5), 1.15 * CEL - r);
      float cob = smoothstep(r + aa, r - aa, length(P - c));
      if (cob <= 0.0) continue;
      float tc = fract(h.w * 13.7);
      vec3 cor = dourado ? GOLD : conta ? (tc < 0.5 ? WHITE : CYAN) : (tc < 0.12 ? DEEP : tc < 0.67 ? BLUE : tc < 0.95 ? CYAN : WHITE);
      float b = conta ? (0.9 + 0.3 * q.y) * ganho : (0.7 + 0.3 * q.y) * (dourado ? 0.8 : ganho);
      float rit = 0.8 + sd * 2.5 + uPensa * 1.5;         // pensando: cintila mais rápido
      float tw = 1.0 + (0.9 + uPensa * 0.6) * pow(0.5 + 0.5 * sin(uTime * rit + sd * 50.0), 12.0);
      float boca = dourado ? 1.0 + uFala * (0.7 + 0.6 * sin(uTime * 9.0 + sd * 3.0)) : 1.0;   // falando: o dourado pulsa
      acc += cor * b * (0.7 + 0.5 * visP) * tw * boca * pulso * cob;
    }
    // exposição fixa, calibrada na tela cheia pra dar o mesmo brilho do modo partículas. Lá a exposição muda com a
    // tela (compensa a sobreposição, que cresce quando a figura encolhe); aqui cada px amostra a pele e não muda
    gl_FragColor = vec4(acc * mix(0.28, 1.15, k) * BRILHO, 1.0);
  }`;
}

// renderer, camera, a malha guia (gerarMalha({ recuo: 0, guia: true })), os uniformes comuns da cena e a forma
// (paleta, queixo, marcas). Devolve o quadro de tela (vai na cena) e o que o laço chama a cada quadro.
export function criarPreenchimento({ renderer, camera, malha, uniforms, forma }) {
  const gbuf = new THREE.WebGLRenderTarget(1, 1, {
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthTexture: new THREE.DepthTexture(1, 1),
  });
  const cenaGuia = new THREE.Scene();
  const guia = new THREE.Mesh(malha, new THREE.ShaderMaterial({ vertexShader: VS_GUIA, fragmentShader: FS_GUIA, side: THREE.DoubleSide }));
  guia.matrixAutoUpdate = false;
  cenaGuia.add(guia);

  const material = new THREE.ShaderMaterial({
    vertexShader: VS_TELA, fragmentShader: shaderPreencher(forma), depthTest: false, depthWrite: false,
    uniforms: {
      ...uniforms,
      tNormal: { value: gbuf.texture }, tDepth: { value: gbuf.depthTexture },
      uInvMVP: { value: new THREE.Matrix4() }, uInvProj: { value: camera.projectionMatrixInverse },
      uRes: { value: new THREE.Vector2(1, 1) },
    },
  });
  const quadro = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quadro.frustumCulled = false;
  quadro.renderOrder = -2;                    // pinta o busto antes de tudo; a máscara e as partículas vêm depois

  const tam = new THREE.Vector2(), corLimpa = new THREE.Color();
  return {
    quadro,
    ajustar() {
      renderer.getDrawingBufferSize(tam);
      gbuf.setSize(tam.x, tam.y);
      material.uniforms.uRes.value.copy(tam);
    },
    // antes de cada composer.render(): a guia na pose do grupo, desenhada na textura
    desenharGuia(grupo) {
      grupo.updateMatrixWorld();
      camera.updateMatrixWorld();
      guia.matrix.copy(grupo.matrixWorld);
      guia.matrixWorldNeedsUpdate = true;
      material.uniforms.uInvMVP.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(grupo.matrixWorld).invert();
      renderer.getClearColor(corLimpa);
      const alfa = renderer.getClearAlpha();
      renderer.setRenderTarget(gbuf);
      renderer.setClearColor(0x000000, 0);
      renderer.clear();
      renderer.render(cenaGuia, camera);
      renderer.setRenderTarget(null);
      renderer.setClearColor(corLimpa, alfa);
    },
    descartar() {
      gbuf.dispose();
      guia.material.dispose();
      material.dispose();
      quadro.geometry.dispose();
    },
  };
}
