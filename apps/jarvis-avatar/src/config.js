/* Ajustes e geometria da cena. Tudo que um integrador costuma mexer fica aqui. */

export const CFG = {
  maxParticles: 0,     // 0 = automático: todas no desktop (~220 mil), até 90 mil no celular
  mouseRadius: 70,     // raio de influência do mouse, em pixels da imagem (a cabeça tem ~365 px de largura)
  bloom: true,         // brilho extra ao falar/pensar (em repouso fica desligado pra não alterar a foto)
  maxDpr: 2,
  voiceLang: 'pt-BR',
  defaultPhrase: 'Olá. Sistemas online. Todas as partículas sob controle. Como posso ajudar?',
};

// geometria medida na imagem de referência (pixels da imagem 1672×941)
export const IMG = { w: 1672, h: 941, horizon: 587, axisX: 832, headTop: 122, chinY: 632, headCY: 377, mouth: [831, 552] };
// cabeça 3D (shaders.js sliceOf + nuvem de partículas em main.js): profundidade/largura de cada fatia do manequim
// e densidade das partículas na superfície (por px²)
export const HEAD3D = {
  PIVOT: 40,               // px: eixo vertical do giro, atrás do centro da cabeça (no pescoço)
  TURN_Y: [640, 700],      // a base do pescoço: acima gira inteiro, abaixo (ombros) fica parado
  EARS: [354, 492],        // linhas onde as orelhas saem do crânio na vista de frente
  HALO: 6, HALO_MAX: 70,   // px: brilho do contorno além do alfa · até onde o brilho fora da silhueta acompanha a borda
  MESH_IN: 10,             // px: a malha escura fica esse tanto pra dentro do sólido (não aparece como mancha na borda)
  LUM: 110,                // soma r+g+b mínima pra um pixel de vista virar partícula (o escuro é a malha)
  // vistas giradas pra direita de quem olha (giro em rad, medido pela silhueta e pelo dourado); a esquerda é o espelho.
  // A de 90° também dá a profundidade da cabeça (model3d.buildSolid)
  VIEWS: [
    { src: 'assets/img/vistas/giro16.webp', yaw: 0.28 },
    { src: 'assets/img/vistas/giro41.webp', yaw: 0.72 },
    { src: 'assets/img/vistas/giro48.webp', yaw: 0.84 },
    { src: 'assets/img/vistas/giro90.webp', yaw: 1.5708 },
  ],
};
export const HEAD_RX = 182, HEAD_RY = (IMG.chinY - IMG.headTop) / 2;   // semi-eixos do crânio
export const NECK_Y = IMG.chinY + 110;                                  // base do pescoço: pivô da cabeça

export const FONT = `'Sora','Avenir Next','Segoe UI',ui-sans-serif,system-ui,sans-serif`;

// Camadas da foto, na ordem que start() espera. O build (build.py) troca estes caminhos por data URIs.
export const ASSETS = [
  'assets/img/fundo.webp',      // fundo: a original + a reconstrução atrás da figura
  'assets/img/mascaras.png',    // r: brilhos pontuais · g: água · b: via láctea
  'assets/img/corpo-rgb.webp',  // brilho da figura (base escura + contorno neon e halo), pré-multiplicado
  'assets/img/corpo-alfa.png',  // alfa da base escura
  'assets/img/emissao.png',     // cor de cada partícula
  'assets/img/info.png',        // r: relevo · g: borda · b: marca da partícula (>200 = aura)
  ...HEAD3D.VIEWS.map((v) => v.src),   // vistas giradas da cabeça
];

export const PARAMS = new URLSearchParams(location.search);
export const STILL = PARAMS.has('estatico');     // sem animação ambiente, pra comparar com a foto
// ?giro=0.3[,aceno] fixa a pose da cabeça (rad): bancada pra ajustar o giro 3D sem depender do mouse
export const GIRO = PARAMS.has('giro') ? PARAMS.get('giro').split(',').map((v) => Number(v) || 0) : null;
export const SO_FUNDO = PARAMS.has('sofundo');   // só o cenário vivo, sem o humanoide (base pra encaixar novos assets)
export const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
export const COARSE = matchMedia('(pointer: coarse)').matches;
export const MOBILE = COARSE && Math.min(screen.width, screen.height) < 820;
