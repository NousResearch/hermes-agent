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
  K: 1.15, DENS: 0.22,
  // perfil do manequim: [y, meia-largura] em px da imagem, medidos no alfa do corpo (sem as orelhas) e alisados.
  // Entre os pontos, Catmull-Rom: o contorno sai liso, sem os calombos de uma borda medida linha a linha.
  PROFILE: [[118, 0], [125, 38], [140, 84], [155, 111], [170, 130], [185, 145], [200, 159], [215, 169], [230, 177],
    [245, 184], [260, 190], [290, 197], [320, 200], [350, 200], [380, 199], [410, 197], [440, 193], [470, 186],
    [500, 178], [530, 163], [560, 147], [590, 135], [620, 130], [660, 131], [700, 136], [740, 142], [800, 150]],
  EAR: { cy: 415, ry: 54, out: 26 },   // orelha: centro e meia-altura (y) e quanto sai do crânio (px)
};
export const HEAD_RX = 182, HEAD_RY = (IMG.chinY - IMG.headTop) / 2;   // semi-eixos do crânio
export const NECK_Y = IMG.chinY + 110;                                  // base do pescoço: pivô da cabeça

export const FONT = `'Sora','Avenir Next','Segoe UI',ui-sans-serif,system-ui,sans-serif`;

// Camadas da foto, na ordem que start() espera. O build (build.py) troca estes caminhos por data URIs.
export const ASSETS = [
  'assets/img/fundo.webp',      // fundo: a original + a reconstrução atrás da figura
  'assets/img/mascaras.png',    // r: brilhos pontuais · g: água · b: via láctea
  'assets/img/corpo-rgb.webp',  // base escura da figura, pré-multiplicada (+ brilho do contorno)
  'assets/img/corpo-alfa.png',  // alfa da base escura
  'assets/img/emissao.png',     // cor de cada partícula
  'assets/img/info.png',        // r: relevo · g: borda · b: marca da partícula (>200 = aura)
];

export const PARAMS = new URLSearchParams(location.search);
export const STILL = PARAMS.has('estatico');     // sem animação ambiente, pra comparar com a foto
// ?giro=0.3[,aceno] fixa a pose da cabeça (rad): bancada pra ajustar o giro 3D sem depender do mouse
export const GIRO = PARAMS.has('giro') ? PARAMS.get('giro').split(',').map((v) => Number(v) || 0) : null;
export const SO_FUNDO = PARAMS.has('sofundo');   // só o cenário vivo, sem o humanoide (base pra encaixar novos assets)
export const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
export const COARSE = matchMedia('(pointer: coarse)').matches;
export const MOBILE = COARSE && Math.min(screen.width, screen.height) < 820;
