// Orb parameter model, presets, and style→flow mapping.
//
// Ported from https://github.com/LerSent001/orb (src/presets.ts).
// Copyright (c) LerSent001. Licensed under the MIT License.
// Adapted for hermes-agent: English naming, repo lint style.

export type OrbStyleName =
  | 'siri'
  | 'voiceWave'
  | 'spectrum'
  | 'aurora'
  | 'frost'
  | 'plasma'
  | 'chrome'
  | 'opal'
  | 'blueDrop'
  | 'violetEmber'
  | 'refractiveBlob'
  | 'particleRibbon'
  | 'chromaticMetal'

export interface OrbParams {
  style: OrbStyleName
  glassEnabled: boolean
  speed: number
  radius: number
  contourDeform: number
  bandDensity: number
  chromaticShift: number
  metalScale: number
  metalStretch: number
  metalAngle: number
  metalOffset: number
  metalPhase: number
  metalEvolution: number
  metalRoughness: number
  metalDepth: number
  particleDensity: number
  ribbonCount: number
  ribbonWidth: number
  ribbonTwist: number
  ribbonFold: number
  ribbonBreath: number
  particleSize: number
  particleBloom: number
  zoom: number
  warp: number
  ridgeAmt: number
  sharp: number
  shade: number
  sheen: number
  gloss: number
  glassOpacity: number
  shellMidAlpha: number
  shellEdgeAlpha: number
  exposure: number
  edgeSoftness: number
  edgeGlow: number
  colorA: string
  colorB: string
  colorC: string
  colorD: string
  highlightColor: string
  shellInner: string
  shellMid: string
  shellEdge: string
  sheenColor: string
  specColor: string
  canvasColor: string
  glowColor: string
}

const basePreset: Omit<OrbParams, 'style'> = {
  glassEnabled: true,
  speed: 1,
  radius: 0.72,
  contourDeform: 0,
  bandDensity: 2,
  chromaticShift: 0.42,
  metalScale: 0.77,
  metalStretch: 0.23,
  metalAngle: 65,
  metalOffset: 0,
  metalPhase: 0,
  metalEvolution: 1,
  metalRoughness: 0.22,
  metalDepth: 0.25,
  particleDensity: 0.72,
  ribbonCount: 5,
  ribbonWidth: 0.42,
  ribbonTwist: 1.25,
  ribbonFold: 0.55,
  ribbonBreath: 0.3,
  particleSize: 1.2,
  particleBloom: 0.7,
  zoom: 0.3,
  warp: 3,
  ridgeAmt: 0.5,
  sharp: 2.2,
  shade: 0.3,
  sheen: 0.36,
  gloss: 0.28,
  glassOpacity: 0.42,
  shellMidAlpha: 0.2,
  shellEdgeAlpha: 0.22,
  exposure: 1,
  edgeSoftness: 0.005,
  edgeGlow: 0,
  colorA: '#F7FBFF',
  colorB: '#D6E8F7',
  colorC: '#A8C8F0',
  colorD: '#6F9EE8',
  highlightColor: '#FFFFFF',
  shellInner: '#FFFFFF',
  shellMid: '#D6E8F7',
  shellEdge: '#6F9EE8',
  sheenColor: '#EAF4FF',
  specColor: '#DCEAFF',
  canvasColor: '#000000',
  glowColor: '#6F9EE8',
};

const stylePresets: Record<OrbStyleName, Omit<OrbParams, 'style'>> = {
  siri: {
    ...basePreset,
    speed: 0.82,
    zoom: 0.36,
    warp: 3.2,
    ridgeAmt: 0.5,
    sharp: 2.2,
    shade: 0.12,
    sheen: 0.28,
    gloss: 0.24,
    glassOpacity: 0.44,
    shellMidAlpha: 0.18,
    shellEdgeAlpha: 0.18,
    exposure: 2,
    colorA: '#FFD86B',
    colorB: '#82F4FF',
    colorC: '#FF7BD5',
    colorD: '#8E6CFF',
    shellMid: '#9BF4FF',
    shellEdge: '#C5A9FF',
    canvasColor: '#030409',
    glowColor: '#956CFF',
  },
  voiceWave: {
    ...basePreset,
    speed: 0.95,
    radius: 0.7,
    contourDeform: 0.1,
    zoom: 0.36,
    warp: 2.6,
    ridgeAmt: 0.46,
    shade: 0.08,
    sheen: 0.22,
    gloss: 0.36,
    glassOpacity: 0.48,
    shellMidAlpha: 0.18,
    shellEdgeAlpha: 0.2,
    exposure: 1.35,
    colorA: '#09030E',
    colorB: '#CE2CCB',
    colorC: '#FF5C71',
    colorD: '#7B53FF',
    highlightColor: '#FFD9F0',
    shellMid: '#E48BFF',
    shellEdge: '#FF7890',
    sheenColor: '#FFF1FA',
    specColor: '#E7D9FF',
    canvasColor: '#020105',
    glowColor: '#CE2CCB',
  },
  aurora: {
    ...basePreset,
    speed: 3,
    contourDeform: 0.08,
    zoom: 0.4,
    warp: 4.2,
    ridgeAmt: 0.62,
    sharp: 2.1,
    shade: 0.18,
    exposure: 1.18,
    colorA: '#030816',
    colorB: '#20F0B6',
    colorC: '#32A8FF',
    colorD: '#A34BFF',
    shellMid: '#32A8FF',
    shellEdge: '#20F0B6',
    canvasColor: '#010207',
    glowColor: '#20F0B6',
  },
  plasma: {
    ...basePreset,
    speed: 1.32,
    contourDeform: 0.05,
    zoom: 0.55,
    warp: 5.4,
    ridgeAmt: 0.78,
    sharp: 4.2,
    shade: 0.16,
    exposure: 1.25,
    colorA: '#06020E',
    colorB: '#0099FF',
    colorC: '#258BFF',
    colorD: '#1375FF',
    shellInner: '#FFFFFF',
    shellMid: '#1951C2',
    shellEdge: '#00E9FF',
    sheenColor: '#EAF4FF',
    specColor: '#DCEAFF',
    canvasColor: '#020105',
    glowColor: '#0099FF',
  },
  chrome: {
    ...basePreset,
    speed: 2,
    zoom: 0.36,
    warp: 3.8,
    ridgeAmt: 0.44,
    sharp: 5.2,
    shade: 0.58,
    exposure: 1.08,
    colorA: '#FFFFFF',
    colorB: '#B9C0CA',
    colorC: '#343A43',
    colorD: '#030405',
    shellMid: '#B9C0CA',
    shellEdge: '#FFFFFF',
    canvasColor: '#050608',
    glowColor: '#FFFFFF',
  },
  opal: {
    ...basePreset,
    speed: 1.5,
    zoom: 0.3,
    warp: 2.8,
    ridgeAmt: 0.36,
    sharp: 2,
    shade: 0.1,
    sheen: 0.3,
    gloss: 0.26,
    glassOpacity: 0.38,
    shellMidAlpha: 0.2,
    shellEdgeAlpha: 0.2,
    exposure: 1.12,
    colorA: '#FFF6E8',
    colorB: '#6EF2CF',
    colorC: '#FF91D8',
    colorD: '#756BFF',
    shellMid: '#CDE5FF',
    shellEdge: '#D9C8FF',
    canvasColor: '#07080D',
    glowColor: '#9E8CFF',
  },
  spectrum: {
    ...basePreset,
    speed: 1.8,
    contourDeform: 0.03,
    zoom: 0.46,
    warp: 4.4,
    ridgeAmt: 0.72,
    shade: 0.06,
    sheen: 0.26,
    gloss: 0.24,
    glassOpacity: 0.4,
    shellMidAlpha: 0.18,
    shellEdgeAlpha: 0.18,
    exposure: 1.5,
    colorA: '#FFFFFF',
    colorB: '#1677FF',
    colorC: '#F249A0',
    colorD: '#35E6B2',
    shellMid: '#66E8FF',
    shellEdge: '#D26CFF',
    canvasColor: '#03040A',
    glowColor: '#1677FF',
  },
  frost: {
    ...basePreset,
    speed: 2.22,
    contourDeform: 0.04,
    zoom: 0.36,
    warp: 3.7,
    ridgeAmt: 0.45,
    sharp: 2.05,
    shade: 0.3,
    sheen: 0.34,
    gloss: 0.28,
    glassOpacity: 0.42,
    shellMidAlpha: 0.2,
    shellEdgeAlpha: 0.22,
    exposure: 1,
    colorA: '#F7FBFF',
    colorB: '#D6E8F7',
    colorC: '#A8C8F0',
    colorD: '#6F9EE8',
    shellMid: '#D6E8F7',
    shellEdge: '#6F9EE8',
    canvasColor: '#000000',
    glowColor: '#6F9EE8',
  },
  blueDrop: {
    ...basePreset,
    speed: 0.9,
    radius: 0.74,
    contourDeform: 0.08,
    zoom: 0.48,
    warp: 2.65,
    ridgeAmt: 0.42,
    sharp: 2.4,
    shade: 0.16,
    sheen: 0.22,
    gloss: 0.42,
    glassOpacity: 0.66,
    shellMidAlpha: 0.32,
    shellEdgeAlpha: 0.24,
    exposure: 1.24,
    colorA: '#020B1D',
    colorB: '#0756B8',
    colorC: '#1EC8FF',
    colorD: '#DDFBFF',
    highlightColor: '#EAFBFF',
    shellInner: '#F6FDFF',
    shellMid: '#4FD7FF',
    shellEdge: '#466DFF',
    sheenColor: '#DDFBFF',
    specColor: '#A8D9FF',
    canvasColor: '#010207',
    glowColor: '#168DFF',
  },
  violetEmber: {
    ...basePreset,
    speed: 1.12,
    radius: 0.72,
    contourDeform: 0.04,
    zoom: 0.58,
    warp: 4.7,
    ridgeAmt: 0.73,
    sharp: 3.3,
    shade: 0.18,
    sheen: 0.2,
    gloss: 0.34,
    glassOpacity: 0.62,
    shellMidAlpha: 0.28,
    shellEdgeAlpha: 0.24,
    exposure: 1.28,
    colorA: '#100016',
    colorB: '#4A0E8F',
    colorC: '#A52EFF',
    colorD: '#F1A7FF',
    highlightColor: '#FFD6FF',
    shellInner: '#FCF5FF',
    shellMid: '#C257FF',
    shellEdge: '#6C2DFF',
    sheenColor: '#F8E6FF',
    specColor: '#D4B7FF',
    canvasColor: '#030006',
    glowColor: '#A52EFF',
  },
  refractiveBlob: {
    ...basePreset,
    speed: 0.76,
    radius: 0.73,
    contourDeform: 0.16,
    zoom: 0.46,
    warp: 3.65,
    ridgeAmt: 0.58,
    sharp: 2.7,
    shade: 0.14,
    sheen: 0.14,
    gloss: 0.52,
    glassOpacity: 0.82,
    shellMidAlpha: 0.42,
    shellEdgeAlpha: 0.2,
    exposure: 1.2,
    colorA: '#1B102B',
    colorB: '#7056A8',
    colorC: '#BFA5F5',
    colorD: '#F1E8FF',
    highlightColor: '#FFFFFF',
    shellInner: '#F6F0FF',
    shellMid: '#D9C7FF',
    shellEdge: '#B59AE8',
    sheenColor: '#FFFFFF',
    specColor: '#E9DEFF',
    canvasColor: '#050208',
    glowColor: '#B18CFF',
  },
  particleRibbon: {
    ...basePreset,
    glassEnabled: true,
    speed: 0.72,
    radius: 0.66,
    particleDensity: 1,
    ribbonCount: 4,
    ribbonWidth: 0.48,
    ribbonTwist: 1.15,
    ribbonFold: 0.6,
    ribbonBreath: 0.38,
    particleSize: 1.12,
    particleBloom: 1.22,
    shade: 0.12,
    sheen: 0.28,
    gloss: 0.24,
    glassOpacity: 0.44,
    shellMidAlpha: 0.18,
    shellEdgeAlpha: 0.18,
    exposure: 1.48,
    colorA: '#63F1FF',
    colorB: '#4A9DFF',
    colorC: '#8566FF',
    colorD: '#F15DE1',
    highlightColor: '#F5FBFF',
    shellInner: '#FFFFFF',
    shellMid: '#9BF4FF',
    shellEdge: '#C5A9FF',
    sheenColor: '#EAF4FF',
    specColor: '#DCEAFF',
    canvasColor: '#010208',
    glowColor: '#765CFF',
  },
  chromaticMetal: {
    ...basePreset,
    speed: 1.12,
    radius: 0.72,
    bandDensity: 2,
    chromaticShift: 0.42,
    metalScale: 0.77,
    metalStretch: 0.23,
    metalAngle: 65,
    metalOffset: 0,
    metalPhase: 0,
    metalEvolution: 1,
    metalRoughness: 0.16,
    metalDepth: 0.38,
    shade: 0.1,
    sheen: 0.14,
    gloss: 0.46,
    glassOpacity: 0.54,
    shellMidAlpha: 0.2,
    shellEdgeAlpha: 0.16,
    exposure: 1.08,
    colorA: '#FBFCFB',
    colorB: '#7F8683',
    colorC: '#D6DAD8',
    colorD: '#33373A',
    highlightColor: '#FFFFFF',
    shellInner: '#F7FCFF',
    shellMid: '#6EDCFF',
    shellEdge: '#FF806D',
    sheenColor: '#F7FCFF',
    specColor: '#D9F3FF',
    canvasColor: '#050606',
    glowColor: '#BDEFFF',
  },
};

export const orbStyleNames: readonly OrbStyleName[] = [
  'siri', 'voiceWave', 'particleRibbon', 'blueDrop', 'violetEmber',
  'refractiveBlob', 'chromaticMetal', 'aurora', 'frost', 'chrome', 'opal',
  'spectrum', 'plasma',
]

// Flow-program index each style selects inside the WGSL shader.
export const styleFlowIndexes: Record<OrbStyleName, number> = {
  siri: 9, voiceWave: 19, aurora: 10, plasma: 11, chrome: 12, opal: 13,
  spectrum: 14, frost: 15, blueDrop: 20, violetEmber: 21, refractiveBlob: 23,
  particleRibbon: 24, chromaticMetal: 22,
}

/** Default orb: the glass-liquid Siri-style preset shipped out of the box. */
export const defaultOrbParams: OrbParams = {
  style: 'siri',
  ...stylePresets.siri,
}

/** Params for a named style preset (falls back to the default when unknown). */
export function orbPresetForStyle(style: string): OrbParams {
  const name = (orbStyleNames as readonly string[]).includes(style)
    ? (style as OrbStyleName)
    : defaultOrbParams.style

  return { style: name, ...stylePresets[name] }
}
