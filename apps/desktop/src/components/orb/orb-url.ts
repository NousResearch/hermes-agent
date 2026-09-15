// Parses lersent001/orb configurator URLs into orb params.
//
// The configurator serializes its entire state into the URL hash
// (https://lersent001.github.io/orb/#effect=orb-glass-liquid&style=siri&...),
// so "bring your own orb" is just: paste the URL, parse the params, render.
//
// Ported from the hash read/write logic in
// https://github.com/LerSent001/orb (src/App.tsx).
// Copyright (c) LerSent001. Licensed under the MIT License.

import { defaultOrbParams, type OrbParams, orbPresetForStyle, type OrbStyleName } from './orb-params'

interface NumericSpec {
  key: keyof OrbParams
  max: number
  min: number
}

// Clamp ranges mirror the editor sliders; unknown keys are ignored.
const numericSpecs: readonly NumericSpec[] = [
  { key: 'speed', min: 0, max: 3 },
  { key: 'radius', min: 0.3, max: 0.95 },
  { key: 'contourDeform', min: 0, max: 1 },
  { key: 'zoom', min: 0.05, max: 1 },
  { key: 'warp', min: 0, max: 6 },
  { key: 'ridgeAmt', min: 0, max: 1 },
  { key: 'sharp', min: 0.5, max: 6 },
  { key: 'bandDensity', min: 1, max: 6 },
  { key: 'metalDepth', min: 0, max: 1 },
  { key: 'metalRoughness', min: 0, max: 1 },
  { key: 'chromaticShift', min: 0, max: 1 },
  { key: 'metalScale', min: 0.2, max: 2 },
  { key: 'metalStretch', min: 0, max: 1 },
  { key: 'metalAngle', min: -180, max: 180 },
  { key: 'metalOffset', min: -1, max: 1 },
  { key: 'metalPhase', min: 0, max: 1 },
  { key: 'metalEvolution', min: 0, max: 2 },
  { key: 'particleDensity', min: 0.2, max: 1 },
  { key: 'ribbonCount', min: 2, max: 6 },
  { key: 'ribbonWidth', min: 0.1, max: 0.8 },
  { key: 'ribbonTwist', min: 0.1, max: 3 },
  { key: 'ribbonFold', min: 0, max: 1.2 },
  { key: 'ribbonBreath', min: 0, max: 0.8 },
  { key: 'particleSize', min: 0.6, max: 2.5 },
  { key: 'particleBloom', min: 0, max: 2 },
  { key: 'shade', min: 0, max: 1.5 },
  { key: 'exposure', min: 0.2, max: 3 },
  { key: 'sheen', min: 0, max: 2 },
  { key: 'gloss', min: 0, max: 2 },
  { key: 'glassOpacity', min: 0, max: 1 },
  { key: 'shellMidAlpha', min: 0, max: 1 },
  { key: 'shellEdgeAlpha', min: 0, max: 1 },
  { key: 'edgeSoftness', min: 0.005, max: 0.15 },
  { key: 'edgeGlow', min: 0, max: 1 }
]

const colorKeys = [
  'colorA',
  'colorB',
  'colorC',
  'colorD',
  'highlightColor',
  'shellInner',
  'shellMid',
  'shellEdge',
  'sheenColor',
  'specColor',
  'canvasColor',
  'glowColor'
] as const

const EFFECT_PARAM = 'orb-glass-liquid'

// Keys the parser understands — anything else in the fragment is ignored.
// A fragment with none of these carries no orb, so it isn't a valid orb URL.
const recognizedKeys: ReadonlySet<string> = new Set([
  'effect',
  'style',
  'glass',
  ...numericSpecs.map(spec => spec.key),
  ...colorKeys
])

function hasAnyOrbParam(search: URLSearchParams): boolean {
  for (const key of search.keys()) {
    if (recognizedKeys.has(key)) {
      return true
    }
  }

  return false
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function normalizeColor(value: string): string | null {
  return /^#[0-9a-f]{6}$/i.test(value) ? value.toUpperCase() : null
}

function paramsFromSearch(search: URLSearchParams): OrbParams {
  const params = orbPresetForStyle(search.get('style') ?? '')

  const glass = search.get('glass')

  if (glass === '1') {params.glassEnabled = true}

  if (glass === '0') {params.glassEnabled = false}

  for (const spec of numericSpecs) {
    const raw = search.get(spec.key)

    if (raw === null) {continue}
    const value = Number(raw)

    if (Number.isFinite(value)) {
      const numeric = params as unknown as Record<NumericSpec['key'], number>
      numeric[spec.key] = clamp(value, spec.min, spec.max)
    }
  }

  for (const key of colorKeys) {
    const raw = search.get(key)

    if (raw === null) {continue}
    const color = normalizeColor(raw)

    if (color) {
      const palette = params as unknown as Record<(typeof colorKeys)[number], string>
      palette[key] = color
    }
  }

  return params
}

export type OrbUrlParseResult = { params: OrbParams } | { error: string }

/**
 * Parse a configurator URL (or a bare `#...` hash / query fragment) into orb
 * params. Starts from the named style preset, then applies whatever params
 * the URL carries — so a partial URL still renders a sane orb.
 */
export function parseOrbConfigUrl(input: string): OrbUrlParseResult {
  const text = input.trim()

  if (!text) {
    return { error: 'empty' }
  }

  let hash = ''

  try {
    const url = new URL(text)
    hash = url.hash.slice(1)

    if (!hash && url.search.length > 1) {
      hash = url.search.slice(1)
    }
  } catch {
    // Not a full URL — treat the input itself as the param fragment.
    hash = text.startsWith('#') ? text.slice(1) : text
  }

  if (!hash) {
    return { error: 'no-params' }
  }

  const search = new URLSearchParams(hash)
  const effect = search.get('effect')

  if (effect !== null && effect !== EFFECT_PARAM) {
    return { error: 'not-an-orb-url' }
  }

  if (!hasAnyOrbParam(search)) {
    return { error: 'no-params' }
  }

  return { params: paramsFromSearch(search) }
}

/** The built-in default orb, used when no custom URL is configured. */
export function defaultOrbConfig(): OrbParams {
  return { ...defaultOrbParams }
}

export type { OrbParams, OrbStyleName }
