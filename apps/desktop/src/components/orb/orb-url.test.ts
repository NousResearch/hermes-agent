import { describe, expect, it } from 'vitest'

import { defaultOrbConfig, parseOrbConfigUrl } from './orb-url'

// The configurator encodes the whole orb in the URL hash:
// https://lersent001.github.io/orb/#effect=orb-glass-liquid&...
describe('parseOrbConfigUrl', () => {
  it('parses a full configurator URL', () => {
    const result = parseOrbConfigUrl(
      'https://lersent001.github.io/orb/#effect=orb-glass-liquid&style=siri&glass=1&speed=0.82&radius=0.72&colorA=%23FFD86B&colorB=%2382F4FF&colorC=%23FF7BD5&colorD=%238E6CFF'
    )

    expect('params' in result).toBe(true)

    if (!('params' in result)) {return}
    expect(result.params.style).toBe('siri')
    expect(result.params.glassEnabled).toBe(true)
    expect(result.params.speed).toBeCloseTo(0.82)
    expect(result.params.radius).toBeCloseTo(0.72)
    expect(result.params.colorA).toBe('#FFD86B')
    expect(result.params.colorB).toBe('#82F4FF')
    expect(result.params.colorC).toBe('#FF7BD5')
    expect(result.params.colorD).toBe('#8E6CFF')
  })

  it('accepts a bare hash or query fragment', () => {
    for (const fragment of [
      '#effect=orb-glass-liquid&style=aurora',
      'effect=orb-glass-liquid&style=aurora',
      '?effect=orb-glass-liquid&style=aurora'
    ]) {
      const result = parseOrbConfigUrl(fragment)
      expect('params' in result).toBe(true)

      if ('params' in result) {
        expect(result.params.style).toBe('aurora')
      }
    }
  })

  it('rejects non-orb URLs and empty input', () => {
    expect(parseOrbConfigUrl('')).toEqual({ error: 'empty' })
    expect(parseOrbConfigUrl('   ')).toEqual({ error: 'empty' })
    expect(parseOrbConfigUrl('https://example.com/')).toEqual({ error: 'no-params' })
    expect(parseOrbConfigUrl('#foo=bar')).toEqual({ error: 'no-params' })
  })

  it('rejects a URL with the wrong effect', () => {
    const result = parseOrbConfigUrl('https://lersent001.github.io/orb/#effect=some-other-thing&style=siri')
    expect(result).toEqual({ error: 'not-an-orb-url' })
  })

  it('clamps numeric values to the editor ranges', () => {
    const result = parseOrbConfigUrl('#effect=orb-glass-liquid&speed=99&radius=-5&contourDeform=0.5')
    expect('params' in result).toBe(true)

    if (!('params' in result)) {return}
    expect(result.params.speed).toBe(3)
    expect(result.params.radius).toBe(0.3)
    expect(result.params.contourDeform).toBeCloseTo(0.5)
  })

  it('ignores malformed numbers and unknown parameters', () => {
    const result = parseOrbConfigUrl('#effect=orb-glass-liquid&speed=banana&frobnicate=1')
    expect('params' in result).toBe(true)

    if (!('params' in result)) {return}
    expect(result.params.speed).toBe(defaultOrbConfig().speed)
  })

  it('ignores malformed colors', () => {
    const result = parseOrbConfigUrl('#effect=orb-glass-liquid&colorA=banana&colorB=%23FFD86B')
    expect('params' in result).toBe(true)

    if (!('params' in result)) {return}
    expect(result.params.colorA).toBe(defaultOrbConfig().colorA)
    expect(result.params.colorB).toBe('#FFD86B')
  })

  it('inherits the selected style preset for missing parameters', () => {
    const result = parseOrbConfigUrl('#effect=orb-glass-liquid&style=aurora')
    expect('params' in result).toBe(true)

    if (!('params' in result)) {return}
    // Aurora's speed (3) differs from siri's (0.82); a param the URL didn't
    // mention comes from the named preset, not the default.
    expect(result.params.style).toBe('aurora')
    expect(result.params.speed).toBe(3)
    expect(result.params.speed).not.toBe(defaultOrbConfig().speed)
  })

  it('parses the glass toggle', () => {
    const on = parseOrbConfigUrl('#effect=orb-glass-liquid&glass=1')
    const off = parseOrbConfigUrl('#effect=orb-glass-liquid&glass=0')
    expect('params' in on && on.params.glassEnabled).toBe(true)
    expect('params' in off && off.params.glassEnabled).toBe(false)
  })

  it('falls back to the siri preset for an unknown style', () => {
    const result = parseOrbConfigUrl('#effect=orb-glass-liquid&style=not-a-style')
    expect('params' in result).toBe(true)

    if (!('params' in result)) {return}
    expect(result.params.style).toBe('siri')
  })
})

describe('defaultOrbConfig', () => {
  it('returns a fresh siri preset each call', () => {
    const a = defaultOrbConfig()
    const b = defaultOrbConfig()
    expect(a).not.toBe(b)
    expect(a.style).toBe('siri')
    expect(a).toEqual(b)
  })
})
