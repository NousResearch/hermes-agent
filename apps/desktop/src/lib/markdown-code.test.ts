import { describe, expect, it } from 'vitest'

import { isLikelyProseCodeBlock, isLikelyProseFence } from './markdown-code'

describe('isLikelyProseCodeBlock', () => {
  it('detects prose that Streamdown mislabels as an unknown language', () => {
    expect(
      isLikelyProseCodeBlock(
        'heads',
        [
          '- Pure white (`#ffffff`), roughness 0.55, no emissive',
          '- Black wireframe edges at 35% opacity',
          '',
          'Want the bunny gone, or want me to keep riffing on it?'
        ].join('\n')
      )
    ).toBe(true)
  })

  it('keeps ordinary text fences with prose as prose', () => {
    const prose = ['Morning update', 'Checks await review', 'Results appear soon'].join('\n')

    expect(isLikelyProseFence('text', prose)).toBe(true)
    expect(isLikelyProseCodeBlock('text', prose)).toBe(true)
  })

  it('keeps fenced file lists as code blocks', () => {
    const files = ['gateway-event.ts', 'process-result-visibility.test.tsx', 'tui_gateway/server.py + tests'].join('\n')

    expect(isLikelyProseFence('text', files)).toBe(false)
    expect(isLikelyProseCodeBlock('text', files)).toBe(false)
  })

  it('keeps real code blocks', () => {
    expect(isLikelyProseCodeBlock('ts', 'const value = { bunny: true };\nreturn value')).toBe(false)
  })
})
