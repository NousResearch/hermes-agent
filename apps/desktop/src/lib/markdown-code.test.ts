import { describe, expect, it } from 'vitest'

import { isLikelyProseCodeBlock, isLikelyProseFence } from './markdown-code'

describe('isLikelyProseFence', () => {
  it('keeps explicit text fences out of prose downgrade so the language tag cannot leak', () => {
    expect(
      isLikelyProseFence(
        'text',
        [
          'apps/desktop/src/components/assistant-ui/thread.tsx',
          'apps/desktop/src/lib/markdown-code.ts',
          'apps/desktop/src/styles.css'
        ].join('\n')
      )
    ).toBe(false)
  })

  it('still detects prose fences with list-marker info strings', () => {
    expect(isLikelyProseFence('- Notes', ['first point', 'second point'].join('\n'))).toBe(true)
  })
})

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

  it('keeps real code blocks', () => {
    expect(isLikelyProseCodeBlock('ts', 'const value = { bunny: true };\nreturn value')).toBe(false)
  })
})
