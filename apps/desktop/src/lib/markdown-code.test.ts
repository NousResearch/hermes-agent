import { describe, expect, it } from 'vitest'

import { isLikelyProseCodeBlock } from './markdown-code'

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

  it('treats short RTL text fences as prose, not LTR code cards', () => {
    expect(isLikelyProseCodeBlock('text', 'אני פונה אליכם כי אתם עוסקים ב OPEN AI ובעבודה עם מודלים.')).toBe(
      true
    )
  })

  it('treats RTL text fences with URLs/emails/punctuation as prose', () => {
    expect(
      isLikelyProseCodeBlock(
        'text',
        [
          'אני פונה אליכם כי אתם עוסקים ב-OPEN AI ובעבודה עם מודלים.',
          'אפשר ליצור קשר ב-test@example.com או דרך https://example.com/path?x=1.',
          'האם זה משהו שאפשר לקדם?'
        ].join('\n')
      )
    ).toBe(true)
  })
})
