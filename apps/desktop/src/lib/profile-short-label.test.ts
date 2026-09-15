import { describe, expect, it } from 'vitest'

import { profileShortLabel } from './profile-short-label'

describe('profileShortLabel', () => {
  it.each([
    ['研究助手', '研究助'],
    ['developer', 'dev'],
    ['OpenAI', 'Ope'],
    ['42 组', '42'],
    ['🔵 dev', '🔵 d'],
    ['🚀 开发', '🚀 开'],
    ['👨‍👩‍👧‍👦 family', '👨‍👩‍👧‍👦 f'],
    ['   ', '?']
  ])('returns a readable three-grapheme rail label for %j', (label, expected) => {
    expect(profileShortLabel(label, 3)).toBe(expected)
  })

  it('keeps a joined emoji family intact as one compact glyph', () => {
    expect(profileShortLabel('👨‍👩‍👧‍👦 family')).toBe('👨‍👩‍👧‍👦')
  })
})
