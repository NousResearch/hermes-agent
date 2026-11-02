import { describe, expect, it } from 'vitest'

import { truncateText } from './job-state'

describe('truncateText', () => {
  it('truncates on Unicode code-point boundaries', () => {
    const result = truncateText('😀😀😀😀', 3)

    expect(result).toBe('😀😀😀…')
    expect(Array.from(result)).toEqual(['😀', '😀', '😀', '…'])
  })

  it('does not add an ellipsis at the limit', () => {
    expect(truncateText('😀😀😀', 3)).toBe('😀😀😀')
  })

  // Grapheme clusters, not code points: the same shapes profile-glyph.test.tsx pins for the
  // profile rail. A job name is raw user input, so these are plausible values, not exotica.
  it('never cuts a grapheme cluster mid-sequence', () => {
    const cases: Array<[string, number, string]> = [
      ['🇨🇳xy', 1, '🇨🇳…'],
      ['🇨🇳xy', 3, '🇨🇳xy'],
      ['👍🏽x', 1, '👍🏽…'],
      ['e\u0301x', 1, 'e\u0301…'],
      ['👨‍👩‍👧‍👦xyz', 1, '👨‍👩‍👧‍👦…'],
      ['👨‍👩‍👧‍👦xyz', 3, '👨‍👩‍👧‍👦xy…'],
      ['1️⃣ 每日简报', 1, '1️⃣…']
    ]

    for (const [value, max, expected] of cases) {
      expect(truncateText(value, max)).toBe(expected)
    }
  })
})
