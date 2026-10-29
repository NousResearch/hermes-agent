import { describe, expect, it } from 'vitest'

import { inlineTextPrefix } from './text'

describe('inlineTextPrefix', () => {
  it('keeps whitespace and UTF-16 truncation identical to a normalized text prefix', () => {
    const inputs = [
      '',
      ' \n\t ',
      '  one\n\n two\tthree  ',
      'one\u00a0two\u2028three\ufeff',
      '  İΣ 😀 café\t終わり  ',
      'x'.repeat(100_000),
      `start${' \n\t'.repeat(10_000)}end`,
      'one two three\n'.repeat(10_000)
    ]

    for (const input of inputs) {
      for (const limit of [0, 1, 2, 3, 4, 8, 72, 180, input.length + 1]) {
        expect(inlineTextPrefix(input, limit)).toBe(input.replace(/\s+/g, ' ').trim().slice(0, limit))
      }
    }
  })
})
