import { describe, expect, it } from 'vitest'

import { ago } from './ui'

describe('ago', () => {
  it('omits non-finite timestamps instead of throwing', () => {
    expect(() => ago(Number.POSITIVE_INFINITY)).not.toThrow()
    expect(ago(Number.POSITIVE_INFINITY)).toBeNull()
    expect(ago(Number.NEGATIVE_INFINITY)).toBeNull()
  })
})
