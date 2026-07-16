import { describe, expect, it } from 'vitest'

import { countLabel } from './chrome'

describe('countLabel', () => {
  it('marks pagination lower bounds without presenting them as exact totals', () => {
    expect(countLabel(40, 41, true)).toBe('40+')
  })
})
