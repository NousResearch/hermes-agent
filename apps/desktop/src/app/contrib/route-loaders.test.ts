import { describe, expect, it } from 'vitest'

import { shouldWarmCommonRoutes } from './route-loaders'

describe('common route warmup eligibility', () => {
  it('runs only in the connected primary window', () => {
    expect(shouldWarmCommonRoutes(true, false)).toBe(true)
    expect(shouldWarmCommonRoutes(false, false)).toBe(false)
    expect(shouldWarmCommonRoutes(true, true)).toBe(false)
  })
})
