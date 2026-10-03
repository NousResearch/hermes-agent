import { describe, expect, it } from 'vitest'

import { petFacingTransform } from './pet-facing'

describe('petFacingTransform', () => {
  it('mirrors a pet on the left so it faces right toward the center', () => {
    expect(petFacingTransform(100, 100, 800)).toBe('scaleX(-1)')
  })

  it('keeps a pet on the right facing left toward the center', () => {
    expect(petFacingTransform(600, 100, 800)).toBe('none')
  })

  it('uses the right-half orientation at the exact midpoint', () => {
    expect(petFacingTransform(350, 100, 800)).toBe('none')
  })
})
