import { describe, expect, it } from 'vitest'

import { didPetPointerMove } from './pet-pointer-gesture'

describe('didPetPointerMove', () => {
  it('keeps small hand jitter as a click', () => {
    expect(didPetPointerMove(10, 10, 10, 10)).toBe(false)
    expect(didPetPointerMove(10, 10, 13, 10)).toBe(false)
  })

  it('classifies travel beyond the click slop as a drag', () => {
    expect(didPetPointerMove(10, 10, 14, 10)).toBe(true)
    expect(didPetPointerMove(10, 10, 13, 13)).toBe(true)
  })
})
