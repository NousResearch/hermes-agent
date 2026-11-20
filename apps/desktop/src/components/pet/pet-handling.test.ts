import { describe, expect, it } from 'vitest'

import { dropTarget } from './pet-handling'
import { groundTop, type Ledge } from './roam-geometry'

// Poke-vs-carry (isDragMove) is exercised end to end in floating-pet-handling.test.tsx.
describe('dropTarget', () => {
  const petH = 100
  const floor: Ledge = { left: 0, right: 1000, y: 600 }
  const shelf: Ledge = { left: 100, right: 400, y: 300 }

  it('lands on the ledge right below the release point, falling past ledges it is not over', () => {
    expect(dropTarget([floor, shelf], 200, 50, petH)).toBe(groundTop(shelf, petH))
    expect(dropTarget([floor, shelf], 800, 50, petH)).toBe(groundTop(floor, petH))
  })

  it('stays put when released already standing, or below every surface', () => {
    expect(dropTarget([floor, shelf], 200, groundTop(shelf, petH), petH)).toBeNull()
    expect(dropTarget([floor], 200, 5000, petH)).toBeNull()
  })
})
