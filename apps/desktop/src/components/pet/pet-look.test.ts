import { describe, expect, it } from 'vitest'

import { lookCellForVector, shouldEnablePetGaze, supportsPetLookDirections } from './pet-look'

describe('supportsPetLookDirections', () => {
  it('requires the validated v2 capability from the gateway', () => {
    expect(supportsPetLookDirections({ lookDirectionCount: 16, spriteVersionNumber: 2 })).toBe(true)
    expect(supportsPetLookDirections({ lookDirectionCount: 0, spriteVersionNumber: 2 })).toBe(false)
    expect(supportsPetLookDirections({ lookDirectionCount: 16, spriteVersionNumber: 1 })).toBe(false)
    expect(supportsPetLookDirections({})).toBe(false)
  })
})

describe('shouldEnablePetGaze', () => {
  const v2 = { lookDirectionCount: 16, spriteVersionNumber: 2 }

  it('only enables gaze while the agent and roam loop are both idle', () => {
    expect(shouldEnablePetGaze(v2, true, null, 0)).toBe(true)
    expect(shouldEnablePetGaze(v2, false, null, 0)).toBe(false)
    expect(shouldEnablePetGaze(v2, true, 'jump', 0)).toBe(false)
    expect(shouldEnablePetGaze(v2, true, 'run', 1)).toBe(false)
  })
})

describe('lookCellForVector', () => {
  it.each([
    ['up', 0, -100, 9, 0],
    ['up-right', 100, -100, 9, 2],
    ['right', 100, 0, 9, 4],
    ['down-right', 100, 100, 9, 6],
    ['down', 0, 100, 10, 0],
    ['down-left', -100, 100, 10, 2],
    ['left', -100, 0, 10, 4],
    ['up-left', -100, -100, 10, 6]
  ])('maps %s clockwise into the v2 atlas', (_name, dx, dy, row, column) => {
    expect(lookCellForVector(dx as number, dy as number, 10)).toEqual({ column, row })
  })

  it('signals the fixed neutral frame inside the pointer deadzone', () => {
    expect(lookCellForVector(3, 4, 5)).toBeNull()
  })

  it('rounds to the nearest 22.5-degree direction', () => {
    const radians = (22.5 * Math.PI) / 180
    expect(lookCellForVector(Math.sin(radians) * 100, -Math.cos(radians) * 100, 0)).toEqual({ column: 1, row: 9 })
  })
})
