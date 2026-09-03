import { describe, expect, it } from 'vitest'

import { initialPetWalk, petBob, type PetWalkInput, stepPetWalk } from './pet-walk'

const input = (overrides: Partial<PetWalkInput> = {}): PetWalkInput => ({
  stripWidth: 600,
  width: 40,
  pointerX: null,
  speed: 100,
  lookRadius: 60,
  random: () => 0.99, // never pause, never turn
  ...overrides
})

describe('stepPetWalk', () => {
  it('walks at the given speed and turns at the edges', () => {
    let s = { ...initialPetWalk(540, 1), decide: 100 }
    s = stepPetWalk(s, 0.1, input())
    expect(s.x).toBeCloseTo(550)
    expect(s.dir).toBe(1)

    s = stepPetWalk(s, 0.1, input())
    expect(s.x).toBe(560) // clamped to stripWidth - width, and turned
    expect(s.dir).toBe(-1)

    s = stepPetWalk({ ...s, x: 5 }, 0.1, input())
    expect(s.x).toBe(0)
    expect(s.dir).toBe(1)
  })

  it('stops and faces the pointer when it is close, then wanders off after a beat', () => {
    const s0 = { ...initialPetWalk(100, 1), decide: 100 }
    const looking = stepPetWalk(s0, 0.1, input({ pointerX: 60 }))

    expect(looking.looking).toBe(true)
    expect(looking.dir).toBe(-1)
    expect(looking.x).toBe(100)

    const released = stepPetWalk(looking, 0.1, input({ pointerX: null, random: () => 0 }))
    expect(released.looking).toBe(false)
    expect(released.idle).toBeCloseTo(0.6)
    expect(released.x).toBe(100)

    const still = stepPetWalk(released, 0.3, input())
    expect(still.x).toBe(100)
    expect(still.idle).toBeCloseTo(0.3)
  })

  it('pauses or turns on a decision roll', () => {
    const paused = stepPetWalk({ ...initialPetWalk(100, 1), decide: 0 }, 0.1, input({ random: () => 0.1 }))
    expect(paused.idle).toBeGreaterThan(0)
    expect(paused.x).toBe(100)

    const turned = stepPetWalk({ ...initialPetWalk(100, 1), decide: 0 }, 0.1, input({ random: () => 0.45 }))
    expect(turned.dir).toBe(-1)
    expect(turned.x).toBeCloseTo(90)
  })
})


describe('modes', () => {
  it('stands still in stand mode, even mid-walk', () => {
    const s = stepPetWalk({ ...initialPetWalk(100, 1), decide: 100 }, 0.5, input({ mode: 'stand' }))

    expect(s.x).toBe(100)
    expect(s.dir).toBe(1)
  })

  it('paces faster and ignores pauses in pace mode', () => {
    const paused = { ...initialPetWalk(100, 1), decide: 100, idle: 5 }
    const s = stepPetWalk(paused, 0.1, input({ mode: 'pace' }))

    expect(s.x).toBeCloseTo(100 + 100 * 2.6 * 0.1)
  })
})

describe('petBob', () => {
  it('bobs only while walking', () => {
    expect(petBob({ ...initialPetWalk(0), idle: 1 })).toBe(0)
    expect(petBob({ ...initialPetWalk(0), looking: true })).toBe(0)
    expect(petBob({ ...initialPetWalk(0), walked: 1 / 6 })).toBeCloseTo(2)
  })
})
