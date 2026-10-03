import { describe, expect, it } from 'vitest'
import { stableArray, stableRecord } from '@/lib/stable-array'

describe('stableArray', () => {
  it('returns prev reference when equal', () => {
    const prev = [1, 2, 3]
    const next = [1, 2, 3]
    expect(stableArray(prev, next)).toBe(prev)
  })

  it('returns new frozen array when different', () => {
    const prev = [1, 2]
    const next = [1, 2, 3]
    const result = stableArray(prev, next)
    expect(result).not.toBe(prev)
    expect(Object.isFrozen(result)).toBe(true)
  })

  it('handles empty arrays', () => {
    const prev: number[] = []
    const next: number[] = []
    expect(stableArray(prev, next)).toBe(prev)
  })
})

describe('stableRecord', () => {
  it('returns prev reference when equal', () => {
    const prev = { a: 1 }
    const next = { a: 1 }
    expect(stableRecord(prev, next)).toBe(prev)
  })

  it('returns new frozen record when different', () => {
    const prev = { a: 1 }
    const next = { a: 2 }
    const result = stableRecord(prev, next)
    expect(result).not.toBe(prev)
    expect(Object.isFrozen(result)).toBe(true)
  })
})
