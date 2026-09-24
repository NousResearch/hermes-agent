import { beforeEach, describe, expect, it } from 'vitest'

import { persistNumber, persistStringArray, storedNumber, storedStringArray } from './storage'

describe('numeric storage', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('round-trips a number', () => {
    persistNumber('test.limit', 12)

    expect(window.localStorage.getItem('test.limit')).toBe('12')
    expect(storedNumber('test.limit', 3)).toBe(12)
  })

  it('falls back when the key is absent or the stored text is not a number', () => {
    expect(storedNumber('test.missing', 3)).toBe(3)

    window.localStorage.setItem('test.limit', 'not-a-number')

    expect(storedNumber('test.limit', 3)).toBe(3)
  })
})

describe('string array storage', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('removes the key for an empty array', () => {
    window.localStorage.setItem('test.order', JSON.stringify(['a']))

    persistStringArray('test.order', [])

    expect(window.localStorage.getItem('test.order')).toBeNull()
    expect(storedStringArray('test.order')).toEqual([])
  })

  it('persists non-empty arrays', () => {
    persistStringArray('test.order', ['a', 'b'])

    expect(window.localStorage.getItem('test.order')).toBe(JSON.stringify(['a', 'b']))
    expect(storedStringArray('test.order')).toEqual(['a', 'b'])
  })
})
