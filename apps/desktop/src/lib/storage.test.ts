import { beforeEach, describe, expect, it } from 'vitest'

import { persistStringArray, storedStringArray } from './storage'

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
