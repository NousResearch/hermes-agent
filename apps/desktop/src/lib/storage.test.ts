import { beforeEach, describe, expect, it } from 'vitest'

import { parseStringRecord, persistStringArray, storedStringArray, storedStringRecord } from './storage'

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

describe('string record storage', () => {
  it('keeps only string entries and reads anything absent or malformed as empty', () => {
    expect(parseStringRecord(JSON.stringify({ a: 'x', b: 1, c: null, d: '' }))).toEqual({ a: 'x', d: '' })

    for (const raw of [null, '', 'null', '"x"', '[["a","x"]]', '{not json']) {
      expect(parseStringRecord(raw)).toEqual({})
    }
  })

  it('reads a stored record through the same parse', () => {
    window.localStorage.setItem('test.record', JSON.stringify({ a: 'x', b: 2 }))
    window.localStorage.setItem('test.broken', '{')

    expect(storedStringRecord('test.record')).toEqual({ a: 'x' })
    expect(storedStringRecord('test.broken')).toEqual({})
    expect(storedStringRecord('test.absent')).toEqual({})
  })
})
