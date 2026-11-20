import { describe, expect, it } from 'vitest'

import { LruCache } from './lru-cache'

describe('LruCache', () => {
  it('never grows past its ceiling', () => {
    const cache = new LruCache<number, string>(3)

    for (let key = 0; key < 100; key += 1) {
      cache.set(key, `v${key}`)
    }

    expect(cache.size).toBe(3)
    expect([...cache.keys()]).toEqual([97, 98, 99])
  })

  it('evicts the least recently READ entry, not the least recently written', () => {
    const cache = new LruCache<string, number>(2)

    cache.set('a', 1)
    cache.set('b', 2)
    cache.get('a')
    cache.set('c', 3)

    expect(cache.get('a')).toBe(1)
    expect(cache.get('b')).toBeUndefined()
    expect(cache.get('c')).toBe(3)
  })

  it('overwrites in place rather than evicting a live neighbour', () => {
    const cache = new LruCache<string, number>(2)

    cache.set('a', 1)
    cache.set('b', 2)
    cache.set('a', 9)

    expect(cache.size).toBe(2)
    expect(cache.get('a')).toBe(9)
    expect(cache.get('b')).toBe(2)
  })

  it('answers membership without promoting the entry', () => {
    const cache = new LruCache<string, number>(2)

    cache.set('a', 1)
    cache.set('b', 2)
    expect(cache.has('a')).toBe(true)
    cache.set('c', 3)

    expect(cache.has('a')).toBe(false)
    expect(cache.has('b')).toBe(true)
  })

  it('frees the slot on delete', () => {
    const cache = new LruCache<string, number>(2)

    cache.set('a', 1)

    expect(cache.delete('a')).toBe(true)
    expect(cache.delete('a')).toBe(false)
    expect(cache.size).toBe(0)

    cache.set('b', 2)
    cache.set('c', 3)

    expect([...cache.keys()]).toEqual(['b', 'c'])
  })
})
