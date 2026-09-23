import { describe, expect, it } from 'vitest'

import {
  HIGHLIGHT_CACHE_MAX_CHARS,
  HIGHLIGHT_CACHE_MAX_ENTRIES,
  HighlightCache,
  highlightCacheKey
} from '@/components/chat/shiki-highlight-cache'

describe('highlightCacheKey', () => {
  it('separates scope, language and code so distinct blocks never collide', () => {
    const a = highlightCacheKey('scope-1', 'ts', 'const x = 1')
    const b = highlightCacheKey('scope-1', 'ts', 'const x = 2')

    expect(a).not.toBe(b)
    expect(highlightCacheKey('scope-1', 'js', 'const x = 1')).not.toBe(a)
    expect(highlightCacheKey('scope-2', 'ts', 'const x = 1')).not.toBe(a)
  })
})

describe('HighlightCache', () => {
  it('round-trips an entry and refreshes recency on get', () => {
    const cache = new HighlightCache(3, 1000)

    cache.set('a', 'A')
    cache.set('b', 'B')
    cache.set('c', 'C')
    // Touch the oldest entry so it becomes the newest.
    expect(cache.get('a')).toBe('A')
    cache.set('d', 'D')

    // 'b' is now the least recently used and must be evicted first.
    expect(cache.has('b')).toBe(false)
    expect(cache.get('a')).toBe('A')
    expect(cache.get('c')).toBe('C')
    expect(cache.get('d')).toBe('D')
  })

  it('evicts oldest-first past the entry cap', () => {
    const cache = new HighlightCache(2, 1_000_000)

    cache.set('a', 'A')
    cache.set('b', 'B')
    cache.set('c', 'C')

    expect(cache.size).toBe(2)
    expect(cache.has('a')).toBe(false)
    expect(cache.has('b')).toBe(true)
    expect(cache.has('c')).toBe(true)
  })

  it('evicts oldest-first past the total-char cap', () => {
    const cache = new HighlightCache(HIGHLIGHT_CACHE_MAX_ENTRIES, 10)

    cache.set('a', '12345')
    cache.set('b', '123456')

    expect(cache.size).toBe(1)
    expect(cache.has('a')).toBe(false)
    expect(cache.has('b')).toBe(true)
    expect(cache.totalChars).toBeLessThanOrEqual(10)
  })

  it('replaces an existing key in place without double-counting chars', () => {
    const cache = new HighlightCache(2, 100)

    cache.set('a', '12345')
    cache.set('a', '1234567890')

    expect(cache.size).toBe(1)
    expect(cache.totalChars).toBe(10)
  })

  it('stays within both caps for a large burst of unique blocks', () => {
    const cache = new HighlightCache(HIGHLIGHT_CACHE_MAX_ENTRIES, HIGHLIGHT_CACHE_MAX_CHARS)

    for (let i = 0; i < 2_000; i++) {
      cache.set(`block-${i}`, `html-${i}`.repeat(100))
    }

    expect(cache.size).toBeLessThanOrEqual(HIGHLIGHT_CACHE_MAX_ENTRIES)
    expect(cache.totalChars).toBeLessThanOrEqual(HIGHLIGHT_CACHE_MAX_CHARS)
  })

  it('clear drops everything', () => {
    const cache = new HighlightCache()

    cache.set('a', 'A')
    cache.clear()

    expect(cache.size).toBe(0)
    expect(cache.totalChars).toBe(0)
    expect(cache.get('a')).toBeUndefined()
  })
})
