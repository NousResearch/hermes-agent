import { describe, expect, it } from 'vitest'
import { BUILTIN_PERSONALITIES } from '@/lib/personalities'

describe('BUILTIN_PERSONALITIES', () => {
  it('is a non-empty array', () => {
    expect(Array.isArray(BUILTIN_PERSONALITIES)).toBe(true)
    expect(BUILTIN_PERSONALITIES.length).toBeGreaterThan(0)
  })

  it('contains expected personalities', () => {
    for (const p of ['helpful', 'concise', 'creative']) {
      expect(BUILTIN_PERSONALITIES).toContain(p)
    }
  })

  it('all entries are strings', () => {
    for (const p of BUILTIN_PERSONALITIES) {
      expect(typeof p).toBe('string')
    }
  })
})
