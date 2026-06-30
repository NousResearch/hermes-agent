import { describe, expect, it } from 'vitest'

import { Codecs } from './persisted'

describe('Codecs.stringArray', () => {
  it('decodes a JSON array of strings', () => {
    expect(Codecs.stringArray.decode('["a","b","c"]')).toEqual(['a', 'b', 'c'])
  })

  it('filters out non-strings and empty strings on decode', () => {
    expect(Codecs.stringArray.decode('["a","",123,null,"b"]')).toEqual(['a', 'b'])
  })

  it('returns empty array for non-array JSON', () => {
    expect(Codecs.stringArray.decode('"oops"')).toEqual([])
    expect(Codecs.stringArray.decode('{}')).toEqual([])
  })

  it('encodes a non-empty array to JSON', () => {
    expect(Codecs.stringArray.encode(['a', 'b'])).toBe('["a","b"]')
  })

  it('encodes an empty array to null (removes key)', () => {
    expect(Codecs.stringArray.encode([])).toBeNull()
  })
})

describe('Codecs.stringArrayPreserved', () => {
  it('decodes identically to stringArray', () => {
    expect(Codecs.stringArrayPreserved.decode('["a","b"]')).toEqual(['a', 'b'])
    expect(Codecs.stringArrayPreserved.decode('[]')).toEqual([])
    expect(Codecs.stringArrayPreserved.decode('"oops"')).toEqual([])
  })

  it('encodes a non-empty array to JSON', () => {
    expect(Codecs.stringArrayPreserved.encode(['a', 'b'])).toBe('["a","b"]')
  })

  it('encodes an empty array to "[]" (preserves key)', () => {
    expect(Codecs.stringArrayPreserved.encode([])).toBe('[]')
  })

  it('round-trips an empty array through encode → decode', () => {
    const encoded = Codecs.stringArrayPreserved.encode([])
    expect(encoded).toBe('[]')
    // Simulating what persistentAtom does: readKey returns "[]", decode returns []
    expect(Codecs.stringArrayPreserved.decode(encoded!)).toEqual([])
  })
})
