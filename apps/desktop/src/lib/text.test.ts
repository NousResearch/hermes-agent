import { describe, expect, it } from 'vitest'

import { foldIncludes, searchFold } from './text'

describe('searchFold', () => {
  it('folds separator characters to spaces, both directions equivalent', () => {
    expect(searchFold('qwen3.8-flash')).toBe('qwen3 8 flash')
    expect(searchFold('Qwen3.8 Flash')).toBe('qwen3 8 flash')
    expect(searchFold('GPT-5')).toBe('gpt 5')
    expect(searchFold('gpt 5')).toBe('gpt 5')
    expect(searchFold('Q4_K_XL')).toBe('q4 k xl')
  })

  it('is length-preserving (1 char in, 1 char out) so highlight indices survive', () => {
    for (const text of ['qwen3.8-flash', 'GPT-5', 'a--b__c..d', 'Grok 4.5 Retro', '']) {
      expect(searchFold(text).length).toBe(text.length)
    }
  })

  it('already-folded text is unchanged (idempotent)', () => {
    expect(searchFold('qwen3 8 flash')).toBe('qwen3 8 flash')
  })
})

describe('foldIncludes', () => {
  it('matches across the hyphen/space/dot/underscore separator difference', () => {
    expect(foldIncludes('Qwen3.8 Flash', 'qwen3.8-flash')).toBe(true)
    expect(foldIncludes('qwen3.8-flash', 'qwen3 8 flash')).toBe(true)
    expect(foldIncludes('GPT-5', 'gpt 5')).toBe(true)
    expect(foldIncludes('gpt-5', 'GPT.5')).toBe(true)
    expect(foldIncludes('Q4_K_XL', 'q4 k xl')).toBe(true)
  })

  it('is a superset of the plain-lowercase includes it replaces', () => {
    // Any query that matched before must still match.
    expect(foldIncludes('qwen3.8-flash', 'qwen3.8')).toBe(true)
    expect(foldIncludes('Qwen3.8 Flash', 'flash')).toBe(true)
    expect(foldIncludes('nous-portal', 'nous')).toBe(true)
  })

  it('still rejects genuinely different text', () => {
    expect(foldIncludes('Gemini Flash', 'qwen')).toBe(false)
    expect(foldIncludes('gpt-5', 'gpt-6')).toBe(false)
  })
})
