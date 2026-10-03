import { describe, expect, it } from 'vitest'
import { modelSearchText } from '@/lib/model-search-text'

describe('modelSearchText', () => {
  it('returns id unchanged when no alias', () => {
    expect(modelSearchText('gpt-4o')).toBe('gpt-4o')
  })

  it('expands k3 alias', () => {
    const result = modelSearchText('k3')
    expect(result).toContain('k3')
    expect(result).toContain('kimi')
  })

  it('expands x-preview-f-free alias', () => {
    const result = modelSearchText('x-preview-f-free')
    expect(result).toContain('ox-alpha')
  })

  it('empty string returns empty', () => {
    expect(modelSearchText('')).toBe('')
  })

  it('trims whitespace', () => {
    expect(modelSearchText('  gpt-4o  ')).toBe('gpt-4o')
  })
})
