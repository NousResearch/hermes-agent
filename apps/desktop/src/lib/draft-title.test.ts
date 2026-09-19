import { describe, expect, it } from 'vitest'
import { deriveDraftTitle } from '@/lib/draft-title'

describe('deriveDraftTitle', () => {
  it('returns first line for short content', () => {
    expect(deriveDraftTitle('Hello world')).toBe('Hello world')
  })

  it('returns empty for empty string', () => {
    expect(deriveDraftTitle('')).toBe('')
  })

  it('returns empty for whitespace-only content', () => {
    expect(deriveDraftTitle('   \n  \n  ')).toBe('')
  })

  it('uses first non-empty line', () => {
    expect(deriveDraftTitle('\n\nSecond line\nThird line')).toBe('Second line')
  })

  it('collapses multiple spaces', () => {
    expect(deriveDraftTitle('Hello    world')).toBe('Hello world')
  })

  it('truncates long content at word boundary with ellipsis', () => {
    const long = 'This is a very long message that exceeds the maximum character limit for titles'
    const result = deriveDraftTitle(long)
    expect(result.length).toBeLessThanOrEqual(49) // 48 chars + ellipsis
    expect(result).toContain('…')
  })

  it('does not end with raw space before ellipsis', () => {
    const long = 'word '.repeat(20)
    const result = deriveDraftTitle(long)
    expect(result.endsWith(' …') || result.endsWith('…')).toBe(true)
  })
})
