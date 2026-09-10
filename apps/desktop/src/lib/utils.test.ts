import { describe, expect, it } from 'vitest'
import { cn } from '@/lib/utils'

describe('cn', () => {
  it('joins class names', () => {
    expect(cn('a', 'b')).toContain('a')
    expect(cn('a', 'b')).toContain('b')
  })

  it('handles empty input', () => {
    expect(cn()).toBe('')
  })

  it('deduplicates conflicting tailwind classes', () => {
    const result = cn('p-4', 'p-8')
    expect(result).toContain('p-8')
    expect(result).not.toContain('p-4')
  })
})
