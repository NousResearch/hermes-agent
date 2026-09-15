import { describe, expect, it } from 'vitest'
import { capitalize } from '@/lib/text'

describe('capitalize additional', () => {
  it('single char', () => {
    expect(capitalize('a')).toBe('A')
  })

  it('preserves rest of string', () => {
    expect(capitalize('hELLO')).toBe('HELLO')
  })
})
