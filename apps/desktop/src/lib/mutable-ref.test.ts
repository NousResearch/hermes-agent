import { describe, expect, it } from 'vitest'
import { setMutableRef } from '@/lib/mutable-ref'

describe('setMutableRef', () => {
  it('sets ref.current to value', () => {
    const ref = { current: 'old' }
    setMutableRef(ref, 'new')
    expect(ref.current).toBe('new')
  })

  it('works with objects', () => {
    const obj = { a: 1 }
    const ref = { current: null as typeof obj | null }
    setMutableRef(ref, obj)
    expect(ref.current).toBe(obj)
  })

  it('works with numbers', () => {
    const ref = { current: 0 }
    setMutableRef(ref, 42)
    expect(ref.current).toBe(42)
  })
})
