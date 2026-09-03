import { describe, expect, it } from 'vitest'
import { rafCoalesce } from '@/lib/raf-coalesce'

describe('rafCoalesce', () => {
  it('push stores value, finish applies it', () => {
    let applied: number | null = null
    const rc = rafCoalesce<number>((v) => { applied = v })
    rc.push(42)
    expect(applied).toBeNull() // not yet applied
    rc.finish()
    expect(applied).toBe(42)
  })

  it('multiple pushes keep only latest', () => {
    let applied: string | null = null
    const rc = rafCoalesce<string>((v) => { applied = v })
    rc.push('first')
    rc.push('second')
    rc.finish()
    expect(applied).toBe('second')
  })

  it('finish without push is safe', () => {
    const rc = rafCoalesce<number>(() => {})
    expect(() => rc.finish()).not.toThrow()
  })
})
