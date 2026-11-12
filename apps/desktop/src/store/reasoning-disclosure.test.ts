import { describe, expect, it } from 'vitest'

import { localShowReasoning } from './reasoning-disclosure'

describe('localShowReasoning', () => {
  it('reads the local per-platform key over the global one', () => {
    expect(localShowReasoning({ show_reasoning: true, platforms: { cli: { show_reasoning: false } } })).toBe(false)
    expect(localShowReasoning({ show_reasoning: false, platforms: { cli: { show_reasoning: true } } })).toBe(true)
  })

  it('falls back to the global key, and ignores other platforms', () => {
    expect(localShowReasoning({ show_reasoning: false })).toBe(false)
    expect(localShowReasoning({ show_reasoning: true, platforms: { telegram: { show_reasoning: false } } })).toBe(true)
    expect(localShowReasoning(undefined)).toBeUndefined()
  })
})
