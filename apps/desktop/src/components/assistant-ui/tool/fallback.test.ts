import { describe, expect, it } from 'vitest'

import { shouldBoundToolGroup } from './fallback'

describe('shouldBoundToolGroup', () => {
  it('bounds long runs of ordinary tool calls', () => {
    expect(shouldBoundToolGroup(3, false)).toBe(true)
  })

  it('never bounds a run containing clarify', () => {
    expect(shouldBoundToolGroup(3, true)).toBe(false)
  })
})
