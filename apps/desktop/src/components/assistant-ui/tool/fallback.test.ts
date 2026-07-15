import { describe, expect, it } from 'vitest'

import { shouldBoundToolGroup, UNBOUNDABLE_TOOLS } from './fallback'

describe('shouldBoundToolGroup', () => {
  it('bounds long runs of ordinary tool calls', () => {
    expect(shouldBoundToolGroup(3, false)).toBe(true)
  })

  it('leaves short runs unbounded', () => {
    expect(shouldBoundToolGroup(2, false)).toBe(false)
  })

  it('never bounds a run holding an unboundable tool', () => {
    expect(shouldBoundToolGroup(3, true)).toBe(false)
  })
})

describe('UNBOUNDABLE_TOOLS', () => {
  it('exempts clarify forms and generated images from the window', () => {
    expect(UNBOUNDABLE_TOOLS.has('clarify')).toBe(true)
    expect(UNBOUNDABLE_TOOLS.has('image_generate')).toBe(true)
  })
})
