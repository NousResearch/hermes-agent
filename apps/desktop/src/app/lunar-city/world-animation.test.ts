import { describe, expect, it } from 'vitest'

import { resolveWorldAnimation } from './world-animation'

describe('resolveWorldAnimation', () => {
  it('turns blocked responders into a looping repair clip', () => {
    expect(
      resolveWorldAnimation(
        { state: 'repairing', animationTags: ['repair', 'extinguish'] },
        { cosmetic: { intensity: 3 } }
      )
    ).toEqual({ clip: 'repair', intensity: 3, loop: false, tags: ['repair', 'extinguish'] })
  })

  it('turns a stable merge into a non-looping celebration', () => {
    expect(
      resolveWorldAnimation(
        { state: 'celebrating', animationTags: ['celebration.citywide', 'dance'] },
        { cosmetic: { intensity: 3 } }
      )
    ).toEqual({ clip: 'celebrate', intensity: 3, loop: false, tags: ['celebration.citywide', 'dance'] })
  })

  it('has an idle fallback for an unknown state at the boundary', () => {
    expect(
      resolveWorldAnimation(
        { state: 'idle', animationTags: [] },
        { cosmetic: { intensity: 0 } }
      ).clip
    ).toBe('idle')
  })
})
