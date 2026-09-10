import { describe, expect, it } from 'vitest'

import { resolveWorldAnimation, resolveWorldAnimationRuntime } from './world-animation'

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

  it('produces the renderer runtime contract with actor and target selectors', () => {
    expect(
      resolveWorldAnimationRuntime(
        {
          actor: { agentId: 'worker-a' },
          animationTags: ['repair'],
          personality: 'methodical',
          state: 'repairing',
          target: { taskId: 'task-7' }
        },
        { cosmetic: { intensity: 2 } }
      )
    ).toEqual({
      actorSelector: 'agent:worker-a',
      clip: 'repair',
      fallbackClip: 'idle',
      intensity: 2,
      loop: false,
      tags: ['repair'],
      targetSelector: 'task:task-7'
    })
  })
})
