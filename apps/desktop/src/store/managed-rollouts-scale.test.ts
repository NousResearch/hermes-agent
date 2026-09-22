import { describe, expect, it, vi } from 'vitest'

import { $managedRollouts, _resetManagedRolloutsForTests, _setManagedRolloutsBridgeForTests, pollManagedRollouts } from './managed-rollouts'

describe('managed rollout store scale fixture', () => {
  it('accepts a bounded 500-target snapshot without changing revision semantics', async () => {
    const targets = Array.from({ length: 500 }, (_, index) => ({ installId: `install-${index}`, phase: 'ready' }))
    const read = vi.fn().mockResolvedValue({ revision: 7, snapshot: { revision: 7, rolloutId: 'r-scale', phase: 'ready', data: { targets } } })
    _resetManagedRolloutsForTests()
    _setManagedRolloutsBridgeForTests({ read, command: vi.fn() })

    await pollManagedRollouts()

    expect($managedRollouts.get().revision).toBe(7)
    expect(($managedRollouts.get().snapshot?.data.targets as unknown[])).toHaveLength(500)
    expect(read).toHaveBeenCalledWith(null)
  })
})
