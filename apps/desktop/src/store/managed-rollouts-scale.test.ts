import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $managedRollouts,
  _resetManagedRolloutsForTests,
  _setManagedRolloutsBridgeForTests,
  pollManagedRollouts
} from './managed-rollouts'

const targets = Array.from({ length: 500 }, (_, index) => ({
  installId: `install-${index}`,
  phase: index === 499 ? 'completed' : 'ready'
}))

function response(revision: number, phase = 'ready', snapshot = true) {
  return {
    revision,
    snapshot: snapshot
      ? { revision, rolloutId: 'r-scale', phase, data: { targets } }
      : null
  }
}

describe('managed rollout store scale fixture', () => {
  afterEach(() => {
    _resetManagedRolloutsForTests()
  })

  it('measures a bounded 500-target snapshot instead of relying on row count alone', async () => {
    const first = response(7)
    const encodedBytes = new TextEncoder().encode(JSON.stringify(first)).byteLength
    const read = vi.fn().mockResolvedValue(first)
    _setManagedRolloutsBridgeForTests({ read, command: vi.fn() })

    await pollManagedRollouts()

    expect(encodedBytes).toBeLessThan(8 * 1024 * 1024)
    expect($managedRollouts.get().revision).toBe(7)
    expect($managedRollouts.get().snapshot?.data.targets).toHaveLength(500)
    expect(read).toHaveBeenCalledWith(null)
  })

  it('bounds unchanged-revision traffic and never drops the terminal snapshot', async () => {
    const read = vi
      .fn()
      .mockResolvedValueOnce(response(7))
      .mockResolvedValueOnce(response(7, 'completed', false))
      .mockResolvedValueOnce(response(7, 'completed', false))
    _setManagedRolloutsBridgeForTests({ read, command: vi.fn() })

    await pollManagedRollouts()
    await pollManagedRollouts()
    await pollManagedRollouts()

    expect(read).toHaveBeenCalledTimes(3)
    expect(read).toHaveBeenNthCalledWith(1, null)
    expect(read).toHaveBeenNthCalledWith(2, 7)
    expect(read).toHaveBeenNthCalledWith(3, 7)
    expect($managedRollouts.get()).toMatchObject({ status: 'ready', revision: 7 })
    expect($managedRollouts.get().snapshot?.phase).toBe('ready')
  })
})
