import { describe, expect, it, vi } from 'vitest'

import { createSshBackendRetirement } from './ssh-backend-retirement'

describe('SSH backend retirement', () => {
  it('retains live and uncertain owners in pool and bootstrap recovery', async () => {
    const owner = { pid: 123 }
    const entry = {}
    const isAlive = vi.fn(async () => true)
    const forgetEntry = vi.fn()
    const teardownOwner = vi.fn()

    const retirement = createSshBackendRetirement({
      getOwner: () => owner, isAlive, getEntry: () => entry, forgetEntry, teardownOwner,
      withDrain: async (_key, cleanup) => cleanup()
    })

    await retirement.retirePool('ssh', entry)
    await retirement.retireIfDead('ssh')
    expect(forgetEntry).not.toHaveBeenCalled()
    expect(teardownOwner).not.toHaveBeenCalled()
    isAlive.mockRejectedValueOnce(new Error('SSH unavailable'))
    await expect(retirement.retirePool('ssh', entry)).rejects.toThrow('SSH unavailable')
    expect(teardownOwner).not.toHaveBeenCalled()
  })

  it('checks the owner and entry again inside the drain and after PID confirmation', async () => {
    let owner = { pid: 123 }
    let entry = {}
    const initial = entry
    const forgetEntry = vi.fn()
    const teardownOwner = vi.fn()
    let replaceInDrain = true

    const retirement = createSshBackendRetirement({
      getOwner: () => owner,
      isAlive: async () => { entry = {};

 return false },
      getEntry: () => entry, forgetEntry, teardownOwner,
      withDrain: async (_key, cleanup) => {
        if (replaceInDrain) { owner = { pid: 456 } }
        await cleanup()
      }
    })

    await retirement.retirePool('ssh', initial)
    expect(forgetEntry).not.toHaveBeenCalled()
    replaceInDrain = false
    await retirement.retirePool('ssh', entry)
    expect(forgetEntry).not.toHaveBeenCalled()
    expect(teardownOwner).not.toHaveBeenCalled()
  })
})
