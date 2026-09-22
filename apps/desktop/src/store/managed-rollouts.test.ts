import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $managedRollouts,
  _resetManagedRolloutsForTests,
  _setManagedRolloutsBridgeForTests,
  pollManagedRollouts,
  sendManagedRolloutCommand,
  startManagedRolloutPolling,
  stopManagedRolloutPolling
} from './managed-rollouts'

afterEach(() => {
  _resetManagedRolloutsForTests()
  vi.useRealTimers()
})

const snapshot = (revision: number) => ({ revision, rolloutId: 'r1', phase: 'running', data: {} })

describe('managed rollout renderer store', () => {
  it('initializes only through the modern managed-rollouts endpoint', async () => {
    const read = vi.fn().mockResolvedValue({ revision: 1, snapshot: snapshot(1) })
    _setManagedRolloutsBridgeForTests({ read, command: vi.fn() })

    await pollManagedRollouts()

    expect(read).toHaveBeenCalledWith(null)
    expect($managedRollouts.get()).toMatchObject({ status: 'ready', revision: 1 })
  })

  it('polls by revision and does not replace state with an older response', async () => {
    const read = vi.fn()
      .mockResolvedValueOnce({ revision: 3, snapshot: snapshot(3) })
      .mockResolvedValueOnce({ revision: 2, snapshot: snapshot(2) })
    _setManagedRolloutsBridgeForTests({ read, command: vi.fn() })

    await pollManagedRollouts()
    await pollManagedRollouts()

    expect(read).toHaveBeenLastCalledWith(3)
    expect($managedRollouts.get().revision).toBe(3)
  })

  it('reconciles a terminal null response without clearing the last snapshot', async () => {
    const read = vi.fn()
      .mockResolvedValueOnce({ revision: 4, snapshot: snapshot(4) })
      .mockResolvedValueOnce({ revision: 4, snapshot: null })
    _setManagedRolloutsBridgeForTests({ read, command: vi.fn() })

    await pollManagedRollouts()
    await pollManagedRollouts()

    expect($managedRollouts.get()).toMatchObject({ status: 'ready', revision: 4, snapshot: snapshot(4) })
  })

  it('coalesces overlapping command retries and cleans polling on stop', async () => {
    let resolve!: (value: unknown) => void
    const pending = new Promise(resolvePromise => { resolve = resolvePromise })
    const command = vi.fn().mockReturnValue(pending)
    const read = vi.fn().mockResolvedValue({ revision: 1, snapshot: snapshot(1) })
    _setManagedRolloutsBridgeForTests({ read, command })

    const first = sendManagedRolloutCommand({ requestId: 'r1', action: 'pause' })
    const second = sendManagedRolloutCommand({ requestId: 'r1', action: 'pause' })
    expect(command).toHaveBeenCalledTimes(1)
    resolve({ accepted: true })
    await Promise.all([first, second])

    vi.useFakeTimers()
    const stop = startManagedRolloutPolling()
    stop()
    stopManagedRolloutPolling()
    await vi.runOnlyPendingTimersAsync()
    expect(command).toHaveBeenCalledTimes(1)
  })

  it('stays fail-closed when the modern capability is unavailable', async () => {
    const command = vi.fn()
    const capabilities = vi.fn().mockResolvedValue({
      available: false,
      reason: 'trusted-assurance-provider-unavailable'
    })
    _setManagedRolloutsBridgeForTests({ capabilities, command })

    await pollManagedRollouts()

    expect(capabilities).toHaveBeenCalledTimes(1)
    expect(command).not.toHaveBeenCalled()
    expect($managedRollouts.get()).toMatchObject({
      status: 'unsupported',
      error: 'trusted-assurance-provider-unavailable'
    })
  })

  it('rechecks an unavailable capability on a later poll for reconnect recovery', async () => {
    const capabilities = vi.fn()
      .mockResolvedValueOnce({ available: false, reason: 'temporarily-unavailable' })
      .mockResolvedValueOnce({ available: true })
    const read = vi.fn().mockResolvedValue({ revision: 1, snapshot: snapshot(1) })
    _setManagedRolloutsBridgeForTests({ capabilities, read, command: vi.fn() })

    await pollManagedRollouts()
    await pollManagedRollouts()

    expect(capabilities).toHaveBeenCalledTimes(2)
    expect(read).toHaveBeenCalledWith(null)
    expect($managedRollouts.get()).toMatchObject({ status: 'ready', revision: 1 })
  })

  it('ignores a late response after polling is stopped', async () => {
    let resolve!: (value: unknown) => void
    const read = vi.fn().mockReturnValue(new Promise(value => { resolve = value }))
    _setManagedRolloutsBridgeForTests({ read, command: vi.fn() })

    const pending = pollManagedRollouts()
    stopManagedRolloutPolling()
    resolve({ revision: 2, snapshot: snapshot(2) })
    await pending

    expect($managedRollouts.get()).toMatchObject({ status: 'loading', revision: null, snapshot: null })
  })

  it('requires a stable request id before crossing the command bridge', async () => {
    const command = vi.fn()
    _setManagedRolloutsBridgeForTests({ command })

    await expect(sendManagedRolloutCommand({ action: 'pause' })).rejects.toThrow('request-id-required')
    expect(command).not.toHaveBeenCalled()
  })

  it('rejects a different in-flight action instead of coalescing it', async () => {
    const command = vi.fn().mockReturnValue(new Promise(() => undefined))
    _setManagedRolloutsBridgeForTests({ read: vi.fn(), command })

    void sendManagedRolloutCommand({ requestId: 'r1', action: 'pause' })
    await expect(sendManagedRolloutCommand({ requestId: 'r2', action: 'stop' })).rejects.toThrow('command-conflict')
    expect(command).toHaveBeenCalledTimes(1)
  })
})
