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
import { availableCapability, rolloutSnapshot } from './managed-rollouts.test-fixtures'

afterEach(() => {
  _resetManagedRolloutsForTests()
  vi.useRealTimers()
})

const snapshot = rolloutSnapshot

const commandPayload = (requestId: string, action: 'pause' | 'stop' = 'pause') => ({
  id: '11111111-1111-4111-8111-111111111111',
  expectedRevision: 1,
  requestId,
  action,
  installId: null,
  reason: null,
  promotionPolicy: null
})

const acceptedCommand = { ok: true, id: '11111111-1111-4111-8111-111111111111', revision: 2, code: null, message: null, changes: [] }

describe('managed rollout renderer store', () => {
  it('initializes only through the modern managed-rollouts endpoint', async () => {
    const read = vi.fn().mockResolvedValue({ revision: 1, snapshot: snapshot(1) })
    _setManagedRolloutsBridgeForTests({
      capabilities: vi.fn().mockResolvedValue(availableCapability),
      activeRevision: vi.fn().mockResolvedValue(1),
      read,
      command: vi.fn()
    })

    await pollManagedRollouts()

    expect(read).toHaveBeenCalledWith(null)
    expect($managedRollouts.get()).toMatchObject({ status: 'ready', revision: 1 })
  })

  it('polls by revision and does not replace state with an older response', async () => {
    const read = vi.fn()
      .mockResolvedValueOnce({ revision: 3, snapshot: snapshot(3) })
      .mockResolvedValueOnce({ revision: 2, snapshot: snapshot(2) })

    const activeRevision = vi.fn().mockResolvedValueOnce(3).mockResolvedValueOnce(2)
    _setManagedRolloutsBridgeForTests({
      capabilities: vi.fn().mockResolvedValue(availableCapability),
      activeRevision,
      read,
      command: vi.fn()
    })

    await pollManagedRollouts()
    await pollManagedRollouts()

    expect(read).toHaveBeenLastCalledWith(null)
    expect($managedRollouts.get().revision).toBe(3)
  })

  it('uses activeRevision to avoid clearing the last snapshot on unchanged polls', async () => {
    const read = vi.fn().mockResolvedValue({ revision: 4, snapshot: snapshot(4) })
    const activeRevision = vi.fn().mockResolvedValue(4)
    _setManagedRolloutsBridgeForTests({
      capabilities: vi.fn().mockResolvedValue(availableCapability),
      activeRevision,
      read,
      command: vi.fn()
    })

    await pollManagedRollouts()
    await pollManagedRollouts()

    expect($managedRollouts.get()).toMatchObject({ status: 'ready', revision: 4, snapshot: snapshot(4) })
  })

  it('coalesces overlapping command retries and cleans polling on stop', async () => {
    let resolve!: (value: unknown) => void
    const pending = new Promise(resolvePromise => { resolve = resolvePromise })
    const command = vi.fn().mockReturnValue(pending)
    const read = vi.fn().mockResolvedValue({ revision: 1, snapshot: snapshot(1) })
    _setManagedRolloutsBridgeForTests({
      capabilities: vi.fn().mockResolvedValue(availableCapability),
      activeRevision: vi.fn().mockResolvedValue(1),
      read,
      command
    })

    const first = sendManagedRolloutCommand(commandPayload('22222222-2222-4222-8222-222222222222'))
    const second = sendManagedRolloutCommand(commandPayload('22222222-2222-4222-8222-222222222222'))
    expect(command).toHaveBeenCalledTimes(1)
    resolve(acceptedCommand)
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
      ...availableCapability,
      available: false,
      reason: 'trusted-assurance-provider-unavailable',
      maxConcurrency: 0,
      maxInstallations: 0
    })

    _setManagedRolloutsBridgeForTests({
      capabilities,
      activeRevision: vi.fn().mockResolvedValue(null),
      read: vi.fn(),
      command
    })

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
      .mockResolvedValueOnce({ ...availableCapability, available: false, reason: 'temporarily-unavailable', maxConcurrency: 0, maxInstallations: 0 })
      .mockResolvedValueOnce(availableCapability)

    const read = vi.fn().mockResolvedValue({ revision: 1, snapshot: snapshot(1) })
    _setManagedRolloutsBridgeForTests({
      capabilities,
      activeRevision: vi.fn().mockResolvedValue(1),
      read,
      command: vi.fn()
    })

    await pollManagedRollouts()
    await pollManagedRollouts()

    expect(capabilities).toHaveBeenCalledTimes(2)
    expect(read).toHaveBeenCalledWith(null)
    expect($managedRollouts.get()).toMatchObject({ status: 'ready', revision: 1 })
  })

  it('ignores a late response after polling is stopped', async () => {
    let resolve!: (value: unknown) => void
    const read = vi.fn().mockReturnValue(new Promise(value => { resolve = value }))
    _setManagedRolloutsBridgeForTests({
      capabilities: vi.fn().mockResolvedValue(availableCapability),
      activeRevision: vi.fn().mockResolvedValue(1),
      read,
      command: vi.fn()
    })

    const pending = pollManagedRollouts()
    stopManagedRolloutPolling()
    resolve({ revision: 2, snapshot: snapshot(2) })
    await pending

    expect($managedRollouts.get()).toMatchObject({ status: 'loading', revision: null, snapshot: null })
  })

  it('requires a stable request id before crossing the command bridge', async () => {
    const command = vi.fn()
    _setManagedRolloutsBridgeForTests({
      capabilities: vi.fn().mockResolvedValue(availableCapability),
      activeRevision: vi.fn().mockResolvedValue(null),
      read: vi.fn(),
      command
    })

    await expect(sendManagedRolloutCommand({ ...commandPayload(''), requestId: '' })).rejects.toThrow('request-id-required')
    expect(command).not.toHaveBeenCalled()
  })

  it('rejects a different in-flight action instead of coalescing it', async () => {
    const command = vi.fn().mockReturnValue(new Promise(() => undefined))
    _setManagedRolloutsBridgeForTests({
      capabilities: vi.fn().mockResolvedValue(availableCapability),
      activeRevision: vi.fn().mockResolvedValue(null),
      read: vi.fn(),
      command
    })

    void sendManagedRolloutCommand(commandPayload('22222222-2222-4222-8222-222222222222')).catch(() => undefined)
    await expect(sendManagedRolloutCommand(commandPayload('33333333-3333-4333-8333-333333333333', 'stop'))).rejects.toThrow('command-conflict')
    expect(command).toHaveBeenCalledTimes(1)
  })

  it('fetches a settled full snapshot when active revision becomes null', async () => {
    const read = vi.fn().mockResolvedValue({ revision: 1, snapshot: snapshot(1) })
    const get = vi.fn().mockResolvedValue(snapshot(2, { phase: 'completed' }))
    _setManagedRolloutsBridgeForTests({
      capabilities: vi.fn().mockResolvedValue(availableCapability),
      activeRevision: vi.fn().mockResolvedValueOnce(1).mockResolvedValueOnce(null),
      read,
      get,
      command: vi.fn()
    })

    await pollManagedRollouts()
    await pollManagedRollouts()

    expect(get).toHaveBeenCalledWith('11111111-1111-4111-8111-111111111111')
    expect($managedRollouts.get()).toMatchObject({ status: 'ready', active: false, revision: 2 })
    expect($managedRollouts.get().snapshot?.phase).toBe('completed')
  })

  it('rechecks capability after it was available and refuses newly unavailable admission', async () => {
    const capabilities = vi.fn()
      .mockResolvedValueOnce(availableCapability)
      .mockResolvedValueOnce({ ...availableCapability, available: false, reason: 'review-manifest-unavailable', maxConcurrency: 0, maxInstallations: 0 })

    _setManagedRolloutsBridgeForTests({
      capabilities,
      activeRevision: vi.fn().mockResolvedValue(null),
      read: vi.fn(),
      command: vi.fn()
    })

    await pollManagedRollouts()
    await pollManagedRollouts()

    expect(capabilities).toHaveBeenCalledTimes(2)
    expect($managedRollouts.get()).toMatchObject({ status: 'unsupported', error: 'review-manifest-unavailable' })
  })
})
