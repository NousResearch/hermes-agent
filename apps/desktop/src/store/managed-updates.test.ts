import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopManagedConnectionUpdateResult } from '@/global'

const {
  $managedUpdates,
  _resetManagedUpdatesForTests,
  isManagedUpdateBusyMessage,
  managedUpdatesSupported,
  runManagedUpdate
} = await import('./managed-updates')

const updateManaged = vi.fn<(id: string) => Promise<DesktopManagedConnectionUpdateResult>>()

function managedResult(over: Partial<DesktopManagedConnectionUpdateResult> = {}): DesktopManagedConnectionUpdateResult {
  return {
    connectionId: 'linux-ssh',
    correlationId: '0c44e2da-993e-4d96-ab45-2d0f73365d61',
    exitCode: 0,
    ok: true,
    outcome: 'updated',
    receipt: {
      correlationId: '0c44e2da-993e-4d96-ab45-2d0f73365d61',
      outcome: 'success',
      postVersion: '1.1.0',
      preVersion: '1.0.0'
    },
    restoreOk: true,
    scopes: [{ profile: 'default', restored: true }],
    updateOk: true,
    ...over
  }
}

beforeEach(() => {
  _resetManagedUpdatesForTests()
  updateManaged.mockReset().mockResolvedValue(managedResult())
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
    connections: { updateManaged }
  }
})

describe('managedUpdatesSupported', () => {
  it('is false on an older Electron main without the transactional bridge', () => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { connections: {} }
    expect(managedUpdatesSupported()).toBe(false)
  })

  it('is true when the preload bridge exposes updateManaged', () => {
    expect(managedUpdatesSupported()).toBe(true)
  })
})

describe('runManagedUpdate', () => {
  it('routes through the managed drain/update/restore bridge and lands on updated with the receipt', async () => {
    const pending = runManagedUpdate('linux-ssh')

    expect($managedUpdates.get()['linux-ssh']).toMatchObject({ status: 'updating' })

    const state = await pending

    expect(updateManaged).toHaveBeenCalledWith('linux-ssh')
    expect(state).toMatchObject({ alreadyRunning: false, connectionId: 'linux-ssh', status: 'updated' })
    expect(state.receipt).toMatchObject({
      correlationId: '0c44e2da-993e-4d96-ab45-2d0f73365d61',
      outcome: 'success',
      postVersion: '1.1.0'
    })
    expect($managedUpdates.get()['linux-ssh']).toMatchObject({ status: 'updated' })
  })

  it('joins a repeat click to the in-flight promise instead of double-dispatching', async () => {
    let resolve!: (value: DesktopManagedConnectionUpdateResult) => void
    updateManaged.mockReturnValue(
      new Promise<DesktopManagedConnectionUpdateResult>(next => {
        resolve = next
      })
    )

    const first = runManagedUpdate('linux-ssh')
    const second = runManagedUpdate('linux-ssh')

    expect(second).toBe(first)
    expect(updateManaged).toHaveBeenCalledTimes(1)

    resolve(managedResult())
    await expect(first).resolves.toMatchObject({ status: 'updated' })
  })

  it('keeps updated-but-restore-failed truthful as partial', async () => {
    updateManaged.mockResolvedValue(
      managedResult({
        error: 'work profile did not come back',
        ok: false,
        outcome: 'restore-failed',
        restoreOk: false,
        scopes: [
          { profile: 'default', restored: true },
          { error: 'ssh dial timed out', profile: 'work', restored: false }
        ]
      })
    )

    const state = await runManagedUpdate('linux-ssh')

    expect(state).toMatchObject({
      message: 'work profile did not come back',
      status: 'partial'
    })
    expect(state.scopes).toHaveLength(2)
  })

  it('surfaces the managed-update-in-progress refusal as busy, not a scary failure', async () => {
    updateManaged.mockResolvedValue(
      managedResult({
        message: 'A managed update is already in progress.',
        ok: false,
        outcome: 'refused',
        receipt: null,
        scopes: [],
        updateOk: false
      })
    )

    const state = await runManagedUpdate('linux-ssh')

    expect(state).toMatchObject({
      alreadyRunning: true,
      message: 'A managed update is already in progress.',
      status: 'refused'
    })
  })

  it('maps a thrown managed-update-in-progress IPC envelope to the same busy state', async () => {
    const error: Error & { code?: string } = new Error(
      "Error invoking remote method 'hermes:connections:update-managed': " +
        'SSH connection "linux-ssh" is paused while its managed update is in progress.'
    )

    error.code = 'managed-update-in-progress'
    updateManaged.mockRejectedValue(error)

    const state = await runManagedUpdate('linux-ssh')

    expect(state).toMatchObject({ alreadyRunning: true, status: 'refused' })
  })

  it('keeps a non-busy refusal refused with its exact message', async () => {
    updateManaged.mockResolvedValue(
      managedResult({
        message: 'Only registered Desktop-managed SSH connections can use this update lifecycle.',
        ok: false,
        outcome: 'refused',
        receipt: null,
        updateOk: false
      })
    )

    const state = await runManagedUpdate('remote-a')

    expect(state).toMatchObject({
      alreadyRunning: false,
      message: 'Only registered Desktop-managed SSH connections can use this update lifecycle.',
      status: 'refused'
    })
  })

  it('lands a failed update on failed with the stop reason as fallback message', async () => {
    updateManaged.mockResolvedValue(
      managedResult({
        error: undefined,
        exitCode: 1,
        message: undefined,
        ok: false,
        outcome: 'update-failed',
        receipt: {
          correlationId: 'run-x',
          outcome: 'failed',
          stopReason: 'launcher exited 1'
        },
        restoreOk: true,
        updateOk: false
      })
    )

    const state = await runManagedUpdate('linux-ssh')

    expect(state).toMatchObject({ message: 'launcher exited 1', status: 'failed' })
  })

  it('fails closed when the bridge is missing instead of pretending to update', async () => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { connections: {} }

    const state = await runManagedUpdate('linux-ssh')

    expect(state).toMatchObject({ status: 'failed' })
    expect(updateManaged).not.toHaveBeenCalled()
  })
})

describe('isManagedUpdateBusyMessage', () => {
  it('recognizes every busy-gate phrasing and rejects unrelated failures', () => {
    expect(isManagedUpdateBusyMessage('A managed update is already in progress.')).toBe(true)
    expect(isManagedUpdateBusyMessage('code managed-update-in-progress')).toBe(true)
    expect(isManagedUpdateBusyMessage('ssh dial timed out')).toBe(false)
    expect(isManagedUpdateBusyMessage(null)).toBe(false)
  })
})
