import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'

const storage = new Map<string, string>()

vi.mock('@/lib/storage', () => ({
  persistString: (key: string, value: null | string) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, value)
    }
  },
  storedString: (key: string) => storage.get(key) ?? null
}))

const notifySpy = vi.fn()
const dismissSpy = vi.fn()

vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => notifySpy(...args),
  dismissNotification: (...args: unknown[]) => dismissSpy(...args)
}))

const checkHermesUpdateSpy = vi.fn()
const updateHermesSpy = vi.fn()
const getActionStatusSpy = vi.fn()

vi.mock('@/hermes', () => ({
  checkHermesUpdate: (...args: unknown[]) => checkHermesUpdateSpy(...args),
  updateHermes: (...args: unknown[]) => updateHermesSpy(...args),
  getActionStatus: (...args: unknown[]) => getActionStatusSpy(...args)
}))

const { maybeNotifyUpdateAvailable, checkBackendUpdates, $backendUpdateStatus, applyBackendUpdate, $backendUpdateApply } = await import('./updates')
const { setConnection } = await import('./session')

const status = (over: Partial<DesktopUpdateStatus> = {}): DesktopUpdateStatus => ({
  supported: true,
  behind: 3,
  targetSha: 'sha-a',
  fetchedAt: 0,
  ...over
})

const lastToast = () => notifySpy.mock.calls.at(-1)?.[0] as { onDismiss: () => void }

describe('maybeNotifyUpdateAvailable', () => {
  beforeEach(() => {
    storage.clear()
    notifySpy.mockClear()
    vi.useRealTimers()
  })

  it('shows when an update is available and not snoozed', () => {
    maybeNotifyUpdateAvailable(status())
    expect(notifySpy).toHaveBeenCalledTimes(1)
  })

  it('stays quiet for new commits once the toast was closed', () => {
    maybeNotifyUpdateAvailable(status())
    lastToast().onDismiss() // user closes it → cooldown starts
    notifySpy.mockClear()

    // A different commit lands while still within the cooldown window.
    maybeNotifyUpdateAvailable(status({ targetSha: 'sha-b', behind: 9 }))
    expect(notifySpy).not.toHaveBeenCalled()
  })

  it('re-shows once the cooldown elapses', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)

    maybeNotifyUpdateAvailable(status())
    lastToast().onDismiss()
    notifySpy.mockClear()

    vi.setSystemTime(25 * 60 * 60 * 1000) // > 24h cooldown
    maybeNotifyUpdateAvailable(status({ targetSha: 'sha-b' }))
    expect(notifySpy).toHaveBeenCalledTimes(1)
  })

  it('does nothing when already up to date', () => {
    maybeNotifyUpdateAvailable(status({ behind: 0 }))
    expect(notifySpy).not.toHaveBeenCalled()
  })
})

describe('checkBackendUpdates', () => {
  beforeEach(() => {
    storage.clear()
    notifySpy.mockClear()
    checkHermesUpdateSpy.mockReset()
    $backendUpdateStatus.set(null)
    vi.useRealTimers()
  })

  const setRemote = (on: boolean) =>
    setConnection({
      baseUrl: 'http://box:9119',
      isFullscreen: false,
      mode: on ? 'remote' : 'local',
      nativeOverlayWidth: 0,
      token: 't',
      wsUrl: 'ws://box:9119',
      logs: [],
      windowButtonPosition: null
    })

  it('maps the backend /update/check onto the backend status, including commits', async () => {
    setRemote(true)
    checkHermesUpdateSpy.mockResolvedValue({
      install_method: 'git',
      current_version: '0.16.0',
      behind: 2,
      update_available: true,
      can_apply: true,
      update_command: 'hermes update',
      message: null,
      commits: [{ sha: 'abc1234', summary: 'feat: x', author: 'a', at: 1 }]
    })

    const result = await checkBackendUpdates()

    expect(checkHermesUpdateSpy).toHaveBeenCalled()
    expect(result?.behind).toBe(2)
    expect(result?.commits?.[0]?.sha).toBe('abc1234')
    expect(result?.supported).toBe(true)
    expect($backendUpdateStatus.get()?.commits?.[0]?.summary).toBe('feat: x')
  })

  it('honours can_apply=false (docker/nix): not supported, carries message', async () => {
    setRemote(true)
    checkHermesUpdateSpy.mockResolvedValue({
      install_method: 'docker',
      current_version: '0.16.0',
      behind: null,
      update_available: false,
      can_apply: false,
      update_command: 'docker pull ...',
      message: 'Docker images are immutable.'
    })

    const result = await checkBackendUpdates()

    expect(result?.supported).toBe(false)
    expect(result?.message).toBe('Docker images are immutable.')
  })

  it('is a no-op in local mode (backend check only runs when remote)', async () => {
    setRemote(false)
    await checkBackendUpdates()
    expect(checkHermesUpdateSpy).not.toHaveBeenCalled()
  })
})

describe('applyBackendUpdate recovery', () => {
  beforeEach(() => {
    storage.clear()
    checkHermesUpdateSpy.mockReset()
    updateHermesSpy.mockReset()
    getActionStatusSpy.mockReset()
    $backendUpdateApply.set({ applying: false, stage: 'idle', message: '', percent: null, error: null, command: null, log: [] })
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('waits for the backend to return after the restart drops the connection, then clears the overlay', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'update', pid: 1 })
    getActionStatusSpy.mockRejectedValue(new Error('ECONNREFUSED'))
    checkHermesUpdateSpy.mockResolvedValue({ install_method: 'git', current_version: '0.16.0', behind: 0, update_available: false, can_apply: true, update_command: 'hermes update', message: null })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(5000)
    const result = await promise

    expect(result.ok).toBe(true)
    expect($backendUpdateApply.get().stage).toBe('idle')
    expect($backendUpdateApply.get().applying).toBe(false)
  })

  it('surfaces an error when the backend never comes back after the restart', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'update', pid: 1 })
    getActionStatusSpy.mockRejectedValue(new Error('ECONNREFUSED'))
    checkHermesUpdateSpy.mockRejectedValue(new Error('ECONNREFUSED'))

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(70000)
    const result = await promise

    expect(result.ok).toBe(false)
    expect($backendUpdateApply.get().stage).toBe('error')
  })
})

