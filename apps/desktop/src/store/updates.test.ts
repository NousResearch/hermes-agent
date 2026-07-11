import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'

const storage = new Map<string, string>()

vi.mock('@/lib/storage', () => ({
  persistBoolean: (key: string, value: boolean) => {
    storage.set(key, String(value))
  },
  persistString: (key: string, value: null | string) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, value)
    }
  },
  storedBoolean: (key: string, fallback: boolean) => {
    const value = storage.get(key)

    return value === undefined ? fallback : value === 'true'
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

const {
  maybeNotifyUpdateAvailable,
  checkBackendUpdates,
  $backendUpdateStatus,
  applyBackendUpdate,
  $backendUpdateApply,
  reportBackendContract,
  applyUpdates,
  $updateApply,
  $updateOverlayOpen,
  resetUpdateApplyState,
  startUpdatePoller,
  stopUpdatePoller,
  $updateStatus
} = await import('./updates')

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
    expect(notifySpy.mock.calls[0]?.[0]).toMatchObject({ icon: 'gift' })
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

describe('reportBackendContract', () => {
  beforeEach(() => {
    storage.clear()
    notifySpy.mockClear()
    dismissSpy.mockClear()
    vi.useRealTimers()
  })

  it('dismisses the toast when the backend meets the contract', () => {
    reportBackendContract(5)
    expect(dismissSpy).toHaveBeenCalledWith('backend-contract-skew')
    expect(notifySpy).not.toHaveBeenCalled()
  })

  it('warns when the backend is behind (or reports no contract)', () => {
    reportBackendContract(undefined)
    expect(notifySpy).toHaveBeenCalledTimes(1)
    reportBackendContract(4)
    expect(notifySpy).toHaveBeenCalledTimes(2)
  })

  it('stays quiet on later session opens once the user closed it', () => {
    reportBackendContract(1)
    lastToast().onDismiss() // user closes it → cooldown starts
    notifySpy.mockClear()

    // Opening another pre-existing session re-runs the check within cooldown.
    reportBackendContract(1)
    expect(notifySpy).not.toHaveBeenCalled()
  })

  it('reminds again after the cooldown elapses', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)

    reportBackendContract(1)
    lastToast().onDismiss()
    notifySpy.mockClear()

    vi.setSystemTime(25 * 60 * 60 * 1000) // > 24h cooldown
    reportBackendContract(1)
    expect(notifySpy).toHaveBeenCalledTimes(1)
  })

  it('clears the snooze once the backend catches up, so a regression warns again', () => {
    reportBackendContract(1)
    lastToast().onDismiss()
    notifySpy.mockClear()

    reportBackendContract(5) // backend updated → satisfied, snooze cleared
    reportBackendContract(4) // a later regression must warn immediately
    expect(notifySpy).toHaveBeenCalledTimes(1)
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
    expect(result?.updateAvailable).toBe(true)
    expect(result?.commits?.[0]?.sha).toBe('abc1234')
    expect(result?.supported).toBe(true)
    expect($backendUpdateStatus.get()?.commits?.[0]?.summary).toBe('feat: x')
  })

  it('preserves backend update_available when the backend cannot count commits', async () => {
    setRemote(true)
    checkHermesUpdateSpy.mockResolvedValue({
      install_method: 'nixos',
      current_version: '0.16.0',
      behind: -1,
      update_available: true,
      can_apply: false,
      update_command: 'managed outside dashboard',
      message: 'Update available.'
    })

    const result = await checkBackendUpdates()

    expect(result?.behind).toBe(0)
    expect(result?.updateAvailable).toBe(true)
    expect(result?.targetSha).toBe('backend:0.16.0')
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

describe('applyUpdates terminal state', () => {
  const applyMock = vi.fn()

  beforeEach(() => {
    storage.clear()
    notifySpy.mockClear()
    dismissSpy.mockClear()
    applyMock.mockReset()
    resetUpdateApplyState()
    $updateOverlayOpen.set(true)
    ;(globalThis as unknown as { window: unknown }).window = {
      hermesDesktop: { updates: { apply: applyMock } }
    }
    vi.useRealTimers()
  })

  afterEach(() => {
    delete (globalThis as unknown as { window?: unknown }).window
  })

  it('holds the restart view when a relauncher hands off (no close, no toast)', async () => {
    applyMock.mockResolvedValue({ ok: true, handedOff: true })

    const result = await applyUpdates()

    expect(result.handedOff).toBe(true)
    // The detached relauncher will quit + reopen us; keep "applying" until then.
    expect($updateApply.get().applying).toBe(true)
    expect($updateOverlayOpen.get()).toBe(true)
    expect(notifySpy).not.toHaveBeenCalled()
  })

  it('closes the overlay + toasts when updated but not relaunched in place', async () => {
    // The Linux AppImage / dev-run path: backend + GUI updated, no in-place
    // relaunch. Must not strand the overlay on a closeless spinner.
    applyMock.mockResolvedValue({ ok: true, backendUpdated: true })

    await applyUpdates()

    expect($updateOverlayOpen.get()).toBe(false)
    expect($updateApply.get().applying).toBe(false)
    expect($updateApply.get().stage).toBe('idle')
    expect(notifySpy).toHaveBeenCalledTimes(1)
    expect(notifySpy.mock.calls[0]?.[0]).toMatchObject({ kind: 'success' })
  })

  it('lands on a closeable error state when the apply resolves not-ok', async () => {
    applyMock.mockResolvedValue({ ok: false, error: 'rebuild-failed', message: 'rebuild failed' })

    await applyUpdates()

    expect($updateApply.get().applying).toBe(false)
    expect($updateApply.get().stage).toBe('error')
    expect($updateApply.get().error).toBe('rebuild-failed')
  })

  it('keeps the manual command state for CLI installs with no staged updater', async () => {
    applyMock.mockResolvedValue({ ok: true, manual: true, command: 'hermes update' })

    await applyUpdates()

    expect($updateApply.get().stage).toBe('manual')
    expect($updateApply.get().command).toBe('hermes update')
    expect($updateOverlayOpen.get()).toBe(true)
    expect(notifySpy).not.toHaveBeenCalled()
  })

  it('lands on the guiSkew terminal state for a GUI/backend skew (AppImage/.deb/.rpm), without claiming a GUI update', async () => {
    // Linux: backend updated, but the running desktop package was NOT replaced.
    // Must NOT toast "loads next launch" — that's the dishonest message #45205
    // guards against. Lands on a closeable guiSkew view instead.
    applyMock.mockResolvedValue({
      ok: true,
      backendUpdated: true,
      guiUpdated: false,
      guiSkew: true,
      message: 'Backend updated, but the desktop app package was not changed.'
    })

    const result = await applyUpdates()

    expect(result.guiUpdated).toBe(false)
    expect($updateApply.get().stage).toBe('guiSkew')
    expect($updateApply.get().applying).toBe(false)
    expect($updateApply.get().message).toMatch(/desktop app package was not changed/)
    // Overlay stays open on a closeable terminal view; no "all set" toast.
    expect($updateOverlayOpen.get()).toBe(true)
    expect(notifySpy).not.toHaveBeenCalled()
  })

  it('lands on a closeable manual-restart state when the rebuilt sandbox blocks auto-relaunch', async () => {
    // Under release/*-unpacked but chrome-sandbox isn't launchable: don't quit
    // into a dead app — keep a working window on a closeable manual state.
    applyMock.mockResolvedValue({
      ok: true,
      backendUpdated: true,
      guiUpdated: false,
      manualRestart: true,
      sandboxBlocked: true,
      message: 'Backend updated. Quit and reopen Hermes to finish.'
    })

    const result = await applyUpdates()

    expect(result.manualRestart).toBe(true)
    expect($updateApply.get().stage).toBe('manual')
    expect($updateApply.get().command).toBeNull()
    expect($updateApply.get().message).toMatch(/Quit and reopen/)
    expect($updateOverlayOpen.get()).toBe(true)
    expect(notifySpy).not.toHaveBeenCalled()
  })
})

describe('applyBackendUpdate recovery', () => {
  beforeEach(() => {
    storage.clear()
    checkHermesUpdateSpy.mockReset()
    updateHermesSpy.mockReset()
    getActionStatusSpy.mockReset()
    $backendUpdateApply.set({
      applying: false,
      stage: 'idle',
      message: '',
      percent: null,
      error: null,
      command: null,
      log: []
    })
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('waits for the backend to return after the restart drops the connection, then clears the overlay', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'update', pid: 1 })
    getActionStatusSpy.mockRejectedValue(new Error('ECONNREFUSED'))
    checkHermesUpdateSpy.mockResolvedValue({
      install_method: 'git',
      current_version: '0.16.0',
      behind: 0,
      update_available: false,
      can_apply: true,
      update_command: 'hermes update',
      message: null
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(5000)
    const result = await promise

    expect(result.ok).toBe(true)
    expect($backendUpdateApply.get().stage).toBe('idle')
    expect($backendUpdateApply.get().applying).toBe(false)
  })

  it('surfaces backend update action log lines while the action is running', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'update', pid: 1 })
    getActionStatusSpy
      .mockResolvedValueOnce({
        exit_code: null,
        lines: ['Pulling updates...', 'Installing dependencies...'],
        name: 'update',
        pid: 1,
        running: true
      })
      .mockRejectedValueOnce(new Error('ECONNREFUSED'))
    checkHermesUpdateSpy.mockResolvedValue({
      install_method: 'git',
      current_version: '0.16.0',
      behind: 0,
      update_available: false,
      can_apply: true,
      update_command: 'hermes update',
      message: null
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(1500)

    expect($backendUpdateApply.get().message).toBe('Installing dependencies...')
    expect($backendUpdateApply.get().log.map(entry => entry.message)).toEqual([
      'Pulling updates...',
      'Installing dependencies...'
    ])

    await vi.advanceTimersByTimeAsync(5000)
    await promise
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

describe('startUpdatePoller', () => {
  const checkMock = vi.fn()
  const onProgressMock = vi.fn()
  const listeners: Record<string, Function> = {}

  beforeEach(() => {
    storage.clear()
    checkMock.mockReset()
    onProgressMock.mockReset()
    Object.keys(listeners).forEach(k => delete listeners[k])
    checkMock.mockResolvedValue({
      supported: true,
      behind: 5,
      targetSha: 'sha-abc',
      fetchedAt: 0
    })
    $updateStatus.set(null)
    ;(globalThis as unknown as { window: unknown }).window = {
      hermesDesktop: { updates: { check: checkMock, onProgress: onProgressMock } },
      addEventListener: vi.fn((event: string, handler: Function) => {
        listeners[event] = handler
      }),
      removeEventListener: vi.fn()
    }
    vi.useFakeTimers()
    stopUpdatePoller()
  })

  afterEach(() => {
    stopUpdatePoller()
    delete (globalThis as unknown as { window?: unknown }).window
    vi.useRealTimers()
  })

  it('calls checkUpdates() on startup so the version pill populates immediately', async () => {
    startUpdatePoller()

    // checkUpdates() is async — flush microtasks without advancing the 30-min interval.
    await vi.advanceTimersByTimeAsync(0)

    expect(checkMock).toHaveBeenCalled()
    expect($updateStatus.get()?.behind).toBe(5)
  })

  it('calls checkUpdates() on each interval tick', async () => {
    startUpdatePoller()
    await vi.advanceTimersByTimeAsync(0)
    checkMock.mockClear()

    await vi.advanceTimersByTimeAsync(30 * 60 * 1000)

    expect(checkMock).toHaveBeenCalled()
  })

  it('calls checkUpdates() when the window regains focus', async () => {
    startUpdatePoller()
    await vi.advanceTimersByTimeAsync(0)
    checkMock.mockClear()

    // Invoke the registered focus handler directly (the mock window doesn't
    // propagate DOM events, so call the stored listener).
    listeners['focus']?.()

    await vi.advanceTimersByTimeAsync(0)

    expect(checkMock).toHaveBeenCalled()
  })
})
