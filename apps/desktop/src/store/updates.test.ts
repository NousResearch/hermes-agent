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

// updates.ts reads the connections registry (multi-target gating + the
// everything-flow fan-out). Mock the store shallowly: the real module drags in
// the profile/gateway-switch graph, which is far heavier than these tests need.
const { atom: registryAtom } = await import('nanostores')
const $mockConnectionsRegistry = registryAtom<unknown>(null)

vi.mock('@/store/connections', () => ({
  $connectionsRegistry: $mockConnectionsRegistry,
  refreshConnectionsRegistry: () => Promise.resolve($mockConnectionsRegistry.get())
}))

const checkHermesUpdateSpy = vi.fn()
const updateHermesSpy = vi.fn()
const getActionStatusSpy = vi.fn()

vi.mock('@/hermes', () => ({
  checkHermesUpdate: (...args: unknown[]) => checkHermesUpdateSpy(...args),
  updateHermes: (...args: unknown[]) => updateHermesSpy(...args),
  getActionStatus: (...args: unknown[]) => getActionStatusSpy(...args)
}))

// A successful backend apply must nudge the gateway reconnect handler — the
// update restarted the gateway process, and over tunnels the old socket dies
// without a close event (users force-quit to recover). Mock the tiny registry
// module so the assertion is direct.
const reconnectGatewaySpy = vi.fn().mockResolvedValue(undefined)

vi.mock('@/store/gateway-reconnect', () => ({
  reconnectGateway: (...args: unknown[]) => reconnectGatewaySpy(...args)
}))

const {
  maybeNotifyUpdateAvailable,
  checkBackendUpdates,
  $backendUpdateStatus,
  applyBackendUpdate,
  $backendUpdateApply,
  reportBackendContract,
  applyUpdates,
  applyEverythingUpdate,
  hasMultipleUpdateTargets,
  $updateApply,
  $updateEverything,
  $updateOverlayOpen,
  $updateOverlayTarget,
  requestActiveUpdate,
  resetUpdateApplyState,
  startUpdatePoller,
  stopUpdatePoller,
  $updateStatus
} = await import('./updates')

const { setConnection } = await import('./session')

const registryOf = (ids: string[]) => ({
  version: 2,
  primary: ids[0] ?? 'local',
  secureTokenStorage: true,
  connections: ids.map(id => ({ id, kind: id === 'local' ? 'local' : 'remote', label: id }))
})

const status = (over: Partial<DesktopUpdateStatus> = {}): DesktopUpdateStatus => ({
  supported: true,
  behind: 3,
  targetSha: 'sha-a',
  fetchedAt: 0,
  ...over
})

const lastToast = () => notifySpy.mock.calls.at(-1)?.[0] as { onDismiss: () => void }

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

  // FAIL-BEFORE: a shallow installer clone reports behind:null + updateAvailable
  // (exact count unknowable without a merge-base). The guard treated null as 0
  // and silently swallowed the notification entirely.
  it('still notifies with generic copy when the exact behind count is unknown', () => {
    maybeNotifyUpdateAvailable(status({ behind: null, updateAvailable: true }))
    expect(notifySpy).toHaveBeenCalledTimes(1)
    expect(notifySpy.mock.calls[0]?.[0]).toMatchObject({ message: 'A new update is available.' })
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
    reportBackendContract(6)
    expect(dismissSpy).toHaveBeenCalledWith('backend-contract-skew')
    expect(notifySpy).not.toHaveBeenCalled()
  })

  it('warns when the backend is behind (or reports no contract)', () => {
    reportBackendContract(undefined)
    expect(notifySpy).toHaveBeenCalledTimes(1)
    reportBackendContract(1)
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

    reportBackendContract(6) // backend updated → satisfied, snooze cleared
    reportBackendContract(5) // a later regression must warn immediately
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

// The ⌘K "Update Hermes" row. It used to call applyBackendUpdate() flat, which
// in local mode aimed at the backend checkout instead of the client and, with
// no overlay open, showed nothing at all.
describe('requestActiveUpdate', () => {
  const applyClientMock = vi.fn()
  const checkClientMock = vi.fn()

  beforeEach(() => {
    storage.clear()
    notifySpy.mockClear()
    dismissSpy.mockClear()
    applyClientMock.mockReset().mockResolvedValue({ ok: true, handedOff: true })
    checkClientMock.mockReset().mockResolvedValue(status({ behind: 0 }))
    updateHermesSpy.mockReset().mockResolvedValue({ ok: true, name: 'update' })
    checkHermesUpdateSpy.mockReset().mockResolvedValue({
      install_method: 'git',
      current_version: '0.4.2',
      behind: 0,
      update_available: false,
      can_apply: true,
      update_command: null,
      message: null
    })
    getActionStatusSpy.mockReset().mockResolvedValue({ lines: [], running: false, exit_code: 0 })
    resetUpdateApplyState()
    $updateStatus.set(null)
    $backendUpdateStatus.set(null)
    $updateOverlayOpen.set(false)
    ;(globalThis as unknown as { window: unknown }).window = {
      hermesDesktop: { updates: { apply: applyClientMock, check: checkClientMock } }
    }
    vi.useRealTimers()
  })

  afterEach(async () => {
    // Drain any backend apply this suite kicked off: applyBackendUpdate() now
    // memoizes the in-flight run, so a dangling promise here would be handed
    // to the next suite's tests instead of a fresh run.
    await vi.waitFor(() => expect($backendUpdateApply.get().applying).toBe(false), { timeout: 5000 })
    // Remote-mode rows now route through the everything-flow; drain its tail
    // (fan-out + client check) so it can't bleed into the next test.
    await vi.waitFor(() => expect($updateEverything.get().running).toBe(false), { timeout: 5000 })
    setRemote(false)
    delete (globalThis as unknown as { window?: unknown }).window
  })

  it('applies the CLIENT update in local mode, never the backend', async () => {
    setRemote(false)
    $updateStatus.set(status({ behind: 3 }))

    requestActiveUpdate()
    await vi.waitFor(() => expect(applyClientMock).toHaveBeenCalled())

    expect(updateHermesSpy).not.toHaveBeenCalled()
    expect($updateOverlayTarget.get()).toBe('client')
  })

  it('applies the BACKEND update in remote mode', async () => {
    setRemote(true)
    $backendUpdateStatus.set(status({ behind: 3 }))

    requestActiveUpdate()
    await vi.waitFor(() => expect(updateHermesSpy).toHaveBeenCalled())

    expect(applyClientMock).not.toHaveBeenCalled()
    expect($updateOverlayTarget.get()).toBe('backend')
  })

  it('always opens the overlay, so selecting the row is never a silent no-op', () => {
    setRemote(false)
    $updateStatus.set(status({ behind: 3 }))

    requestActiveUpdate()

    expect($updateOverlayOpen.get()).toBe(true)
  })

  it('opens the overlay to re-check instead of applying when already current', () => {
    setRemote(false)
    $updateStatus.set(status({ behind: 0, updateAvailable: false }))

    requestActiveUpdate()

    expect($updateOverlayOpen.get()).toBe(true)
    expect(applyClientMock).not.toHaveBeenCalled()
    expect(updateHermesSpy).not.toHaveBeenCalled()
  })

  it('applies on a backend that reports an update it cannot count commits for', async () => {
    setRemote(true)
    $backendUpdateStatus.set(status({ behind: 0, updateAvailable: true }))

    requestActiveUpdate()
    await vi.waitFor(() => expect(updateHermesSpy).toHaveBeenCalled())
  })
})

// The everything-flow: on multi-target installs (remote mode / multi-connection
// registry) "update" must mean every machine — active backend, other registered
// sources via the Electron fan-out, and the client LAST. Before this flow,
// every remote-mode affordance updated only the backend, so users "updated"
// forever while the desktop app itself stayed weeks stale (the Aug 2026
// mac-app-on-v0.20.0 report).
describe('applyEverythingUpdate', () => {
  const applyClientMock = vi.fn()
  const checkClientMock = vi.fn()
  const updateAllMock = vi.fn()

  beforeEach(() => {
    storage.clear()
    notifySpy.mockClear()
    dismissSpy.mockClear()
    applyClientMock.mockReset().mockResolvedValue({ ok: true, handedOff: true })
    checkClientMock.mockReset().mockResolvedValue(status({ behind: 0, updateAvailable: false }))
    updateAllMock.mockReset().mockResolvedValue({ ok: true, results: [] })
    updateHermesSpy.mockReset().mockResolvedValue({ ok: true, name: 'update' })
    checkHermesUpdateSpy.mockReset().mockResolvedValue({
      install_method: 'git',
      current_version: '0.4.2',
      behind: 0,
      update_available: false,
      can_apply: true,
      update_command: null,
      message: null
    })
    getActionStatusSpy.mockReset().mockResolvedValue({ lines: [], running: false, exit_code: 0 })
    resetUpdateApplyState()
    $updateStatus.set(null)
    $backendUpdateStatus.set(null)
    $updateOverlayOpen.set(false)
    $mockConnectionsRegistry.set(null)
    ;(globalThis as unknown as { window: unknown }).window = {
      hermesDesktop: {
        updates: { apply: applyClientMock, check: checkClientMock },
        connections: { updateAll: updateAllMock }
      }
    }
    vi.useRealTimers()
  })

  afterEach(async () => {
    await vi.waitFor(() => expect($updateEverything.get().running).toBe(false), { timeout: 5000 })
    await vi.waitFor(() => expect($backendUpdateApply.get().applying).toBe(false), { timeout: 5000 })
    setRemote(false)
    $mockConnectionsRegistry.set(null)
    delete (globalThis as unknown as { window?: unknown }).window
  })

  it('gates on multiple targets: remote mode OR a multi-connection registry', () => {
    setRemote(false)
    $mockConnectionsRegistry.set(null)
    expect(hasMultipleUpdateTargets()).toBe(false)

    setRemote(true)
    expect(hasMultipleUpdateTargets()).toBe(true)

    setRemote(false)
    $mockConnectionsRegistry.set(registryOf(['local', 'vps']))
    expect(hasMultipleUpdateTargets()).toBe(true)
  })

  it('remote mode: updates the backend, then the still-behind client — the stale-GUI gap', async () => {
    setRemote(true)
    $backendUpdateStatus.set(status({ behind: 3 }))
    // The client is ALSO behind; the old flow never touched it.
    checkClientMock.mockResolvedValue(status({ behind: 7, updateAvailable: true }))

    await applyEverythingUpdate()

    expect(updateHermesSpy).toHaveBeenCalledTimes(1)
    expect(applyClientMock).toHaveBeenCalledTimes(1)
  })

  it('remote mode: skips the client apply when the client is already current', async () => {
    setRemote(true)
    $backendUpdateStatus.set(status({ behind: 3 }))
    checkClientMock.mockResolvedValue(status({ behind: 0, updateAvailable: false }))

    await applyEverythingUpdate()

    expect(updateHermesSpy).toHaveBeenCalledTimes(1)
    expect(applyClientMock).not.toHaveBeenCalled()
  })

  it('fans out to other registered connections, excluding local and the active backend', async () => {
    setRemote(true)
    $mockConnectionsRegistry.set(registryOf(['local', 'vps', 'homelab']))
    $backendUpdateStatus.set(status({ behind: 1 }))

    await applyEverythingUpdate()

    expect(updateAllMock).toHaveBeenCalledTimes(1)
    const options = updateAllMock.mock.calls[0]?.[0] as { excludeIds: string[] }
    expect(options.excludeIds).toContain('local')
  })

  it('local mode with a multi-connection registry: fans out and updates the client, no active-backend leg', async () => {
    setRemote(false)
    $mockConnectionsRegistry.set(registryOf(['local', 'vps']))
    checkClientMock.mockResolvedValue(status({ behind: 2, updateAvailable: true }))
    $updateStatus.set(status({ behind: 2, updateAvailable: true }))

    await applyEverythingUpdate()

    expect(updateHermesSpy).not.toHaveBeenCalled()
    expect(updateAllMock).toHaveBeenCalledTimes(1)
    expect(applyClientMock).toHaveBeenCalledTimes(1)
  })

  it('a failed backend leg does not strand the fan-out or the client', async () => {
    setRemote(true)
    $mockConnectionsRegistry.set(registryOf(['local', 'vps']))
    updateHermesSpy.mockRejectedValue(new Error('backend gone'))
    checkClientMock.mockResolvedValue(status({ behind: 4, updateAvailable: true }))

    await applyEverythingUpdate()

    expect(updateAllMock).toHaveBeenCalledTimes(1)
    expect(applyClientMock).toHaveBeenCalledTimes(1)
  })

  it('surfaces per-row fan-out outcomes as notifications', async () => {
    setRemote(false)
    $mockConnectionsRegistry.set(registryOf(['local', 'vps', 'dead-box']))
    updateAllMock.mockResolvedValue({
      ok: true,
      results: [
        { connectionId: 'vps', label: 'vps', kind: 'remote', ok: true, detail: 'update started' },
        { connectionId: 'dead-box', label: 'dead-box', kind: 'remote', ok: false, error: 'ECONNREFUSED' }
      ]
    })

    await applyEverythingUpdate()

    const titles = notifySpy.mock.calls.map(call => (call[0] as { title?: string }).title)
    expect(titles).toContain('vps')
    expect(titles).toContain('dead-box')
  })

  it('memoizes the in-flight run so a double click cannot double-dispatch', async () => {
    setRemote(false)
    $mockConnectionsRegistry.set(registryOf(['local', 'vps']))

    const first = applyEverythingUpdate()
    const second = applyEverythingUpdate()

    expect(second).toBe(first)
    await Promise.all([first, second])
    expect(updateAllMock).toHaveBeenCalledTimes(1)
  })

  it('requestActiveUpdate routes through the everything-flow when EITHER target is behind', async () => {
    setRemote(true)
    // Backend current, client behind — the exact case the old remote-only
    // ternary missed entirely.
    $backendUpdateStatus.set(status({ behind: 0, updateAvailable: false }))
    $updateStatus.set(status({ behind: 5, updateAvailable: true }))
    checkClientMock.mockResolvedValue(status({ behind: 5, updateAvailable: true }))

    requestActiveUpdate()

    // The everything-flow runs the backend leg first (a 1.5s status poll on
    // real timers) before reaching the client apply — give it room.
    await vi.waitFor(() => expect(applyClientMock).toHaveBeenCalled(), { timeout: 5000 })
  })
})

// Post-backend-update client nudge: a successful backend apply re-checks the
// CLIENT and warns when it is still behind, with a one-click client update.
describe('client nudge after a backend update', () => {
  const applyClientMock = vi.fn()
  const checkClientMock = vi.fn()

  beforeEach(() => {
    storage.clear()
    notifySpy.mockClear()
    dismissSpy.mockClear()
    applyClientMock.mockReset().mockResolvedValue({ ok: true, handedOff: true })
    checkClientMock.mockReset()
    updateHermesSpy.mockReset().mockResolvedValue({ ok: true, name: 'update' })
    checkHermesUpdateSpy.mockReset().mockResolvedValue({
      install_method: 'git',
      current_version: '0.4.2',
      behind: 0,
      update_available: false,
      can_apply: true,
      update_command: null,
      message: null
    })
    getActionStatusSpy.mockReset().mockResolvedValue({ lines: [], running: false, exit_code: 0 })
    resetUpdateApplyState()
    $updateStatus.set(null)
    $backendUpdateStatus.set(null)
    $mockConnectionsRegistry.set(null)
    setRemote(true)
    ;(globalThis as unknown as { window: unknown }).window = {
      hermesDesktop: { updates: { apply: applyClientMock, check: checkClientMock } }
    }
    vi.useRealTimers()
  })

  afterEach(async () => {
    await vi.waitFor(() => expect($backendUpdateApply.get().applying).toBe(false), { timeout: 5000 })
    setRemote(false)
    delete (globalThis as unknown as { window?: unknown }).window
  })

  it('warns when the client is still behind after the backend caught up', async () => {
    checkClientMock.mockResolvedValue(status({ behind: 6, updateAvailable: true }))

    await applyBackendUpdate()

    await vi.waitFor(() => {
      const ids = notifySpy.mock.calls.map(call => (call[0] as { id?: string }).id)
      expect(ids).toContain('client-update-after-backend')
    })
  })

  it('stays silent when the client is already current', async () => {
    checkClientMock.mockResolvedValue(status({ behind: 0, updateAvailable: false }))

    await applyBackendUpdate()

    // Give the fire-and-forget nudge a beat to (not) fire.
    await new Promise(resolve => setTimeout(resolve, 50))
    const ids = notifySpy.mock.calls.map(call => (call[0] as { id?: string }).id)
    expect(ids).not.toContain('client-update-after-backend')
  })

  it('nudges a gateway reconnect after the backend caught up (tunnel socket may be dead)', async () => {
    checkClientMock.mockResolvedValue(status({ behind: 0, updateAvailable: false }))
    reconnectGatewaySpy.mockClear()

    await applyBackendUpdate()

    expect(reconnectGatewaySpy).toHaveBeenCalledTimes(1)
  })

  it('does not nudge a reconnect when the backend apply failed', async () => {
    reconnectGatewaySpy.mockClear()
    getActionStatusSpy.mockReset().mockResolvedValue({ lines: [], running: false, exit_code: 1 })

    await applyBackendUpdate()

    expect(reconnectGatewaySpy).not.toHaveBeenCalled()
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

  it('preserves structured safe blockers for the close-and-update prompt', async () => {
    const blockers = [
      {
        pid: 47484,
        name: 'python.exe',
        cmdline: 'python.exe -m http.server 8766',
        kind: 'local-preview' as const,
        safeToStop: true,
        label: 'Example Preview',
        port: 8766
      }
    ]

    applyMock.mockResolvedValue({ ok: false, error: 'venv-blocked', message: 'blocked', blockers })

    await applyUpdates()

    expect($updateApply.get().error).toBe('venv-blocked')
    expect($updateApply.get().blockers).toEqual(blockers)
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
    $backendUpdateStatus.set(null)
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
    const actionId = 'd'.repeat(32)
    updateHermesSpy.mockResolvedValue({ action_id: actionId, ok: true, name: 'update', pid: 1 })
    getActionStatusSpy.mockRejectedValueOnce(new Error('ECONNREFUSED')).mockResolvedValueOnce({
      exit_code: null,
      lines: [`=== hermes-update completed ${actionId} ===`],
      name: 'update',
      pid: null,
      running: false
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(5000)
    const result = await promise

    expect(result.ok).toBe(true)
    expect($backendUpdateApply.get().stage).toBe('idle')
    expect($backendUpdateApply.get().applying).toBe(false)
  })

  it('surfaces backend update action log lines while the action is running', async () => {
    const actionId = 'e'.repeat(32)
    updateHermesSpy.mockResolvedValue({ action_id: actionId, ok: true, name: 'update', pid: 1 })
    getActionStatusSpy
      .mockResolvedValueOnce({
        exit_code: null,
        lines: ['Pulling updates...', 'Installing dependencies...'],
        name: 'update',
        pid: 1,
        running: true
      })
      .mockRejectedValueOnce(new Error('ECONNREFUSED'))
      .mockResolvedValueOnce({
        exit_code: null,
        lines: [`=== hermes-update completed ${actionId} ===`],
        name: 'update',
        pid: null,
        running: false
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

  it('keeps waiting past the old 45-second cutoff while the update action is running', async () => {
    const actionId = 'f'.repeat(32)
    updateHermesSpy.mockResolvedValue({ action_id: actionId, ok: true, name: 'hermes-update', pid: 1 })

    for (let attempt = 0; attempt < 31; attempt += 1) {
      getActionStatusSpy.mockResolvedValueOnce({
        exit_code: null,
        lines: ['=== hermes-update started now ===', `step ${attempt}`],
        name: 'hermes-update',
        pid: 1,
        running: true
      })
    }

    getActionStatusSpy.mockRejectedValueOnce(new Error('ECONNREFUSED')).mockResolvedValueOnce({
      exit_code: null,
      lines: [`=== hermes-update completed ${actionId} ===`],
      name: 'hermes-update',
      pid: null,
      running: false
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(46500)

    expect($backendUpdateApply.get().applying).toBe(true)
    expect($backendUpdateApply.get().stage).toBe('pull')

    await vi.advanceTimersByTimeAsync(5000)
    await expect(promise).resolves.toMatchObject({ ok: true })
  })

  it('treats a successful no-op as complete without waiting for a restart', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockResolvedValue({
      exit_code: 0,
      lines: ['stale output from another run', '=== hermes-update started now ===', '✓ Already up to date!'],
      name: 'hermes-update',
      pid: 1,
      running: false
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(1500)
    const result = await promise

    expect(result.ok).toBe(true)
    expect($backendUpdateApply.get().stage).toBe('idle')
  })

  it('treats a successful dependency repair as complete without waiting for a restart', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockResolvedValue({
      exit_code: 0,
      lines: ['=== hermes-update started now ===', '✓ Dependencies repaired!', '✓ Update complete!'],
      name: 'hermes-update',
      pid: 1,
      running: false
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(1500)
    await expect(promise).resolves.toMatchObject({ ok: true })
    expect($backendUpdateApply.get().stage).toBe('idle')
  })

  it('trusts the current action exit code without parsing its output', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockResolvedValue({
      exit_code: 0,
      lines: ['✓ Already up to date!'],
      name: 'hermes-update',
      pid: 1,
      running: false
    })
    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(1500)
    await expect(promise).resolves.toMatchObject({ ok: true })
    expect(checkHermesUpdateSpy).not.toHaveBeenCalled()
  })

  it('waits for current-action completion proof after the backend restarts', async () => {
    const actionId = 'a'.repeat(32)
    updateHermesSpy.mockResolvedValue({ action_id: actionId, ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy
      .mockRejectedValueOnce(new Error('ECONNREFUSED'))
      .mockResolvedValueOnce({
        exit_code: null,
        lines: ['Update complete!', `=== hermes-update completed ${'c'.repeat(32)} ===`],
        name: 'hermes-update',
        pid: null,
        running: false
      })
      .mockResolvedValueOnce({
        exit_code: null,
        lines: ['Update complete!', `=== hermes-update completed ${actionId} ===`],
        name: 'hermes-update',
        pid: null,
        running: false
      })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(5000)
    await expect(promise).resolves.toMatchObject({ ok: true })
    expect(checkHermesUpdateSpy).not.toHaveBeenCalled()
  })

  it('accepts its terminal receipt when a verbose update pushes the start marker out of the log tail', async () => {
    const actionId = 'b'.repeat(32)
    updateHermesSpy.mockResolvedValue({ action_id: actionId, ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockRejectedValueOnce(new Error('ECONNREFUSED')).mockResolvedValueOnce({
      exit_code: null,
      lines: ['final build output', 'Update complete!', `=== hermes-update completed ${actionId} ===`],
      name: 'hermes-update',
      pid: null,
      running: false
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(5000)

    await expect(promise).resolves.toMatchObject({ ok: true })
    expect(getActionStatusSpy).toHaveBeenCalledWith('hermes-update', 2000)
  })

  it('proves a pre-action-ID backend reached its requested commit after restart', async () => {
    $backendUpdateStatus.set({
      behind: 2,
      commits: [{ at: 1, author: 'Nous', sha: 'requested-target', summary: 'target' }],
      fetchedAt: 1,
      supported: true,
      targetSha: 'backend:0.18.2',
      updateAvailable: true
    })
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockRejectedValueOnce(new Error('ECONNREFUSED')).mockResolvedValue({
      exit_code: null,
      lines: ['verbose output', 'Update complete!'],
      name: 'hermes-update',
      pid: null,
      running: false
    })
    checkHermesUpdateSpy
      .mockResolvedValueOnce({
        behind: null,
        can_apply: true,
        commits: [],
        current_version: '0.18.2',
        install_method: 'git',
        message: 'offline',
        update_available: false,
        update_command: 'hermes update'
      })
      .mockResolvedValueOnce({
        behind: 1,
        can_apply: true,
        commits: [{ at: 2, author: 'Nous', sha: 'newer-commit', summary: 'newer' }],
        current_version: '0.18.2',
        install_method: 'git',
        message: null,
        update_available: true,
        update_command: 'hermes update'
      })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(5000)

    await expect(promise).resolves.toMatchObject({ ok: true })
    expect(checkHermesUpdateSpy).toHaveBeenCalledTimes(2)
  })

  it('proves a fast pre-action-ID packaged update by its changed version', async () => {
    $backendUpdateStatus.set({
      behind: 1,
      commits: [],
      fetchedAt: 1,
      supported: true,
      targetSha: 'backend:0.18.2',
      updateAvailable: true
    })
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockResolvedValue({
      exit_code: null,
      lines: ['verbose output without a retained start marker'],
      name: 'hermes-update',
      pid: null,
      running: false
    })
    checkHermesUpdateSpy.mockResolvedValue({
      behind: -1,
      can_apply: true,
      commits: [],
      current_version: '0.18.3',
      install_method: 'pip',
      message: null,
      update_available: true,
      update_command: 'hermes update'
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(1500)

    await expect(promise).resolves.toMatchObject({ ok: true })
    expect(checkHermesUpdateSpy).toHaveBeenCalledWith(true)
  })

  it('resumes action polling after a transient status failure', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy
      .mockRejectedValueOnce(new Error('ECONNRESET'))
      .mockResolvedValueOnce({
        exit_code: null,
        lines: ['=== hermes-update started now ===', 'still running'],
        name: 'hermes-update',
        pid: 1,
        running: true
      })
      .mockResolvedValueOnce({
        exit_code: 0,
        lines: ['=== hermes-update started now ===', 'Update complete!'],
        name: 'hermes-update',
        pid: 1,
        running: false
      })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(5000)
    await expect(promise).resolves.toMatchObject({ ok: true })
    expect(getActionStatusSpy).toHaveBeenCalledTimes(3)
  })

  it('restores the fixed action deadline after reconnecting', async () => {
    updateHermesSpy.mockResolvedValue({ action_id: 'a'.repeat(32), ok: true, name: 'hermes-update', pid: 1 })

    const running = {
      exit_code: null,
      lines: ['still running'],
      name: 'hermes-update',
      pid: 1,
      running: true
    }

    for (let attempt = 0; attempt < 119; attempt += 1) {
      getActionStatusSpy.mockResolvedValueOnce(running)
    }

    getActionStatusSpy.mockRejectedValueOnce(new Error('ECONNRESET')).mockResolvedValue(running)

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(6 * 60 * 1000 + 1500)

    await expect(promise).resolves.toMatchObject({ error: 'apply-failed', ok: false })
    expect($backendUpdateApply.get().stage).toBe('error')
  })

  it('shares one in-flight update between concurrent apply requests', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockResolvedValue({
      exit_code: 0,
      lines: ['=== hermes-update started now ===', '✓ Already up to date!'],
      name: 'hermes-update',
      pid: 1,
      running: false
    })

    const first = applyBackendUpdate()
    const second = applyBackendUpdate()

    expect(second).toBe(first)
    await vi.advanceTimersByTimeAsync(1500)
    await Promise.all([first, second])
    expect(updateHermesSpy).toHaveBeenCalledTimes(1)
  })

  it('fails closed when the update action never reaches a terminal state', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockResolvedValue({
      exit_code: null,
      lines: ['=== hermes-update started now ===', 'still running'],
      name: 'hermes-update',
      pid: 1,
      running: true
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(6 * 60 * 1000 + 1500)
    await expect(promise).resolves.toMatchObject({ ok: false, error: 'apply-failed' })
    expect($backendUpdateApply.get().stage).toBe('error')
  })

  it('fails immediately when the update action exits nonzero', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockResolvedValue({
      exit_code: 1,
      lines: ['=== hermes-update started now ===', 'update failed'],
      name: 'hermes-update',
      pid: 1,
      running: false
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(1500)
    await expect(promise).resolves.toMatchObject({ ok: false, error: 'apply-failed' })
    expect(checkHermesUpdateSpy).not.toHaveBeenCalled()
    expect($backendUpdateApply.get().stage).toBe('error')
  })

  it('surfaces an error when the backend never comes back after the restart', async () => {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'update', pid: 1 })
    getActionStatusSpy.mockRejectedValue(new Error('ECONNREFUSED'))
    checkHermesUpdateSpy.mockRejectedValue(new Error('ECONNREFUSED'))

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(250000)
    const result = await promise

    expect(result.ok).toBe(false)
    expect($backendUpdateApply.get().stage).toBe('error')
  }, 10000)
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
