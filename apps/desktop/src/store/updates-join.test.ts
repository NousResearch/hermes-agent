import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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
  // store/session persists its exact owner hints through the JSON helpers.
  readJson: (key: string) => {
    const value = storage.get(key)

    try {
      return value === undefined ? null : JSON.parse(value)
    } catch {
      return null
    }
  },
  writeJson: (key: string, value: unknown) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, JSON.stringify(value))
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
  $backendUpdateApply,
  $updateOverlayOpen,
  $updateOverlayTarget,
  joinRunningBackendUpdate,
  startUpdatePoller,
  stopUpdatePoller
} = await import('./updates')

const { setConnection } = await import('./session')

// A window that connects (or reloads) while its backend is updating joins that update instead of
// showing a normal app. Split out of updates.test.ts, which sits at its file-size cap.
describe('startUpdatePoller rejoins a running backend update', () => {
  beforeEach(() => {
    storage.clear()
    ;(globalThis as unknown as { window: unknown }).window = {
      hermesDesktop: {
        updates: { check: vi.fn().mockResolvedValue({ supported: true, behind: 0, fetchedAt: 0 }), onProgress: vi.fn() }
      },
      addEventListener: vi.fn(),
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

  it('rejoins a backend update that is still running when a window connects to that backend', async () => {
    checkHermesUpdateSpy.mockReset().mockResolvedValue({ behind: 0, update_available: false })
    getActionStatusSpy
      .mockReset()
      .mockResolvedValue({ exit_code: null, lines: [], name: 'hermes-update', pid: 9, running: true })
    $updateOverlayOpen.set(false)

    setConnection({
      baseUrl: 'http://rejoin:9119',
      isFullscreen: false,
      mode: 'remote',
      nativeOverlayWidth: 0,
      token: 't',
      wsUrl: 'ws://rejoin:9119',
      logs: [],
      windowButtonPosition: null
    })
    startUpdatePoller()
    await vi.advanceTimersByTimeAsync(0)

    expect(getActionStatusSpy).toHaveBeenCalledWith('hermes-update', 1)
    expect($updateOverlayOpen.get()).toBe(true)
    expect($backendUpdateApply.get().applying).toBe(true)
    getActionStatusSpy
      .mockReset()
      .mockResolvedValue({ exit_code: 0, lines: [], name: 'hermes-update', pid: 9, running: false })
    await vi.advanceTimersByTimeAsync(1500)
  })
})

describe('joinRunningBackendUpdate', () => {
  const idle = {
    applying: false,
    stage: 'idle' as const,
    message: '',
    percent: null,
    error: null,
    command: null,
    log: []
  }

  beforeEach(() => {
    updateHermesSpy.mockReset()
    getActionStatusSpy.mockReset()
    checkHermesUpdateSpy.mockReset()
    $backendUpdateApply.set(idle)
    $updateOverlayOpen.set(false)
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('opens the progress overlay on an update a reloaded window finds running, and follows it to the end', async () => {
    const actionId = 'c'.repeat(32)
    const startedAt = new Date().toISOString()
    // After the update restarted the gateway, only the receipt still says the action runs.
    getActionStatusSpy
      .mockResolvedValueOnce({
        exit_code: null,
        lines: [],
        name: 'hermes-update',
        pid: null,
        running: false,
        receipt: { action_id: actionId, outcome: 'running', started_at: startedAt, finished_at: null }
      })
      .mockResolvedValueOnce({
        exit_code: null,
        lines: [`=== hermes-update completed ${actionId} ===`],
        name: 'hermes-update',
        pid: null,
        running: false
      })

    await joinRunningBackendUpdate()

    expect($updateOverlayOpen.get()).toBe(true)
    expect($updateOverlayTarget.get()).toBe('backend')
    expect($backendUpdateApply.get().applying).toBe(true)
    expect(updateHermesSpy).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(1500)

    expect($backendUpdateApply.get().stage).toBe('idle')
    expect($updateOverlayOpen.get()).toBe(false)
  })

  it('joins an update the spawning gateway still reports running', async () => {
    getActionStatusSpy.mockResolvedValue({ exit_code: null, lines: [], name: 'hermes-update', pid: 4, running: true })

    await joinRunningBackendUpdate()

    expect($updateOverlayOpen.get()).toBe(true)
    expect($backendUpdateApply.get().applying).toBe(true)
    // Let the joined update finish, so no follow carries into the next test.
    getActionStatusSpy.mockResolvedValue({ exit_code: 0, lines: [], name: 'hermes-update', pid: 4, running: false })
    await vi.advanceTimersByTimeAsync(1500)
    expect($backendUpdateApply.get().applying).toBe(false)
  })

  it.each([
    ['nothing running', { exit_code: 0, lines: [], name: 'hermes-update', pid: null, running: false }],
    [
      'a finished receipt',
      {
        exit_code: null,
        lines: [],
        name: 'hermes-update',
        pid: null,
        running: false,
        receipt: {
          action_id: 'd'.repeat(32),
          outcome: 'success',
          started_at: new Date().toISOString(),
          finished_at: new Date().toISOString()
        }
      }
    ],
    [
      'a running receipt from a killed update long ago',
      {
        exit_code: null,
        lines: [],
        name: 'hermes-update',
        pid: null,
        running: false,
        receipt: {
          action_id: 'e'.repeat(32),
          outcome: 'running',
          started_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          finished_at: null
        }
      }
    ],
    [
      'a running receipt with no action id (a CLI update)',
      {
        exit_code: null,
        lines: [],
        name: 'hermes-update',
        pid: null,
        running: false,
        receipt: { action_id: null, outcome: 'running', started_at: new Date().toISOString(), finished_at: null }
      }
    ]
  ])('stays idle for %s', async (_label, status) => {
    getActionStatusSpy.mockResolvedValue(status)

    await joinRunningBackendUpdate()

    // It asked, so the idle answer is a decision on this status, not a join already in flight.
    expect(getActionStatusSpy).toHaveBeenCalledWith('hermes-update', 1)

    expect($updateOverlayOpen.get()).toBe(false)
    expect($backendUpdateApply.get().applying).toBe(false)
  })

  it('stays idle when the status read fails', async () => {
    getActionStatusSpy.mockRejectedValue(new Error('503 gateway_not_ready'))

    await joinRunningBackendUpdate()

    expect($updateOverlayOpen.get()).toBe(false)
  })
})
