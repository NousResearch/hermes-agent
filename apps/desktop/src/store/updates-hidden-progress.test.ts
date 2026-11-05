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

// Stub: the real reconnect module's graph is not needed here.
const reconnectGatewaySpy = vi.fn().mockResolvedValue(undefined)

vi.mock('@/store/gateway-reconnect', () => ({
  reconnectGateway: (...args: unknown[]) => reconnectGatewaySpy(...args)
}))

const {
  $backendUpdateApply,
  $updateApply,
  $updateOverlayOpen,
  $updateOverlayTarget,
  applyBackendUpdate,
  applyUpdates
} = await import('./updates')

// A backend update runs while this window stays: the person may hide its progress to keep using
// their chats. What needs them brings it back; what does not leaves it hidden. Split out of
// updates.test.ts, which sits at its file-size cap.
describe('hidden backend update progress', () => {
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
    $updateApply.set(idle)
    $updateOverlayOpen.set(false)
    $updateOverlayTarget.set('client')
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  async function pressAndHide(): Promise<{ pressed: Promise<unknown> }> {
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 4, action_id: 'a'.repeat(32) })
    getActionStatusSpy.mockResolvedValue({ exit_code: null, lines: [], name: 'hermes-update', pid: 4, running: true })
    const pressed = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(1500)
    $updateOverlayOpen.set(false) // hidden to answer a chat's approval while the update drains

    return { pressed }
  }

  it('shows the progress again when the update the person hid fails', async () => {
    const { pressed } = await pressAndHide()
    getActionStatusSpy.mockResolvedValue({ exit_code: 1, lines: [], name: 'hermes-update', pid: 4, running: false })
    await vi.advanceTimersByTimeAsync(1500)
    await pressed

    expect($backendUpdateApply.get().stage).toBe('error')
    expect($updateOverlayTarget.get()).toBe('backend')
    expect($updateOverlayOpen.get()).toBe(true)
  })

  it('leaves the progress hidden when the update the person hid succeeds', async () => {
    const { pressed } = await pressAndHide()
    getActionStatusSpy.mockResolvedValue({ exit_code: 0, lines: [], name: 'hermes-update', pid: 4, running: false })
    await vi.advanceTimersByTimeAsync(1500)
    await pressed

    expect($backendUpdateApply.get().applying).toBe(false)
    expect($updateOverlayOpen.get()).toBe(false)
  })

  it.each([
    [
      'refused',
      () => updateHermesSpy.mockResolvedValue({ ok: false, name: 'hermes-update', message: 'managed' }),
      'manual'
    ],
    ['rejected', () => updateHermesSpy.mockRejectedValue(new Error('offline')), 'error']
  ])('shows a hidden Update press again when its start is %s', async (_label, arrange, stage) => {
    let release: () => void = () => undefined
    const gate = new Promise<void>(resolve => (release = resolve))
    arrange()
    const outcome = updateHermesSpy.getMockImplementation()
    updateHermesSpy.mockImplementation(async () => {
      await gate

      return outcome?.()
    })

    const pressed = applyBackendUpdate()
    $updateOverlayOpen.set(false) // hidden while the start request is in flight
    release()
    await pressed

    expect($backendUpdateApply.get().stage).toBe(stage)
    expect($updateOverlayOpen.get()).toBe(true)
  })

  it('a client update that finishes leaves a hidden backend update running', async () => {
    const live = { ...idle, applying: true, stage: 'restart' as const }
    $backendUpdateApply.set(live)
    ;(globalThis as unknown as { window: unknown }).window = {
      hermesDesktop: { updates: { apply: vi.fn().mockResolvedValue({ ok: true, backendUpdated: true }) } }
    }

    try {
      await applyUpdates()
    } finally {
      delete (globalThis as unknown as { window?: unknown }).window
    }

    expect($updateApply.get().stage).toBe('idle')
    expect($backendUpdateApply.get()).toEqual(live)
  })
})
