import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { deferred } from '@/test/deferred'

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

// Import-safety stubs: updates.ts pulls these modules in at load time.
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), dismissNotification: vi.fn() }))
vi.mock('@/store/gateway-reconnect', () => ({ reconnectGateway: vi.fn().mockResolvedValue(undefined) }))

// The real connections store drags in the profile/gateway-switch graph.
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

const {
  $backendUpdateApply,
  $updateApply,
  $updateOverlayOpen,
  $updateOverlayTarget,
  applyBackendUpdate,
  applyUpdates,
  resetUpdateApplyState
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
    const gate = deferred()
    arrange()
    const outcome = updateHermesSpy.getMockImplementation()
    updateHermesSpy.mockImplementation(async () => {
      await gate.promise

      return outcome?.()
    })

    const pressed = applyBackendUpdate()
    $updateOverlayOpen.set(false) // hidden while the start request is in flight
    gate.resolve()
    await pressed

    expect($backendUpdateApply.get().stage).toBe(stage)
    expect($updateOverlayOpen.get()).toBe(true)
  })

  it('a pressed update whose own run the backend reports interrupted fails at once', async () => {
    const own = 'b'.repeat(32)
    updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update', pid: 4, action_id: own })
    // The dashboard restarted under its run, so it no longer has the process; the receipt says it died.
    getActionStatusSpy.mockResolvedValue({
      exit_code: null,
      lines: [],
      name: 'hermes-update',
      pid: null,
      running: false,
      receipt: { action_id: own, outcome: 'interrupted', started_at: new Date().toISOString(), finished_at: null }
    })

    const pressed = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(1500)
    await pressed

    expect($backendUpdateApply.get().stage).toBe('error')
    expect($updateOverlayOpen.get()).toBe(true)
  })

  it("does not replace a running client update's progress with a backend failure, and shows it after", async () => {
    const { pressed } = await pressAndHide()

    const clientApply = deferred<unknown>()

    ;(globalThis as unknown as { window: unknown }).window = {
      hermesDesktop: { updates: { apply: vi.fn(() => clientApply.promise) } }
    }

    try {
      $updateOverlayTarget.set('client')
      $updateOverlayOpen.set(true)
      const client = applyUpdates()

      getActionStatusSpy.mockResolvedValue({ exit_code: 1, lines: [], name: 'hermes-update', pid: 4, running: false })
      await vi.advanceTimersByTimeAsync(1500)
      await pressed

      expect($backendUpdateApply.get().stage).toBe('error')
      expect($updateOverlayTarget.get()).toBe('client')
      expect($updateApply.get().applying).toBe(true)

      clientApply.resolve({ ok: true, backendUpdated: true })
      await client
    } finally {
      delete (globalThis as unknown as { window?: unknown }).window
    }

    expect($updateOverlayTarget.get()).toBe('backend')
    expect($updateOverlayOpen.get()).toBe(true)
  })

  it.each([
    ['restarting into the new version', { applying: false, stage: 'restart' as const }],
    ['showing its own failure', { applying: false, stage: 'error' as const }]
  ])('a backend failure waits while the client update is %s, and shows when that view closes', async (_l, client) => {
    const { pressed } = await pressAndHide()
    $updateApply.set({ ...idle, ...client })
    $updateOverlayTarget.set('client')
    $updateOverlayOpen.set(true)

    getActionStatusSpy.mockResolvedValue({ exit_code: 1, lines: [], name: 'hermes-update', pid: 4, running: false })
    await vi.advanceTimersByTimeAsync(1500)
    await pressed

    expect($updateOverlayTarget.get()).toBe('client')

    // The person closes the client's view (the overlay's own close path).
    $updateOverlayOpen.set(false)
    resetUpdateApplyState('client')

    expect($updateOverlayTarget.get()).toBe('backend')
    expect($updateOverlayOpen.get()).toBe(true)
  })

  it("a hidden backend update that succeeds leaves a running client update's progress open", async () => {
    const { pressed } = await pressAndHide()
    $updateApply.set({ ...idle, applying: true, stage: 'prepare' })
    $updateOverlayTarget.set('client')
    $updateOverlayOpen.set(true)

    getActionStatusSpy.mockResolvedValue({ exit_code: 0, lines: [], name: 'hermes-update', pid: 4, running: false })
    await vi.advanceTimersByTimeAsync(1500)
    await pressed

    expect($updateOverlayTarget.get()).toBe('client')
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
