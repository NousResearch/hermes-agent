import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { registerGatewayReconnect } from '@/store/gateway-reconnect'
import { $notifications, clearNotifications } from '@/store/notifications'

// The REST layer is the only seam these flows own; the confirm dialog and
// notifications stay real. getStatus answers a standalone gateway so
// confirmSharedGatewayRestart() takes the silent no-dialog path.
const getActionStatus = vi.fn()
const restartGateway = vi.fn()
const startGateway = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getActionStatus: (name: string, lines?: number, owner?: HermesApi.ResolvedOwner) =>
    getActionStatus(name, lines, owner),
  getStatus: async () => ({}),
  restartGateway: () => restartGateway(),
  startGateway: () => startGateway()
}))

import { $gatewayRestarting, runGatewayRestart, runGatewayStart, watchGatewayRestartOutcome } from './system-actions'

// Mirrors POLL_INTERVAL_MS × POLL_ATTEMPTS in system-actions.ts: the poll
// window the flows own. Driven as one advance so timers AND promise
// microtasks interleave the way the real loop runs.
const POLL_WINDOW_MS = 18 * 1_200

const settlePollWindow = () => vi.advanceTimersByTimeAsync(POLL_WINDOW_MS + 1_000)

beforeEach(() => {
  clearNotifications()
  vi.useFakeTimers()
  restartGateway.mockResolvedValue({ ok: true, pid: 4242, name: 'gateway-restart' })
  startGateway.mockResolvedValue({ ok: true, pid: 4242, name: 'gateway-start' })
})

afterEach(() => {
  vi.useRealTimers()
  vi.clearAllMocks()
})

// A backend that is down for the restart itself: polls refuse until `healthy`
// attempts in, then answer — the exact window #123111 reports. The status
// endpoint 404s for an action the (new) registry never saw and errors while
// the backend is down, so a REFUSAL is "no answer", not a terminal verdict.
const backendDownUntil = (healthyAt: number) => {
  let polls = 0

  getActionStatus.mockImplementation(async () => {
    polls += 1

    if (polls < healthyAt) {
      throw new Error('fetch failed')
    }

    return { name: 'gateway-restart', running: true, exit_code: null, pid: 4242, lines: [] }
  })
}

// A permanently dead backend: every poll for the whole window is refused.
const refusedBackend = () => getActionStatus.mockRejectedValue(new Error('fetch failed'))

// A replacement process whose in-memory action registry never saw this action
// id (the registry died with the process the restart replaced).
const freshProcess = () =>
  getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: null, pid: null, lines: [] })

describe('runGatewayRestart during the restart window (#123111)', () => {
  it('resolves success when the status poll is refused mid-window, then answers', async () => {
    backendDownUntil(4)

    const outcome = runGatewayRestart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(true)
    expect(getActionStatus).toHaveBeenCalled()
  })

  it('resolves success when a fresh process reports the action unknown', async () => {
    freshProcess()

    const outcome = runGatewayRestart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(true)
  })

  it('fails when the poll window is refused end to end — the restart never confirmed', async () => {
    refusedBackend()

    const outcome = runGatewayRestart()

    await settlePollWindow()
    // Draining the whole budget with zero answered polls must NOT resolve
    // success: the callers' failure banners stay up and the user sees the
    // failure toast instead of a cleared banner over a down gateway.
    await expect(outcome).resolves.toBe(false)
  })

  it('still surfaces a real failure: a recorded non-zero exit', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: 1, pid: null, lines: [] })

    const outcome = runGatewayRestart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
  })

  it('retains the last nonempty child-output cause in the actual failure notification', async () => {
    getActionStatus.mockResolvedValue({
      name: 'gateway-restart',
      running: false,
      exit_code: 1,
      pid: null,
      lines: ['starting gateway', '  bind failed: address already in use  ', '  ']
    })

    const outcome = runGatewayRestart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
    expect($notifications.get()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'error', message: 'bind failed: address already in use' })
      ])
    )
  })

  it('hands reconnection to the gateway reconnect owner after a confirmed restart', async () => {
    backendDownUntil(4)
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)

    try {
      const outcome = runGatewayRestart()

      await settlePollWindow()
      await expect(outcome).resolves.toBe(true)
      expect(handler).toHaveBeenCalledOnce()
    } finally {
      off()
    }
  })

  it('hands reconnection to the owner even when the recorded exit failed', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: 1, pid: null, lines: [] })
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)

    try {
      const outcome = runGatewayRestart()

      await settlePollWindow()
      await expect(outcome).resolves.toBe(false)
      expect(handler).toHaveBeenCalledOnce()
    } finally {
      off()
    }
  })
})

describe('watchGatewayRestartOutcome (backend-spawned restart)', () => {
  it('resolves true across a refused-then-answered poll window and reconnects', async () => {
    backendDownUntil(4)
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)

    try {
      const outcome = watchGatewayRestartOutcome()

      await settlePollWindow()
      await expect(outcome).resolves.toBe(true)
      expect(handler).toHaveBeenCalledOnce()
    } finally {
      off()
    }
  })

  it('resolves false when the poll window is refused end to end', async () => {
    refusedBackend()

    const outcome = watchGatewayRestartOutcome()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
  })

  it('resolves false on a recorded non-zero exit', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: 1, pid: null, lines: [] })

    const outcome = watchGatewayRestartOutcome()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
  })
})

// Exercise the action owner, not a restart relabelled as Start.
describe('runGatewayStart', () => {
  it('polls the accepted start and hands recovery to the reconnect owner', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-start', running: false, exit_code: 0, pid: null, lines: [] })
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)

    try {
      const outcome = runGatewayStart()

      expect($gatewayRestarting.get()).toBe(true)
      await settlePollWindow()
      await expect(outcome).resolves.toBe(true)
      expect(startGateway).toHaveBeenCalledOnce()
      expect(restartGateway).not.toHaveBeenCalled()
      expect(getActionStatus).toHaveBeenCalledWith(
        'gateway-start',
        200,
        expect.objectContaining({ connectionId: null, profile: null })
      )
      expect(handler).toHaveBeenCalledOnce()
      expect($gatewayRestarting.get()).toBe(false)
    } finally {
      off()
    }
  })

  it('preserves the recorded start failure cause and clears its busy indicator', async () => {
    getActionStatus.mockResolvedValue({
      name: 'gateway-start',
      running: false,
      exit_code: 2,
      pid: null,
      lines: ['launching', '  invalid bot credentials  ', '']
    })
    const outcome = runGatewayStart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
    expect($notifications.get()).toEqual(
      expect.arrayContaining([expect.objectContaining({ kind: 'error', message: 'invalid bot credentials' })])
    )
    expect($gatewayRestarting.get()).toBe(false)
  })

  it('uses the localized Start failure when the failed child has no output', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-start', running: false, exit_code: 1, pid: null, lines: ['  '] })
    const outcome = runGatewayStart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
    expect($notifications.get()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'error',
          title: 'Messaging gateway start failed.',
          message: 'Messaging gateway start failed.'
        })
      ])
    )
  })

  it('does not report success for an entirely unanswered start poll window', async () => {
    refusedBackend()
    const outcome = runGatewayStart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
    expect($gatewayRestarting.get()).toBe(false)
  })

  it('does not poll a declined start request and still reconnects safely', async () => {
    startGateway.mockRejectedValue(new Error('profile is already served by the shared gateway'))
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)

    try {
      await expect(runGatewayStart()).resolves.toBe(false)
      expect(getActionStatus).not.toHaveBeenCalled()
      expect(handler).toHaveBeenCalledOnce()
      expect($gatewayRestarting.get()).toBe(false)
    } finally {
      off()
    }
  })
})
