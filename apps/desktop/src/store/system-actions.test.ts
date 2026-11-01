import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { registerGatewayReconnect } from '@/store/gateway-reconnect'

// The REST layer is the only seam these flows own; the confirm dialog and
// notifications stay real. getStatus answers a standalone gateway so
// confirmSharedGatewayRestart() takes the silent no-dialog path.
const getActionStatus = vi.fn()
const restartGateway = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getActionStatus: (name: string, timeout?: number) => getActionStatus(name, timeout),
  getStatus: async () => ({}),
  restartGateway: () => restartGateway()
}))

import { runGatewayRestart, watchGatewayRestartOutcome } from './system-actions'

// Mirrors POLL_INTERVAL_MS × POLL_ATTEMPTS in system-actions.ts: the poll
// window the flows own. Driven as one advance so timers AND promise
// microtasks interleave the way the real loop runs.
const POLL_WINDOW_MS = 18 * 1_200

const settlePollWindow = () => vi.advanceTimersByTimeAsync(POLL_WINDOW_MS + 1_000)

beforeEach(() => {
  vi.useFakeTimers()
  restartGateway.mockResolvedValue({ ok: true, pid: 4242, name: 'gateway-restart' })
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
