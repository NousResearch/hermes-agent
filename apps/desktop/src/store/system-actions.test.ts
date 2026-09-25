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

// The backend accepted the restart POST and then went down for the restart
// itself: every poll refused — the exact window #123111 reports.
const refusedBackend = () => getActionStatus.mockRejectedValue(new Error('fetch failed'))

// A replacement process whose in-memory action registry never saw this action
// id (the registry died with the process the restart replaced).
const freshProcess = () =>
  getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: null, pid: null, lines: [] })

describe('runGatewayRestart during the restart window (#123111)', () => {
  it('resolves success when the status poll is refused mid-restart', async () => {
    refusedBackend()

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

  it('still surfaces a real failure: a recorded non-zero exit', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: 1, pid: null, lines: [] })

    const outcome = runGatewayRestart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
  })

  it('hands reconnection to the gateway reconnect owner after the restart', async () => {
    refusedBackend()
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
  it('resolves true across a refused poll window and reconnects', async () => {
    refusedBackend()
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

  it('resolves false on a recorded non-zero exit', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: 1, pid: null, lines: [] })

    const outcome = watchGatewayRestartOutcome()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
  })
})
