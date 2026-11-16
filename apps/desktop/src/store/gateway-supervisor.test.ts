import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { parkSecondaryReconnects, unparkSecondaryReconnects } from './gateway-park'
import {
  armRouteRetry,
  cancelRouteRetry,
  dialPrimaryRoute,
  dialRoute,
  forgetRoute,
  gatewayWakeInFlight,
  PRIMARY_ROUTE,
  registerGatewayPoolRoute,
  registerGatewayPrimaryRoute,
  resetGatewaySupervisorForTests,
  resetRouteLadder,
  routeDialInFlight,
  routeRetryPending,
  secondaryRouteId,
  superviseWake
} from './gateway-supervisor'

/** Let the supervisor's queued work run (a dial starts on a microtask). */
async function settle(): Promise<void> {
  for (let tick = 0; tick < 20 && gatewayWakeInFlight(); tick += 1) {
    await Promise.resolve()
  }

  for (let tick = 0; tick < 4; tick += 1) {
    await Promise.resolve()
  }
}

describe('gateway supervisor (#C5)', () => {
  beforeEach(() => {
    resetGatewaySupervisorForTests()
    unparkSecondaryReconnects()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    resetGatewaySupervisorForTests()
    unparkSecondaryReconnects()
    vi.useRealTimers()
  })

  // C5 acceptance, half two: "zero duplicate concurrent dials to the same
  // route". The booking is synchronous, so two callers in one tick cannot both
  // open a socket — the second joins the first's in-flight promise.
  it('books ONE in-flight dial per route and hands the same promise to the next caller', async () => {
    let release: () => void = () => undefined

    const gate = new Promise<void>(resolve => {
      release = resolve
    })

    const run = vi.fn(async () => {
      await gate
    })

    const first = dialRoute(PRIMARY_ROUTE, run)
    const second = dialRoute(PRIMARY_ROUTE, run)

    expect(second).toBe(first)

    await settle()

    expect(run).toHaveBeenCalledTimes(1)
    expect(routeDialInFlight(PRIMARY_ROUTE)).toBe(true)

    release()
    await first

    expect(routeDialInFlight(PRIMARY_ROUTE)).toBe(false)

    await dialRoute(PRIMARY_ROUTE, run)

    expect(run).toHaveBeenCalledTimes(2)
  })

  it('does not let one route\'s dial block or join another route\'s', async () => {
    const run = vi.fn(async () => undefined)

    await Promise.all([dialRoute(PRIMARY_ROUTE, run), dialRoute(secondaryRouteId('coder'), run)])

    expect(run).toHaveBeenCalledTimes(2)
  })

  it('rejects a primary dial when no boot owner is mounted', async () => {
    await expect(dialPrimaryRoute()).rejects.toThrow('Gateway reconnect is unavailable')
  })

  // The ladder: ONE timer per route, full-jitter exponential shape, expanding
  // per attempt.
  it('arms one timer per route and widens the delay with each attempt', async () => {
    vi.useFakeTimers()
    // Ceiling rather than a random point inside it, so the rungs are exact.
    vi.spyOn(Math, 'random').mockReturnValue(1)

    const run = vi.fn()

    armRouteRetry('route', run)
    armRouteRetry('route', run)

    expect(routeRetryPending('route')).toBe(true)

    await vi.advanceTimersByTimeAsync(300)
    expect(run).toHaveBeenCalledTimes(1)
    expect(routeRetryPending('route')).toBe(false)

    // Second rung: twice the first ceiling.
    armRouteRetry('route', run)
    await vi.advanceTimersByTimeAsync(599)
    expect(run).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1)
    expect(run).toHaveBeenCalledTimes(2)

    // A healthy open drops the route back to the first rung.
    resetRouteLadder('route')
    armRouteRetry('route', run)
    await vi.advanceTimersByTimeAsync(300)
    expect(run).toHaveBeenCalledTimes(3)

    cancelRouteRetry('route')
    armRouteRetry('route', run)
    expect(routeRetryPending('route')).toBe(true)

    forgetRoute('route')
    expect(routeRetryPending('route')).toBe(false)
    expect(routeDialInFlight('route')).toBe(false)
  })

  // #C2a: an armed ladder is not an excuse to dial from a hidden window.
  it('skips an armed retry that a park landed on', async () => {
    vi.useFakeTimers()

    const run = vi.fn()

    armRouteRetry('route', run)
    parkSecondaryReconnects()
    await vi.advanceTimersByTimeAsync(20_000)

    expect(run).not.toHaveBeenCalled()
    expect(routeRetryPending('route')).toBe(false)
  })

  // C5 acceptance, half one: one wake event → exactly one pre-IPC chain. A real
  // resume fires several signals (power resume, network online, the window
  // becoming visible); each one used to start its own redial family. They land
  // on ONE action now — one pool nudge, one primary dial.
  it('takes ONE wake action for a signal burst', async () => {
    const dial = vi.fn(async (_options?: { source?: string }) => undefined)
    const nudge = vi.fn()

    registerGatewayPrimaryRoute({ dial, reportsOpen: () => true })
    registerGatewayPoolRoute({ nudge })

    superviseWake(true)
    superviseWake(true)
    superviseWake(true)

    expect(nudge).toHaveBeenCalledTimes(1)

    await settle()

    expect(dial).toHaveBeenCalledTimes(1)
    expect(dial.mock.calls[0]?.[0]).toEqual({ source: 'wake' })
  })

  // Coalescing is per burst, not a global suppress: a wake that lands after the
  // previous action settled is a new recovery, and must recover.
  it('takes a fresh action for a signal that lands after the previous one settled', async () => {
    const dial = vi.fn(async () => undefined)

    registerGatewayPrimaryRoute({ dial, reportsOpen: () => false })

    superviseWake(true)
    await settle()
    superviseWake(true)
    await settle()

    expect(dial).toHaveBeenCalledTimes(2)
  })

  it('gates the destructive sweep behind its holdoff but always runs the cheap half', async () => {
    const forced: boolean[] = []

    registerGatewayPoolRoute({ nudge: ({ forceOpenSockets }) => forced.push(forceOpenSockets) })
    registerGatewayPrimaryRoute({ dial: vi.fn(), reportsOpen: () => true })

    superviseWake(true)
    await settle()
    superviseWake(true)
    await settle()

    expect(forced).toEqual([true, false])
  })

  it('a soft signal on a socket that reports open nudges the pool but dials nothing', async () => {
    const dial = vi.fn()
    const nudge = vi.fn()

    registerGatewayPrimaryRoute({ dial, reportsOpen: () => true })
    registerGatewayPoolRoute({ nudge })

    superviseWake(false)
    await settle()

    expect(nudge).toHaveBeenCalledTimes(1)
    expect(dial).not.toHaveBeenCalled()
  })

  it('a soft signal still dials a primary that does not report open', async () => {
    const dial = vi.fn(async () => undefined)

    registerGatewayPrimaryRoute({ dial, reportsOpen: () => false })

    superviseWake(false)
    await settle()

    expect(dial).toHaveBeenCalledTimes(1)
  })

  it('a wake with no route registered is a no-op, not a rejection', async () => {
    superviseWake(true)
    await settle()
  })
})
