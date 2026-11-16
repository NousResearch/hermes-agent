import { act, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_ROUTE, routeRetryPending } from '@/store/gateway-supervisor'
import { $gatewayState } from '@/store/session'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'
import {
  advanceBackoff,
  fakeDesktop,
  flushAsync,
  Harness,
  powerResume,
  resetGatewayBootHarness,
  setVisibility,
  teardownGatewayBootHarness
} from '../../../test/gateway-boot-harness'

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

// #C5 acceptance (C1 telemetry, half one): ONE wake event → exactly ONE pre-IPC
// chain. The wake path used to hold three redial families — the primary ladder,
// the pool's per-entry backoff timers and the wake dial — and each re-minted its
// own ticket, measured at up to nine sockets inside one five-second burst.
//
// These cases drive the REAL hook + REAL HermesGateway through the fake socket,
// so what is asserted is the number of DIALS the renderer performs, counted at
// the two observable ends of the pre-IPC chain: the ticket mint
// (`getGatewayWsUrl`) and the socket itself.
describe('useGatewayBoot connection supervisor (#C5)', () => {
  beforeEach(resetGatewayBootHarness)
  afterEach(teardownGatewayBootHarness)

  it('one wake burst is exactly one ticket mint and one dial', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    const mintsAfterBoot = desktop.getGatewayWsUrl.mock.calls.length
    const socketsAfterBoot = FakeWebSocket.instances.length

    // The socket after a sleep is presumptively dead whatever it reports, so
    // this is the case the wake path exists for.
    FakeWebSocket.pingMode = 'silent'

    // A real resume is a BURST of signals, and it fires them in the same tick:
    // the OS resumes (power resume), the interface comes back (`online`) and the
    // window becomes visible again. Three families answered all three.
    act(() => {
      powerResume?.()
      window.dispatchEvent(new Event('online'))
      setVisibility('visible')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot + 1)
    expect(FakeWebSocket.instances.length).toBe(socketsAfterBoot + 1)
    expect($gatewayState.get()).toBe('open')
  })

  it('a signal that arrives while the pre-IPC chain is in flight joins it', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    const mintsAfterBoot = desktop.getGatewayWsUrl.mock.calls.length
    const socketsAfterBoot = FakeWebSocket.instances.length

    // Hold the wake's own ticket mint open so a second signal provably lands
    // INSIDE the chain. A cold boot resolves it too fast to prove anything.
    let releaseMint: () => void = () => undefined

    const held = new Promise<void>(resolve => {
      releaseMint = resolve
    })

    desktop.getGatewayWsUrl.mockImplementationOnce(async (conn?: { wsUrl?: string }) => {
      await held

      return conn?.wsUrl ?? 'wss://vps.example.com/api/ws?token=t'
    })

    act(() => window.dispatchEvent(new Event('online')))
    await flushAsync()

    // Still mid-chain: these three must join the dial in flight, not open their
    // own socket each.
    act(() => {
      window.dispatchEvent(new Event('online'))
      powerResume?.()
      window.dispatchEvent(new Event('focus'))
    })
    await flushAsync()

    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot + 1)

    releaseMint()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot + 1)
    expect(FakeWebSocket.instances.length).toBe(socketsAfterBoot + 1)
    expect($gatewayState.get()).toBe('open')
  })

  // The ownership itself: the window's primary retry ladder is the supervisor's,
  // and the redial runs through it. The Base cannot even resolve this module —
  // that is the deliverable — so the receipt for this case is the import, not a
  // red assertion on the old code.
  it('a dropped primary arms its retry on the supervisor and heals through it', async () => {
    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    expect(routeRetryPending(PRIMARY_ROUTE)).toBe(false)

    act(() => FakeWebSocket.instances.at(-1)!.drop())
    await flushAsync()

    // The close edge armed the supervisor's ONE timer for the window's route.
    expect(routeRetryPending(PRIMARY_ROUTE)).toBe(true)

    await advanceBackoff()

    expect($gatewayState.get()).toBe('open')
    expect(routeRetryPending(PRIMARY_ROUTE)).toBe(false)
  })
})
