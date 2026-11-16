import { act, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { parkSecondaryReconnects } from '@/store/gateway-park'
import { $gatewayState } from '@/store/session'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'
import {
  fakeDesktop,
  flushAsync,
  Harness,
  parseReconnectEpisode,
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

// #C2a: a backgrounded window is PARKED (ladders disarmed, nothing minted) and a
// returning one DIALS its replacement generation instead of probing the socket
// it suspects — the same hook, the same fake socket, its own suite so the wake
// contract is in one place.
describe('useGatewayBoot wake path (#C2a)', () => {
  beforeEach(resetGatewayBootHarness)
  afterEach(teardownGatewayBootHarness)

  // #C2a: the wake path DIALS. A socket that still reports OPEN after a sleep is
  // presumptively dead — the OS dropped the transport while the renderer's
  // WebSocket object kept readyState OPEN and no close ever fired — so a wake
  // mints and connects the replacement at once and retires the socket it
  // replaced, instead of spending the 5 s liveness-probe budget proving what
  // the user is already experiencing. The old contract asserted the opposite
  // (an untouched socket); that assertion pinned the probe, not the protection.
  it('wake dial: a signal replaces the primary generation in one dial, without a probe', async () => {
    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const socketCountBefore = FakeWebSocket.instances.length
    const staleSocket = FakeWebSocket.instances.at(-1)!

    // Half-open: the socket answers NOTHING. Under the old contract this cost
    // the full 5 s probe budget before any redial was considered.
    FakeWebSocket.pingMode = 'silent'

    act(() => window.dispatchEvent(new Event('online')))

    // The signal must not tear the working socket down: the replacement is
    // dialed first, and the old generation is retired only once it is open.
    expect(staleSocket.readyState).toBe(FakeWebSocket.OPEN)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(FakeWebSocket.instances.length).toBe(socketCountBefore + 1)
    expect(FakeWebSocket.instances.at(-1)!.readyState).toBe(FakeWebSocket.OPEN)
    // Nothing rode the old socket, so it is retired as soon as the replacement
    // is open — one dial, no probe, no backoff ladder.
    expect(staleSocket.readyState).toBe(FakeWebSocket.CLOSED)
    expect($gatewayState.get()).toBe('open')
  })

  // #C2a acceptance: a hidden window pays nothing. Both ladders are disarmed and
  // no wake signal is allowed to act, so ten minutes in the background mints
  // ZERO tickets — counted at the C1 telemetry's pre-IPC half, the
  // `getGatewayWsUrl` mint the episode line reports, and cross-checked against
  // the episode log itself (an episode would mean a dial happened).
  it('park: a hidden window mints nothing over ten minutes in the background', async () => {
    const desktop = fakeDesktop()
    const episodeLines: string[] = []

    desktop.logLine.mockImplementation((line: string) => episodeLines.push(line))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const mintsAfterBoot = desktop.getGatewayWsUrl.mock.calls.length

    // The user leaves. Cover both deaths: a silent (half-open) one that fires no
    // close at all, and a real close the ladder would normally redial in 300 ms.
    setVisibility('hidden')
    FakeWebSocket.pingMode = 'silent'
    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600_000)
    })

    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot)
    expect(FakeWebSocket.instances.length).toBe(1)

    FakeWebSocket.instances[0].drop()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600_000)
    })

    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot)
    expect(FakeWebSocket.instances.length).toBe(1)
    expect($gatewayState.get()).toBe('closed')
    expect(episodeLines.map(parseReconnectEpisode).filter(Boolean)).toHaveLength(0)

    // The user comes back: exactly ONE dial, and the replacement is live.
    setVisibility('visible')
    FakeWebSocket.pingMode = 'pong'
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot + 1)
    expect($gatewayState.get()).toBe('open')
  })

  // A previous mount parked the window and never came back (the visibility
  // change landed while it was hidden). The latch is a function of the document,
  // so binding the signals re-reads it: a visible window must never inherit a
  // stale park and sit with its ladders latched off.
  it('re-reads the document on mount: a stale park cannot latch a visible window off', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    parkSecondaryReconnects()

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    const staleSocket = FakeWebSocket.instances.at(-1)!

    FakeWebSocket.pingMode = 'silent'
    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(FakeWebSocket.instances.length).toBe(2)
    expect(staleSocket.readyState).toBe(FakeWebSocket.CLOSED)
    expect($gatewayState.get()).toBe('open')
  })

  // #C2a acceptance: the returning user is back on a live socket after ONE dial,
  // and the C1 telemetry's detect_ms — the redial decision, measured from the
  // signal that triggered it — stays inside a quarter second. It used to be
  // multi-second: detection waited out LIVENESS_PROBE_TIMEOUT_MS (5 s) before a
  // redial was even considered.
  it('wake dial: a return from a hidden window is one dial with detect_ms inside 250 ms', async () => {
    const desktop = fakeDesktop()
    const episodeLines: string[] = []

    desktop.logLine.mockImplementation((line: string) => episodeLines.push(line))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const mintsAfterBoot = desktop.getGatewayWsUrl.mock.calls.length
    const staleSocket = FakeWebSocket.instances.at(-1)!

    // Away for a while, socket silently dead (no close event: the half-open case
    // the wake path exists for).
    setVisibility('hidden')
    FakeWebSocket.pingMode = 'silent'
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600_000)
    })
    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot)

    // Back. One signal, one dial, and the dial is done inside the quarter second
    // the old probe could not even have started deciding in.
    // A real gateway advertises its keepalive on `gateway.ready`; the pong that
    // answers the replacement's first ping is the C1 `usable` round trip.
    FakeWebSocket.readyPayload = { heartbeat: true }
    setVisibility('visible')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot + 1)
    expect(staleSocket.readyState).toBe(FakeWebSocket.CLOSED)
    expect(FakeWebSocket.instances.at(-1)!.readyState).toBe(FakeWebSocket.OPEN)
    expect($gatewayState.get()).toBe('open')

    // Let the replacement's first response land (the C1 usable stamp) so the
    // episode writes its phase table, then read detect_ms off it.
    FakeWebSocket.pingMode = 'pong'
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })

    const episode = episodeLines.map(parseReconnectEpisode).find(Boolean)

    expect(episode?.close_reason).toBe('wake')
    expect(Number(episode?.detect_ms)).toBeLessThanOrEqual(250)
  })
})
