/**
 * Park/episode contract (t_de06ee1e, the C1 telemetry decision).
 *
 * `detect_ms` is the time from a signal a PRESENT user could feel to the redial
 * decision, and parked time is never inside an episode. Three holes this file
 * pins closed, all through the REAL hook and the REAL client over the fake
 * socket:
 *
 *   1. a close parked before its backoff fires used to drag the whole park into
 *      the return dial's detect_ms (the park now SETTLES the open episode, and
 *      the return reports `parked_close=yes`);
 *   2. a kept generation's dying socket used to file its phases into the live
 *      episode — overwriting the successor's numbers, or settling it on a dead
 *      socket's clock (only the generation that IS the window's primary may
 *      report);
 *   3. a wake dial whose successor never opened used to leave its episode for
 *      an unrelated later event to date (`outcome=abandoned` now).
 */
import { act, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $gatewayState } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

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

type Episode = Record<string, string>

/** A transcript frame on `sock`, as the backend writes it mid-turn. */
function emitDelta(sock: FakeWebSocket, seq: number) {
  emitFrame(sock, {
    jsonrpc: '2.0',
    method: 'event',
    params: { type: 'message.delta', session_id: 's-ep', seq, payload: { text: `d${seq}` } }
  })
}

function emitFrame(sock: FakeWebSocket, frame: Record<string, unknown>) {
  ;(sock as unknown as { emit: (type: string, ev: unknown) => void }).emit('message', {
    data: JSON.stringify(frame)
  })
}

// Every request id each socket was sent, oldest first: answering one is how a
// live backend makes a generation application-usable (`client.capabilities` is
// already sent on every open), and it is the ONLY way to reach `usable` without
// waiting out a keepalive interval.
const outstanding = new WeakMap<FakeWebSocket, unknown[]>()

function captureRequests(): () => void {
  const original = FakeWebSocket.prototype.send

  FakeWebSocket.prototype.send = function (this: FakeWebSocket, data: string) {
    try {
      const frame = JSON.parse(data) as { id?: unknown; method?: unknown }

      if (frame.id !== undefined && frame.id !== null && typeof frame.method === 'string') {
        outstanding.set(this, [...(outstanding.get(this) ?? []), frame.id])
      }
    } catch {
      // Non-JSON frames carry no request to answer.
    }

    return original.call(this, data)
  }

  return () => {
    FakeWebSocket.prototype.send = original
  }
}

/** Answer `sock`'s oldest outstanding request: its generation becomes usable. */
function answerRequest(sock: FakeWebSocket, result: Record<string, unknown> = {}) {
  const id = outstanding.get(sock)?.shift()

  if (!id) {
    throw new Error('the socket has no outstanding request to answer')
  }

  emitFrame(sock, { id, jsonrpc: '2.0', result })
}

function episodesOf(lines: string[]): Episode[] {
  return lines.map(parseReconnectEpisode).filter((episode): episode is Episode => episode !== null)
}

/** A window with a turn in flight: a generation replaced by a wake is KEPT. */
function holdTurnBusy() {
  act(() => {
    publishSessionState('rt-episode', {
      ...createClientSessionState(null),
      busy: true,
      storedSessionId: 's-ep'
    })
  })
}

describe('useGatewayBoot parked episodes (#C1 telemetry contract)', () => {
  let restoreSends: () => void

  beforeEach(() => {
    resetGatewayBootHarness()
    restoreSends = captureRequests()
  })

  afterEach(() => {
    restoreSends()
    clearAllSessionStates()
    teardownGatewayBootHarness()
  })

  it('a close parked before its backoff fires is settled by the park; the return dials inside 50 ms', async () => {
    const desktop = fakeDesktop()
    const lines: string[] = []

    desktop.logLine.mockImplementation((line: string) => lines.push(line))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    const booted = FakeWebSocket.instances.at(-1)!
    const mintsAfterBoot = desktop.getGatewayWsUrl.mock.calls.length

    // The socket dies while the user is looking at it: the close opens the
    // episode, and its redial is one backoff rung away.
    act(() => booted.drop())
    await flushAsync()

    // The user leaves first. The park ends the wait — and the parked time is
    // NOT part of any episode.
    setVisibility('hidden')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600_000)
    })

    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot)
    expect(FakeWebSocket.instances.length).toBe(1)

    // Back on a real backend: one dial, then the usable round trip.
    FakeWebSocket.readyPayload = { heartbeat: true }
    setVisibility('visible')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })

    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot + 1)

    const episodes = episodesOf(lines)

    expect(episodes.map(episode => episode.outcome)).toEqual(['parked', 'usable'])
    // The parked line is the wait the park ended — no total, and not itself a
    // return dial.
    expect(episodes[0].total_ms).toBe('n/a')
    expect(episodes[0].parked_close).toBe('no')
    // The return is a COLD redial of a primary that died unwatched, and its
    // clock starts at the return, not at the close ten minutes earlier.
    expect(episodes[1].close_reason).toBe('wake')
    expect(episodes[1].parked_close).toBe('yes')
    expect(Number(episodes[1].detect_ms)).toBeLessThan(50)
  })

  it('a close that lands while parked opens no episode; the return redials it exactly once', async () => {
    const desktop = fakeDesktop()
    const lines: string[] = []

    desktop.logLine.mockImplementation((line: string) => lines.push(line))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    const booted = FakeWebSocket.instances.at(-1)!
    const mintsAfterBoot = desktop.getGatewayWsUrl.mock.calls.length

    setVisibility('hidden')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100)
    })

    // A real close, nobody watching.
    act(() => booted.drop())
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600_000)
    })

    // R1: no episode opens while parked, so there is nothing to report yet.
    expect(episodesOf(lines)).toEqual([])
    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot)

    FakeWebSocket.readyPayload = { heartbeat: true }
    setVisibility('visible')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })

    const episodes = episodesOf(lines)

    expect(episodes).toHaveLength(1)
    expect(episodes[0].close_reason).toBe('wake')
    // The primary died while nobody watched: this is the cold redial of it.
    expect(episodes[0].parked_close).toBe('yes')
    expect(Number(episodes[0].detect_ms)).toBeLessThan(50)
    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot + 1)
  })

  it('a park that left the primary alive is not a parked close: the return replaces a live socket', async () => {
    const desktop = fakeDesktop()
    const lines: string[] = []

    desktop.logLine.mockImplementation((line: string) => lines.push(line))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    const socketCountAfterBoot = FakeWebSocket.instances.length

    // Away, socket half-open (reports OPEN, answers nothing): no close edge, so
    // nothing died where anyone could see it.
    setVisibility('hidden')
    FakeWebSocket.pingMode = 'silent'
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600_000)
    })
    expect(episodesOf(lines)).toEqual([])

    // Back: the wake REPLACES the socket it suspects, under the old one.
    FakeWebSocket.readyPayload = { heartbeat: true }
    FakeWebSocket.pingMode = 'pong'
    setVisibility('visible')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })

    expect(FakeWebSocket.instances.length).toBe(socketCountAfterBoot + 1)

    const episodes = episodesOf(lines)

    expect(episodes).toHaveLength(1)
    expect(episodes[0].close_reason).toBe('wake')
    // QA reads this to tell "redial a dead primary on return" from "replace a
    // half-open live primary": this is the latter.
    expect(episodes[0].parked_close).toBe('no')
  })

  it('a kept generation dying during the keep window never settles the live episode', async () => {
    const desktop = fakeDesktop()
    const lines: string[] = []

    desktop.logLine.mockImplementation((line: string) => lines.push(line))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    const booted = FakeWebSocket.instances.at(-1)!

    // A watermark on the boot generation, and a turn in flight: every
    // replacement from here is a CONTINUATION kept SERVED for that turn.
    emitDelta(booted, 1)
    await flushAsync()

    holdTurnBusy()

    // Wake 1: the successor inherits the watermark, so its open starts a replay
    // whose barrier is still outstanding when it becomes usable. It is usable
    // and has never emitted — exactly the "kept generation" shape.
    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(FakeWebSocket.instances.length).toBe(2)

    const kept = FakeWebSocket.instances[1]

    act(() => answerRequest(kept))
    await flushAsync()
    // No episode line yet: the wake's episode is open and the replacement's
    // phases are still barred.
    expect(episodesOf(lines)).toEqual([])

    // Wake 2: a successor under the kept generation, and the episode stays open
    // for it (this is the dial the user is waiting on).
    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(FakeWebSocket.instances.length).toBe(3)

    const successor = FakeWebSocket.instances[2]

    // The KEPT generation dies. Its dying socket reports what it reached — with
    // a usable stamp on it — and none of that is the live episode's business.
    act(() => kept.drop())
    await flushAsync()

    expect(episodesOf(lines)).toEqual([])

    // The successor's own usable is what ends the episode, with ITS phases.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })
    act(() => answerRequest(successor))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(11_000)
    })

    const episodes = episodesOf(lines)

    expect(episodes).toHaveLength(1)
    expect(episodes[0].outcome).toBe('usable')
    // The successor answered ~2 s after its socket opened; the kept generation
    // answered immediately after its own. A usable_ms in the thousands is the
    // successor's report, filed after `promote()` — the guard's ordering pin.
    expect(Number(episodes[0].usable_ms)).toBeGreaterThan(1_000)
    expect(episodes[0].parked_close).toBe('no')
  })

  it('a wake dial whose successor never opens settles its episode as abandoned', async () => {
    const desktop = fakeDesktop()
    const lines: string[] = []

    desktop.logLine.mockImplementation((line: string) => lines.push(line))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    // The remote is gone: the replacement socket errors before it opens, and
    // the live socket is left alone.
    FakeWebSocket.mode = 'fail'
    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(FakeWebSocket.instances.length).toBe(2)
    expect($gatewayState.get()).toBe('open')

    const episodes = episodesOf(lines)

    expect(episodes).toHaveLength(1)
    expect(episodes[0].outcome).toBe('abandoned')
    expect(episodes[0].close_reason).toBe('wake')
    expect(episodes[0].parked_close).toBe('no')
    // Settled at the dial's own failure, not dated by whatever the next
    // unrelated usable or begin would have been.
    expect(Number(episodes[0].detect_ms)).toBeLessThan(50)
  })
})
