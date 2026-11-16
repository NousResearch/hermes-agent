/**
 * Round-2 wake contract (#C2a, review of PR elight/hermes-agent#7).
 *
 * The reviewer proved three ways the first cut turned "dial, don't probe" into
 * "cold-boot over the live socket": a replacement with no replay cursor loses
 * the half-open window; a generation kept for a running turn was detached
 * before it was kept, so the turn's deltas hit a deaf socket; and every
 * alt-tab focus swapped a healthy primary. These are behaviour tests for
 * exactly those three, with the assertions the round-1 suite lacked: the
 * replay frame CONTENT and its watermark, DELIVERY on a kept socket, and a
 * socket/mint COUNT across repeated focus.
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
  resetGatewayBootHarness,
  teardownGatewayBootHarness
} from '../../../test/gateway-boot-harness'

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))
vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

interface SentFrame {
  method?: string
  params?: Record<string, unknown>
}

const sent: SentFrame[] = []

/** Capture every JSON-RPC frame ANY socket of this test sends. */
function spySends(): () => void {
  const original = FakeWebSocket.prototype.send

  FakeWebSocket.prototype.send = function (this: FakeWebSocket, data: string) {
    try {
      sent.push(JSON.parse(data) as SentFrame)
    } catch {
      // Non-JSON frames carry no method to pin.
    }

    return original.call(this, data)
  }

  return () => {
    FakeWebSocket.prototype.send = original
  }
}

/** A transcript frame on `sock`, as the backend writes it mid-turn. */
function emitDelta(sock: FakeWebSocket, seq: number) {
  ;(sock as unknown as { emit: (type: string, ev: unknown) => void }).emit('message', {
    data: JSON.stringify({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's-cont', seq, payload: { text: `d${seq}` } }
    })
  })
}

describe('useGatewayBoot wake continuity (#C2a r2)', () => {
  let restoreSends: () => void

  beforeEach(() => {
    resetGatewayBootHarness()
    sent.length = 0
    restoreSends = spySends()
  })

  afterEach(() => {
    restoreSends()
    clearAllSessionStates()
    teardownGatewayBootHarness()
  })

  // A replacement generation is a CONTINUATION, not a cold boot. The wake
  // inherits the dying socket's per-session watermarks, so its first act on
  // `open` is asking for the window the half-open socket could not deliver —
  // at the watermark the old socket actually reached, or every event of the
  // gap is silently lost (round-1 defect P1: zero `session.events.since`).
  it('wake replacement replays from the inherited watermark, not a cold boot', async () => {
    render(<Harness />)
    await flushAsync()
    const old = FakeWebSocket.instances.at(-1)!
    emitDelta(old, 1)
    emitDelta(old, 2)
    await flushAsync()

    FakeWebSocket.pingMode = 'silent'
    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(FakeWebSocket.instances.length).toBe(2)
    const since = sent.filter(frame => frame.method === 'session.events.since')
    expect(since.length).toBeGreaterThan(0)
    // The continuation proof: THIS session, at the cursor the old socket set.
    expect(
      since.some(frame => frame.params?.session_id === 's-cont' && frame.params?.last_seen === 2)
    ).toBe(true)
  })

  // #95327 with a DELIVERY assertion: "kept open" is worthless unless the kept
  // generation is still SERVED. The wake replaces the primary, a running turn
  // keeps the old socket alive — and the turn's next delta on THAT socket must
  // still reach the app, until the claim drains and the generation retires
  // (detach in the same breath as the close: listener lifetime = socket
  // lifetime; round-1 defect P2: the kept socket was deaf).
  it('#95327: a wake-kept generation stays served until its turn drains', async () => {
    const delivered: number[] = []

    render(
      <Harness
        handleGatewayEvent={event => {
          if (event.type === 'message.delta') {
            delivered.push(1)
          }
        }}
      />
    )
    await flushAsync()
    const old = FakeWebSocket.instances.at(-1)!

    act(() => {
      publishSessionState('rt-cont', {
        ...createClientSessionState(null),
        storedSessionId: 's-cont',
        busy: true
      })
    })
    emitDelta(old, 1)
    await flushAsync()
    expect(delivered.length).toBe(1)

    // Hard wake while the turn rides the old socket: replacement in, promoted,
    // and the old generation KEPT — busy work pins it.
    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })
    expect(FakeWebSocket.instances.length).toBe(2)
    expect(old.readyState).toBe(FakeWebSocket.OPEN)
    expect($gatewayState.get()).toBe('open')

    // The running turn keeps streaming on the socket the backend still writes
    // to — and the app still hears it.
    emitDelta(old, 2)
    await flushAsync()
    expect(delivered.length).toBe(2)

    // The turn drains: the kept generation closes, and closing it detaches it
    // — nothing more is delivered, and no listener outlives the socket.
    act(() => {
      publishSessionState('rt-cont', {
        ...createClientSessionState(null),
        storedSessionId: 's-cont',
        busy: false
      })
    })
    await flushAsync()
    expect(old.readyState).toBe(FakeWebSocket.CLOSED)
    emitDelta(old, 3)
    await flushAsync()
    expect(delivered.length).toBe(2)
  })

  // #94769's churn class moved onto the PRIMARY by round-1's wake: focus is a
  // SOFT signal — an alt-tab says nothing about the transport — so N focus
  // events on a healthy open socket must mint ZERO tickets and open ZERO
  // sockets (round-1 defect P3: 3 focus events, 3 replacement dials).
  it('#94769: plain focus never swaps a healthy primary socket', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    render(<Harness />)
    await flushAsync()
    const socketsAfterBoot = FakeWebSocket.instances.length
    const mintsAfterBoot = desktop.getGatewayWsUrl.mock.calls.length

    for (let altTab = 0; altTab < 3; altTab += 1) {
      act(() => window.dispatchEvent(new Event('focus')))
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2_000)
      })
    }

    expect(FakeWebSocket.instances.length).toBe(socketsAfterBoot)
    expect(desktop.getGatewayWsUrl.mock.calls.length).toBe(mintsAfterBoot)
    expect($gatewayState.get()).toBe('open')
    // And the socket the user was on is still the window's own.
    expect(FakeWebSocket.instances.at(-1)!.readyState).toBe(FakeWebSocket.OPEN)
  })
})
