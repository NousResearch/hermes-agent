/**
 * #C3 (t_406dbcb3) integration proof: the reconnect critical path no longer
 * waits for the two list refreshes.
 *
 * Drives the REAL hook + REAL HermesGateway through the shared fake socket.
 * The scenario is the ladder reconnect after a drop, with a replay watermark
 * so the new generation opens an actual replay barrier — exactly the state in
 * which the old `await refreshHermesConfig(); await refreshSessions()` made
 * the composer wait for two full list round trips AFTER the socket was open.
 *
 * The pins:
 *   1. While the replay barrier is still outstanding, no refresh has fired
 *      (they wait behind the barrier, not merely behind connect).
 *   2. When the barrier settles, the C1 episode completes as usable — and the
 *      refresh pair has merely STARTED: the config fetch is still in flight
 *      and sessions has not even begun when the composer is already usable.
 *      (The awaited path would have flipped that: usable ⇒ lists landed.)
 *   3. The serial config→sessions order the old path had is preserved.
 */
import { act, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $gatewayState } from '@/store/session'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'
import {
  flushAsync,
  Harness,
  parseReconnectEpisode,
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
  id?: number | string
  method?: string
  params?: Record<string, unknown>
}

/**
 * Answer the frames the fake must answer for a generation to reach
 * application-usable: `client.capabilities` immediately (the first RESPONSE
 * frame is what stamps usable) and `session.events.since` only when the test
 * pulls the trigger (that response releases the replay barrier).
 */
function installResponder() {
  const original = FakeWebSocket.prototype.send
  const heldReplay: Array<{ socket: FakeWebSocket; id: number | string }> = []

  FakeWebSocket.prototype.send = function (this: FakeWebSocket, data: string) {
    let frame: SentFrame

    try {
      frame = JSON.parse(data) as SentFrame
    } catch {
      return original.call(this, data)
    }

    const socket = this

    if (frame.method === 'client.capabilities' && frame.id !== undefined) {
      const id = frame.id
      // Async like a real peer: the connect() continuation (finish →
      // schedule) runs before the first response arrives.
      setTimeout(() => {
        ;(socket as unknown as { emit: (t: string, e: unknown) => void }).emit('message', {
          data: JSON.stringify({ jsonrpc: '2.0', id, result: {} })
        })
      }, 0)
    } else if (frame.method === 'session.events.since' && frame.id !== undefined) {
      heldReplay.push({ socket, id: frame.id })
    }

    return original.call(this, data)
  }

  return {
    held: () => heldReplay.length,
    releaseReplay: () => {
      const pending = heldReplay.splice(0)

      for (const { socket, id } of pending) {
        setTimeout(() => {
          ;(socket as unknown as { emit: (t: string, e: unknown) => void }).emit('message', {
            data: JSON.stringify({ jsonrpc: '2.0', id, result: { events: [], latest_seq: 0 } })
          })
        }, 0)
      }
    },
    restore: () => {
      FakeWebSocket.prototype.send = original
    }
  }
}

/** Transcript frame with a seq, so the reconnecting client holds a watermark. */
function emitDelta(sock: FakeWebSocket, seq: number) {
  ;(sock as unknown as { emit: (type: string, ev: unknown) => void }).emit('message', {
    data: JSON.stringify({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's-defer', seq, payload: { text: `d${seq}` } }
    })
  })
}

function episodeLines(): Array<Record<string, string>> {
  const logLine = (window as unknown as { hermesDesktop: { logLine: ReturnType<typeof vi.fn> } }).hermesDesktop
    .logLine

  return logLine.mock.calls
    .map(call => parseReconnectEpisode(String(call[0])))
    .filter((line): line is Record<string, string> => line !== null)
}

describe('reconnect defers the list refreshes (#C3)', () => {
  beforeEach(() => {
    resetGatewayBootHarness()
  })

  afterEach(() => {
    teardownGatewayBootHarness()
  })

  it('completes the reconnect at usable while the lists are still in flight', async () => {
    const responder = installResponder()

    // The boot pair must resolve (boot legitimately awaits its lists); the
    // RECONNECT pair hangs on gates the test controls.
    const releaseRefreshes: Array<() => void> = []
    let configCalls = 0
    let sessionsCalls = 0

    const gated = (sink: Array<() => void>) =>
      new Promise<void>(resolve => {
        sink.push(resolve)
      })

    const refreshHermesConfig = vi.fn(async () => {
      configCalls += 1
      if (configCalls > 1) {
        await gated(releaseRefreshes)
      }
    })
    const refreshSessions = vi.fn(async () => {
      sessionsCalls += 1
      if (sessionsCalls > 1) {
        await gated(releaseRefreshes)
      }
    })

    render(<Harness refreshHermesConfig={refreshHermesConfig} refreshSessions={refreshSessions} />)
    await flushAsync()

    expect(configCalls).toBe(1)
    expect(sessionsCalls).toBe(1)
    expect($gatewayState.get()).toBe('open')

    // A watermark: the client saw seq 1, so the next generation's open opens
    // a REAL replay barrier (session.events.since outstanding).
    emitDelta(FakeWebSocket.instances.at(-1)!, 1)
    await flushAsync()

    // The socket dies; the ladder redials.
    act(() => {
      FakeWebSocket.instances.at(-1)!.drop()
    })
    await flushAsync()
    await vi.advanceTimersByTimeAsync(1000)
    await flushAsync()

    expect(FakeWebSocket.instances.length).toBe(2)
    expect(responder.held()).toBeGreaterThan(0)

    // 1. Barrier outstanding → nothing refreshed yet (usable arrived via the
    //    answered capabilities, but the refresh waits behind the barrier).
    expect(configCalls).toBe(1)
    expect(sessionsCalls).toBe(1)
    expect(episodeLines().some(line => line.outcome === 'usable')).toBe(false)

    // The replay answer releases the barrier: usable+barrier is the emit point
    // for the phases, the episode settles, and the deferred pair merely STARTS.
    responder.releaseReplay()
    await vi.advanceTimersByTimeAsync(5)
    await flushAsync()

    // 2. The episode completed — the composer is usable at the barrier — while
    //    the config fetch is STILL pending and sessions has not started.
    const usable = episodeLines().find(line => line.outcome === 'usable')

    expect(usable).toBeDefined()
    expect(configCalls).toBe(2)
    expect(sessionsCalls).toBe(1)
    expect($gatewayState.get()).toBe('open')

    // 3. The old serial order still holds once the in-flight fetch lands.
    act(() => {
      releaseRefreshes.splice(0).forEach(release => release())
    })
    await flushAsync()

    expect(sessionsCalls).toBe(2)
    responder.restore()
  })

  it('never fires a refresh before the reconnecting generation is usable', async () => {
    const responder = installResponder()
    const calls: string[] = []

    const refreshHermesConfig = vi.fn(async () => {
      calls.push('config')
    })
    const refreshSessions = vi.fn(async () => {
      calls.push('sessions')
    })

    render(<Harness refreshHermesConfig={refreshHermesConfig} refreshSessions={refreshSessions} />)
    await flushAsync()
    calls.length = 0

    emitDelta(FakeWebSocket.instances.at(-1)!, 3)
    await flushAsync()

    act(() => {
      FakeWebSocket.instances.at(-1)!.drop()
    })

    // Let the redial OPEN but answer nothing (the responder needs a timer tick
    // to answer capabilities; the barrier needs the test's trigger).
    await vi.advanceTimersByTimeAsync(1000)
    // The capabilities answer lands here: usable stamps, but the barrier for
    // session.events.since is held by the responder.
    await vi.advanceTimersByTimeAsync(1)
    await flushAsync()

    expect(calls).toEqual([])

    responder.releaseReplay()
    await vi.advanceTimersByTimeAsync(1)
    await flushAsync()

    expect(calls).toEqual(['config', 'sessions'])
    responder.restore()
  })
})
