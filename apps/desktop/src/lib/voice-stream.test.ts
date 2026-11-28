import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { openDictationStream } from './voice-stream'

// The wait for the final frame after `eos` is the last leg of a dictation with
// no bound of its own (regression test for #147333): without a deadline a live
// session that never answers (or whose frames never arrive) leaves `stop()`
// pending forever, which holds the composer in "transcribing" — every voice
// control disabled, `dictate()` a no-op — for the rest of the session. The host
// bounds its own provider flush (`tools/transcription_streaming.py`,
// `_FINALIZE_TIMEOUT_S`), so this deadline only ever covers a frame that never
// reached this renderer.

class FakeGatewaySocket {
  static OPEN = 1

  binaryType = ''
  closed = false
  readyState = 1
  sent: Array<ArrayBuffer | string> = []
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onopen: (() => void) | null = null

  close = vi.fn(() => {
    this.closed = true
    this.readyState = 3
  })

  send = vi.fn((data: ArrayBuffer | string) => {
    this.sent.push(data)
  })

  emitClose() {
    this.onclose?.()
  }

  emitMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }

  emitOpen() {
    this.onopen?.()
  }
}

let socket: FakeGatewaySocket

vi.mock('@/api/canonical-protocol', () => ({
  openGatewaySocket: () => socket
}))

vi.mock('@/lib/sibling-ws-url', () => ({
  resolveSiblingWsUrl: async () => 'ws://gateway.test/api/audio/transcribe-stream'
}))

const OWNER = { connectionId: 'gateway-a', profile: 'default' }

/** Open one live dictation session over the fake socket. */
async function openSession() {
  socket = new FakeGatewaySocket()
  const opening = openDictationStream(OWNER, 48_000)

  await vi.advanceTimersByTimeAsync(0)
  socket.emitOpen()

  const session = await opening

  expect(session).not.toBeNull()

  return session!
}

describe('openDictationStream finalize', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('resolves with the transcript from the final frame', async () => {
    const session = await openSession()
    const stopping = session.stop()

    expect(socket.sent[socket.sent.length - 1]).toBe(JSON.stringify({ eos: true }))

    socket.emitMessage({ transcript: '  hello world  ', type: 'final' })

    expect(await stopping).toBe('hello world')
    expect(socket.close).toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('fails the finalize once the deadline passes with no final frame', async () => {
    const session = await openSession()
    const stopping = session.stop()
    let rejected: unknown = null

    void stopping.catch(error => {
      rejected = error
    })

    // Still waiting one second short of the deadline.
    await vi.advanceTimersByTimeAsync(59_000)
    expect(rejected).toBeNull()

    await vi.advanceTimersByTimeAsync(1_000)

    expect(rejected).toBeInstanceOf(Error)
    expect((rejected as Error).message).toBe('live transcription timed out after 60s (no final frame)')
    // The session is done with: the socket is dropped and no timer is left behind.
    expect(socket.close).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('clears the deadline when the connection closes first', async () => {
    const session = await openSession()
    const stopping = session.stop()

    socket.emitClose()

    await expect(stopping).rejects.toThrow('live transcription connection closed')
    // Nothing left armed: a deadline surviving here would fire at a settled session.
    expect(vi.getTimerCount()).toBe(0)
  })

  it('drops the deadline on cancel', async () => {
    const session = await openSession()
    const stopping = session.stop()
    let settled = false

    void stopping.then(
      () => {
        settled = true
      },
      () => {
        settled = true
      }
    )

    session.cancel()

    expect(socket.close).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)

    await vi.advanceTimersByTimeAsync(60_000)
    // Cancel settles nothing: the caller already gave up on this take.
    expect(settled).toBe(false)
  })
})
