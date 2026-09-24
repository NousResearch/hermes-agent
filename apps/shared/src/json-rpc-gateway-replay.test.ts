import { beforeEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

/**
 * Minimal EventTarget-based WebSocket stand-in so the seq-tracking and
 * replay-resume logic can be driven with real dispatch semantics.
 */
class FakeWebSocket extends EventTarget {
  static OPEN = 1
  static instances: FakeWebSocket[] = []

  readyState = 0
  sent: string[] = []
  url: string

  constructor(url: string) {
    super()
    this.url = url
    FakeWebSocket.instances.push(this)
  }

  send(data: string): void {
    this.sent.push(data)
  }

  close(): void {
    this.readyState = 3
    this.dispatchEvent(new CloseEvent('close'))
  }

  // Test drivers
  open(): void {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
  }

  serverFrame(obj: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(obj) }))
  }

  lastRequest(): { id: string; method: string; params: Record<string, unknown> } {
    const last = this.sent[this.sent.length - 1]

    return JSON.parse(last ?? '{}')
  }
}

let sockets: FakeWebSocket[]

const makeClient = () => {
  const client = new JsonRpcGatewayClient({
    socketFactory: url => new FakeWebSocket(url) as unknown as WebSocket,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs: 1000
  })

  return client
}

describe('JsonRpcGatewayClient event-seq tracking + replay resume', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    sockets = FakeWebSocket.instances as unknown as FakeWebSocket[]
  })

  it('records per-session seq watermarks from live events', async () => {
    const client = makeClient()
    const p = client.connect('ws://x')
    sockets[0].open()
    await p

    sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 4 } })
    sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } }) // out of order / late
    sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'tool.start', session_id: 's2', seq: 9 } })
    sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'skin.changed' } }) // no sid/seq

    expect(client.getSeqWatermarks()).toEqual({ s1: 4, s2: 9 })
    client.close()
  })

  it('fetches replay on reconnect for sessions it has watermarks for', async () => {
    const client = makeClient()

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first

    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.start', session_id: 's1', seq: 1 } })
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 5 } })

    // Drop and reconnect.
    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    // The reconnect triggered a replay fetch — flush microtasks.
    await vi.waitFor(() => {
      const req = sock.lastRequest()
      expect(req.method).toBe('session.events.since')
      expect(req.params).toMatchObject({ session_id: 's1', last_seen: 5 })
    })

    client.close()
  })

  it('dispatches replayed events through the normal handler path', async () => {
    const client = makeClient()
    const seen: string[] = []
    client.on('tool.complete', e => seen.push(`live:${String((e.payload as { n?: number }).n)}`))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 3 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    await vi.waitFor(async () => {
      const req = sock.lastRequest()
      expect(req.method).toBe('session.events.since')
      // Answer the replay request with two missed events.
      sock.serverFrame({
        jsonrpc: '2.0',
        id: req.id,
        result: {
          events: [
            { type: 'tool.complete', session_id: 's1', seq: 4, payload: { n: 1 } },
            { type: 'tool.complete', session_id: 's1', seq: 5, payload: { n: 2 } }
          ],
          latest_seq: 5,
          truncated: false,
          count: 2
        }
      })
      await Promise.resolve()
      expect(seen).toEqual(['live:1', 'live:2'])
    })

    expect(client.getSeqWatermarks().s1).toBe(5)
    client.close()
  })

  it('does not attempt replay when nothing was ever observed', async () => {
    const client = makeClient()
    const p = client.connect('ws://x')
    sockets[0].open()
    await p
    // No events ever seen → close+reconnect must NOT fire a replay RPC.
    client.invalidate('drop')
    const p2 = client.connect('ws://x')
    sockets[sockets.length - 1].open()
    await p2
    await new Promise(r => setTimeout(r, 20))

    expect(sockets[sockets.length - 1].sent).toHaveLength(0)
    client.close()
  })

  it('replayed seqs advance watermarks but never regress them', async () => {
    const client = makeClient()
    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'status.update', session_id: 's1', seq: 10 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    await vi.waitFor(() => {
      expect(sock.lastRequest().method).toBe('session.events.since')
    })
    // Replay returns a STALE frame (seq 2 < watermark 10): watermark must hold.
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: { events: [{ type: 'status.update', session_id: 's1', seq: 2 }], latest_seq: 10, truncated: false, count: 1 }
    })
    await Promise.resolve()
    expect(client.getSeqWatermarks().s1).toBe(10)
    client.close()
  })

  it('rejects envelope-shaped replay elements (the #94219 server-shape bug)', async () => {
    const client = makeClient()
    const seen: string[] = []
    client.on('message.delta', () => seen.push('delta'))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 1 } })
    expect(seen).toEqual(['delta']) // the pre-drop live frame

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    await vi.waitFor(() => {
      expect(sock.lastRequest().method).toBe('session.events.since')
    })
    const req = sock.lastRequest()
    // Pre-fix servers returned FULL JSON-RPC envelopes. The client must not
    // dispatch those blindly — and this documents why the server now sends
    // bare event objects.
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: {
        events: [{ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } }],
        latest_seq: 2,
        truncated: false,
        count: 1
      }
    })
    await Promise.resolve()
    // Envelope-shaped replay elements must add nothing beyond the live frame.
    expect(seen).toEqual(['delta'])
    client.close()
  })

  it('holds live frames racing the replay fetch — no double dispatch, no skipped gap', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', e => seen.push((e as unknown as { seq: number }).seq))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } })
    expect(seen).toEqual([2]) // pre-drop live frame dispatches normally

    client.invalidate('drop')
    let openBarrier: Promise<boolean> | undefined
    client.onState(state => {
      if (state === 'open') {
        // Synchronous open listeners start history reads; the barrier must
        // already exist, and a live frame here must not beat the gap.
        openBarrier = client.sessionReplayBarrier('s1')
        sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 5 } })
      }
    })
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    expect(openBarrier).toBeInstanceOf(Promise)
    expect(client.sessionReplayBarrier('unobserved')).toBeUndefined()
    let settled = false

    const barrierChecked = openBarrier!.then(valid => {
      // Resolves only after replayed AND parked frames have dispatched.
      expect(valid).toBe(true)
      expect(seen).toEqual([2, 3, 4, 5, 6])
      settled = true
    })

    await Promise.resolve()
    expect(settled).toBe(false) // connect() does not wait for replay

    await vi.waitFor(() => {
      expect(sock.lastRequest().method).toBe('session.events.since')
    })

    // LIVE frames 5 and 6 arrive while the replay (which carries 3,4,5) is
    // still in flight. They must be parked, not dispatched ahead of the gap.
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 5 } })
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 6 } })
    expect(seen).toEqual([2]) // still only the pre-drop frame — 5/6 parked

    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: {
        events: [
          { type: 'message.delta', session_id: 's1', seq: 3 },
          { type: 'message.delta', session_id: 's1', seq: 4 },
          { type: 'message.delta', session_id: 's1', seq: 5 }
        ],
        latest_seq: 5,
        truncated: false,
        count: 3
      }
    })

    // In-order, exactly once: replayed 3,4,5 then the parked live 6 —
    // the parked duplicate of 5 is seq-gated out.
    await vi.waitFor(() => {
      expect(seen).toEqual([2, 3, 4, 5, 6])
    })
    await barrierChecked
    expect(client.sessionReplayBarrier('s1')).toBeUndefined()
    expect(client.getSeqWatermarks().s1).toBe(6)
    client.close()
  })

  it('restarts replay after its socket is invalidated before rejected cleanup runs', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', e => seen.push((e as unknown as { seq: number }).seq))

    const first = client.connect('ws://x')
    sockets[0].open()
    await first
    sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 1 } })

    client.invalidate('first drop')
    // History can arrive in the disconnected gap, before any replay exists.
    await expect(client.sessionReplayBarrier('s1')).resolves.toBe(false)
    const second = client.connect('ws://x')
    sockets[1].open()
    await second
    await vi.waitFor(() => expect(sockets[1].lastRequest().method).toBe('session.events.since'))
    const oldBarrier = client.sessionReplayBarrier('s1')

    // The old request rejects asynchronously after detach. Open the
    // replacement before its cleanup runs; it must own a fresh replay.
    client.invalidate('second drop')
    const third = client.connect('ws://x')
    sockets[2].open()
    const replacementBarrier = client.sessionReplayBarrier('s1')
    await third
    // A read waiting on the lost socket is abandoned; the replacement owns a new barrier.
    await expect(oldBarrier).resolves.toBe(false)
    expect(replacementBarrier).toBeInstanceOf(Promise)
    expect(replacementBarrier).not.toBe(oldBarrier)
    await vi.waitFor(() => expect(sockets[2].lastRequest().method).toBe('session.events.since'))

    const request = sockets[2].lastRequest()
    expect(request.params).toEqual({ session_id: 's1', last_seen: 1 })

    // A live frame racing the new replay is parked by the NEW hold; the stale
    // replay's cleanup must neither flush it nor advance the watermark past
    // the gap it never recovered.
    sockets[2].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 3 } })
    await Promise.resolve()
    expect(seen).toEqual([1])
    expect(client.getSeqWatermarks()).toEqual({ s1: 1 })

    sockets[2].serverFrame({
      jsonrpc: '2.0', id: request.id,
      result: { events: [{ type: 'message.delta', session_id: 's1', seq: 2 }], latest_seq: 2, truncated: false, count: 1 }
    })

    await vi.waitFor(() => expect(seen).toEqual([1, 2, 3]))
    await expect(replacementBarrier).resolves.toBe(true)
    expect(client.getSeqWatermarks()).toEqual({ s1: 3 })
    client.close()
  })

  it.each(['timeout', 'unsupported'] as const)(
    'settles sessions independently and releases fresh frames on replay %s',
    async fallback => {
      vi.useFakeTimers()
      const client = makeClient()
      const seen: string[] = []
      client.onEvent(event => seen.push(`${event.session_id}:${event.type}:${event.seq}`))

      try {
        const first = client.connect('ws://x')
        sockets[0].open()
        await first

        sockets[0].serverFrame({
          jsonrpc: '2.0',
          method: 'event',
          params: { type: 'gateway.ready', payload: { replay_epoch: 'stable' } }
        })

        for (const sid of ['slow', 'fast']) {
          sockets[0].serverFrame({
            jsonrpc: '2.0',
            method: 'event',
            params: { type: 'session.info', session_id: sid, seq: 1 }
          })
        }

        seen.length = 0
        client.invalidate()
        const second = client.connect('ws://x')
        sockets[1].open()
        await second
        const slow = client.sessionReplayBarrier('slow')!
        const fast = client.sessionReplayBarrier('fast')!
        expect(slow).toBeInstanceOf(Promise)
        expect(fast).toBeInstanceOf(Promise)
        const requests = sockets[1].sent.map(text => JSON.parse(text))
        const slowRequest = requests.find(request => request.params.session_id === 'slow')
        const fastRequest = requests.find(request => request.params.session_id === 'fast')
        const slowSettled = vi.fn()
        void slow.then(slowSettled)

        // Both ordinary parked starts and wire-marked replayed starts are fresh.
        for (const [sid, seq] of [
          ['slow', 2],
          ['fast', 3]
        ] as const) {
          sockets[1].serverFrame({
            jsonrpc: '2.0',
            method: 'event',
            params: { type: 'message.start', session_id: sid, seq, replayed: sid === 'fast' }
          })
        }

        sockets[1].serverFrame({
          jsonrpc: '2.0',
          method: 'event',
          params: { type: 'message.start', session_id: 'new', seq: 1 }
        })
        expect(seen).toEqual(['new:message.start:1'])

        const fastSettled = fast.then(valid => {
          expect(valid).toBe(true)
          expect(seen).toEqual(['new:message.start:1', 'fast:message.complete:2', 'fast:message.start:3'])
        })

        sockets[1].serverFrame({
          jsonrpc: '2.0',
          id: fastRequest.id,
          result: {
            epoch: 'stable',
            events: [{ type: 'message.complete', session_id: 'fast', seq: 2 }]
          }
        })
        await fastSettled
        expect(client.sessionReplayBarrier('fast')).toBeUndefined()
        expect(client.sessionReplayBarrier('slow')).toBe(slow)
        expect(slowSettled).not.toHaveBeenCalled()
        sockets[1].serverFrame({
          jsonrpc: '2.0',
          method: 'event',
          params: { type: 'message.delta', session_id: 'fast', seq: 4 }
        })
        expect(seen.at(-1)).toBe('fast:message.delta:4')

        if (fallback === 'timeout') {
          // The replay deadline is bounded independently of the 120s RPC default.
          await vi.advanceTimersByTimeAsync(9_999)
          expect(slowSettled).not.toHaveBeenCalled()
          await vi.advanceTimersByTimeAsync(1)
        } else {
          sockets[1].serverFrame({
            jsonrpc: '2.0',
            id: slowRequest.id,
            error: { code: -32601, message: 'method not found' }
          })
        }

        await expect(slow).resolves.toBe(true)
        expect(seen.at(-1)).toBe('slow:message.start:2')
        expect(client.sessionReplayBarrier('slow')).toBeUndefined()
        // Late replies after fallback cannot resurrect the old replay window.
        sockets[1].serverFrame({
          jsonrpc: '2.0',
          id: slowRequest.id,
          result: {
            events: [{ type: 'message.delta', session_id: 'slow', seq: 99 }]
          }
        })
        await Promise.resolve()
        expect(client.getSeqWatermarks()).toEqual({ slow: 2, fast: 4, new: 1 })
      } finally {
        client.close()
        vi.useRealTimers()
      }
    }
  )

  // A backend restart announces a new epoch over the still-open socket. No
  // replay can cover the old numbering, so waiting history reads must proceed
  // (the reconnect backstop read, #94779) instead of being silently dropped.
  it.each(['ready', 'response'] as const)(
    'lets waiting history reads proceed when %s announces a new epoch, after parked frames',
    async via => {
      const client = makeClient()
      const seen: number[] = []
      client.on('message.delta', event => seen.push(event.seq!))

      try {
        const first = client.connect('ws://x')
        sockets[0].open()
        await first
        sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'A' } } })

        for (const sid of ['s1', 's2']) {
          sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: sid, seq: 97 } })
        }

        client.invalidate()
        const second = client.connect('ws://x')
        sockets[1].open()
        await second
        const barriers = ['s1', 's2'].map(sid => client.sessionReplayBarrier(sid))
        const requests = sockets[1].sent.map(text => JSON.parse(text))
        // A fresh live frame parked behind the replay must survive the revoke.
        sockets[1].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 1 } })
        expect(seen).toEqual([97, 97])

        const seenAtResolve = Promise.all(barriers).then(valid => ({ valid, seen: [...seen] }))

        if (via === 'ready') {
          sockets[1].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'B' } } })
        } else {
          sockets[1].serverFrame({ jsonrpc: '2.0', id: requests[0].id, result: { events: [], epoch: 'B' } })
        }

        await expect(seenAtResolve).resolves.toEqual({ valid: [true, true], seen: [97, 97, 1] })
        expect(client.sessionReplayBarrier('s1')).toBeUndefined()
        expect(client.getSeqWatermarks()).toEqual({ s1: 1 })

        // A late old-epoch response cannot restore the revoked window.
        for (const request of requests) {
          sockets[1].serverFrame({
            jsonrpc: '2.0',
            id: request.id,
            result: { events: [{ type: 'message.delta', session_id: request.params.session_id, seq: 98 }], epoch: 'A' }
          })
        }

        await Promise.resolve()
        expect(seen).toEqual([97, 97, 1])
      } finally {
        client.close()
      }
    }
  )

  it('clears stale watermarks when the backend epoch changes (restart poisoning)', async () => {
    const client = makeClient()

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    // Learn epoch A and a high watermark.
    sock.serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'gateway.ready', payload: { replay_epoch: 'epoch-A' } }
    })
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 97 } })
    expect(client.getSeqWatermarks()).toEqual({ s1: 97 })

    // Backend restarts: reconnect, replay under a NEW epoch returns nothing
    // (fresh process, empty ring) — pre-fix the client kept watermark 97 and
    // silently believed it missed nothing, forever.
    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    await vi.waitFor(() => {
      expect(sock.lastRequest().method).toBe('session.events.since')
    })
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: { events: [], latest_seq: 0, truncated: false, count: 0, epoch: 'epoch-B' }
    })

    await vi.waitFor(() => {
      expect(client.getSeqWatermarks()).toEqual({})
    })

    // New-epoch events build fresh watermarks from scratch.
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 3 } })
    expect(client.getSeqWatermarks()).toEqual({ s1: 3 })
    client.close()
  })
})
