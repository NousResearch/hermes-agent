// @vitest-environment node
import { expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

// andrexibiza F2: a canonical viewer that observed 1-2 and lost its socket while the owner kept
// 3-5 must replay that window to the SAME owner profile on reconnect. The shared client keeps the
// cursor and its replay epoch across the transport replacement; the epoch's owner must outlive the
// socket with it, or the replay is refused locally with no request and no gap.
test('a canonical reconnect replays the retained window to the epoch owner\'s profile', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sid = 'review-chat'
  const epoch = 'epoch-alpha'

  const frame = (seq: number, type: string, payload: Record<string, unknown>) =>
    ({ type, session_id: sid, seq, replay_epoch: epoch, payload })

  const ring = [
    frame(1, 'message.delta', { text: 'before ' }), frame(2, 'tool.start', { tool_id: 't0', name: 'read_file' }),
    frame(3, 'message.delta', { text: 'during ' }), frame(4, 'tool.start', { tool_id: 't1', name: 'read_file' }),
    frame(5, 'tool.complete', { tool_id: 't1', name: 'read_file' }), frame(6, 'message.delta', { text: 'after' })
  ]

  const calls: any[] = []
  const peers: any[] = []
  const push = (peer: any, event: unknown) => peer.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: event }))

  server.on('connection', (peer: any) => {
    peers.push(peer)
    peer.on('message', (bytes: Buffer) => {
      const request = JSON.parse(bytes.toString())
      calls.push(request)
      const { method, params } = request

      // The owner only answers its own profile; an unrouted replay is a different session.
      const result = method === 'session.events.since'
        ? (params.profile === 'alpha' && params.replay_epoch === epoch
            ? { events: ring.filter(event => event.seq > params.last_seen && event.seq <= 5), latest_seq: 5, replay_epoch: epoch, snapshot_required: false }
            : { events: [], latest_seq: 0, replay_epoch: 'other', snapshot_required: true })
        : { session_id: sid, revision: 1, execution_generation: 1, replay_epoch: epoch, last_sequence: peers.length === 1 ? 2 : 5, prompts: [] }

      peer.send(JSON.stringify({ jsonrpc: '2.0', id: request.id, result }))
    })
  })

  const client = new HermesGateway()
  const seen: Array<{ seq?: number; type: string; profile?: string }> = []
  client.onAny(event => { if (event.session_id === sid) { seen.push({ seq: event.seq, type: event.type, profile: event.profile }) } })
  const url = `ws://127.0.0.1:${(server.address() as { port: number }).port}/api/ws?native_dial=fixture&ticket=one-use`

  try {
    await client.connect(url)
    await client.request('session.resume', { session_id: sid, profile: 'alpha' })
    push(peers[0], ring[0])
    push(peers[0], ring[1])
    await vi.waitFor(() => expect(seen.map(event => event.seq)).toEqual([1, 2]))

    // The viewer's transport drops; the owner keeps producing 3-5 into its ring.
    const closed = new Promise<void>(resolve => client.onState(state => { if (state === 'closed') { resolve() } }))
    peers[0].terminate()
    await closed

    await client.connect(url)
    await vi.waitFor(() => expect(seen.map(event => event.seq)).toEqual([1, 2, 3, 4, 5]))
    push(peers[1], ring[5])
    await vi.waitFor(() => expect(seen.map(event => event.seq)).toEqual([1, 2, 3, 4, 5, 6]))

    const replays = calls.filter(call => call.method === 'session.events.since')
    expect(replays).toHaveLength(1)
    expect(replays[0].params).toMatchObject({ session_id: sid, profile: 'alpha', last_seen: 2, replay_epoch: epoch })
    expect(seen.filter(event => event.type === 'tool.complete')).toHaveLength(1)
    expect(new Set(seen.map(event => event.profile))).toEqual(new Set(['alpha']))
  } finally {
    client.close()

    for (const peer of server.clients) { peer.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
