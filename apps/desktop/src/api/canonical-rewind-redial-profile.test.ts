// @vitest-environment node
import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// Review (#141492, RT-40): on a named launch profile the owner names itself in the snapshot's
// `info.profile_name`, and a redial forgets that route. A repeated /undo or /retry after a lost
// rewind reply must still find the retained rewind (keyed under the named profile) and re-present
// it: the replay-only probe is answered locally only after the attach has re-learned the owner.
test('a named-profile owner re-presents a lost rewind after a redial', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const mutations: any[] = []
  let revision = 4

  server.on('connection', (peer: any) => {
    peer.on('message', (bytes: Buffer) => {
      const { id, method, params } = JSON.parse(bytes.toString())

      if (method === 'session.mutate') {
        mutations.push(params)

        // The first rewind commits on the owner, and its reply is lost with the socket.
        if (mutations.length === 1) {
          revision += 1

          return peer.terminate()
        }
      }

      const result =
        method === 'session.mutate'
          ? {
              session_id: 's',
              revision,
              operation: 'rewind',
              rewound_count: 2,
              target_message: { id: 14, content: 'second question' }
            }
          : {
              session_id: 's',
              stored_session_id: 's',
              revision,
              execution_generation: 2,
              info: { profile_name: 'work' },
              messages: [],
              prompts: []
            }

      peer.send(JSON.stringify({ jsonrpc: '2.0', id, result }))
    })
  })

  const client = new HermesGateway()
  const url = `ws://127.0.0.1:${(server.address() as { port: number }).port}/api/ws?native_dial=fixture&ticket=one-use`

  try {
    await client.connect(url)
    await client.request('session.resume', { session_id: 's' })

    const closed = new Promise<void>(resolve =>
      client.onState(state => {
        if (state === 'closed') {
          resolve()
        }
      })
    )

    await expect(
      client.request('session.rewind', { session_id: 's', target_message_id: 14, derived_target: true })
    ).rejects.toThrow()
    await closed

    await client.connect(url)
    const replayed = await client.request<{ rewound_count?: number }>('session.rewind', {
      session_id: 's',
      derived_target: true,
      replay_only: true
    })

    expect(mutations).toHaveLength(2)
    expect(mutations[1]).toEqual(mutations[0])
    expect(mutations[1]).toMatchObject({ profile: 'work', operation: 'rewind', payload: { target_message_id: 14 } })
    expect(replayed.rewound_count).toBe(2)
  } finally {
    client.close()

    for (const peer of server.clients) {
      peer.terminate()
    }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
