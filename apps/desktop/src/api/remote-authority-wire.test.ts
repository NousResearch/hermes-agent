// @vitest-environment node
import { JsonRpcGatewayClient } from '@hermes/shared'
import { expect, test, vi } from 'vitest'

import {
  type FixtureFrame,
  type FixtureSocket,
  listen,
  onTransportState,
  publicUrl,
  readyFrame,
  reply,
  shut,
  transportOpened
} from '../test/ws-gateway-fixture'

import { GATEWAY_NOT_CONNECTED_MESSAGE, HermesGateway } from './client'

test('a public authority dial negotiates before create, submit and model mutation', async () => {
  const server = await listen()
  const sent: FixtureFrame[] = []
  server.on('connection', (socket: FixtureSocket) => {
    const ready = setTimeout(() => socket.send(readyFrame()), 30)
    socket.on('close', () => clearTimeout(ready))
    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)
      const params = frame.params ?? {}
      let result: Record<string, unknown> = {}
      let error: { code: number; message: string } | undefined

      if (frame.method === 'session.create') {
        if (params.source !== 'gui' || 'cols' in params || !params.request_id) {
          error = { code: 4001, message: 'invalid_params' }
        } else {
          result = {
            session_id: 's',
            stored_session_id: 's',
            revision: 1,
            execution_generation: 2,
            info: { desktop_protocol: 'hermes-gateway-v1', profile_name: 'worker' }
          }
        }
      } else if (frame.method === 'prompt.submit') {
        result = {
          admission_id: 'admitted',
          status: 'queued',
          ref: { session_id: 's', profile_id: '/profiles/worker' }
        }
      } else if (frame.method === 'config.set') {
        error = { code: 4001, message: 'use_session_mutation_model' }
      } else if (frame.method === 'session.mutate') {
        result = { session_id: 's', revision: 2, operation: 'model', model: 'picked' }
      }

      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, ...(error ? { error } : { result }) }))

      if (frame.method === 'prompt.submit') {
        socket.send(
          JSON.stringify({
            jsonrpc: '2.0',
            method: 'event',
            params: {
              type: 'message.start',
              session_id: 's',
              payload: {}
            }
          })
        )
      }
    })
  })
  const client = new HermesGateway()
  let incomingProfile: string | undefined
  client.on('message.start', event => {
    incomingProfile = event.profile
  })

  try {
    await client.connect(
      `ws://127.0.0.1:${(server.address() as { port: number }).port}/api/ws?ticket=remote-session-ticket`
    )
    expect(client.sessionWire).toBe('canonical')
    await expect(
      client.request('session.create', { source: 'desktop', profile: 'worker', cols: 96, fast: false })
    ).resolves.toMatchObject({ session_id: 's' })
    await expect(
      client.request('prompt.submit', { session_id: 's', profile: 'worker', text: 'hello', submission_id: 'one' })
    ).resolves.toMatchObject({ session_id: 's', submission_id: 'one', admission_id: 'admitted' })
    await vi.waitFor(() => expect(incomingProfile).toBe('worker'))
    await expect(
      client.request('config.set', {
        session_id: 's',
        profile: 'worker',
        key: 'model',
        value: 'picked --provider custom'
      })
    ).resolves.toMatchObject({ ok: true })
    expect(sent.find(frame => frame.method === 'session.mutate')?.params).toMatchObject({
      operation: 'model',
      expected_revision: 1
    })
    expect(sent.some(frame => frame.method === 'config.set')).toBe(false)
    // Remote auth stays ticket based: negotiating the wire grants no private native-dial credential.
    expect(sent.some(frame => frame.method === 'client.capabilities' && frame.params.desktop_local)).toBe(false)
  } finally {
    await shut(client, server)
  }
})

// A request made while the socket is open but not yet ready waits for the wire, inside its own
// abort signal and deadline; a deadline answers in the channel's own timeout text.
test.each(['abort', 'deadline', 'ready'] as const)('a request waiting for public readiness: %s', async exit => {
  const server = await listen()
  const sent: Array<{ method: string; afterReady: boolean }> = []

  let announceReady = () => {}
  server.on('connection', (socket: FixtureSocket) => {
    let ready = false

    announceReady = () => {
      ready = true
      socket.send(readyFrame())
    }
    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())
      sent.push({ method: frame.method, afterReady: ready })
      reply(socket, frame)
    })
  })
  const client = new HermesGateway()
  const opened = transportOpened(client)
  const connected = client.connect(publicUrl(server))
  void connected.catch(() => {})

  try {
    await opened
    const controller = new AbortController()

    const waiting =
      exit === 'abort'
        ? client.request('session.create', { source: 'desktop' }, undefined, controller.signal)
        : client.request('session.create', { source: 'desktop' }, exit === 'deadline' ? 40 : undefined)

    if (exit === 'ready') {
      announceReady()
      await waiting
      expect(sent.filter(frame => frame.method === 'session.create')).toEqual([
        { method: 'session.create', afterReady: true }
      ])

      return
    }

    if (exit === 'abort') {
      controller.abort()
    }
    await expect(waiting).rejects.toThrow(
      exit === 'abort' ? /abort/i : /^request timed out after \d+s: session\.create$/
    )
    announceReady()
    await connected
    await expect(client.request('ping')).resolves.toEqual({})
    expect(sent.map(frame => frame.method)).toContain('ping')
    expect(sent.map(frame => frame.method)).not.toContain('session.create')
  } finally {
    await shut(client, server)
  }
})

test("a closed-handler redial survives the previous socket's close until its own readiness", async () => {
  const server = await listen()
  let connections = 0
  server.on('connection', (socket: FixtureSocket) => {
    if (++connections === 1) {
      setTimeout(() => socket.terminate(), 20)

      return
    }

    setTimeout(() => socket.send(readyFrame()), 20)
    socket.on('message', bytes => reply(socket, JSON.parse(bytes.toString())))
  })
  const client = new HermesGateway()
  const url = publicUrl(server)
  let redial: Promise<void> | undefined
  let opened = false

  // A subscriber that redials from its `closed` handler, as the reconnect ladders do.
  const off = onTransportState(client, state => {
    if (state === 'open') {
      opened = true
    }

    if (state === 'closed' && opened && !redial) {
      redial = client.connect(url)
    }
  })

  try {
    await expect(client.connect(url)).rejects.toThrow('before protocol readiness')
    await vi.waitFor(() => expect(redial).toBeDefined())
    await expect(redial).resolves.toBeUndefined()
    expect(client.connectionState).toBe('open')
    await expect(client.request('ping')).resolves.toEqual({})
    expect(connections).toBe(2)
  } finally {
    off()
    await shut(client, server)
  }
})

// The request layer reconnects, and connector reads retry, only on the transport's own answers.
test.each(['refused', 'closed before ready'] as const)(
  'a public dial %s keeps the transport failure for connect and for requests',
  async failure => {
    const server = await listen()
    server.on('connection', (socket: FixtureSocket) => {
      setTimeout(() => socket.terminate(), 10)
    })
    const url = publicUrl(server)

    if (failure === 'refused') {
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
    const client = new HermesGateway()

    try {
      await expect(client.connect(url)).rejects.toThrow(
        failure === 'refused' ? 'Could not connect to Hermes gateway' : 'before protocol readiness'
      )
      await expect(client.request('config.get')).rejects.toThrow(GATEWAY_NOT_CONNECTED_MESSAGE)
    } finally {
      if (failure === 'refused') {
        client.close()
      } else {
        await shut(client, server)
      }
    }
  }
)

test('a readiness timeout fails within the connect budget and leaves transport errors behind it', async () => {
  const server = await listen()
  server.on('connection', (socket: FixtureSocket) => {
    socket.on('message', bytes => reply(socket, JSON.parse(bytes.toString())))
  })
  const client = new HermesGateway()
  const opened = transportOpened(client)
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })

  try {
    const rejected = expect(client.connect(publicUrl(server))).rejects.toThrow('protocol readiness timed out')
    await opened
    const waiting = expect(client.request('config.get', {}, 0)).rejects.toThrow(GATEWAY_NOT_CONNECTED_MESSAGE)
    await vi.advanceTimersByTimeAsync(15_001)
    await rejected
    await waiting
    expect(client.connectionState).toBe('closed')
    await expect(client.request('config.get')).rejects.toThrow(GATEWAY_NOT_CONNECTED_MESSAGE)
  } finally {
    vi.useRealTimers()
    await shut(client, server)
  }
})

// An open socket that never becomes ready is not a healthy connection: subscribers (the
// reconnect ladders' stable-open clocks) hear `open` only once the wire is known.
test('state subscribers hear open only after readiness', async () => {
  const server = await listen()

  let announceReady = () => {}
  server.on('connection', (socket: FixtureSocket) => {
    announceReady = () => socket.send(readyFrame())
  })
  const client = new HermesGateway()
  const heard: string[] = []
  const off = client.onState(state => {
    heard.push(state)
  })
  const opened = transportOpened(client)
  const connected = client.connect(publicUrl(server))

  try {
    await opened
    expect(heard).not.toContain('open')
    expect(client.connectionState).toBe('connecting')
    expect(client.sessionWire).toBeNull()
    announceReady()
    await connected
    expect(heard.filter(state => state === 'open')).toHaveLength(1)
    expect(client.connectionState).toBe('open')
  } finally {
    off()
    await shut(client, server)
  }
})

test('a never-ready peer closes without ever reporting open', async () => {
  const server = await listen()
  let peer: FixtureSocket | undefined
  server.on('connection', (socket: FixtureSocket) => {
    peer = socket
  })
  const client = new HermesGateway()
  const heard: string[] = []
  const off = client.onState(state => {
    heard.push(state)
  })
  const opened = transportOpened(client)
  const connected = expect(client.connect(publicUrl(server))).rejects.toThrow('before protocol readiness')

  try {
    await opened
    peer?.terminate()
    await connected
    expect(heard).not.toContain('open')
    expect(heard.at(-1)).toBe('closed')
  } finally {
    off()
    await shut(client, server)
  }
})

test('the first ready frame latches the wire for its connection', async () => {
  const server = await listen()
  const creates: Array<Record<string, unknown>> = []
  server.on('connection', (socket: FixtureSocket) => {
    socket.send(readyFrame())
    socket.send(readyFrame({}))
    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())

      if (frame.method === 'session.create') {
        creates.push(frame.params)
      }
      reply(socket, frame)
    })
  })
  const client = new HermesGateway()
  let readies = 0
  client.on('gateway.ready', () => {
    readies += 1
  })

  try {
    await client.connect(publicUrl(server))
    await vi.waitFor(() => expect(readies).toBe(2))
    expect(client.sessionWire).toBe('canonical')
    await client.request('session.create', { source: 'desktop', request_id: 'kept' })
    expect(creates[0]).toHaveProperty('request_id', 'kept')
  } finally {
    await shut(client, server)
  }
})

test('a settled readiness sends a request without waiting a turn', async () => {
  const server = await listen()
  server.on('connection', (socket: FixtureSocket) => {
    socket.send(readyFrame({}))
    socket.on('message', bytes => reply(socket, JSON.parse(bytes.toString())))
  })
  const client = new HermesGateway()
  const send = vi.spyOn(JsonRpcGatewayClient.prototype, 'request')

  try {
    await client.connect(publicUrl(server))
    expect(client.sessionWire).toBe('legacy')
    send.mockClear()
    const answered = client.request('ping')
    expect(send).toHaveBeenCalledTimes(1)
    await expect(answered).resolves.toEqual({})
  } finally {
    send.mockRestore()
    await shut(client, server)
  }
})

// Held `open` is delivered one handler at a time: an earlier handler can unsubscribe a later one,
// or close the client, and the later one must then not hear `open`.
test.each(['unsubscribe', 'close'] as const)(
  'a held open is not delivered after an earlier handler %ss',
  async action => {
    const server = await listen()

    let announceReady = () => {}
    server.on('connection', (socket: FixtureSocket) => {
      announceReady = () => socket.send(readyFrame())
    })
    const client = new HermesGateway()
    const heard: string[] = []

    let offLater = () => {}

    const offFirst = client.onState(state => {
      if (state !== 'open') {
        return
      }

      if (action === 'unsubscribe') {
        offLater()
      } else {
        client.close()
      }
    })

    offLater = client.onState(state => {
      heard.push(`${state}:${client.connectionState}`)
    })
    const opened = transportOpened(client)
    const connected = client.connect(publicUrl(server))
    void connected.catch(() => {})

    try {
      await opened
      announceReady()
      await connected.catch(() => {})
      await vi.waitFor(() => expect(client.connectionState).toBe(action === 'close' ? 'closed' : 'open'))
      expect(heard.filter(entry => entry.startsWith('open'))).toEqual([])
    } finally {
      offFirst()
      offLater()
      await shut(client, server)
    }
  }
)
