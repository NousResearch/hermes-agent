// @vitest-environment node
import { expect, test } from 'vitest'

import { type FixtureFrame, type FixtureSocket, listen, publicUrl, readyFrame, reply, shut } from '../test/ws-gateway-fixture'

import { HermesGateway } from './client'

const controls = [
  { method: 'session.redirect', status: 'redirected' },
  { method: 'session.steer', status: 'queued' }
] as const

// Execution snapshots are learned per socket and owner profile, not just the stored ID.
test.each(controls)('$method attaches to the current owner before fencing a busy correction', async ({ method, status }) => {
  const server = await listen()
  const frames: FixtureFrame[] = []
  let connection = 0
  server.on('connection', (socket: FixtureSocket) => {
    const dial = ++connection

    socket.on('message', bytes => {
      const frame: FixtureFrame = JSON.parse(bytes.toString())

      if (frame.method !== 'client.capabilities') { frames.push(frame) }
      const profile = String(frame.params.profile)
      const generation = profile === 'alpha' ? (dial === 1 ? 7 : 11) : 29

      if (frame.method === 'session.resume') {
        reply(socket, frame, {
          session_id: 'same', stored_session_id: 'same', revision: 2, execution_generation: generation,
          authority_epoch: 1, replay_epoch: `${profile}-${dial}`, last_sequence: 0,
          subscription_id: `${profile}-${dial}`, running: true, message_count: 0,
          messages: [], pending: [], prompts: [], info: { profile_name: profile, desktop_protocol: 'hermes-gateway-v1' }
        })
      } else if (frame.method === method) {
        const reason = frame.params.execution_generation !== generation ? 'stale_generation' : null

        if (reason) {
          socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, error: { code: 4001, message: reason, data: { reason } } }))
        } else {
          reply(socket, frame, { status, text: frame.params.text })
        }
      } else {
        reply(socket, frame)
      }
    })
    socket.send(readyFrame())
  })
  const client = new HermesGateway()
  const url = publicUrl(server)
  const text = 'Keep the valid work; check café 🧭\nthen continue.'

  const correct = (profile: string, execution_generation?: number) => client.request(method, {
    session_id: 'same', profile, text, ...(execution_generation === undefined ? {} : { execution_generation })
  })

  const sent = () => frames.filter(frame => frame.method === method || frame.method === 'session.resume')

  try {
    // No native_dial shortcut, prior resume, or stream event supplies the generation.
    await client.connect(url)
    expect(client.sessionWire).toBe('canonical')
    await expect(correct('alpha')).resolves.toEqual({ status, text })
    expect(sent().map(frame => frame.method)).toEqual(['session.resume', method])
    expect(sent()[0].params).toEqual({ session_id: 'same', profile: 'alpha', defer_history: true, omit_messages: true })
    expect(sent()[1].params).toEqual({ session_id: 'same', profile: 'alpha', text, execution_generation: 7 })

    frames.length = 0
    await expect(correct('alpha')).resolves.toEqual({ status, text })
    expect(sent().map(frame => frame.method)).toEqual([method])

    // Equal stored IDs in different profile islands must attach separately. An explicit
    // observed generation is never silently upgraded to the snapshot's newer execution.
    frames.length = 0
    await expect(correct('beta', 12)).rejects.toThrow('stale_generation')
    expect(sent().map(frame => frame.method)).toEqual(['session.resume', method])
    expect(sent()[0].params.profile).toBe('beta')
    expect(sent()[1].params).toEqual({ session_id: 'same', profile: 'beta', text, execution_generation: 12 })
    await expect(correct('beta')).resolves.toEqual({ status, text })
    await expect(correct('alpha')).resolves.toEqual({ status, text })
    expect(sent().filter(frame => frame.method === method).map(frame => frame.params.execution_generation)).toEqual([12, 29, 7])
    expect(sent().filter(frame => frame.method === 'session.resume')).toHaveLength(1)

    client.close()
    await client.connect(url)
    frames.length = 0
    await expect(correct('alpha')).resolves.toEqual({ status, text })
    expect(connection).toBe(2)
    expect(sent().map(frame => frame.method)).toEqual(['session.resume', method])
    expect(sent()[1].params).toEqual({ session_id: 'same', profile: 'alpha', text, execution_generation: 11 })
  } finally {
    await shut(client, server)
  }
})

// Attachment refusal is terminal for canonical controls; legacy peers still get the original verb.
test.each(controls.flatMap(control => [
  { ...control, wire: 'canonical' as const },
  { ...control, wire: 'legacy' as const }
]))('$method respects $wire attachment admission', async ({ method, status, wire }) => {
  const server = await listen()
  const frames: FixtureFrame[] = []
  server.on('connection', (socket: FixtureSocket) => {
    socket.on('message', bytes => {
      const frame: FixtureFrame = JSON.parse(bytes.toString())

      if (frame.method !== 'client.capabilities') { frames.push(frame) }

      if (frame.method === 'session.resume') {
        socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, error: { code: 4001, message: 'permission_denied' } }))
      } else {
        reply(socket, frame, { status, text: frame.params.text })
      }
    })
    socket.send(readyFrame({ session_authority: wire === 'canonical' }))
  })
  const client = new HermesGateway()

  const params = { session_id: 'same', profile: 'alpha', text: 'Preserve this correction',
    ...(wire === 'canonical' ? { execution_generation: 7 } : {}) }

  try {
    await client.connect(publicUrl(server))
    expect(client.sessionWire).toBe(wire)

    if (wire === 'canonical') {
      await expect(client.request(method, params)).rejects.toThrow('permission_denied')
      expect(frames.map(frame => frame.method)).toEqual(['session.resume'])
      expect(frames[0].params).toEqual({ session_id: 'same', profile: 'alpha', defer_history: true, omit_messages: true })
    } else {
      await expect(client.request(method, params)).resolves.toEqual({ status, text: params.text })
      expect(frames.map(frame => frame.method)).toEqual([method])
      expect(frames[0].params).toEqual(params)
    }
  } finally {
    await shut(client, server)
  }
})
