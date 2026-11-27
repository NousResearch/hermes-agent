// @vitest-environment node
import { expect, test } from 'vitest'
import type { ServerRequest } from '@hermes/shared'
import { HermesGateway } from './client'

const identity = { profile_id: '/profiles/work', authority_epoch: 1, stream_id: 'stream', execution_generation: 1 }
const batch = {
  request_id: 'srq-b',
  identity,
  questions: [
    { qid: 'q0', question: 'Colour?', choices: ['red', 'blue'], multi_select: true },
    { qid: 'q1', question: 'Size?', multi_select: false }
  ],
  answers: {},
  replayed: false
}

async function fixture(run: (f: any) => Promise<void>) {
  const ws = 'ws'
  const { WebSocketServer } = await import(ws)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const calls: any[] = []
  const cards: ServerRequest[] = []
  const events: any[] = []
  let socket: any
  let snapshots: any[] = []
  let delay: (() => void) | undefined
  let mutationReply: any = undefined
  server.on('connection', (s: any) => {
    socket = s
    s.send(
      JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: { type: 'gateway.ready', payload: { session_authority: true } }
      })
    )
    s.on('message', (bytes: Buffer) => {
      const f = JSON.parse(bytes.toString())
      if (!f.method) return
      calls.push(f)
      const reply = (result: any) => s.send(JSON.stringify({ jsonrpc: '2.0', id: f.id, result }))
      if (f.method === 'client.capabilities')
        return reply({ canonical_clarify_batch: true, canonical_clarify_projection_v1: true })
      if (f.method === 'session.resume' || f.method === 'session.events.since') {
        const result = {
          session_id: 's',
          revision: 1,
          execution_generation: 1,
          replay_epoch: f.params.profile === 'beta' ? 'beta-epoch' : 'epoch',
          last_sequence: 0,
          info: {
            profile_id: f.params.profile === 'beta' ? '/profiles/beta' : '/profiles/work',
            profile_name: f.params.profile === 'beta' ? 'beta' : 'work'
          },
          prompts: [],
          clarify_batches: snapshots
        }
        if (delay) {
          const keep = delay
          delay = () => {
            reply(result)
            keep()
          }
          return
        }
        return reply(result)
      }
      if (mutationReply !== undefined) return reply(mutationReply)
      return reply(f.method === 'clarify.batch.lock' ? { status: 'ok', remaining: ['q1'] } : { status: 'ok' })
    })
  })
  const client = new HermesGateway()
  client.onRequest(r => {
    cards.push(r)
    return true
  })
  client.onAny(e => events.push(e))
  try {
    await client.connect(`ws://127.0.0.1:${(server.address() as any).port}/api/ws`)
    await expect.poll(() => calls.filter(f => f.params?.canonical_clarify_projection_v1).length).toBe(1)
    await client.request('session.resume', { session_id: 's', profile: 'work' })
    await run({
      client,
      calls,
      cards,
      events,
      emit: (type: string, payload: any, generation = 1) =>
        socket.send(
          JSON.stringify({
            jsonrpc: '2.0',
            method: 'event',
            params: { type, session_id: 's', replay_epoch: 'epoch', execution_generation: generation, payload }
          })
        ),
      snapshot: (value: any[]) => {
        snapshots = value
      },
      reply: (value: any) => {
        mutationReply = value
      },
      reconnect: async () => {
        client.close()
        await client.connect(`ws://127.0.0.1:${(server.address() as any).port}/api/ws`)
        await new Promise(r => setTimeout(r, 40))
        await client.request('session.resume', { session_id: 's', profile: 'work' })
      },
      hold: () => {
        delay = () => {}
      },
      release: () => delay?.()
    })
  } finally {
    client.close()
    for (const s of server.clients) s.terminate()
    await new Promise<void>(r => server.close(() => r()))
  }
}

test.each([{ status: 'ok', remaining: [] }, { status: 'expired' }])(
  'terminal lock %j invalidates old callbacks and blocks replay resurrection',
  async reply =>
    fixture(async f => {
      f.emit('clarify.batch.request', batch)
      await expect.poll(() => f.cards.length).toBe(1)
      f.reply(reply)
      await f.client.request('clarify.lock', { request_id: f.cards[0].id, question_id: 'q0', answer: null })
      f.cards[0].respond({ answers: {}, outcome: 'cancelled' })
      f.emit('clarify.batch.request', batch)
      await new Promise(r => setTimeout(r, 40))
      expect(f.calls.filter((c: any) => c.method === 'clarify.batch.respond')).toHaveLength(0)
      expect(f.cards).toHaveLength(1)
    })
)

test.each([
  { status: 'ok', remaining: ['q1'], unknown: true },
  { status: 'expired', remaining: 42 },
  { status: 'ok', remaining: ['q1', 'q1'] },
  { status: 'ok', remaining: ['q0'] }
])('malformed acknowledgement %j recovers by read without retrying the write', async reply =>
  fixture(async f => {
    f.emit('clarify.batch.request', batch)
    await expect.poll(() => f.cards.length).toBe(1)
    f.reply(reply)
    const resumes = f.calls.filter((c: any) => c.method === 'session.resume').length
    await expect(
      f.client.request('clarify.lock', { request_id: f.cards[0].id, question_id: 'q0', answer: '' })
    ).rejects.toThrow('Malformed')
    expect(f.calls.filter((c: any) => c.method === 'clarify.batch.lock')).toHaveLength(1)
    expect(f.calls.filter((c: any) => c.method === 'session.resume')).toHaveLength(resumes + 1)
  })
)

test('reconnect restores locks without fresh attention and rejects the old card callback', async () =>
  fixture(async f => {
    f.emit('clarify.batch.request', batch)
    await expect.poll(() => f.cards.length).toBe(1)
    const old = f.cards[0]
    f.snapshot([{ ...batch, answers: { q0: null }, replayed: true }])
    await f.reconnect()
    expect(f.cards).toHaveLength(2)
    expect(f.cards[1].replayed).toBe(true)
    old.respond({ answers: { q1: 'stale' }, outcome: 'submitted' })
    await new Promise(r => setTimeout(r, 30))
    expect(f.calls.filter((c: any) => c.method === 'clarify.batch.respond')).toHaveLength(0)
    expect(f.cards[1].params.answers).toEqual({ q0: null })
  }))

test.each([
  { ...batch, identity: { ...identity, profile_id: '/foreign' } },
  { ...batch, identity: { ...identity, execution_generation: true } },
  { ...batch, questions: [batch.questions[0], batch.questions[0]] },
  { ...batch, answers: { unknown: 'x' } },
  { ...batch, unknown: true },
  { ...batch, questions: [{ ...batch.questions[0], qid: ' q0' }] },
  { ...batch, questions: [{ ...batch.questions[0], question: '  ' }] }
])('malformed or foreign projection is not a card', async value =>
  fixture(async f => {
    f.emit('clarify.batch.request', value)
    await new Promise(r => setTimeout(r, 40))
    expect(f.cards).toHaveLength(0)
  })
)

test('opening envelope and captured generation must agree', async () =>
  fixture(async f => {
    f.emit('clarify.batch.request', batch, 2)
    await new Promise(r => setTimeout(r, 40))
    expect(f.cards).toHaveLength(0)
  }))

test('generated optional choices and multi-select fields preserve native defaults', async () =>
  fixture(async f => {
    f.emit('clarify.batch.request', { ...batch, questions: [{ qid: 'q0', question: 'Default?', choices: null }] })
    await expect.poll(() => f.cards.length).toBe(1)
  }))

test('local withdrawal retains its captured owner when equal session IDs are attached', async () =>
  fixture(async f => {
    f.emit('clarify.batch.request', batch)
    await expect.poll(() => f.cards.length).toBe(1)
    await f.client.request('session.resume', { session_id: 's', profile: 'beta' })
    f.client.close()
    expect(f.events.some((e: any) => e.type === 'request.cancel' && e.payload.id === f.cards[0].id)).toBe(true)
  }))

test('whole canonical batch retains native panel locks and full structured result', async () =>
  fixture(async f => {
    f.emit('clarify.batch.request', batch)
    await expect.poll(() => f.cards.length).toBe(1)
    const card = f.cards[0]
    expect(card.method).toBe('clarify')
    expect(card.params.questions).toEqual(batch.questions)
    await f.client.request('clarify.lock', { request_id: card.id, question_id: 'q0', answer: ['red', 'blue'] })
    expect(f.calls.find((c: any) => c.method === 'clarify.batch.lock').params).toEqual({
      session_id: 's',
      profile: 'work',
      request_id: 'srq-b',
      identity,
      question_id: 'q0',
      answer: ['red', 'blue']
    })
    card.respond({ answers: { q1: '' }, outcome: 'submitted' })
    await expect.poll(() => f.calls.filter((c: any) => c.method === 'clarify.batch.respond').length).toBe(1)
    expect(f.calls.find((c: any) => c.method === 'clarify.batch.respond').params.result).toEqual({
      answers: { q1: '' },
      outcome: 'submitted'
    })
  }))

test('snapshot restores accepted locks silently and exact cancellation fences delayed recovery', async () =>
  fixture(async f => {
    f.snapshot([{ ...batch, answers: { q0: '["red"]' }, replayed: true }])
    await f.client.request('session.resume', { session_id: 's', profile: 'work' })
    expect(f.cards).toHaveLength(1)
    expect(f.cards[0].replayed).toBe(true)
    expect(f.cards[0].params.answers).toEqual({ q0: '["red"]' })
    f.hold()
    const pending = f.client.request('session.resume', { session_id: 's', profile: 'work' })
    await new Promise(r => setTimeout(r, 30))
    f.emit('request.cancel', { id: batch.request_id, method: 'clarify', reason: 'resolved', identity })
    await new Promise(r => setTimeout(r, 30))
    f.release()
    await pending
    expect(f.cards).toHaveLength(1)
    expect(f.events.filter((e: any) => e.type === 'request.cancel').at(-1).payload.id).toBe(f.cards[0].id)
    const writes = f.calls.filter((c: any) => c.method.startsWith('clarify.')).length
    f.cards[0].respond({ answers: { q1: 'late' }, outcome: 'submitted' })
    await new Promise(r => setTimeout(r, 30))
    expect(f.calls.filter((c: any) => c.method.startsWith('clarify.'))).toHaveLength(writes)
  }))
