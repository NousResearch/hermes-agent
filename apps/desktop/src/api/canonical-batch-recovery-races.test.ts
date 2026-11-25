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
  let watermark = 0
  let registryWatermark: any = undefined
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
          last_sequence: watermark,
          clarify_batch_watermark: registryWatermark,
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
      watermark: (value: number) => { watermark = value },
      registryWatermark: (value: any) => { registryWatermark = value },
      emit: (type: string, payload: any, generation = 1, seq = 1, epoch = 'epoch') =>
        socket.send(
          JSON.stringify({
            jsonrpc: '2.0',
            method: 'event',
            params: { type, session_id: 's', replay_epoch: epoch, execution_generation: generation, seq, payload }
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


test('review: an opening newer than the authoritative snapshot is fresh', async () =>
 fixture(async f => {
  f.watermark(20)
  await f.client.request('session.resume', {session_id:'s',profile:'work'})
  f.emit('clarify.batch.request', batch, 1, 21)
  await expect.poll(() => f.cards.length).toBe(1)
  expect(f.cards[0].replayed).toBe(false)
 }))

test('review: live/snapshot overlap is one silent restored card', async () =>
 fixture(async f => {
  f.watermark(20)
  f.snapshot([{...batch,replayed:true,answers:{q0:'red'}}])
  await f.client.request('session.resume', {session_id:'s',profile:'work'})
  f.emit('clarify.batch.request', batch, 1, 19)
  await new Promise(r=>setTimeout(r,60))
  expect(f.cards).toHaveLength(1)
  expect(f.cards[0].replayed).toBe(true)
  expect(f.cards[0].params.answers).toEqual({q0:'red'})
 }))

test('review: authoritative empty snapshot fences a queued older opening before cancellation arrives', async () =>
 fixture(async f => {
  f.watermark(20)
  f.snapshot([])
  await f.client.request('session.resume', {session_id:'s',profile:'work'})
  f.emit('clarify.batch.request', batch, 1, 19)
  await new Promise(r=>setTimeout(r,60))
  // The authoritative read is after this request's terminal settlement. The
  // fanout mailbox can lag its request-local reply. A later cancellation must
  // not be needed to avoid rendering/attention for an already closed batch.
  expect(f.cards).toHaveLength(0)
 }))

test('review: old renderer lock capability is invalid after reconnect restores same request', async () =>
 fixture(async f => {
  f.emit('clarify.batch.request', batch, 1, 1)
  await expect.poll(() => f.cards.length).toBe(1)
  const oldId = f.cards[0].id
  const oldRendererLock = () => f.client.request('clarify.lock', {
   request_id:oldId,question_id:'q0',answer:'stale pre-reconnect draft'
  })
  f.snapshot([{...batch,replayed:true}])
  await f.reconnect()
  expect(f.cards).toHaveLength(2)
  const before=f.calls.filter((c:any)=>c.method==='clarify.batch.lock').length
  await expect(oldRendererLock()).rejects.toThrow()
  expect(f.calls.filter((c:any)=>c.method==='clarify.batch.lock')).toHaveLength(before)
 }))


test('registry watermark, not earlier attach cursor, fences a queued opening', async () => fixture(async f => {
  f.registryWatermark({replay_epoch:'epoch',last_sequence:20})
  await f.client.request('session.resume',{session_id:'s',profile:'work'})
  f.emit('clarify.batch.request',batch,1,19)
  await new Promise(r=>setTimeout(r,60))
  expect(f.cards).toHaveLength(0)
}))

test('new stream/replay epoch never inherits the prior epoch numeric watermark', async () => fixture(async f => {
  f.watermark(20)
  await f.client.request('session.resume',{session_id:'s',profile:'work'})
  f.emit('clarify.batch.request',{...batch,identity:{...identity,stream_id:'replacement',authority_epoch:2}},1,1,'replacement-epoch')
  await expect.poll(()=>f.cards.length).toBe(1)
  expect(f.cards[0].replayed).toBe(false)
}))

test('another owner with equal session ID does not inherit recovery watermark', async () => fixture(async f => {
  f.watermark(20)
  await f.client.request('session.resume',{session_id:'s',profile:'work'})
  f.watermark(0)
  await f.client.request('session.resume',{session_id:'s',profile:'beta'})
  f.emit('clarify.batch.request',{...batch,identity:{...identity,profile_id:'/profiles/beta'}},1,1,'beta-epoch')
  await expect.poll(()=>f.cards.length).toBe(1)
}))

test('newer opening during delayed empty read is retained; queued older opening is not', async () => fixture(async f => {
  f.watermark(20); f.hold()
  const read=f.client.request('session.resume',{session_id:'s',profile:'work'})
  await new Promise(r=>setTimeout(r,30))
  f.emit('clarify.batch.request',{...batch,request_id:'newer'},1,21)
  await expect.poll(()=>f.cards.length).toBe(1)
  f.release();await read
  expect(f.events.filter((e:any)=>e.type==='request.cancel')).toHaveLength(0)
  f.emit('clarify.batch.request',batch,1,19)
  await new Promise(r=>setTimeout(r,60))
  expect(f.cards).toHaveLength(1)
}))

test('same-identity replacement revokes old lock ID and withdraws exactly the old card', async () => fixture(async f => {
  f.emit('clarify.batch.request',batch,1,1)
  await expect.poll(()=>f.cards.length).toBe(1)
  const old=f.cards[0]
  f.watermark(20);f.snapshot([{...batch,replayed:true,answers:{q0:'red'}}])
  await f.client.request('session.resume',{session_id:'s',profile:'work'})
  expect(f.cards).toHaveLength(2)
  expect(f.cards[1].id).not.toBe(old.id)
  expect(f.events.filter((e:any)=>e.type==='request.cancel').map((e:any)=>e.payload.id)).toEqual([old.id])
  const count=f.calls.filter((c:any)=>c.method==='clarify.batch.lock').length
  await expect(f.client.request('clarify.lock',{request_id:old.id,question_id:'q1',answer:'stale'})).rejects.toThrow()
  expect(f.calls.filter((c:any)=>c.method==='clarify.batch.lock')).toHaveLength(count)
  f.reply({status:'ok',remaining:[]})
  await f.client.request('clarify.lock',{request_id:f.cards[1].id,question_id:'q1',answer:'new'})
  expect(f.calls.filter((c:any)=>c.method==='clarify.batch.lock')).toHaveLength(count+1)
}))


test('accepted partial lock during delayed recovery is not overwritten by the older snapshot', async () => fixture(async f => {
  f.emit('clarify.batch.request',batch,1,1)
  await expect.poll(()=>f.cards.length).toBe(1)
  f.watermark(20);f.snapshot([{...batch,replayed:true}]);f.hold()
  const read=f.client.request('session.resume',{session_id:'s',profile:'work'})
  await new Promise(r=>setTimeout(r,30))
  await f.client.request('clarify.lock',{request_id:f.cards[0].id,question_id:'q0',answer:'red'})
  f.release();await read
  expect(f.cards).toHaveLength(1)
  expect(f.events.filter((e:any)=>e.type==='request.cancel')).toHaveLength(0)
}))
