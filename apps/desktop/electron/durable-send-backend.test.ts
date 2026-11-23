import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { expect, test } from 'vitest'

import { drainDurableSends } from './durable-send-delivery'
import { durableRpcTransport } from './durable-send-rpc'
import { DurableSendStore } from './durable-send-store'
import { createDurableBackendPeer as peer } from './durable-send-test-peer'
import type { DurableSendIntent, DurableSendOwner } from './durable-send-types'

async function ownerOf(p: ReturnType<typeof peer>): Promise<DurableSendOwner> {
  return (
    await durableRpcTransport(p.rpc).describe({
      connectionId: 'fixture',
      profile: 'default',
      gatewayId: '',
      principalId: ''
    })
  ).owner
}

function enqueue(root: string, owner: DurableSendOwner, target?: string) {
  const store = new DurableSendStore(root, 'fixture-origin')
  const content = Buffer.from('original attachment bytes — not a pathname or thumbnail')
  const source = path.join(root, 'source.txt')
  fs.writeFileSync(source, content)

  const file = {
    id: randomUUID(),
    name: 'source.txt',
    mediaType: 'text/plain',
    size: content.length,
    sha256: createHash('sha256').update(content).digest('hex')
  }

  const intent: DurableSendIntent = {
    id: randomUUID(),
    owner,
    text: 'Read the preserved attachment',
    files: [file],
    target: target
      ? { sessionId: target }
      : { draftId: randomUUID(), createRequestId: randomUUID(), params: { source: 'desktop' } }
  }

  store.begin(intent)
  store.append(intent.id, file.id, 0, fs.readFileSync(source))
  store.commit(intent.id)
  fs.unlinkSync(source)

  return { store, intent, content }
}

test('real protocol: offline enqueue, deleted source, upload ACK loss, both restarts, admission ACK loss and exact replay', async () => {
  const p = peer()
  const owner = await ownerOf(p)
  const { store, intent, content } = enqueue(p.root, owner)
  const transport = durableRpcTransport(p.rpc)
  p.online(false)
  await drainDurableSends(store, transport)
  expect(store.read(intent.id).state).toBe('waiting')
  expect(await p.raw('fixture.ledger', { request_id: intent.id })).toEqual([])
  p.online(true)
  p.drop('durable.create')
  await drainDurableSends(store, transport, Date.now() + 50_000)
  expect(store.read(intent.id)).toMatchObject({ state: 'waiting' })
  expect(store.read(intent.id).sessionId).toBeUndefined()
  p.drop('durable.upload')
  await drainDurableSends(store, transport, Date.now() + 100_000)
  expect(store.read(intent.id).state).toBe('waiting')
  const sessionId = store.read(intent.id).sessionId!
  await p.raw('fixture.restart')
  const restored = new DurableSendStore(p.root, 'fixture-origin')
  p.drop('durable.submit')
  await drainDurableSends(restored, transport, Date.now() + 200_000)
  expect(restored.read(intent.id).submitStarted).toBe(true)
  expect(restored.read(intent.id).state).toBe('waiting')
  const before = await p.raw('fixture.ledger', { request_id: intent.id })
  expect(before).toHaveLength(1)
  const text = before[0].payload.text as string
  const materialized = JSON.parse(text.slice(text.indexOf('@file:') + 6))
  expect(fs.readFileSync(materialized)).toEqual(content)
  await p.raw('fixture.reconnect')
  const again = new DurableSendStore(p.root, 'fixture-origin')
  await drainDurableSends(again, transport, Date.now() + 300_000)
  expect(again.read(intent.id)).toMatchObject({ state: 'confirmed', sessionId, admissionId: before[0].admission_id })
  expect(await p.raw('fixture.ledger', { request_id: intent.id })).toHaveLength(1)
  expect(p.calls.filter(call => call.method === 'durable.create')).toHaveLength(2)
  expect(p.calls.filter(call => call.method === 'durable.create')[0].params).toEqual(
    p.calls.filter(call => call.method === 'durable.create')[1].params
  )

  for (const call of p.calls) {
    expect(call.owner).toMatchObject({ connectionId: 'fixture', profile: 'default' })
  }

  // A subsequent message to an existing chat never invokes create or inherits the prior request id.
  const next = enqueue(p.root, owner, sessionId)
  await drainDurableSends(next.store, transport, Date.now() + 400_000)
  expect(next.store.read(next.intent.id).state).toBe('confirmed')
  expect(await p.raw('fixture.ledger', { request_id: next.intent.id })).toHaveLength(1)
}, 60_000)

test('real owner restart fences a claimed admission UNKNOWN without executing or minting another message', async () => {
  const p = peer()
  const owner = await ownerOf(p)
  const { store, intent } = enqueue(p.root, owner)
  const transport = durableRpcTransport(p.rpc)
  p.drop('durable.submit')
  await drainDurableSends(store, transport)
  const sessionId = store.read(intent.id).sessionId!
  expect((await p.raw('fixture.claim', { session_id: sessionId })).status).toBe('started')
  await p.raw('fixture.restart')
  await drainDurableSends(new DurableSendStore(p.root, 'fixture-origin'), transport, Date.now() + 100_000)
  expect(store.read(intent.id)).toMatchObject({ state: 'blocked', problem: 'uncertain', submitStarted: true })
  const submissions = p.calls.filter(call => call.method === 'durable.submit').length
  await drainDurableSends(store, transport, Date.now() + 200_000)
  expect(p.calls.filter(call => call.method === 'durable.submit')).toHaveLength(submissions)
  expect(await p.raw('fixture.ledger', { request_id: intent.id })).toHaveLength(1)
}, 60_000)

test('a different authenticated principal cannot drain an existing offline queue', async () => {
  const p = peer()
  const owner = await ownerOf(p)
  const { store, intent } = enqueue(p.root, owner)
  await p.raw('fixture.principal', { user: 'different-human' })
  await drainDurableSends(store, durableRpcTransport(p.rpc))
  expect(store.read(intent.id)).toMatchObject({ state: 'blocked', problem: 'capability' })
  expect(
    p.calls.some(
      call => call.method === 'durable.upload' || call.method === 'durable.submit' || call.method === 'durable.create'
    )
  ).toBe(false)
  expect(await p.raw('fixture.ledger', { request_id: intent.id })).toEqual([])
}, 60_000)
