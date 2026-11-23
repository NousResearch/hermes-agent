import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test } from 'vitest'

import { drainDurableSends, type DurableSendTransport } from './durable-send-delivery'
import { DurableSendStore } from './durable-send-store'
import type { DurableSendIntent } from './durable-send-types'

const homes: string[] = []
afterEach(() => homes.splice(0).forEach(home => fs.rmSync(home, { recursive: true, force: true })))

function fixture() {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'durable-retirement-'))
  homes.push(home)
  const store = new DurableSendStore(home, 'fixture')
  const owner = { connectionId: 'vps', profile: 'default', gatewayId: 'store', principalId: 'human' }
  const intent = (): DurableSendIntent => ({ id: randomUUID(), owner, target: { sessionId: 'sid' }, text: 'test', files: [] })

  return { home, store, intent }
}

test('1025 confirmed sends do not exhaust permanent metadata quota; compacted identities never replay', () => {
  const { home, store, intent } = fixture()
  const uncertain = intent()
  store.begin(uncertain); store.commit(uncertain.id)
  store.update(uncertain.id, { state: 'blocked', problem: 'uncertain', submitStarted: true })
  let oldest!: DurableSendIntent

  for (let i = 0; i < 1025; i++) {
    const item = intent()

    if (!i) {oldest = item}
    store.begin(item); store.commit(item.id)
    store.update(item.id, { state: 'confirmed', admissionId: `ack-${i}` })
    store.cleanup(item.id)
  }

  const reopened = new DurableSendStore(home, 'fixture')
  expect(() => reopened.begin(intent())).not.toThrow()
  expect(reopened.list().filter(entry => entry.state === 'confirmed')).toHaveLength(128)
  expect(reopened.read(uncertain.id)).toMatchObject({ state: 'blocked', problem: 'uncertain' })
  expect(reopened.retiredState(oldest.id)).toBe('confirmed')
  expect(fs.existsSync(path.join(store.root, oldest.id))).toBe(false)
  expect(() => reopened.begin(oldest)).toThrow('already retired')
}, 60_000)

test('cancelled partial upload retains the only local bytes until a fenced remote cancellation ACK', async () => {
  const { store, intent } = fixture()
  const item = intent()
  const bytes = Buffer.from('original file')
  item.files = [{ id: randomUUID(), name: 'original.txt', mediaType: 'text/plain', size: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex') }]
  store.begin(item); store.append(item.id, item.files[0].id, 0, bytes); store.commit(item.id)
  store.update(item.id, { state: 'waiting', attempts: 1, sessionId: 'sid' })
  expect(store.cancel(item.id).cancelPending).toBe(true)
  expect(() => store.begin(intent())).not.toThrow() // a pending cancellation is not a global enqueue lock
  expect(() => store.cleanup(item.id)).toThrow('Unconfirmed')
  let accepted = false
  let released = 0

  const transport: DurableSendTransport = {
    acquire: async () => async () => { released++ },
    describe: async owner => ({ owner, capabilities: ['durable-admission-v1'], uploadReplayVerified: true }),
    cancel: async () => accepted,
    create: async () => { throw new Error('must not create') },
    upload: async () => { throw new Error('must not upload') },
    submit: async () => { throw new Error('must not submit') }
  }

  await drainDurableSends(store, transport)
  expect(store.bytes(item.id, item.files[0].id)).toEqual(bytes)
  accepted = true
  await drainDurableSends(store, transport)
  expect(store.read(item.id)).toMatchObject({ state: 'cancelled', cancelPending: false })
  expect(() => store.bytes(item.id, item.files[0].id)).toThrow()
  expect(released).toBe(2)
})

test.each(['existing', 'fresh'])('fresh and %s second sends share FIFO even across lost admission ACK', async kind => {
  const { store, intent } = fixture()
  const first = intent()
  first.target = { draftId: 'fresh-origin', createRequestId: randomUUID(), params: { source: 'gui' } }
  store.begin(first); store.commit(first.id)
  let loseAck = true
  const submitted: string[] = []

  const transport: DurableSendTransport = {
    describe: async owner => ({ owner, capabilities: ['durable-admission-v1', 'local-cli-create-v1'], uploadReplayVerified: true }),
    create: async () => 'sid', upload: async () => { throw new Error('no files') },
    submit: async (_owner, request) => {
      submitted.push(request.submissionId)

      if (loseAck) {throw new Error('lost ACK')}

      return { admissionId: `ack-${request.submissionId}`, status: 'queued' }
    }
  }

  await drainDurableSends(store, transport, 0)
  const second = intent() // before OR after the UI handoff, this is the same logical conversation

  if (kind === 'fresh') {second.target = first.target}
  store.begin(second); store.commit(second.id)
  await drainDurableSends(store, transport, 1)
  expect(submitted).toEqual([first.id])
  loseAck = false
  await drainDurableSends(store, transport, 100000)
  expect(submitted).toEqual([first.id, first.id, second.id])
  expect(store.read(second.id).state).toBe('confirmed')
})
