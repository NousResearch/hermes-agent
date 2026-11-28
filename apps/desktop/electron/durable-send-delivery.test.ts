import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test, vi } from 'vitest'

import { drainDurableSends, DurableSendRefusal, type DurableSendTransport } from './durable-send-delivery'
import { DurableSendStore } from './durable-send-store'
import type { DurableSendIntent } from './durable-send-types'

const homes: string[] = []
const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex')

function setup() {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'durable-delivery-'))
  homes.push(home)
  const store = new DurableSendStore(home, 'fixture')
  const bytes = Buffer.from('actual attachment\0\nΩ')

  const intent: DurableSendIntent = {
    id: randomUUID(),
    text: 'fixture only',
    target: { sessionId: 'session-a' },
    owner: { connectionId: 'vps-fixture', gatewayId: 'gateway-fixture', profile: 'alpha', principalId: 'user-fixture' },
    files: [
      {
        id: randomUUID(),
        name: 'document.bin',
        mediaType: 'application/octet-stream',
        size: bytes.length,
        sha256: hash(bytes)
      }
    ]
  }

  store.begin(intent)
  store.append(intent.id, intent.files[0].id, 0, bytes)
  store.commit(intent.id)
  const uploads = new Map<string, string>()
  const admissions = new Map<string, string>()

  const transport: DurableSendTransport = {
    describe: vi.fn(async owner => ({
      owner,
      capabilities: ['durable-admission-v1', 'local-cli-create-v1'],
      uploadReplayVerified: true
    })),
    create: vi.fn(async () => 'new-session'),
    upload: vi.fn(async (_owner, request) => {
      expect(hash(request.bytes)).toBe(request.file.sha256)
      const key = `${request.submissionId}/${request.file.id}`
      const reference = uploads.get(key) ?? `fixture-ref-${uploads.size}`
      uploads.set(key, reference)

      return { reference, sha256: request.file.sha256, size: request.file.size }
    }),
    submit: vi.fn(async (_owner, request) => {
      const payload = JSON.stringify(request)

      if (admissions.has(request.submissionId)) {
        expect(admissions.get(request.submissionId)).toBe(payload)
      } else {
        admissions.set(request.submissionId, payload)
      }

      return { admissionId: `admission-${request.submissionId}`, status: 'queued' as const }
    })
  }

  return { home, store, bytes, intent, transport, uploads, admissions }
}

afterEach(() => homes.splice(0).forEach(home => fs.rmSync(home, { recursive: true, force: true })))

test('offline before send retains bytes; reconnect after store recreation confirms and cleans once', async () => {
  const f = setup()
  vi.mocked(f.transport.describe).mockRejectedValueOnce(new Error('offline'))
  await drainDurableSends(f.store, f.transport, 100)
  expect(f.store.read(f.intent.id).state).toBe('waiting')
  expect(f.transport.submit).not.toHaveBeenCalled()
  expect(f.store.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
  const restarted = new DurableSendStore(f.home, 'fixture')
  await drainDurableSends(restarted, f.transport, 100_000)
  expect(restarted.read(f.intent.id).state).toBe('confirmed')
  expect(f.admissions.size).toBe(1)
  expect(() => restarted.bytes(f.intent.id, f.intent.files[0].id)).toThrow()
  await drainDurableSends(restarted, f.transport, 200_000)
  expect(f.transport.submit).toHaveBeenCalledTimes(1)
})

test('receipt lost after server admission replays the identical payload/id, never another execution', async () => {
  const f = setup()
  const original = f.transport.submit
  let lose = true
  f.transport.submit = vi.fn(async (...args: Parameters<DurableSendTransport['submit']>) => {
    const receipt = await original(...args)

    if (lose) {
      lose = false
      throw new Error('lost admission ACK')
    }

    return receipt
  })
  await drainDurableSends(f.store, f.transport, 0)
  expect(f.store.read(f.intent.id).submitStarted).toBe(true)
  expect(f.admissions.size).toBe(1)
  expect(f.store.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
  await drainDurableSends(new DurableSendStore(f.home, 'fixture'), f.transport, 100_000)
  expect(f.transport.submit).toHaveBeenCalledTimes(2)
  expect(f.admissions.size).toBe(1)
  expect(f.transport.upload).toHaveBeenCalledTimes(1)
  expect(f.store.read(f.intent.id).state).toBe('confirmed')
})

test('lost upload ACK reuses its stable identity and exact bytes, never submits text alone', async () => {
  const f = setup()
  const original = f.transport.upload
  let lose = true
  f.transport.upload = vi.fn(async (...args: Parameters<DurableSendTransport['upload']>) => {
    const result = await original(...args)

    if (lose) {
      lose = false
      throw new Error('lost upload ACK')
    }

    return result
  })
  await drainDurableSends(f.store, f.transport, 0)
  expect(f.uploads.size).toBe(1)
  expect(f.transport.submit).not.toHaveBeenCalled()
  await drainDurableSends(new DurableSendStore(f.home, 'fixture'), f.transport, 100_000)
  expect(f.uploads.size).toBe(1)
  expect(f.admissions.size).toBe(1)
  expect(f.transport.upload).toHaveBeenCalledTimes(2)
  const submitted = vi.mocked(f.transport.submit).mock.calls[0][1]
  expect(submitted.attachments).toHaveLength(1)
  expect(submitted.attachments[0].sha256).toBe(hash(f.bytes))
})

test('interrupted upload never submits a partial message', async () => {
  const f = setup()
  vi.mocked(f.transport.upload).mockRejectedValueOnce(new Error('connection interrupted mid-upload'))
  await drainDurableSends(f.store, f.transport, 0)
  expect(f.transport.submit).not.toHaveBeenCalled()
  expect(f.store.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
  await drainDurableSends(f.store, f.transport, 100_000)
  expect(f.admissions.size).toBe(1)
})

test('the second attachment failing retains both bytes and never falls back to the first alone', async () => {
  const f = setup()
  f.store.cancel(f.intent.id)
  const secondBytes = Buffer.from('different second attachment')

  const intent = {
    ...f.intent,
    id: randomUUID(),
    files: [
      ...f.intent.files,
      {
        ...f.intent.files[0],
        id: randomUUID(),
        name: 'second.bin',
        size: secondBytes.length,
        sha256: hash(secondBytes)
      }
    ]
  }

  f.store.begin(intent)
  f.store.append(intent.id, intent.files[0].id, 0, f.bytes)
  f.store.append(intent.id, intent.files[1].id, 0, secondBytes)
  f.store.commit(intent.id)
  const original = f.transport.upload
  let fail = true
  f.transport.upload = vi.fn(async (owner, request) => {
    if (request.file.id === intent.files[1].id && fail) {
      fail = false
      throw new Error('second upload interrupted')
    }

    return original(owner, request)
  })
  await drainDurableSends(f.store, f.transport, 0)
  expect(f.transport.submit).not.toHaveBeenCalled()
  expect(f.store.bytes(intent.id, intent.files[0].id)).toEqual(f.bytes)
  expect(f.store.bytes(intent.id, intent.files[1].id)).toEqual(secondBytes)
  await drainDurableSends(new DurableSendStore(f.home, 'fixture'), f.transport, 100_000)
  expect(f.transport.upload).toHaveBeenCalledTimes(3)
  expect(vi.mocked(f.transport.submit).mock.calls[0][1].attachments).toHaveLength(2)
})

test('wrong uploaded digest blocks admission and retains local bytes', async () => {
  const f = setup()
  vi.mocked(f.transport.upload).mockResolvedValue({ reference: 'bad', size: f.bytes.length, sha256: '0'.repeat(64) })
  await drainDurableSends(f.store, f.transport, 0)
  expect(f.store.read(f.intent.id)).toMatchObject({ state: 'blocked', problem: 'integrity' })
  expect(f.transport.submit).not.toHaveBeenCalled()
  expect(f.store.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
})

test.each(['legacy', 'upload-unverified', 'wrong-owner'])(
  'fails closed for %s, with no legacy fallback',
  async reason => {
    const f = setup()
    vi.mocked(f.transport.describe).mockResolvedValue({
      owner: reason === 'wrong-owner' ? { ...f.intent.owner, profile: 'beta' } : f.intent.owner,
      capabilities: reason === 'legacy' ? [] : ['durable-admission-v1'],
      uploadReplayVerified: reason !== 'upload-unverified'
    })
    await drainDurableSends(f.store, f.transport, 0)
    expect(f.store.read(f.intent.id)).toMatchObject({ state: 'blocked', problem: 'capability' })
    expect(f.transport.upload).not.toHaveBeenCalled()
    expect(f.transport.submit).not.toHaveBeenCalled()
  }
)

test('unknown execution is never replayed automatically or cleaned up', async () => {
  const f = setup()
  vi.mocked(f.transport.submit).mockResolvedValue({ admissionId: 'unknown-admission', status: 'unknown' })
  await drainDurableSends(f.store, f.transport, 0)
  await drainDurableSends(new DurableSendStore(f.home, 'fixture'), f.transport, 100_000)
  expect(f.transport.submit).toHaveBeenCalledTimes(1)
  expect(f.store.read(f.intent.id)).toMatchObject({ state: 'blocked', problem: 'uncertain' })
  expect(f.store.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
})

test('new-session create ACK lost survives restart with the same create identity', async () => {
  const f = setup()
  f.store.cancel(f.intent.id)

  const intent: DurableSendIntent = {
    ...f.intent,
    id: randomUUID(),
    target: { draftId: 'fresh-a', createRequestId: randomUUID(), params: { source: 'desktop', profile: 'alpha' } }
  }

  f.store.begin(intent)
  f.store.append(intent.id, intent.files[0].id, 0, f.bytes)
  f.store.commit(intent.id)
  const creates = new Map<string, string>()
  let lose = true
  f.transport.create = vi.fn(async (_owner, target) => {
    const session = creates.get(target.createRequestId) ?? `session-${creates.size}`
    creates.set(target.createRequestId, session)

    if (lose) {
      lose = false
      throw new Error('lost create ACK')
    }

    return session
  })
  await drainDurableSends(f.store, f.transport, 0)
  expect(f.transport.upload).not.toHaveBeenCalled()
  await drainDurableSends(new DurableSendStore(f.home, 'fixture'), f.transport, 100_000)
  expect(creates.size).toBe(1)
  expect(f.admissions.size).toBe(1)
  expect(f.store.read(intent.id).sessionId).toBe('session-0')
})

test('two window consumers cannot race one item and never follow ambient profile changes', async () => {
  const f = setup()
  let release!: () => void

  const wait = new Promise<void>(resolve => {
    release = resolve
  })

  const original = f.transport.describe
  f.transport.describe = vi.fn(async owner => {
    await wait

    return original(owner)
  })
  const first = drainDurableSends(f.store, f.transport, 0)
  await drainDurableSends(new DurableSendStore(f.home, 'fixture'), f.transport, 0)
  release()
  await first
  expect(f.transport.submit).toHaveBeenCalledTimes(1)
  expect(vi.mocked(f.transport.submit).mock.calls[0][0]).toEqual(f.intent.owner)
})

test('FIFO head failure blocks its chat but not another captured chat', async () => {
  const f = setup()
  const later = { ...f.intent, id: randomUUID(), files: [] }
  const other = { ...later, id: randomUUID(), target: { sessionId: 'chat-b' } }

  for (const intent of [later, other]) {
    f.store.begin(intent)
    f.store.commit(intent.id)
  }

  vi.mocked(f.transport.upload).mockRejectedValueOnce(new Error('upload offline'))
  await drainDurableSends(f.store, f.transport, 0)
  expect(f.store.read(later.id).state).toBe('queued')
  expect(f.store.read(other.id).state).toBe('confirmed')
  expect(f.admissions.size).toBe(1)
  await drainDurableSends(f.store, f.transport, 100_000)
  expect(f.admissions.size).toBe(3)
})

test('permanent refusal is visible and retains payload without repeated execution', async () => {
  const f = setup()
  vi.mocked(f.transport.submit).mockRejectedValue(new DurableSendRefusal('permanent fixture rejection'))
  await drainDurableSends(f.store, f.transport, 0)
  await drainDurableSends(f.store, f.transport, 100_000)
  expect(f.store.read(f.intent.id)).toMatchObject({ state: 'blocked', problem: 'refused' })
  expect(f.transport.submit).toHaveBeenCalledTimes(1)
  expect(f.store.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
})
