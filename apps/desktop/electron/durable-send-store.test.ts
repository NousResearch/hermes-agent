import { spawnSync } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { afterEach, expect, test, vi } from 'vitest'

import { DURABLE_SEND_LIMITS, DurableSendStore } from './durable-send-store'
import type { DurableSendIntent } from './durable-send-types'

const homes: string[] = []
const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex')

const owner = {
  connectionId: 'fixture-vps',
  gatewayId: 'fixture-instance',
  profile: 'alpha',
  principalId: 'fixture-user'
}

function fixture(options: ConstructorParameters<typeof DurableSendStore>[2] = {}) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'durable-store-'))
  homes.push(home)
  const store = new DurableSendStore(home, 'fixture://desktop', options)
  const bytes = Buffer.from('binary attachment\0\xff\nΩ')

  const intent: DurableSendIntent = {
    id: randomUUID(),
    owner,
    target: { sessionId: 'chat-a' },
    text: 'retained text',
    files: [
      {
        id: randomUUID(),
        name: 'example.bin',
        mediaType: 'application/octet-stream',
        size: bytes.length,
        sha256: hash(bytes)
      }
    ]
  }

  return { home, store, intent, bytes }
}

function enqueue(f: ReturnType<typeof fixture>) {
  f.store.begin(f.intent)
  f.store.append(f.intent.id, f.intent.files[0].id, 0, f.bytes)

  return f.store.commit(f.intent.id)
}

afterEach(() => {
  vi.restoreAllMocks()
  homes.splice(0).forEach(home => fs.rmSync(home, { recursive: true, force: true }))
})

test('queued acknowledgement owns actual bytes after reopening, not the original path or buffer', () => {
  const f = fixture()
  const expected = Buffer.from(f.bytes)
  enqueue(f)
  f.bytes.fill(0)
  const reopened = new DurableSendStore(f.home, 'fixture://desktop')
  expect(reopened.read(f.intent.id).state).toBe('queued')
  expect(reopened.bytes(f.intent.id, f.intent.files[0].id)).toEqual(expected)
  expect(new DurableSendStore(f.home, 'fixture://other').list()).toEqual([])
})

test('incomplete attachment never receives a queued acknowledgement', () => {
  const f = fixture()
  f.store.begin(f.intent)
  f.store.append(f.intent.id, f.intent.files[0].id, 0, f.bytes.subarray(0, 3))
  expect(() => f.store.commit(f.intent.id)).toThrow('integrity')
  expect(f.store.read(f.intent.id).state).toBe('staging')
  f.store.append(f.intent.id, f.intent.files[0].id, 3, f.bytes.subarray(3))
  expect(f.store.commit(f.intent.id).state).toBe('queued')
})

test('lost chunk acknowledgement retries identically; changed and out-of-order chunks are refused', () => {
  let fail = true

  const f = fixture({
    checkpoint: point => {
      if (point === 'blob-synced' && fail) {
        fail = false
        throw new Error('lost reply')
      }
    }
  })

  f.store.begin(f.intent)
  expect(() => f.store.append(f.intent.id, f.intent.files[0].id, 0, f.bytes)).toThrow('lost reply')
  expect(f.store.append(f.intent.id, f.intent.files[0].id, 0, f.bytes)).toBe(f.bytes.length)
  expect(() => f.store.append(f.intent.id, f.intent.files[0].id, 0, Buffer.from('DIFFERENT'))).toThrow('Conflicting')
  expect(f.store.commit(f.intent.id).state).toBe('queued')
})

test('a partial local chunk survives a write failure and can be retried in full', () => {
  const f = fixture()
  f.store.begin(f.intent)
  // This is the on-disk state of a process interrupted partway through a chunk.
  const file = path.join(f.store.root, f.intent.id, `${f.intent.files[0].id}.blob`)
  fs.writeFileSync(file, f.bytes.subarray(0, 3), { mode: 0o600 })
  const reopened = new DurableSendStore(f.home, 'fixture://desktop')
  expect(reopened.append(f.intent.id, f.intent.files[0].id, 0, f.bytes)).toBe(f.bytes.length)
  expect(reopened.commit(f.intent.id).state).toBe('queued')
  expect(reopened.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
})

test('an actual manifest write failure leaves no accumulating temporary files', () => {
  const f = fixture()
  f.store.begin(f.intent)
  f.store.append(f.intent.id, f.intent.files[0].id, 0, f.bytes)

  const write = vi.spyOn(fs, 'writeFileSync').mockImplementationOnce(() => {
    throw new Error('ENOSPC write fixture')
  })

  expect(() => f.store.commit(f.intent.id)).toThrow('ENOSPC')
  write.mockRestore()
  expect(fs.readdirSync(path.join(f.store.root, f.intent.id)).filter(name => name.startsWith('.entry-'))).toEqual([])
  expect(f.store.read(f.intent.id).state).toBe('staging')
})

test('quota is reserved before copying and includes staging entries from other windows', () => {
  const f = fixture({ maxBytes: 30 })
  f.store.begin(f.intent)
  const second = structuredClone(f.intent)
  second.id = randomUUID()
  const windowB = new DurableSendStore(f.home, 'fixture://desktop', { maxBytes: 30 })
  expect(() => windowB.begin(second)).toThrow('quota')
  expect(windowB.list()).toHaveLength(1)
})

test('disk failure before publication does not acknowledge queued or erase staging bytes', () => {
  let fail = false

  const f = fixture({
    checkpoint: point => {
      if (fail && point === 'before-publish') {
        throw new Error('ENOSPC fixture')
      }
    }
  })

  f.store.begin(f.intent)
  f.store.append(f.intent.id, f.intent.files[0].id, 0, f.bytes)
  fail = true
  expect(() => f.store.commit(f.intent.id)).toThrow('ENOSPC')
  expect(f.store.read(f.intent.id).state).toBe('staging')
  expect(f.store.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
})

test('lost commit acknowledgement reuses the exact durable item, never a second send', () => {
  let fail = false

  const f = fixture({
    checkpoint: point => {
      if (fail && point === 'after-publish') {
        throw new Error('lost enqueue ACK')
      }
    }
  })

  f.store.begin(f.intent)
  f.store.append(f.intent.id, f.intent.files[0].id, 0, f.bytes)
  fail = true
  expect(() => f.store.commit(f.intent.id)).toThrow('lost enqueue ACK')
  const reopened = new DurableSendStore(f.home, 'fixture://desktop')
  expect(reopened.begin(f.intent).state).toBe('queued')
  expect(reopened.list()).toHaveLength(1)
})

test('same ID with different destination, text or attachment digest is refused', () => {
  const f = fixture()
  enqueue(f)

  for (const change of [
    { owner: { ...owner, profile: 'beta' } },
    { text: 'changed' },
    { target: { sessionId: 'chat-b' } }
  ]) {
    expect(() => f.store.begin({ ...f.intent, ...change })).toThrow('identity conflict')
  }
})

test('new-chat creation identity and parameters survive reopening and cannot split one draft', () => {
  const f = fixture()
  f.intent.target = {
    draftId: '__new__:fixture',
    createRequestId: randomUUID(),
    params: { source: 'desktop', profile: 'alpha' }
  }
  enqueue(f)
  const reopened = new DurableSendStore(f.home, 'fixture://desktop')
  expect(reopened.read(f.intent.id).intent.target).toEqual(f.intent.target)
  expect(() =>
    reopened.begin({ ...f.intent, id: randomUUID(), target: { ...f.intent.target, createRequestId: randomUUID() } })
  ).toThrow('creation identity conflict')
})

test('plain native file corruption never clears or silently downgrades the item', () => {
  const f = fixture()
  enqueue(f)
  fs.writeFileSync(path.join(f.store.root, f.intent.id, `${f.intent.files[0].id}.blob`), 'corrupted')
  expect(() => f.store.bytes(f.intent.id, f.intent.files[0].id)).toThrow('integrity')
  fs.writeFileSync(path.join(f.store.root, f.intent.id, 'entry.json'), '{torn')
  expect(() => f.store.list()).toThrow()
  expect(fs.readFileSync(path.join(f.store.root, f.intent.id, 'entry.json'), 'utf8')).toBe('{torn')
})

test('no expiry: an old queued send remains pending, with FIFO order under identical clocks', () => {
  const f = fixture({ now: () => 1 })
  enqueue(f)
  const second = { ...f.intent, id: randomUUID(), files: [] }
  f.store.begin(second)
  f.store.commit(second.id)
  expect(f.store.list().map(entry => entry.intent.id)).toEqual([f.intent.id, second.id])
  expect(new DurableSendStore(f.home, 'fixture://desktop').read(f.intent.id).state).toBe('queued')
})

test('unconfirmed bytes cannot be cleaned; explicit unsent cancellation can clean them', () => {
  const f = fixture()
  enqueue(f)
  expect(() => f.store.cleanup(f.intent.id)).toThrow('Unconfirmed')
  expect(f.store.cancel(f.intent.id).state).toBe('cancelled')
  expect(() => f.store.bytes(f.intent.id, f.intent.files[0].id)).toThrow()
  expect(f.store.begin(f.intent).state).toBe('cancelled')
})

test('cancellation after a potentially admitted submission preserves its bytes', () => {
  const f = fixture()
  enqueue(f)
  f.store.update(f.intent.id, { state: 'waiting', submitStarted: true })
  expect(() => f.store.cancel(f.intent.id)).toThrow('Reconcile')
  expect(f.store.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
})

test('identity traversal, credentials in creation params, oversize chunks and symlinks are refused', () => {
  const f = fixture()
  expect(() => f.store.begin({ ...f.intent, id: '../outside' })).toThrow()
  expect(() =>
    f.store.begin({
      ...f.intent,
      target: { draftId: 'new', createRequestId: randomUUID(), params: { token: 'not-a-real-secret' } }
    })
  ).toThrow()
  f.store.begin(f.intent)
  expect(() =>
    f.store.append(f.intent.id, f.intent.files[0].id, 0, Buffer.alloc(DURABLE_SEND_LIMITS.chunk + 1))
  ).toThrow()
  const outside = path.join(f.home, 'outside')
  fs.writeFileSync(outside, 'not copied')
  fs.symlinkSync(outside, path.join(f.store.root, f.intent.id, `${f.intent.files[0].id}.blob`))
  expect(() => f.store.append(f.intent.id, f.intent.files[0].id, 0, f.bytes)).toThrow()
  expect(fs.readFileSync(outside, 'utf8')).toBe('not copied')
})

test('a process terminated after fsynced enqueue leaves exact files readable by a fresh process', () => {
  const f = fixture()
  const module = pathToFileURL(path.resolve(import.meta.dirname, 'durable-send-store.ts')).href

  const code = `import { DurableSendStore } from ${JSON.stringify(module)};
    const s=new DurableSendStore(${JSON.stringify(f.home)},'fixture://desktop');
    const intent=${JSON.stringify(f.intent)}; s.begin(intent);
    s.append(intent.id,intent.files[0].id,0,Buffer.from(${JSON.stringify(f.bytes.toString('base64'))},'base64'));
    s.commit(intent.id); process.exit(17);`

  const child = spawnSync(process.execPath, ['--input-type=module', '-e', code], { encoding: 'utf8' })
  expect(child.stderr).toBe('')
  expect(child.status).toBe(17)
  const reopened = new DurableSendStore(f.home, 'fixture://desktop')
  expect(reopened.read(f.intent.id).state).toBe('queued')
  expect(reopened.bytes(f.intent.id, f.intent.files[0].id)).toEqual(f.bytes)
})
