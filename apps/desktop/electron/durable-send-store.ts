import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import type { DurableSendIntent, DurableSendRecord } from './durable-send-types'

const ID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/
const HASH = /^[a-f0-9]{64}$/
const MAX_MANIFEST = 1024 * 1024
export const DURABLE_SEND_LIMITS = Object.freeze({
  file: 64 * 1024 * 1024,
  total: 256 * 1024 * 1024,
  chunk: 1024 * 1024,
  items: 128,
  files: 32
})
const STATES = new Set(['staging', 'queued', 'sending', 'waiting', 'blocked', 'confirmed', 'cancelled'])

const CREATE_FIELDS = new Set([
  'source',
  'profile',
  'cwd',
  'cwd_explicit',
  'cols',
  'model',
  'provider',
  'reasoning_effort',
  'fast',
  'service_tier'
])

const terminal = (entry: DurableSendRecord) => entry.state === 'confirmed' || entry.state === 'cancelled'
const digest = (bytes: Uint8Array | string) => createHash('sha256').update(bytes).digest('hex')
const clone = <T>(value: T): T => structuredClone(value)

function requireId(value: unknown): asserts value is string {
  if (typeof value !== 'string' || !ID.test(value)) {
    throw new Error('Invalid durable send identity')
  }
}

function boundedText(value: unknown, max = 1024): asserts value is string {
  if (typeof value !== 'string' || !value || Buffer.byteLength(value) > max || value.includes('\0')) {
    throw new Error('Invalid durable send field')
  }
}

function exact(value: unknown, fields: string[]): asserts value is Record<string, unknown> {
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    Object.keys(value).some(key => !fields.includes(key))
  ) {
    throw new Error('Invalid durable send shape')
  }
}

export function validateDurableIntent(input: unknown): DurableSendIntent {
  exact(input, ['id', 'owner', 'target', 'text', 'files'])
  requireId(input.id)
  exact(input.owner, ['connectionId', 'gatewayId', 'profile', 'principalId'])

  for (const field of ['connectionId', 'gatewayId', 'profile', 'principalId']) {
    boundedText(input.owner[field])
  }

  const target = input.target
  exact(target, ['sessionId', 'draftId', 'createRequestId', 'params'])

  if ('sessionId' in target) {
    exact(target, ['sessionId'])
    boundedText(target.sessionId)
  } else {
    exact(target, ['draftId', 'createRequestId', 'params'])
    boundedText(target.draftId)
    requireId(target.createRequestId)
    exact(target.params, [...CREATE_FIELDS])

    for (const value of Object.values(target.params)) {
      if (typeof value === 'string') {
        boundedText(value, 16_384)
      } else if (typeof value !== 'boolean' && !(typeof value === 'number' && Number.isFinite(value))) {
        throw new Error('Invalid create parameter')
      }
    }
  }

  if (typeof input.text !== 'string' || Buffer.byteLength(input.text) > 256 * 1024) {
    throw new Error('Message too large')
  }

  if (!Array.isArray(input.files) || input.files.length > DURABLE_SEND_LIMITS.files) {
    throw new Error('Too many attachments')
  }

  const ids = new Set<string>()

  for (const file of input.files) {
    exact(file, ['id', 'name', 'mediaType', 'size', 'sha256'])
    requireId(file.id)
    boundedText(file.name, 1024)
    boundedText(file.mediaType, 256)

    if (ids.has(file.id)) {
      throw new Error('Duplicate attachment identity')
    }

    ids.add(file.id)

    if (
      !Number.isSafeInteger(file.size) ||
      Number(file.size) < 0 ||
      Number(file.size) > DURABLE_SEND_LIMITS.file ||
      typeof file.sha256 !== 'string' ||
      !HASH.test(file.sha256)
    ) {
      throw new Error('Invalid attachment size or digest')
    }
  }

  if (!input.text.trim() && !input.files.length) {
    throw new Error('Empty durable send')
  }

  if (Buffer.byteLength(JSON.stringify(input)) > MAX_MANIFEST / 2) {
    throw new Error('Message metadata too large')
  }

  const intent = clone(input as unknown as DurableSendIntent)

  // Object-key order is not identity. Normalize it before persistence, FIFO lane selection,
  // and explicit retry comparison (including a second renderer's structured clone).
  return {
    id: intent.id,
    owner: {
      connectionId: intent.owner.connectionId,
      gatewayId: intent.owner.gatewayId,
      profile: intent.owner.profile,
      principalId: intent.owner.principalId
    },
    target:
      'sessionId' in intent.target
        ? { sessionId: intent.target.sessionId }
        : {
            draftId: intent.target.draftId,
            createRequestId: intent.target.createRequestId,
            params: Object.fromEntries(Object.entries(intent.target.params).sort(([a], [b]) => a.localeCompare(b)))
          },
    text: intent.text,
    files: intent.files.map(file => ({
      id: file.id,
      name: file.name,
      mediaType: file.mediaType,
      size: file.size,
      sha256: file.sha256
    }))
  }
}

interface Options {
  maxBytes?: number
  maxItems?: number
  /** Fault-injection seam; never supplied by IPC. */
  checkpoint?: (point: 'blob-synced' | 'before-publish' | 'after-publish') => void
  now?: () => number
}

/** Main-process owned. Synchronous mutations serialize across windows. Actual bytes precede
 * the queued acknowledgement; a failed/corrupt read never resets the queue to empty.
 * This store must have one owning Electron process (the application's single-instance lock). */
export class DurableSendStore {
  readonly root: string
  private readonly now: () => number
  private readonly options: Options

  constructor(userData: string, origin: string, options: Options = {}) {
    this.options = options
    this.root = path.join(userData, 'durable-sends-v1', digest(origin))
    this.now = options.now ?? Date.now
    const parent = path.dirname(this.root)
    fs.mkdirSync(userData, { recursive: true })
    this.directory(parent)
    this.directory(this.root)
    this.syncDirectory(parent)
    this.syncDirectory(userData)
  }

  private directory(dir: string): void {
    try {
      fs.mkdirSync(dir, { mode: 0o700 })
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') {
        throw error
      }
    }

    const info = fs.lstatSync(dir)

    if (!info.isDirectory() || info.isSymbolicLink() || (process.platform !== 'win32' && (info.mode & 0o077) !== 0)) {
      throw new Error('Unsafe durable send directory')
    }
  }

  private syncDirectory(dir: string): void {
    // Windows cannot open directories this way. Native power-loss acceptance remains platform-specific.
    if (process.platform === 'win32') {
      return
    }

    const fd = fs.openSync(dir, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW)

    try {
      fs.fsyncSync(fd)
    } finally {
      fs.closeSync(fd)
    }
  }

  private itemDir(id: string): string {
    requireId(id)
    const dir = path.join(this.root, id)
    const info = fs.lstatSync(dir)

    if (!info.isDirectory() || info.isSymbolicLink()) {
      throw new Error('Unsafe durable send item')
    }

    return dir
  }

  private readFile(file: string, max: number): Buffer {
    const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW)

    try {
      const info = fs.fstatSync(fd)

      if (!info.isFile() || info.nlink !== 1 || info.size > max) {
        throw new Error('Unsafe durable send file')
      }

      return fs.readFileSync(fd)
    } finally {
      fs.closeSync(fd)
    }
  }

  private validateEntry(entry: DurableSendRecord, id: string): void {
    if (
      entry.version !== 1 ||
      !STATES.has(entry.state) ||
      entry.intent?.id !== id ||
      !Number.isSafeInteger(entry.attempts) ||
      entry.attempts < 0 ||
      !Number.isFinite(entry.createdAt) ||
      !Number.isFinite(entry.retryAt) ||
      typeof entry.submitStarted !== 'boolean' ||
      (entry.cancelPending !== undefined && typeof entry.cancelPending !== 'boolean')
    ) {
      throw new Error('Corrupt durable send manifest')
    }

    validateDurableIntent(entry.intent)

    if (!entry.uploads || typeof entry.uploads !== 'object' || Array.isArray(entry.uploads)) {
      throw new Error('Corrupt upload manifest')
    }

    for (const [fileId, uploaded] of Object.entries(entry.uploads)) {
      const file = entry.intent.files.find(candidate => candidate.id === fileId)

      if (!file || uploaded.sha256 !== file.sha256 || uploaded.size !== file.size) {
        throw new Error('Corrupt upload identity')
      }

      boundedText(uploaded.reference, 16_384)
    }

    if (entry.sessionId !== undefined) {
      boundedText(entry.sessionId)
    }

    if (entry.admissionId !== undefined) {
      boundedText(entry.admissionId)
    }
  }

  read(id: string): DurableSendRecord {
    const entry = JSON.parse(
      this.readFile(path.join(this.itemDir(id), 'entry.json'), MAX_MANIFEST).toString('utf8')
    ) as DurableSendRecord

    this.validateEntry(entry, id)

    return entry
  }

  private save(entry: DurableSendRecord): void {
    this.validateEntry(entry, entry.intent.id)
    const dir = this.itemDir(entry.intent.id)
    const temp = path.join(dir, `.entry-${randomUUID()}`)
    const data = JSON.stringify(entry)

    if (Buffer.byteLength(data) > MAX_MANIFEST) {
      throw new Error('Message metadata too large')
    }

    const fd = fs.openSync(temp, 'wx', 0o600)

    try {
      try {
        fs.writeFileSync(fd, data)
        fs.fsyncSync(fd)
      } finally {
        fs.closeSync(fd)
      }

      this.options.checkpoint?.('before-publish')
      fs.renameSync(temp, path.join(dir, 'entry.json'))
      this.syncDirectory(dir)
      this.options.checkpoint?.('after-publish')
    } finally {
      if (fs.existsSync(temp)) {
        fs.unlinkSync(temp)
      }
    }
  }

  /** Tiny anti-replay tombstones outlive the bounded full UI receipts. Never recycle IDs. */
  retiredState(id: string): 'confirmed' | 'cancelled' | undefined {
    requireId(id)
    const root = path.join(this.root, 'retired')
    this.directory(root)
    const name = path.join(root, id)

    if (!fs.existsSync(name)) {return undefined}
    const receipt = JSON.parse(this.readFile(name, 256).toString('utf8'))

    if (!['confirmed', 'cancelled'].includes(receipt.state) || !HASH.test(receipt.digest)) {
      throw new Error('Corrupt retirement receipt')
    }

    return receipt.state
  }

  private retireReceipts(): void {
    const records = this.list().filter(entry => terminal(entry) && !entry.cancelPending)
    const root = path.join(this.root, 'retired')
    this.directory(root)

    for (const entry of records.slice(0, Math.max(0, records.length - 128))) {
      const name = path.join(root, entry.intent.id)
      const temporary = path.join(root, `.receipt-${randomUUID()}`)

      try {
        const fd = fs.openSync(temporary, 'wx', 0o600)

        try {
          fs.writeFileSync(fd, JSON.stringify({ state: entry.state, digest: digest(JSON.stringify(entry.intent)) }))
          fs.fsyncSync(fd)
        } finally {
          fs.closeSync(fd)
        }

        fs.renameSync(temporary, name)
        this.syncDirectory(root)
      } finally {
        if (fs.existsSync(temporary)) {fs.unlinkSync(temporary)}
      }

      // Tombstone durable first; a crash cannot revive this intent. No active/unknown entry
      // is eligible, and no cancelled entry is eligible until remote cleanup reconciles.
      fs.rmSync(this.itemDir(entry.intent.id), { recursive: true })
      this.syncDirectory(this.root)
    }
  }

  list(): DurableSendRecord[] {
    return fs
      .readdirSync(this.root)
      .filter(id => ID.test(id))
      .flatMap(id => {
        const dir = this.itemDir(id)

        // A crash before begin published its manifest cannot own any acknowledged bytes.
        if (!fs.existsSync(path.join(dir, 'entry.json'))) {
          if (fs.readdirSync(dir).some(name => !name.startsWith('.entry-'))) {
            throw new Error('Unaccounted durable send data')
          }

          return []
        }

        return [this.read(id)]
      })
      .sort((a, b) => a.createdAt - b.createdAt || a.intent.id.localeCompare(b.intent.id))
  }

  begin(input: unknown): DurableSendRecord {
    const intent = validateDurableIntent(input)

    if (this.retiredState(intent.id)) {throw new Error('Durable send identity already retired')}
    this.retireReceipts()
    const records = this.list()
    const existing = records.find(entry => entry.intent.id === intent.id)

    if (existing) {
      if (JSON.stringify(existing.intent) !== JSON.stringify(intent)) {
        throw new Error('Durable send identity conflict')
      }

      return existing
    }

    const active = records.filter(record => !terminal(record) || record.cancelPending)
    // Terminal receipts remain as idempotency tombstones. Bound metadata too; never silently
    // evict an unconfirmed message, and finish confirmed-byte cleanup before reserving more.
    records.filter(entry => terminal(entry) && !entry.cancelPending).forEach(entry => this.cleanup(entry.intent.id))
    const metadata = records.reduce((sum, entry) => sum + Buffer.byteLength(JSON.stringify(entry)), 0)

    if (
      fs.readdirSync(this.root).filter(id => ID.test(id)).length >= 1024 ||
      metadata + Buffer.byteLength(JSON.stringify(intent)) > 16 * MAX_MANIFEST
    ) {
      throw new Error('Durable send metadata quota exceeded')
    }

    if (!('sessionId' in intent.target)) {
      const next = intent.target

      const related = records.find(
        entry =>
          JSON.stringify(entry.intent.owner) === JSON.stringify(intent.owner) &&
          !('sessionId' in entry.intent.target) &&
          entry.intent.target.draftId === next.draftId
      )

      if (related && JSON.stringify(related.intent.target) !== JSON.stringify(next)) {
        throw new Error('Durable creation identity conflict')
      }
    }

    const reserved = active.reduce(
      (total, entry) => total + entry.intent.files.reduce((sum, file) => sum + file.size, 0),
      0
    )

    const requested = intent.files.reduce((sum, file) => sum + file.size, 0)

    if (
      active.length >= (this.options.maxItems ?? DURABLE_SEND_LIMITS.items) ||
      reserved + requested > (this.options.maxBytes ?? DURABLE_SEND_LIMITS.total)
    ) {
      throw new Error('Durable send storage quota exceeded')
    }

    this.directory(path.join(this.root, intent.id))
    this.syncDirectory(this.root)

    const entry: DurableSendRecord = {
      version: 1,
      intent,
      state: 'staging',
      sessionId: 'sessionId' in intent.target ? intent.target.sessionId : records.find(record =>
        JSON.stringify(record.intent.owner) === JSON.stringify(intent.owner) &&
        JSON.stringify(record.intent.target) === JSON.stringify(intent.target) && record.sessionId)?.sessionId,
      createdAt: Math.max(this.now(), ...records.map(record => record.createdAt + 1)),
      attempts: 0,
      retryAt: 0,
      uploads: {},
      submitStarted: false
    }

    this.save(entry)

    return entry
  }

  append(id: string, fileId: string, offset: number, bytes: Uint8Array): number {
    requireId(fileId)
    const entry = this.read(id)
    const file = entry.intent.files.find(candidate => candidate.id === fileId)

    if (
      entry.state !== 'staging' ||
      !file ||
      !Number.isSafeInteger(offset) ||
      offset < 0 ||
      !(bytes instanceof Uint8Array) ||
      !bytes.byteLength ||
      bytes.byteLength > DURABLE_SEND_LIMITS.chunk ||
      offset + bytes.byteLength > file.size
    ) {
      throw new Error('Invalid durable attachment chunk')
    }

    const name = path.join(this.itemDir(id), `${fileId}.blob`)
    const fd = fs.openSync(name, fs.constants.O_CREAT | fs.constants.O_RDWR | fs.constants.O_NOFOLLOW, 0o600)

    try {
      const info = fs.fstatSync(fd)

      if (!info.isFile() || info.nlink !== 1 || offset > info.size || info.size > file.size) {
        throw new Error('Invalid durable attachment offset')
      }

      // A process can die mid-write, leaving only a prefix of this chunk. Compare that
      // prefix before continuing; replaying the whole chunk is safe after either kind of lost ACK.
      let written = Math.min(bytes.byteLength, info.size - offset)

      if (written > 0) {
        const prior = Buffer.alloc(written)

        if (
          fs.readSync(fd, prior, 0, written, offset) !== written ||
          !prior.equals(Buffer.from(bytes.subarray(0, written)))
        ) {
          throw new Error('Conflicting durable attachment retry')
        }
      }

      while (written < bytes.byteLength) {
        const count = fs.writeSync(fd, bytes, written, bytes.byteLength - written, offset + written)

        if (!count) {
          throw new Error('Incomplete durable attachment write')
        }

        written += count
      }

      fs.fsyncSync(fd)
      this.options.checkpoint?.('blob-synced')

      return fs.fstatSync(fd).size
    } finally {
      fs.closeSync(fd)
    }
  }

  bytes(id: string, fileId: string): Buffer {
    const entry = this.read(id)
    const file = entry.intent.files.find(candidate => candidate.id === fileId)

    if (!file) {
      throw new Error('Unknown durable attachment')
    }

    const name = path.join(this.itemDir(id), `${fileId}.blob`)

    const bytes =
      file.size === 0 && !fs.existsSync(name) ? Buffer.alloc(0) : this.readFile(name, DURABLE_SEND_LIMITS.file)

    if (bytes.byteLength !== file.size || digest(bytes) !== file.sha256) {
      throw new Error('Durable attachment integrity failure')
    }

    return bytes
  }

  commit(id: string): DurableSendRecord {
    const entry = this.read(id)

    if (entry.state !== 'staging') {
      return entry
    }

    for (const file of entry.intent.files) {
      this.bytes(id, file.id)
    }

    entry.state = 'queued'
    this.save(entry)

    return entry
  }

  /** Main-process delivery only: not exposed as renderer IPC. */
  update(
    id: string,
    changes: Partial<
      Pick<
        DurableSendRecord,
        'state' | 'attempts' | 'retryAt' | 'sessionId' | 'uploads' | 'submitStarted' | 'admissionId' | 'problem' | 'cancelPending'
      >
    >
  ): DurableSendRecord {
    const entry = this.read(id)

    if (terminal(entry)) {
      throw new Error('Durable send already retired')
    }

    const updated = { ...entry, ...clone(changes) }

    if (entry.sessionId && updated.sessionId !== entry.sessionId) {
      throw new Error('Durable destination changed')
    }

    if (
      'sessionId' in entry.intent.target &&
      updated.sessionId &&
      updated.sessionId !== entry.intent.target.sessionId
    ) {
      throw new Error('Durable destination changed')
    }

    this.save(updated)

    return updated
  }

  cancel(id: string): DurableSendRecord {
    const entry = this.read(id)

    if (entry.state === 'cancelled') {
      this.cleanup(id)

      return entry
    }

    if (terminal(entry) || entry.state === 'sending' || entry.submitStarted) {
      throw new Error('Reconcile uncertain admission before cancellation')
    }

    const cancelled = this.update(id, { state: 'cancelled',
      cancelPending: !!entry.sessionId && entry.attempts > 0 && entry.intent.files.length > 0 })

    if (!cancelled.cancelPending) {this.cleanup(id)}

    return cancelled
  }

  /** Only the delivery consumer calls this after an authenticated remote cancellation ACK. */
  finishCancellation(id: string): void {
    const entry = this.read(id)

    if (entry.state !== 'cancelled' || entry.submitStarted) {throw new Error('Unsafe cancellation cleanup')}
    entry.cancelPending = false
    this.save(entry)
    this.cleanup(id)
  }

  cleanup(id: string): void {
    const entry = this.read(id)

    if (!terminal(entry) || entry.cancelPending) {
      throw new Error('Unconfirmed attachment cleanup refused')
    }

    const dir = this.itemDir(id)

    for (const file of entry.intent.files) {
      const name = path.join(dir, `${file.id}.blob`)

      // unlink removes the directory entry, not a symlink's target.
      try {
        fs.unlinkSync(name)
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
          throw error
        }
      }
    }

    this.syncDirectory(dir)
  }
}
