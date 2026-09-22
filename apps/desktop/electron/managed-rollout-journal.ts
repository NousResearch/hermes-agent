import fs from 'node:fs'
import path from 'node:path'
import { createHash, randomUUID } from 'node:crypto'

export const JOURNAL_SCHEMA_VERSION = 1 as const
export const DEFAULT_JOURNAL_RETENTION = 200
export const MAX_EVENT_PAGE_SIZE = 50
export const MAX_HISTORY_PAGE_SIZE = 50

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const RECORD_FILE_RE = /^([0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json$/
const INDEX_FILE = 'index.json'
const UNRESOLVED_FILE = 'unresolved.json'
const OWNER_FILE = '.owner'
const TEMP_FILE_RE = /^\.tmp-[^/\\]+$/
const MAX_REQUEST_ID_LENGTH = 256
const MAX_REQUEST_PAYLOAD_BYTES = 256 * 1024
const MAX_REASON_LENGTH = 500

type JournalStats = {
  isFile: () => boolean
  isDirectory: () => boolean
  isSymbolicLink: () => boolean
}

/** The filesystem waist keeps fault injection and the journal storage policy local. */
export interface JournalFs {
  mkdirSync: (filePath: string, options?: { recursive?: boolean }) => unknown
  readdirSync: (filePath: string) => string[]
  lstatSync: (filePath: string) => JournalStats
  readFileSync: (filePath: string, encoding: 'utf8') => string
  writeFileSync: (
    filePath: string,
    data: string,
    options?: { encoding?: BufferEncoding; flag?: string; mode?: number }
  ) => unknown
  renameSync: (from: string, to: string) => unknown
  unlinkSync: (filePath: string) => unknown
  syncFileSync?: (filePath: string) => unknown
  syncDirectorySync?: (filePath: string) => unknown
}

export interface JournalSnapshot extends Record<string, unknown> {
  schemaVersion: typeof JOURNAL_SCHEMA_VERSION
  id: string
  revision: number
  createdAt: string
  updatedAt: string
  phase: string
  finishedAt?: string | null
  archivedAt?: string | null
  attempts: unknown[]
  eventCount: number
}

export interface JournalEventInput extends Record<string, unknown> {
  kind: string
  actor: 'local-operator' | 'system'
  at?: string
  installId?: string | null
  reason?: string | null
  evidenceDigest?: string | null
}

export interface JournalEvent extends JournalEventInput {
  sequence: number
  at: string
}

export type JournalFactKind =
  | 'authorization-committed'
  | 'handoff-accepted'
  | 'detached-intent'
  | 'terminal-receipt'
  | 'settlement-validated'

export interface JournalEvidenceFact extends Record<string, unknown> {
  kind: JournalFactKind
  rolloutId: string
  correlationId: string | null
  installId: string | null
  observedAt: string
  basis: string
}

export interface UnresolvedFence {
  key: string
  rolloutId: string
  installId: string
  correlationId: string | null
  reason: string
  recordedAt: string
}

export interface UnresolvedIndexEntry extends UnresolvedFence {
  tombstone: boolean
  tombstoneAt?: string
}

export interface ArchiveMetadata {
  at: string
  actor: 'local-operator' | 'system'
  reason: string
}

export interface JournalAck {
  accepted: true
  duplicate: boolean
  id: string
  revision: number
  acceptanceId: string
  eventSequences: number[]
}

export interface JournalOwnerLease {
  token: string
  release: () => void
}

export interface JournalRequest {
  requestId: string
  payload: unknown
}

export interface JournalRecord {
  schemaVersion: typeof JOURNAL_SCHEMA_VERSION
  id: string
  generation: string
  snapshot: JournalSnapshot
  events: JournalEvent[]
  requests: Record<string, StoredRequest>
  archive: ArchiveMetadata | null
  unresolved: UnresolvedFence[]
  facts: JournalEvidenceFact[]
  createdAt: string
  updatedAt: string
}

interface StoredRequest {
  payloadDigest: string
  ack: JournalAck
}

export interface JournalSummary {
  id: string
  generation: string
  revision: number
  phase: string
  createdAt: string
  updatedAt: string
  finishedAt: string | null
  targetSha: string | null
  eventCount: number
  archived: boolean
  archive: ArchiveMetadata | null
  unresolvedInstallIds: string[]
  pruned: boolean
  tombstone: boolean
  prunedAt: string | null
  evidenceFacts: JournalEvidenceFact[]
  requestDigests: Record<string, string>
}

interface JournalIndexFile {
  schemaVersion: typeof JOURNAL_SCHEMA_VERSION
  summaries: JournalSummary[]
}

interface UnresolvedIndexFile {
  schemaVersion: typeof JOURNAL_SCHEMA_VERSION
  entries: UnresolvedIndexEntry[]
}

export interface JournalRecordOptions {
  events?: JournalEventInput[]
  unresolved?: UnresolvedFenceChange | UnresolvedFence[]
  archive?: ArchiveMetadata | null
  facts?: JournalEvidenceFact[]
}

export interface UnresolvedFenceChange {
  add?: UnresolvedFence[]
  remove?: string[]
}

export interface JournalRecordInput extends JournalRecordOptions, JournalRequest {
  id: string
  expectedRevision: number
  snapshot: JournalSnapshot
}

export interface JournalCreateOptions extends JournalRecordOptions {
  request?: JournalRequest
}

export interface JournalArchiveInput extends JournalRequest {
  id: string
  expectedRevision: number
  actor: 'local-operator' | 'system'
  reason: string
}

export interface EventPage {
  items: JournalEvent[]
  nextCursor: string | null
}

export interface HistoryPage {
  items: JournalSummary[]
  nextCursor: string | null
}

export interface PruneResult {
  prunedRecordIds: string[]
  retainedRecordIds: string[]
}

export interface ManagedRolloutJournalOptions {
  /** The `managed-rollouts/` directory in the Desktop data directory. */
  directory: string
  fs?: JournalFs
  clock?: () => string
  /** Generates acceptance identities, never record filenames. */
  idFactory?: () => string
  retentionLimit?: number
}

export class JournalError extends Error {
  readonly code: string
  readonly cause?: unknown

  constructor(code: string, message: string, cause?: unknown) {
    super(message)
    this.name = 'JournalError'
    this.code = code
    this.cause = cause
  }
}

export class JournalCorruptionError extends JournalError {
  constructor(message: string, cause?: unknown) {
    super('corrupt-journal', message, cause)
    this.name = 'JournalCorruptionError'
  }
}

const defaultFs: JournalFs = {
  mkdirSync: (filePath, options) => fs.mkdirSync(filePath, options),
  readdirSync: filePath => fs.readdirSync(filePath, { encoding: 'utf8' }),
  lstatSync: filePath => fs.lstatSync(filePath),
  readFileSync: filePath => fs.readFileSync(filePath, 'utf8'),
  writeFileSync: (filePath, data, options) => fs.writeFileSync(filePath, data, options),
  renameSync: (from, to) => fs.renameSync(from, to),
  unlinkSync: filePath => fs.unlinkSync(filePath),
  syncFileSync: filePath => {
    const handle = fs.openSync(filePath, 'r')
    try {
      fs.fsyncSync(handle)
    } finally {
      fs.closeSync(handle)
    }
  },
  syncDirectorySync: filePath => {
    const handle = fs.openSync(filePath, 'r')
    try {
      fs.fsyncSync(handle)
    } finally {
      fs.closeSync(handle)
    }
  }
}

function isMissing(error: unknown): boolean {
  return Boolean(error && typeof error === 'object' && (error as NodeJS.ErrnoException).code === 'ENOENT')
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function canonicalJson(value: unknown, stack = new Set<unknown>()): string {
  if (value === null) return 'null'
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value)
  if (typeof value === 'number') {
    if (!Number.isFinite(value))
      throw new JournalError('invalid-payload', 'Journal payload must contain finite numbers.')
    return JSON.stringify(value)
  }
  if (typeof value !== 'object') {
    throw new JournalError('invalid-payload', 'Journal payload must be JSON data.')
  }
  if (stack.has(value)) throw new JournalError('invalid-payload', 'Journal payload must not be cyclic.')
  stack.add(value)

  let result: string
  if (Array.isArray(value)) {
    result = `[${value.map(item => canonicalJson(item, stack)).join(',')}]`
  } else {
    const entries = Object.keys(value)
      .sort()
      .map(key => `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key], stack)}`)
    result = `{${entries.join(',')}}`
  }

  stack.delete(value)
  return result
}

function digestPayload(payload: unknown): string {
  // The payload is intentionally stored as a digest only. The record remains bounded
  // and a retry can still prove whether it is the same request.
  const serialized = canonicalJson(payload)
  if (Buffer.byteLength(serialized, 'utf8') > MAX_REQUEST_PAYLOAD_BYTES) {
    throw new JournalError('invalid-payload', 'Managed rollout request payload is too large.')
  }
  return createHash('sha256').update(serialized, 'utf8').digest('hex')
}

function validateRolloutId(id: unknown): string {
  if (typeof id !== 'string' || !UUID_RE.test(id)) {
    throw new JournalError('invalid-id', 'Managed rollout id must be a literal UUID.')
  }
  return id
}

function validateRequestId(requestId: unknown): string {
  if (
    typeof requestId !== 'string' ||
    !requestId ||
    requestId.length > MAX_REQUEST_ID_LENGTH ||
    /[\x00\r\n]/.test(requestId)
  ) {
    throw new JournalError('invalid-request', 'Managed rollout requestId is invalid.')
  }
  return requestId
}

function validateIsoLike(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value || value.length > 128 || /[\x00\r\n]/.test(value)) {
    throw new JournalCorruptionError(`Journal ${label} is invalid.`)
  }
  return value
}

function validateReason(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value || value.length > MAX_REASON_LENGTH || /[\x00\r\n]/.test(value)) {
    throw new JournalError('invalid-input', `${label} is invalid.`)
  }
  return value
}

function validateSnapshot(value: unknown, expectedId?: string): JournalSnapshot {
  if (!isPlainObject(value)) throw new JournalCorruptionError('Journal snapshot must be an object.')
  if (value.schemaVersion !== JOURNAL_SCHEMA_VERSION) {
    throw new JournalCorruptionError('Journal snapshot schema version is unsupported.')
  }
  const id = validateRolloutId(value.id)
  if (expectedId && id !== expectedId)
    throw new JournalCorruptionError('Journal snapshot id does not match its record.')
  if (!Number.isSafeInteger(value.revision) || Number(value.revision) < 0) {
    throw new JournalCorruptionError('Journal snapshot revision is invalid.')
  }
  if (typeof value.phase !== 'string' || !value.phase) {
    throw new JournalCorruptionError('Journal snapshot phase is invalid.')
  }
  const createdAt = validateIsoLike(value.createdAt, 'createdAt')
  const updatedAt = validateIsoLike(value.updatedAt, 'updatedAt')
  if (!Array.isArray(value.attempts)) throw new JournalCorruptionError('Journal snapshot attempts must be an array.')
  if (!Number.isSafeInteger(value.eventCount) || Number(value.eventCount) < 0) {
    throw new JournalCorruptionError('Journal snapshot eventCount is invalid.')
  }
  if (value.finishedAt !== undefined && value.finishedAt !== null) validateIsoLike(value.finishedAt, 'finishedAt')
  if (value.archivedAt !== undefined && value.archivedAt !== null) validateIsoLike(value.archivedAt, 'archivedAt')

  // Walk the complete object so functions, undefined, BigInts, and cycles cannot
  // enter a file that a later process cannot parse with the same rules.
  canonicalJson(value)
  return clone({ ...value, id, createdAt, updatedAt }) as JournalSnapshot
}

function validateEventInput(value: unknown, forRead = false): JournalEventInput | JournalEvent {
  if (!isPlainObject(value)) throw new JournalCorruptionError('Journal event must be an object.')
  if (typeof value.kind !== 'string' || !value.kind || value.kind.length > 100) {
    throw new JournalCorruptionError('Journal event kind is invalid.')
  }
  if (value.actor !== 'local-operator' && value.actor !== 'system') {
    throw new JournalCorruptionError('Journal event actor is invalid.')
  }
  if (value.installId !== undefined && value.installId !== null && typeof value.installId !== 'string') {
    throw new JournalCorruptionError('Journal event installId is invalid.')
  }
  if (value.reason !== undefined && value.reason !== null) validateReason(value.reason, 'Journal event reason')
  if (value.evidenceDigest !== undefined && value.evidenceDigest !== null && typeof value.evidenceDigest !== 'string') {
    throw new JournalCorruptionError('Journal event evidenceDigest is invalid.')
  }
  if (value.at !== undefined) validateIsoLike(value.at, 'event at')
  if (forRead) {
    if (!Number.isSafeInteger(value.sequence) || Number(value.sequence) < 1) {
      throw new JournalCorruptionError('Journal event sequence is invalid.')
    }
  }
  canonicalJson(value)
  return value as JournalEventInput | JournalEvent
}

function validateEvidenceFact(value: unknown): JournalEvidenceFact {
  if (!isPlainObject(value)) throw new JournalCorruptionError('Journal evidence fact must be an object.')
  const kinds: JournalFactKind[] = [
    'authorization-committed',
    'handoff-accepted',
    'detached-intent',
    'terminal-receipt',
    'settlement-validated'
  ]
  if (!kinds.includes(value.kind as JournalFactKind)) throw new JournalCorruptionError('Journal evidence fact kind is invalid.')
  validateRolloutId(value.rolloutId)
  if (value.correlationId !== null && typeof value.correlationId !== 'string') {
    throw new JournalCorruptionError('Journal evidence correlationId is invalid.')
  }
  if (value.installId !== null && typeof value.installId !== 'string') {
    throw new JournalCorruptionError('Journal evidence installId is invalid.')
  }
  validateIsoLike(value.observedAt, 'evidence observedAt')
  if (typeof value.basis !== 'string' || !value.basis || value.basis.length > MAX_REASON_LENGTH) {
    throw new JournalCorruptionError('Journal evidence basis is invalid.')
  }
  canonicalJson(value)
  return clone(value) as JournalEvidenceFact
}

function validateFence(value: unknown, forRead = false): UnresolvedFence | UnresolvedIndexEntry {
  if (!isPlainObject(value)) throw new JournalCorruptionError('Unresolved fence must be an object.')
  if (typeof value.key !== 'string' || !value.key || value.key.length > 300) {
    throw new JournalCorruptionError('Unresolved fence key is invalid.')
  }
  validateRolloutId(value.rolloutId)
  if (typeof value.installId !== 'string' || !value.installId || value.installId.length > 256) {
    throw new JournalCorruptionError('Unresolved fence installId is invalid.')
  }
  if (value.correlationId !== null && typeof value.correlationId !== 'string') {
    throw new JournalCorruptionError('Unresolved fence correlationId is invalid.')
  }
  validateReason(value.reason, 'Unresolved fence reason')
  validateIsoLike(value.recordedAt, 'Unresolved fence recordedAt')
  if (forRead) {
    if (typeof value.tombstone !== 'boolean') throw new JournalCorruptionError('Unresolved index tombstone is invalid.')
    if (value.tombstoneAt !== undefined) validateIsoLike(value.tombstoneAt, 'Unresolved tombstoneAt')
  }
  canonicalJson(value)
  return value as unknown as UnresolvedFence | UnresolvedIndexEntry
}

function validateArchive(value: unknown): ArchiveMetadata | null {
  if (value === null) return null
  if (!isPlainObject(value)) throw new JournalCorruptionError('Archive metadata must be an object or null.')
  validateIsoLike(value.at, 'archive at')
  if (value.actor !== 'local-operator' && value.actor !== 'system') {
    throw new JournalCorruptionError('Archive actor is invalid.')
  }
  validateReason(value.reason, 'Archive reason')
  return clone(value) as unknown as ArchiveMetadata
}

function encodeCursor(value: Record<string, unknown>): string {
  return Buffer.from(JSON.stringify(value), 'utf8').toString('base64url')
}

function decodeCursor(value: string, label: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(Buffer.from(value, 'base64url').toString('utf8'))
    if (!isPlainObject(parsed) || parsed.v !== 1) throw new Error('shape')
    return parsed
  } catch (error) {
    throw new JournalError('invalid-cursor', `${label} cursor is invalid.`, error)
  }
}

function compareHistory(a: JournalSummary, b: JournalSummary): number {
  if (a.createdAt !== b.createdAt) return a.createdAt > b.createdAt ? -1 : 1
  return a.id > b.id ? -1 : a.id < b.id ? 1 : 0
}

function targetShaFromSnapshot(snapshot: JournalSnapshot): string | null {
  const target = snapshot.target
  if (!isPlainObject(target) || typeof target.sha !== 'string') return null
  return target.sha
}

function summaryFromRecord(record: JournalRecord): JournalSummary {
  const snapshot = record.snapshot
  return {
    id: record.id,
    generation: record.generation,
    revision: snapshot.revision,
    phase: snapshot.phase,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    finishedAt: typeof snapshot.finishedAt === 'string' ? snapshot.finishedAt : null,
    targetSha: targetShaFromSnapshot(snapshot),
    eventCount: record.events.length,
    archived: record.archive !== null,
    archive: clone(record.archive),
    unresolvedInstallIds: [...new Set(record.unresolved.map(fence => fence.installId))].sort(),
    pruned: false,
    tombstone: false,
    prunedAt: null,
    evidenceFacts: clone(record.facts),
    requestDigests: Object.fromEntries(Object.entries(record.requests).map(([id, request]) => [id, request.payloadDigest]))
  }
}

function isSettled(record: JournalRecord): boolean {
  return new Set(['stopped', 'completed', 'completed-with-exclusions']).has(record.snapshot.phase)
}

export class ManagedRolloutJournal {
  readonly directory: string
  readonly retentionLimit: number
  private readonly fs: JournalFs
  private readonly clock: () => string
  private readonly idFactory: () => string
  private readonly records = new Map<string, JournalRecord>()
  private readonly summaries = new Map<string, JournalSummary>()
  private readonly unresolved = new Map<string, UnresolvedIndexEntry>()
  private tempCounter = 0
  private activeOwnerToken: string | null = null

  constructor(options: ManagedRolloutJournalOptions) {
    this.directory = path.resolve(options.directory)
    this.fs = options.fs ?? defaultFs
    this.clock = options.clock ?? (() => new Date().toISOString())
    this.idFactory = options.idFactory ?? (() => randomUUID())
    this.retentionLimit =
      Number.isSafeInteger(options.retentionLimit) && Number(options.retentionLimit) >= 1
        ? Number(options.retentionLimit)
        : DEFAULT_JOURNAL_RETENTION
    this.open()
  }

  private now(): string {
    const value = this.clock()
    return validateIsoLike(value, 'clock value')
  }

  private recordPath(id: string): string {
    return path.join(this.directory, `${validateRolloutId(id)}.json`)
  }

  private indexPath(): string {
    return path.join(this.directory, INDEX_FILE)
  }

  private unresolvedPath(): string {
    return path.join(this.directory, UNRESOLVED_FILE)
  }

  private ownerPath(): string {
    return path.join(this.directory, OWNER_FILE)
  }

  private withOwner<T>(operation: () => T): T {
    if (this.activeOwnerToken !== null) return operation()
    const lease = this.acquireOwner()
    try {
      return operation()
    } finally {
      lease.release()
    }
  }

  acquireOwner(): JournalOwnerLease {
    if (this.activeOwnerToken !== null) {
      return { token: this.activeOwnerToken, release: () => undefined }
    }
    const owner = this.ownerPath()
    const token = randomUUID()
    try {
      this.fs.writeFileSync(
        owner,
        JSON.stringify({
          pid: process.pid,
          incarnation: process.pid + ':' + process.uptime(),
          ownerId: token,
          acquiredAt: new Date().toISOString()
        }),
        { encoding: 'utf8', flag: 'wx', mode: 0o600 }
      )
    } catch (error) {
      throw new JournalError('owner-unavailable', 'Another process owns the managed rollout journal.', error)
    }
    this.activeOwnerToken = token
    let released = false
    return {
      token,
      release: () => {
        if (released) return
        released = true
        if (this.activeOwnerToken !== token) return
        this.activeOwnerToken = null
        try {
          this.fs.unlinkSync(owner)
        } catch {
          // Preserve fail-closed semantics if ownership cannot be released.
        }
      }
    }
  }

  private ensureDirectory(): void {
    try {
      const stat = this.fs.lstatSync(this.directory)
      if (stat.isSymbolicLink() || !stat.isDirectory()) {
        throw new JournalError('unsafe-path', 'Managed rollout journal directory is not a real directory.')
      }
    } catch (error) {
      if (!isMissing(error)) throw error
      this.fs.mkdirSync(this.directory, { recursive: true })
      const stat = this.fs.lstatSync(this.directory)
      if (stat.isSymbolicLink() || !stat.isDirectory()) {
        throw new JournalError('unsafe-path', 'Managed rollout journal directory is not a real directory.')
      }
    }
  }

  private assertRegular(pathname: string, label: string): void {
    let stat: JournalStats
    try {
      stat = this.fs.lstatSync(pathname)
    } catch (error) {
      throw error
    }
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new JournalCorruptionError(`${label} is not a regular file.`)
    }
  }

  private readOptional(pathname: string, label: string): string | null {
    try {
      this.assertRegular(pathname, label)
    } catch (error) {
      if (isMissing(error)) return null
      throw error
    }
    try {
      return this.fs.readFileSync(pathname, 'utf8')
    } catch (error) {
      throw new JournalCorruptionError(`Could not read ${label}.`, error)
    }
  }

  private parseFile<T>(raw: string, label: string): T {
    try {
      return JSON.parse(raw) as T
    } catch (error) {
      throw new JournalCorruptionError(`${label} is truncated or not valid JSON.`, error)
    }
  }

  private parseRecord(raw: string, expectedId: string): JournalRecord {
    const parsed = this.parseFile<unknown>(raw, `Managed rollout ${expectedId}`)
    if (!isPlainObject(parsed) || parsed.schemaVersion !== JOURNAL_SCHEMA_VERSION) {
      throw new JournalCorruptionError(`Managed rollout ${expectedId} schema is invalid.`)
    }
    const id = validateRolloutId(parsed.id)
    if (id !== expectedId) throw new JournalCorruptionError(`Managed rollout ${expectedId} id is invalid.`)
    if (typeof parsed.generation !== 'string' || !parsed.generation) {
      throw new JournalCorruptionError(`Managed rollout ${expectedId} generation is invalid.`)
    }
    const snapshot = validateSnapshot(parsed.snapshot, expectedId)
    if (!Array.isArray(parsed.events))
      throw new JournalCorruptionError(`Managed rollout ${expectedId} events are invalid.`)
    const events = parsed.events
      .map(item => validateEventInput(item, true) as JournalEvent)
      .sort((a, b) => a.sequence - b.sequence)
    for (let index = 0; index < events.length; index += 1) {
      if (events[index].sequence !== index + 1) {
        throw new JournalCorruptionError(`Managed rollout ${expectedId} event sequence is not contiguous.`)
      }
    }
    if (snapshot.eventCount !== events.length) {
      throw new JournalCorruptionError(`Managed rollout ${expectedId} eventCount does not match its event log.`)
    }
    if (!isPlainObject(parsed.requests))
      throw new JournalCorruptionError(`Managed rollout ${expectedId} request index is invalid.`)
    const requests: Record<string, StoredRequest> = {}
    for (const [requestId, request] of Object.entries(parsed.requests)) {
      validateRequestId(requestId)
      if (
        !isPlainObject(request) ||
        typeof request.payloadDigest !== 'string' ||
        !/^[0-9a-f]{64}$/.test(request.payloadDigest) ||
        !isPlainObject(request.ack)
      ) {
        throw new JournalCorruptionError(`Managed rollout ${expectedId} request entry is invalid.`)
      }
      const ack = request.ack as Record<string, unknown>
      if (
        ack.accepted !== true ||
        typeof ack.duplicate !== 'boolean' ||
        ack.id !== expectedId ||
        !Number.isSafeInteger(ack.revision) ||
        typeof ack.acceptanceId !== 'string' ||
        !ack.acceptanceId ||
        !Array.isArray(ack.eventSequences) ||
        !ack.eventSequences.every(sequence => Number.isSafeInteger(sequence) && Number(sequence) >= 1)
      ) {
        throw new JournalCorruptionError(`Managed rollout ${expectedId} request acknowledgement is invalid.`)
      }
      requests[requestId] = {
        payloadDigest: request.payloadDigest as string,
        ack: clone(request.ack) as unknown as JournalAck
      }
    }
    if (!Array.isArray(parsed.unresolved))
      throw new JournalCorruptionError(`Managed rollout ${expectedId} unresolved list is invalid.`)
    const unresolved = parsed.unresolved.map(item => validateFence(item) as UnresolvedFence)
    const facts = parsed.facts === undefined
      ? []
      : Array.isArray(parsed.facts)
        ? parsed.facts.map(item => validateEvidenceFact(item))
        : (() => { throw new JournalCorruptionError(`Managed rollout ${expectedId} evidence facts are invalid.`) })()
    const archive = validateArchive(parsed.archive)
    const createdAt = validateIsoLike(parsed.createdAt, 'record createdAt')
    const updatedAt = validateIsoLike(parsed.updatedAt, 'record updatedAt')
    canonicalJson(parsed)
    return {
      schemaVersion: JOURNAL_SCHEMA_VERSION,
      id,
      generation: parsed.generation,
      snapshot,
      events,
      requests,
      archive,
      unresolved,
      facts,
      createdAt,
      updatedAt
    }
  }

  private parseIndex(raw: string): JournalIndexFile {
    const parsed = this.parseFile<unknown>(raw, INDEX_FILE)
    if (!isPlainObject(parsed) || parsed.schemaVersion !== JOURNAL_SCHEMA_VERSION || !Array.isArray(parsed.summaries)) {
      throw new JournalCorruptionError('Managed rollout history index schema is invalid.')
    }
    const summaries = parsed.summaries.map(item => {
      if (!isPlainObject(item) || typeof item.id !== 'string') {
        throw new JournalCorruptionError('Managed rollout history summary is invalid.')
      }
      validateRolloutId(item.id)
      if (typeof item.generation !== 'string' || !item.generation) {
        throw new JournalCorruptionError('Managed rollout history generation is invalid.')
      }
      if (
        !Number.isSafeInteger(item.revision) ||
        Number(item.revision) < 1 ||
        typeof item.phase !== 'string' ||
        !item.phase
      ) {
        throw new JournalCorruptionError('Managed rollout history summary revision or phase is invalid.')
      }
      if (typeof item.createdAt !== 'string' || typeof item.updatedAt !== 'string') {
        throw new JournalCorruptionError('Managed rollout history summary timestamps are invalid.')
      }
      validateIsoLike(item.createdAt, 'summary createdAt')
      validateIsoLike(item.updatedAt, 'summary updatedAt')
      if (item.finishedAt !== null) validateIsoLike(item.finishedAt, 'summary finishedAt')
      if (item.targetSha !== null && typeof item.targetSha !== 'string') {
        throw new JournalCorruptionError('Managed rollout history targetSha is invalid.')
      }
      if (!Number.isSafeInteger(item.eventCount) || Number(item.eventCount) < 0) {
        throw new JournalCorruptionError('Managed rollout history eventCount is invalid.')
      }
      if (
        typeof item.archived !== 'boolean' ||
        typeof item.pruned !== 'boolean' ||
        typeof item.tombstone !== 'boolean' ||
        !Array.isArray(item.unresolvedInstallIds) ||
        !item.unresolvedInstallIds.every(installId => typeof installId === 'string')
      ) {
        throw new JournalCorruptionError('Managed rollout history summary flags are invalid.')
      }
      if (item.prunedAt !== null) validateIsoLike(item.prunedAt, 'summary prunedAt')
      const archive = validateArchive(item.archive)
      const evidenceFacts = item.evidenceFacts === undefined
        ? []
        : Array.isArray(item.evidenceFacts)
          ? item.evidenceFacts.map(fact => validateEvidenceFact(fact))
          : (() => { throw new JournalCorruptionError('Managed rollout history evidence facts are invalid.') })()
      const requestDigests = item.requestDigests === undefined
        ? {}
        : isPlainObject(item.requestDigests) &&
            Object.values(item.requestDigests).every(digest => typeof digest === 'string' && /^[0-9a-f]{64}$/.test(digest))
          ? clone(item.requestDigests) as Record<string, string>
          : (() => { throw new JournalCorruptionError('Managed rollout history request digests are invalid.') })()
      return clone({ ...item, archive, evidenceFacts, requestDigests }) as unknown as JournalSummary
    })
    return { schemaVersion: JOURNAL_SCHEMA_VERSION, summaries }
  }

  private parseUnresolvedIndex(raw: string): UnresolvedIndexFile {
    const parsed = this.parseFile<unknown>(raw, UNRESOLVED_FILE)
    if (!isPlainObject(parsed) || parsed.schemaVersion !== JOURNAL_SCHEMA_VERSION || !Array.isArray(parsed.entries)) {
      throw new JournalCorruptionError('Managed rollout unresolved index schema is invalid.')
    }
    const entries = parsed.entries.map(item => validateFence(item, true) as UnresolvedIndexEntry)
    const keys = new Set<string>()
    for (const entry of entries) {
      if (keys.has(entry.key)) throw new JournalCorruptionError('Managed rollout unresolved index has duplicate keys.')
      keys.add(entry.key)
    }
    return { schemaVersion: JOURNAL_SCHEMA_VERSION, entries }
  }

  private open(): void {
    this.ensureDirectory()
    const indexRaw = this.readOptional(this.indexPath(), INDEX_FILE)
    const unresolvedRaw = this.readOptional(this.unresolvedPath(), UNRESOLVED_FILE)
    const index = indexRaw === null ? null : this.parseIndex(indexRaw)
    const unresolvedIndex = unresolvedRaw === null ? null : this.parseUnresolvedIndex(unresolvedRaw)
    const entries = this.fs.readdirSync(this.directory)

    for (const name of entries) {
      if (TEMP_FILE_RE.test(name)) continue
      if (name === INDEX_FILE || name === UNRESOLVED_FILE || name === OWNER_FILE) continue
      const match = RECORD_FILE_RE.exec(name)
      if (!match) throw new JournalCorruptionError(`Unexpected file in managed rollout journal: ${name}`)
      const id = match[1]
      const pathname = path.join(this.directory, name)
      this.assertRegular(pathname, `Managed rollout ${id}`)
      const raw = this.readOptional(pathname, `Managed rollout ${id}`)
      if (raw === null) throw new JournalCorruptionError(`Managed rollout ${id} disappeared while opening.`)
      const record = this.parseRecord(raw, id)
      this.records.set(id, record)
      this.summaries.set(id, summaryFromRecord(record))
    }

    for (const summary of index?.summaries ?? []) {
      const record = this.records.get(summary.id)
      if (record) {
        if (record.generation !== summary.generation) {
          throw new JournalCorruptionError(`History summary ${summary.id} generation does not match its record.`)
        }
        continue
      }
      if (!summary.pruned || !summary.tombstone) {
        throw new JournalCorruptionError(`History summary ${summary.id} has no retained record.`)
      }
      this.summaries.set(summary.id, summary)
    }

    for (const entry of unresolvedIndex?.entries ?? []) this.unresolved.set(entry.key, clone(entry))
    for (const record of this.records.values()) {
      const indexed = this.indexRecord(record, this.summaries, this.unresolved)
      this.summaries.clear()
      for (const [key, value] of indexed.summaries) this.summaries.set(key, value)
      this.unresolved.clear()
      for (const [key, value] of indexed.unresolved) this.unresolved.set(key, value)
    }
  }

  private nextTempPath(target: string): string {
    this.tempCounter += 1
    return path.join(this.directory, `.tmp-${path.basename(target)}-${process.pid}-${this.tempCounter}`)
  }

  private writeJsonAtomic(target: string, value: unknown): void {
    const temp = this.nextTempPath(target)
    const data = JSON.stringify(value)
    try {
      try {
        this.assertRegular(target, path.basename(target))
      } catch (error) {
        if (isMissing(error)) {
          // A new record has no destination yet.
        } else if (error instanceof JournalCorruptionError) {
          throw new JournalError('unsafe-path', error.message, error)
        } else {
          throw error
        }
      }
      this.fs.writeFileSync(temp, data, { encoding: 'utf8', flag: 'wx', mode: 0o600 })
      this.fs.syncFileSync?.(temp)
      this.fs.renameSync(temp, target)
      this.fs.syncDirectorySync?.(this.directory)
    } catch (error) {
      try {
        this.assertRegular(temp, path.basename(temp))
        this.fs.unlinkSync(temp)
      } catch {
        // The temp may already have been moved (a crash-after-rename injection)
        // or may have failed before creation. Never follow a symlink while cleaning.
      }
      if (error instanceof JournalError) throw error
      const message = error instanceof Error ? error.message : String(error)
      throw new JournalError('write-failed', `Atomic journal write failed: ${message}`, error)
    }
  }

  private persistIndexes(summaries: Map<string, JournalSummary>, unresolved: Map<string, UnresolvedIndexEntry>): void {
    const sortedSummaries = [...summaries.values()].sort(compareHistory)
    const sortedUnresolved = [...unresolved.values()].sort((a, b) => a.key.localeCompare(b.key))
    this.writeJsonAtomic(this.indexPath(), {
      schemaVersion: JOURNAL_SCHEMA_VERSION,
      summaries: sortedSummaries
    } satisfies JournalIndexFile)
    this.writeJsonAtomic(this.unresolvedPath(), {
      schemaVersion: JOURNAL_SCHEMA_VERSION,
      entries: sortedUnresolved
    } satisfies UnresolvedIndexFile)
  }

  private rethrowAfterReload(error: unknown): never {
    try {
      this.reload()
    } catch (reloadError) {
      throw new JournalError(
        'corrupt-journal',
        'Managed rollout journal could not be reopened after a write failure.',
        reloadError
      )
    }
    throw error
  }

  private makeEvents(input: JournalEventInput[] | undefined, startSequence: number): JournalEvent[] {
    return (input ?? []).map((item, offset) => {
      validateEventInput(item)
      const at = item.at === undefined ? this.now() : validateIsoLike(item.at, 'event at')
      return {
        ...clone(item),
        sequence: startSequence + offset,
        at
      } as JournalEvent
    })
  }

  private makeSnapshot(
    input: JournalSnapshot,
    id: string,
    revision: number,
    eventCount: number,
    now: string,
    current?: JournalSnapshot
  ): JournalSnapshot {
    const validated = validateSnapshot(input, id)
    const createdAt = current?.createdAt ?? validated.createdAt
    return {
      ...validated,
      id,
      revision,
      createdAt,
      updatedAt: now,
      eventCount
    }
  }

  private applyFenceChange(
    current: UnresolvedFence[],
    change: UnresolvedFenceChange | UnresolvedFence[] | undefined
  ): UnresolvedFence[] {
    if (change === undefined) return clone(current)
    const next = new Map(current.map(item => [item.key, clone(item)]))
    if (Array.isArray(change)) {
      for (const item of change) {
        validateFence(item)
        next.set(item.key, clone(item))
      }
    } else {
      for (const key of change.remove ?? []) next.delete(String(key))
      for (const item of change.add ?? []) {
        validateFence(item)
        next.set(item.key, clone(item))
      }
    }
    return [...next.values()].sort((a, b) => a.key.localeCompare(b.key))
  }

  private validateFacts(facts: JournalEvidenceFact[] | undefined, rolloutId: string): JournalEvidenceFact[] {
    return (facts ?? []).map(fact => {
      const validated = validateEvidenceFact(fact)
      if (validated.rolloutId !== rolloutId) throw new JournalError('invalid-input', 'Evidence fact rollout does not match the journal record.')
      return validated
    })
  }

  private assertFenceRelease(
    current: JournalRecord,
    nextFacts: JournalEvidenceFact[],
    nextUnresolved: UnresolvedFence[]
  ): void {
    const retained = new Set(nextUnresolved.map(fence => fence.key))
    for (const fence of current.unresolved) {
      if (retained.has(fence.key)) continue
      const settled = nextFacts.some(
        fact =>
          fact.kind === 'settlement-validated' &&
          fact.correlationId === fence.correlationId &&
          fact.installId === fence.installId
      )
      if (!settled) throw new JournalError('fence-release-unproven', `Fence ${fence.key} lacks validated settlement evidence.`)
    }
  }

  private prepareAck(id: string, revision: number, eventSequences: number[]): JournalAck {
    const acceptanceId = String(this.idFactory())
    if (!acceptanceId || /[\x00\r\n]/.test(acceptanceId)) {
      throw new JournalError('invalid-acceptance-id', 'Journal acceptance identity is invalid.')
    }
    return {
      accepted: true,
      duplicate: false,
      id,
      revision,
      acceptanceId,
      eventSequences: [...eventSequences]
    }
  }

  private reload(): void {
    const priorRecords = new Map(this.records)
    const priorSummaries = new Map(this.summaries)
    const priorUnresolved = new Map(this.unresolved)
    this.records.clear()
    this.summaries.clear()
    this.unresolved.clear()
    try {
      this.open()
    } catch (error) {
      this.records.clear()
      for (const [key, value] of priorRecords) this.records.set(key, value)
      this.summaries.clear()
      for (const [key, value] of priorSummaries) this.summaries.set(key, value)
      this.unresolved.clear()
      for (const [key, value] of priorUnresolved) this.unresolved.set(key, value)
      throw error
    }
  }

  private indexRecord(
    record: JournalRecord,
    baseSummaries: Map<string, JournalSummary>,
    baseUnresolved: Map<string, UnresolvedIndexEntry>
  ) {
    const summaries = new Map(baseSummaries)
    summaries.set(record.id, summaryFromRecord(record))
    const unresolved = new Map(baseUnresolved)
    const activeFenceKeys = new Set(record.unresolved.map(fence => fence.key))
    for (const [key, entry] of unresolved) {
      if (entry.rolloutId === record.id && !entry.tombstone && !activeFenceKeys.has(key)) {
        unresolved.delete(key)
      }
    }
    for (const fence of record.unresolved) {
      unresolved.set(fence.key, { ...clone(fence), tombstone: false })
    }
    return { summaries, unresolved }
  }

  create(snapshot: JournalSnapshot, options: JournalCreateOptions = {}): JournalAck {
    return this.withOwner(() => this.createOwned(snapshot, options))
  }

  private createOwned(snapshot: JournalSnapshot, options: JournalCreateOptions = {}): JournalAck {
    this.reload()
    const id = validateRolloutId(snapshot.id)
    const request = options.request
    const requestId = request ? validateRequestId(request.requestId) : null
    const payloadDigest = request ? digestPayload(request.payload) : null
    const existingRecord = this.records.get(id)
    if (existingRecord || this.summaries.get(id)?.tombstone) {
      if (existingRecord && requestId !== null && payloadDigest !== null) {
        const existing = existingRecord.requests[requestId]
        if (existing) {
          if (existing.payloadDigest !== payloadDigest) {
            throw new JournalError(
              'request-payload-mismatch',
              `Request ${requestId} was reused with a different payload.`
            )
          }
          return { ...clone(existing.ack), duplicate: true }
        }
      }
      throw new JournalError('already-exists', `Managed rollout ${id} already exists.`)
    }
    const now = this.now()
    const events = this.makeEvents(options.events, 1)
    const normalizedSnapshot = this.makeSnapshot(snapshot, id, 1, events.length, now)
    const requests: Record<string, StoredRequest> = {}
    const record: JournalRecord = {
      schemaVersion: JOURNAL_SCHEMA_VERSION,
      id,
      generation: randomUUID(),
      snapshot: normalizedSnapshot,
      events,
      requests,
      archive: options.archive ? clone(options.archive) : null,
      unresolved: this.applyFenceChange([], options.unresolved),
      facts: this.validateFacts(options.facts, id),
      createdAt: normalizedSnapshot.createdAt,
      updatedAt: now
    }
    if (record.archive) validateArchive(record.archive)
    const ack = this.prepareAck(
      id,
      1,
      events.map(item => item.sequence)
    )
    if (request && requestId !== null && payloadDigest !== null) {
      record.requests[requestId] = { payloadDigest, ack: clone(ack) }
    }

    const indexed = this.indexRecord(record, this.summaries, this.unresolved)
    try {
      this.writeJsonAtomic(this.recordPath(id), record)
      this.persistIndexes(indexed.summaries, indexed.unresolved)
    } catch (error) {
      return this.rethrowAfterReload(error)
    }
    this.records.set(id, record)
    this.summaries.clear()
    for (const [key, value] of indexed.summaries) this.summaries.set(key, value)
    this.unresolved.clear()
    for (const [key, value] of indexed.unresolved) this.unresolved.set(key, value)
    return clone(ack)
  }

  record(input: JournalRecordInput): JournalAck {
    return this.withOwner(() => this.recordOwned(input))
  }

  private recordOwned(input: JournalRecordInput): JournalAck {
    this.reload()
    const id = validateRolloutId(input.id)
    const current = this.records.get(id)
    if (!current) {
      if (this.summaries.get(id)?.tombstone) {
        throw new JournalError('replay-expired', `Managed rollout ${id} was compacted; replay is refused.`)
      }
      throw new JournalError('not-found', `Managed rollout ${id} was not found.`)
    }
    const requestId = validateRequestId(input.requestId)
    const payloadDigest = digestPayload(input.payload)
    const existing = current.requests[requestId]
    if (existing) {
      if (existing.payloadDigest !== payloadDigest) {
        throw new JournalError('request-payload-mismatch', `Request ${requestId} was reused with a different payload.`)
      }
      return { ...clone(existing.ack), duplicate: true }
    }
    if (!Number.isSafeInteger(input.expectedRevision) || input.expectedRevision !== current.snapshot.revision) {
      throw new JournalError(
        'stale-revision',
        `Managed rollout ${id} revision ${current.snapshot.revision} does not match expected ${input.expectedRevision}.`
      )
    }
    const now = this.now()
    const nextRevision = current.snapshot.revision + 1
    const events = this.makeEvents(input.events, current.events.length + 1)
    const nextSnapshot = this.makeSnapshot(
      input.snapshot,
      id,
      nextRevision,
      current.events.length + events.length,
      now,
      current.snapshot
    )
    const nextFacts = [...current.facts, ...this.validateFacts(input.facts, id)]
    const nextUnresolved = this.applyFenceChange(current.unresolved, input.unresolved)
    this.assertFenceRelease(current, nextFacts, nextUnresolved)
    const nextRecord: JournalRecord = {
      schemaVersion: JOURNAL_SCHEMA_VERSION,
      id,
      generation: randomUUID(),
      snapshot: nextSnapshot,
      events: [...current.events, ...events],
      requests: clone(current.requests),
      archive: input.archive === undefined ? clone(current.archive) : clone(input.archive),
      unresolved: nextUnresolved,
      facts: nextFacts,
      createdAt: current.createdAt,
      updatedAt: now
    }
    if (nextRecord.archive) validateArchive(nextRecord.archive)
    const ack = this.prepareAck(
      id,
      nextRevision,
      events.map(item => item.sequence)
    )
    nextRecord.requests[requestId] = { payloadDigest, ack: clone(ack) }
    const indexed = this.indexRecord(nextRecord, this.summaries, this.unresolved)
    try {
      this.writeJsonAtomic(this.recordPath(id), nextRecord)
      this.persistIndexes(indexed.summaries, indexed.unresolved)
    } catch (error) {
      return this.rethrowAfterReload(error)
    }
    this.records.set(id, nextRecord)
    this.summaries.clear()
    for (const [key, value] of indexed.summaries) this.summaries.set(key, value)
    this.unresolved.clear()
    for (const [key, value] of indexed.unresolved) this.unresolved.set(key, value)
    return clone(ack)
  }

  archive(input: JournalArchiveInput): JournalAck {
    const reason = validateReason(input.reason, 'Archive reason')
    const at = this.now()
    const current = this.getRecord(input.id)
    const snapshot = {
      ...current.snapshot,
      archivedAt: at
    } as JournalSnapshot
    return this.record({
      id: input.id,
      expectedRevision: input.expectedRevision,
      requestId: input.requestId,
      payload: input.payload,
      snapshot,
      events: [{ kind: 'archived', actor: input.actor, reason, installId: null, evidenceDigest: null }],
      archive: { at, actor: input.actor, reason }
    })
  }

  getRecord(id: string): JournalRecord {
    const normalized = validateRolloutId(id)
    const record = this.records.get(normalized)
    if (!record) throw new JournalError('not-found', `Managed rollout ${normalized} was not found.`)
    return clone(record)
  }

  read(id: string): JournalRecord {
    return this.getRecord(id)
  }

  events(id: string, options: { cursor?: string | null; limit?: number } = {}): EventPage {
    const record = this.getRecord(id)
    const limit = this.pageLimit(options.limit, MAX_EVENT_PAGE_SIZE)
    let after = 0
    if (options.cursor) {
      const cursor = decodeCursor(options.cursor, 'Event')
      if (
        cursor.kind !== 'events' ||
        cursor.id !== record.id ||
        !Number.isSafeInteger(cursor.sequence) ||
        Number(cursor.sequence) < 1
      ) {
        throw new JournalError('invalid-cursor', 'Event cursor belongs to another rollout or is invalid.')
      }
      after = Number(cursor.sequence)
    }
    const items = record.events.filter(item => item.sequence > after).slice(0, limit)
    const nextCursor =
      items.length === limit && items.length < record.events.length - after
        ? encodeCursor({ v: 1, kind: 'events', id: record.id, sequence: items[items.length - 1].sequence })
        : null
    return { items: clone(items), nextCursor }
  }

  history(options: { cursor?: string | null; limit?: number } = {}): HistoryPage {
    const limit = this.pageLimit(options.limit, MAX_HISTORY_PAGE_SIZE)
    const sorted = [...this.summaries.values()].sort(compareHistory)
    let candidates = sorted
    if (options.cursor) {
      const cursor = decodeCursor(options.cursor, 'History')
      const cursorCreatedAt = cursor.createdAt
      const cursorId = cursor.id
      if (cursor.kind !== 'history' || typeof cursorCreatedAt !== 'string' || typeof cursorId !== 'string') {
        throw new JournalError('invalid-cursor', 'History cursor is invalid.')
      }
      candidates = sorted.filter(
        item => item.createdAt < cursorCreatedAt || (item.createdAt === cursorCreatedAt && item.id < cursorId)
      )
    }
    const items = candidates.slice(0, limit)
    const nextCursor =
      items.length === limit && candidates.length > limit
        ? encodeCursor({
            v: 1,
            kind: 'history',
            createdAt: items[items.length - 1].createdAt,
            id: items[items.length - 1].id
          })
        : null
    return { items: clone(items), nextCursor }
  }

  unresolvedIndex(): UnresolvedIndexEntry[] {
    return clone([...this.unresolved.values()].sort((a, b) => a.key.localeCompare(b.key)))
  }

  hasUnresolvedInstall(installId: string): boolean {
    return [...this.unresolved.values()].some(entry => entry.installId === installId)
  }

  prune(): PruneResult {
    return this.withOwner(() => this.pruneOwned())
  }

  private pruneOwned(): PruneResult {
    this.reload()
    const records = [...this.records.values()]
      .filter(isSettled)
      .sort((a, b) => compareHistory(summaryFromRecord(a), summaryFromRecord(b)))
    const keep = new Set(records.slice(0, this.retentionLimit).map(record => record.id))
    const plans = records
      .filter(record => !keep.has(record.id))
      .map(record => ({
        record,
        prior: this.summaries.get(record.id) ?? summaryFromRecord(record)
      }))
    const retainedRecordIds = records.filter(record => keep.has(record.id)).map(record => record.id)
    if (!plans.length) return { prunedRecordIds: [], retainedRecordIds }

    const nextSummaries = new Map(this.summaries)
    const nextUnresolved = new Map(this.unresolved)
    const now = this.now()
    for (const { record, prior } of plans) {
      const hasFence = record.unresolved.length > 0
      if (hasFence) {
        for (const fence of record.unresolved) {
          nextUnresolved.set(fence.key, {
            ...clone(fence),
            tombstone: true,
            tombstoneAt: now
          })
        }
        nextSummaries.set(record.id, {
          ...prior,
          pruned: true,
          tombstone: true,
          prunedAt: now
        })
      } else {
        nextSummaries.delete(record.id)
      }
    }

    for (const { record } of plans) this.assertRegular(this.recordPath(record.id), `Managed rollout ${record.id}`)
    try {
      // Publish the derived indexes first. If deletion is interrupted, reopening
      // still prefers any surviving record over its tombstone, so no evidence is
      // lost and a later prune can retry the physical deletion.
      this.persistIndexes(nextSummaries, nextUnresolved)
    } catch (error) {
      return this.rethrowAfterReload(error)
    }

    const prunedRecordIds: string[] = []
    try {
      for (const { record } of plans) {
        this.fs.unlinkSync(this.recordPath(record.id))
        prunedRecordIds.push(record.id)
      }
    } catch (error) {
      const wrapped = new JournalError('write-failed', 'Could not prune a managed rollout record.', error)
      return this.rethrowAfterReload(wrapped)
    }

    this.reload()
    return { prunedRecordIds, retainedRecordIds }
  }

  private pageLimit(value: number | undefined, max: number): number {
    if (value === undefined) return max
    if (!Number.isSafeInteger(value) || value < 1 || value > max) {
      throw new JournalError('invalid-page-limit', `Page limit must be an integer from 1 to ${max}.`)
    }
    return value
  }
}

export function createManagedRolloutJournal(options: ManagedRolloutJournalOptions): ManagedRolloutJournal {
  return new ManagedRolloutJournal(options)
}
