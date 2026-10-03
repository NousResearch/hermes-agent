import type { CanonicalGroupBinding } from './canonical-groups'

const STORAGE_KEY = 'hermes.desktop.canonicalGroupSends.v1'
let browserOwner: string | undefined
const snapshots = new WeakMap<PreparedCanonicalGroupSend, string>()
const acknowledged = new Set<string>()
const selected = new Map<string, string>()

export interface PreparedCanonicalGroupSend {
  binding: CanonicalGroupBinding
  params: { room_id: string; event_id: string; payload: Record<string, unknown> }
  journal?: { owner: string; storageKey: string }
  attempted?: boolean
  acknowledged?: boolean
  /** Written while the group was paused: Desktop sends it once the group resumes. */
  held?: boolean
}

export interface RecoverableCanonicalGroupSend {
  entry: PreparedCanonicalGroupSend
  storageKey: string
  expected: string
}

function roomKey(binding: CanonicalGroupBinding): string {
  if (![binding.connectionId, binding.profile, binding.roomId].every(value => typeof value === 'string' && value.trim())) {
    throw new Error('Canonical group Send requires an explicit connection, profile and room')
  }

  return JSON.stringify([binding.connectionId, binding.profile, binding.roomId])
}

function nativeJournal() {
  const desktop = window.hermesDesktop

  if (desktop === undefined) {return undefined}
  const native = desktop?.preparedSubmissions

  if (!native || typeof native.read !== 'function' || typeof native.owner !== 'function' || typeof native.compareSend !== 'function') {
    throw new Error('Atomic draft storage unavailable; update Desktop')
  }

  return { read: native.read, owner: native.owner, compareSend: native.compareSend }
}

async function journalOwner(): Promise<string> {
  const native = nativeJournal()

  if (!native) {return browserOwner ??= crypto.randomUUID()}
  const owner = await native.owner()

  if (typeof owner !== 'string' || !owner.trim()) {throw new Error('Prepared draft ownership is unavailable')}

  return owner
}

async function readJournal(): Promise<Record<string, PreparedCanonicalGroupSend>> {
  const native = nativeJournal()
  const parsed: unknown = JSON.parse(native ? await native.read() : window.localStorage.getItem(STORAGE_KEY) || '{}')

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {throw new Error('Invalid canonical group Send journal')}

  return parsed as Record<string, PreparedCanonicalGroupSend>
}

async function compareJournal(key: string, expected: string | null, entry: string | null): Promise<boolean> {
  const native = nativeJournal()

  if (native) {
    return native.compareSend(key, expected, entry)
  }

  if (!navigator.locks) {throw new Error('Atomic draft storage unavailable in this browser')}

  return navigator.locks.request(STORAGE_KEY, () => {
    const journal = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}')

    if (!journal || typeof journal !== 'object' || Array.isArray(journal)) {throw new Error('Invalid canonical group Send journal')}
    const current = Object.hasOwn(journal, key) ? JSON.stringify(journal[key]) : null

    if (current !== expected) {return false}

    if (entry === null) {delete journal[key]}
    else {Object.defineProperty(journal, key, { value: JSON.parse(entry), enumerable: true, configurable: true })}

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(journal))

    return true
  })
}

async function records(binding: CanonicalGroupBinding): Promise<RecoverableCanonicalGroupSend[]> {
  const room = roomKey(binding)
  const result: RecoverableCanonicalGroupSend[] = []

  for (const [storageKey, entry] of Object.entries(await readJournal())) {
    let key: unknown

    try {key = JSON.parse(storageKey)} catch {continue}

    if (!Array.isArray(key) || !['canonical-group-send-v1', 'canonical-group-send-v2'].includes(key[0]) ||
        JSON.stringify(key.slice(1, 4)) !== room) {continue}

    if (!entry || typeof entry !== 'object' || !entry.binding || roomKey(entry.binding) !== room ||
        entry.params?.room_id !== binding.roomId || typeof entry.params.event_id !== 'string' || !entry.params.event_id ||
        !entry.params.payload || typeof entry.params.payload !== 'object' || Array.isArray(entry.params.payload) ||
        (key[0] === 'canonical-group-send-v1' && key.length !== 4) ||
        (key[0] === 'canonical-group-send-v2' && (key.length !== 5 || key[4] !== entry.params.event_id || !entry.journal)) ||
        (entry.journal && (entry.journal.storageKey !== storageKey || typeof entry.journal.owner !== 'string' || !entry.journal.owner)) ||
        (entry.attempted !== undefined && typeof entry.attempted !== 'boolean') ||
        (entry.acknowledged !== undefined && typeof entry.acknowledged !== 'boolean') ||
        (entry.held !== undefined && typeof entry.held !== 'boolean')) {
      throw new Error('Invalid canonical group Send entry')
    }

    if (entry.acknowledged || acknowledged.has(storageKey)) {continue}
    snapshots.set(entry, JSON.stringify(entry))
    result.push({ entry, storageKey, expected: JSON.stringify(entry) })
  }

  return result
}

function selectionKey(binding: CanonicalGroupBinding, owner: string): string {
  return JSON.stringify([owner, roomKey(binding)])
}

export async function readCanonicalGroupSend(binding: CanonicalGroupBinding): Promise<PreparedCanonicalGroupSend | undefined> {
  const owner = await journalOwner()
  const own = (await records(binding)).filter(record => record.entry.journal?.owner === owner)
  const chosen = selected.get(selectionKey(binding, owner))

  return (chosen ? own.find(record => record.storageKey === chosen) : own.length === 1 ? own[0] : undefined)?.entry
}

export async function listCanonicalGroupSends(binding: CanonicalGroupBinding): Promise<RecoverableCanonicalGroupSend[]> {
  await journalOwner()

  return records(binding)
}

/** Explicit recovery transfers the exact frozen intent; it never sends it. */
export async function claimCanonicalGroupSend(binding: CanonicalGroupBinding, recovery: RecoverableCanonicalGroupSend): Promise<PreparedCanonicalGroupSend> {
  const owner = await journalOwner()
  const record = (await records(binding)).find(record => record.storageKey === recovery.storageKey)

  if (!record || record.expected !== recovery.expected) {throw new Error('Prepared draft changed during recovery')}
  const next = { ...record.entry, journal: { owner, storageKey: record.storageKey } }
  const snapshot = JSON.stringify(next)

  if (!await compareJournal(record.storageKey, record.expected, snapshot)) {throw new Error('Prepared draft changed during recovery')}
  snapshots.set(next, snapshot)
  selected.set(selectionKey(binding, owner), record.storageKey)

  return next
}

export async function prepareCanonicalGroupSend(binding: CanonicalGroupBinding, payload: Record<string, unknown>,
  options: { held?: boolean } = {}): Promise<PreparedCanonicalGroupSend> {
  const scope = { ...binding }
  roomKey(scope)
  const owner = await journalOwner()
  const pending = (await records(scope)).filter(record => record.entry.journal?.owner === owner)
  const chosen = selected.get(selectionKey(scope, owner))
  const existing = chosen ? pending.find(record => record.storageKey === chosen)?.entry : pending.length === 1 ? pending[0].entry : undefined

  if (existing) {
    const requested = { ...payload, thread_id: payload.thread_id ?? existing.params.payload.thread_id }

    if (JSON.stringify(existing.params.payload) !== JSON.stringify(requested)) {throw new Error('A pending Send requires explicit recovery before sending another message')}

    return existing
  }

  if (pending.length) {throw new Error('Multiple prepared drafts require explicit recovery')}
  const eventId = crypto.randomUUID()
  const storageKey = JSON.stringify(['canonical-group-send-v2', scope.connectionId, scope.profile, scope.roomId, eventId])

  const entry: PreparedCanonicalGroupSend = JSON.parse(JSON.stringify({ binding: scope,
    params: { room_id: scope.roomId, event_id: eventId, payload: { ...payload, thread_id: payload.thread_id ?? eventId } },
    journal: { owner, storageKey }, attempted: false, ...options.held ? { held: true } : {} }))

  const snapshot = JSON.stringify(entry)

  if (!await compareJournal(storageKey, null, snapshot)) {throw new Error('Prepared draft changed before write')}
  snapshots.set(entry, snapshot)

  return entry
}

export async function retireCanonicalGroupSend(binding: CanonicalGroupBinding, eventId: string, captured?: PreparedCanonicalGroupSend): Promise<void> {
  const entry = captured ?? await readCanonicalGroupSend(binding)

  if (!entry || entry.params.event_id !== eventId || roomKey(entry.binding) !== roomKey(binding)) {return}
  const owner = await journalOwner()

  if (entry.journal?.owner !== owner) {throw new Error('Prepared draft ownership changed')}
  const expected = snapshots.get(entry)

  if (!expected || !await compareJournal(entry.journal.storageKey, expected, null)) {throw new Error('Prepared draft changed before retirement')}
  snapshots.delete(entry)
  selected.delete(selectionKey(binding, owner))
}

/** Publish uncertainty before dispatch; missing legacy state is already uncertain. */
export async function attemptCanonicalGroupSend(binding: CanonicalGroupBinding, entry: PreparedCanonicalGroupSend): Promise<boolean> {
  const owner = await journalOwner()

  if (!entry.journal || entry.journal.owner !== owner || roomKey(entry.binding) !== roomKey(binding)) {throw new Error('Prepared draft ownership changed')}
  const expected = snapshots.get(entry)

  if (!expected || JSON.stringify(entry) !== expected) {throw new Error('Prepared draft changed before Retry')}
  const fresh = entry.attempted === false
  const next = JSON.stringify({ ...entry, attempted: true })

  if (!await compareJournal(entry.journal.storageKey, expected, next)) {throw new Error('Prepared draft changed before Retry')}
  entry.attempted = true
  snapshots.set(entry, next)

  return fresh
}

/** Valid acceptance is authoritative; local cleanup cannot request new work. */
export async function settleCanonicalGroupSend(binding: CanonicalGroupBinding, entry: PreparedCanonicalGroupSend): Promise<void> {
  if (!entry.journal) {return}
  acknowledged.add(entry.journal.storageKey)

  try {
    const owner = await journalOwner()
    const expected = snapshots.get(entry)

    if (entry.journal.owner !== owner || !expected) {throw new Error('Prepared draft ownership changed')}
    entry.acknowledged = true
    const next = JSON.stringify(entry)

    if (!await compareJournal(entry.journal.storageKey, expected, next)) {throw new Error('Prepared draft changed before acknowledgement')}
    snapshots.set(entry, next)
    await retireCanonicalGroupSend(binding, entry.params.event_id, entry)
    acknowledged.delete(entry.journal.storageKey)
  } catch (error) {
    console.warn('Accepted group Send journal cleanup failed', error)
  }
}

/** A group that continues on another computer keeps its unsent messages: each entry moves to the new route with
 * its original event id, owner and attempt state, so the new host recognizes a message the old one accepted. */
export async function rehomeCanonicalGroupSends(from: CanonicalGroupBinding, to: CanonicalGroupBinding): Promise<void> {
  if (from.roomId !== to.roomId) {throw new Error('A group keeps its room id when it moves')}
  roomKey(to)

  for (const record of await records(from)) {
    const { entry } = record
    const storageKey = JSON.stringify(['canonical-group-send-v2', to.connectionId, to.profile, to.roomId, entry.params.event_id])
    const next = JSON.stringify({ ...entry, binding: { ...to }, journal: { owner: entry.journal?.owner ?? await journalOwner(), storageKey } })

    if (!await compareJournal(storageKey, null, next)) {continue}

    if (!await compareJournal(record.storageKey, record.expected, null)) {await compareJournal(storageKey, next, null)}
  }
}
