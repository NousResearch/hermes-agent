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
  /** Accepted, but not yet stored on another computer (`protected: false`). It stays here until a later status from
   * the running host covers its seq; if the group moves first, it is offered to the new host with the same identity.
   * A new host that refuses it for good (`refused`, its reason) hands it back to the composer as a draft. */
  unsaved?: { seq: number; event_id: string; reoffer?: boolean; refused?: string }
}

/** A valid `groups.send` acceptance. `protected` is absent on hosts that don't wait for other computers. */
export interface AcceptedCanonicalGroupSend { protected?: unknown; event?: { event_id?: unknown; seq?: unknown } }

// These reasons prove only this attempt had no effect, not any earlier attempt.
const TERMINAL_SEND_REFUSALS = new Set(['invalid_params', 'permission_denied', 'unknown_execution', 'stale_generation'])

/** `paused`: the host stored nothing because it paused to stay safe (`room_host_paused`), or because it promised the group to
 * another computer while a move waits (`room_authority_promised`). The message waits for the group to resume, there or on
 * the new host. */
export function sendOutcome(error: unknown): 'refused' | 'retryable' | 'unknown' | 'paused' {
  const failure = error as { code?: unknown; data?: { reason?: unknown } } | null

  if (failure?.code !== 4001) {return 'unknown'}

  if (failure.data?.reason === 'room_host_paused' || failure.data?.reason === 'room_authority_promised') {return 'paused'}

  return TERMINAL_SEND_REFUSALS.has(String(failure.data?.reason)) ? 'refused' : 'retryable'
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

const validUnsaved = (value: PreparedCanonicalGroupSend['unsaved']) => value === undefined || (!!value && typeof value === 'object' &&
  Number.isSafeInteger(value.seq) && value.seq > 0 && typeof value.event_id === 'string' && value.event_id.length > 0 &&
  [undefined, true, false].includes(value.reoffer) && (value.refused === undefined || typeof value.refused === 'string' && value.refused.length > 0))

/** Every valid entry of this room, including accepted ones still in the journal. */
async function journalRecords(binding: CanonicalGroupBinding): Promise<RecoverableCanonicalGroupSend[]> {
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
        [entry.attempted, entry.acknowledged, entry.held].some(flag => flag !== undefined && typeof flag !== 'boolean') || !validUnsaved(entry.unsaved)) {
      throw new Error('Invalid canonical group Send entry')
    }

    snapshots.set(entry, JSON.stringify(entry))
    result.push({ entry, storageKey, expected: JSON.stringify(entry) })
  }

  return result
}

/** Sends the room may still owe; an accepted one, even while it settles, is never offered for Send or Restore. */
async function records(binding: CanonicalGroupBinding): Promise<RecoverableCanonicalGroupSend[]> {
  return (await journalRecords(binding)).filter(({ entry, storageKey }) => !entry.acknowledged && !acknowledged.has(storageKey))
}

async function unsavedRecords(binding: CanonicalGroupBinding): Promise<RecoverableCanonicalGroupSend[]> {
  return (await journalRecords(binding)).filter(({ entry, storageKey }) => entry.acknowledged && entry.unsaved && !acknowledged.has(storageKey))
}

/** `protected: false` with the stored event's identity. Absent or true (older hosts, or saved) settles as before. */
function unsavedOf(accepted: AcceptedCanonicalGroupSend | undefined): PreparedCanonicalGroupSend['unsaved'] {
  const event = accepted?.event

  return accepted?.protected === false && event && Number.isSafeInteger(event.seq) && (event.seq as number) > 0 &&
    typeof event.event_id === 'string' && event.event_id ? { seq: event.seq as number, event_id: event.event_id } : undefined
}

/** A valid acceptance names this Send's identity; anything else leaves the Send unconfirmed. */
export function acceptedCanonicalGroupSend(result: unknown, entry: PreparedCanonicalGroupSend, unconfirmed: string): AcceptedCanonicalGroupSend {
  const value = result as (AcceptedCanonicalGroupSend & { accepted?: unknown; client_event_id?: unknown }) | null | undefined

  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      (Object.hasOwn(value, 'accepted') && value.accepted !== true) || value.client_event_id !== entry.params.event_id) {
    throw new Error(unconfirmed)
  }

  return value
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

/** A first attempt a paused host refused stored nothing: the message is held again, to go out once the group resumes. */
export async function holdCanonicalGroupSend(binding: CanonicalGroupBinding, entry: PreparedCanonicalGroupSend): Promise<void> {
  const owner = await journalOwner()

  if (!entry.journal || entry.journal.owner !== owner || roomKey(entry.binding) !== roomKey(binding)) {throw new Error('Prepared draft ownership changed')}
  const expected = snapshots.get(entry)
  const next = JSON.stringify({ ...entry, attempted: false, held: true })

  if (!expected || JSON.stringify(entry) !== expected || !await compareJournal(entry.journal.storageKey, expected, next)) {
    throw new Error('Prepared draft changed before it was held')
  }

  entry.attempted = false
  entry.held = true
  snapshots.set(entry, next)
}

/** Valid acceptance is authoritative; local cleanup cannot request new work. A message no other computer holds yet
 * stays in the journal, accepted, so it can be offered again if the group moves before a copy has it. */
export async function settleCanonicalGroupSend(binding: CanonicalGroupBinding, entry: PreparedCanonicalGroupSend,
  accepted?: AcceptedCanonicalGroupSend): Promise<void> {
  if (!entry.journal) {return}
  acknowledged.add(entry.journal.storageKey)

  try {
    const owner = await journalOwner()
    const expected = snapshots.get(entry)

    if (entry.journal.owner !== owner || !expected) {throw new Error('Prepared draft ownership changed')}
    entry.acknowledged = true
    entry.unsaved = unsavedOf(accepted)
    const next = JSON.stringify(entry)

    if (!await compareJournal(entry.journal.storageKey, expected, next)) {throw new Error('Prepared draft changed before acknowledgement')}
    snapshots.set(entry, next)

    if (!entry.unsaved) {await retireCanonicalGroupSend(binding, entry.params.event_id, entry)}
    acknowledged.delete(entry.journal.storageKey)
  } catch (error) {
    console.warn('Accepted group Send journal cleanup failed', error)
  }
}

/** Accepted messages that no other computer held yet when they were sent. */
export async function listUnsavedCanonicalGroupSends(binding: CanonicalGroupBinding): Promise<RecoverableCanonicalGroupSend[]> {
  return unsavedRecords(binding)
}

/** The running host reports these stored on enough computers now (`protected_seq`); a message waiting to be offered
 * again after a move, or refused by the new host, is never cleared by a seq from before it. */
export async function clearUnsavedCanonicalGroupSends(binding: CanonicalGroupBinding, protectedSeq: number): Promise<void> {
  for (const record of await unsavedRecords(binding)) {
    const unsaved = record.entry.unsaved!

    if (!unsaved.reoffer && !unsaved.refused && unsaved.seq <= protectedSeq) {await compareJournal(record.storageKey, record.expected, null)}
  }
}

/** The new host refused a message offered again for good: it is never offered again, and waits to be edited. */
export async function refuseReofferedCanonicalGroupSend(record: RecoverableCanonicalGroupSend, reason: string): Promise<void> {
  await compareJournal(record.storageKey, record.expected, JSON.stringify({ ...record.entry,
    unsaved: { ...record.entry.unsaved!, reoffer: false, refused: reason } }))
}

/** Takes a refused message out of the journal for the composer; only one view gets it. */
export async function reclaimRefusedCanonicalGroupSend(record: RecoverableCanonicalGroupSend): Promise<Record<string, unknown> | null> {
  return await compareJournal(record.storageKey, record.expected, null) ? record.entry.params.payload : null
}

/** A message offered again on the new host: done once saved there, or kept with its new seq until it is. */
export async function reofferedCanonicalGroupSend(record: RecoverableCanonicalGroupSend, accepted: AcceptedCanonicalGroupSend): Promise<void> {
  const unsaved = unsavedOf(accepted)

  await compareJournal(record.storageKey, record.expected, unsaved ? JSON.stringify({ ...record.entry, unsaved }) : null)
}

/** A group that continues on another computer keeps its unsent messages: each entry moves to the new route with
 * its original event id, owner and attempt state, so the new host recognizes a message the old one accepted.
 * Accepted messages no other computer held yet move too, to be offered again there. */
export async function rehomeCanonicalGroupSends(from: CanonicalGroupBinding, to: CanonicalGroupBinding): Promise<void> {
  if (from.roomId !== to.roomId) {throw new Error('A group keeps its room id when it moves')}
  roomKey(to)

  for (const record of [...await records(from), ...await unsavedRecords(from)]) {
    const entry = record.entry.unsaved ? { ...record.entry, unsaved: { ...record.entry.unsaved, reoffer: true } } : record.entry
    const storageKey = JSON.stringify(['canonical-group-send-v2', to.connectionId, to.profile, to.roomId, entry.params.event_id])
    const next = JSON.stringify({ ...entry, binding: { ...to }, journal: { owner: entry.journal?.owner ?? await journalOwner(), storageKey } })

    if (!await compareJournal(storageKey, null, next)) {continue}

    if (!await compareJournal(record.storageKey, record.expected, null)) {await compareJournal(storageKey, next, null)}
  }
}
