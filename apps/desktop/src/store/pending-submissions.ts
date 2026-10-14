import { getQueuedPrompts, type QueuedPromptEntry, writeSessionQueue } from './composer-queue'

// One localStorage key per (session, admission). Every Desktop window of the origin shares this
// storage with last-writer-wins per KEY, so a single journal blob let one window's
// read-modify-write silently drop an entry another window had just tracked. Per-entry keys make
// each write touch only the record it owns.
const ENTRY_PREFIX = 'hermes.desktop.pendingSubmissions.v2:'
// The pre-v2 single blob, read for upgrade and drained per session as entries are rewritten.
const LEGACY_KEY = 'hermes.desktop.pendingSubmissions.v1'
interface PendingSubmission {
  id: string
  text: string
  displayText?: string
  status?: string
  /** Where in the session's event order the owner last listed this admission as pending. */
  seen?: PendingSnapshotFence
}

/** A canonical pending snapshot's place in its session's event order: the replay epoch and the
 *  event sequence it reflects (a live `session.info` frame's own seq, or a resume snapshot's
 *  `last_sequence`, both read under the owner's event-stream lock). */
export interface PendingSnapshotFence {
  epoch: string
  sequence: number
}

const entryKey = (session: string, id: string) => `${ENTRY_PREFIX}${JSON.stringify([session, id])}`

const parse = <T>(raw: null | string, fallback: T): T => {
  try {
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

/** Every tracked/observed pending submission of one session, keyed by id. */
export function readPendingSubmissions(session: string): Record<string, PendingSubmission> {
  const entries: Record<string, PendingSubmission> = { ...parse(window.localStorage.getItem(LEGACY_KEY), {} as Record<string, Record<string, PendingSubmission>>)[session] }

  for (let index = 0; index < window.localStorage.length; index += 1) {
    const key = window.localStorage.key(index)

    if (!key?.startsWith(ENTRY_PREFIX)) { continue }
    const [owner, id] = parse(key.slice(ENTRY_PREFIX.length), [] as unknown[])
    const entry = parse<null | PendingSubmission>(window.localStorage.getItem(key), null)

    if (owner === session && typeof id === 'string' && entry) { entries[id] = entry }
  }

  return entries
}

function writeEntries(session: string, before: Record<string, PendingSubmission>, after: Record<string, PendingSubmission>): void {
  for (const id of Object.keys(before)) {
    if (!(id in after)) { window.localStorage.removeItem(entryKey(session, id)) }
  }

  for (const [id, entry] of Object.entries(after)) {
    if (JSON.stringify(before[id]) !== JSON.stringify(entry)) { window.localStorage.setItem(entryKey(session, id), JSON.stringify(entry)) }
  }

  // Upgrade: once this session's entries live in v2 keys, drop its slice of the legacy blob.
  const legacy = parse(window.localStorage.getItem(LEGACY_KEY), {} as Record<string, unknown>)

  if (session in legacy) {
    for (const [id, entry] of Object.entries(after)) { window.localStorage.setItem(entryKey(session, id), JSON.stringify(entry)) }
    delete legacy[session]

    if (Object.keys(legacy).length) { window.localStorage.setItem(LEGACY_KEY, JSON.stringify(legacy)) }
    else { window.localStorage.removeItem(LEGACY_KEY) }
  }
}

// Not an outbox: an uncertain accepted input must never be replayed automatically.
export function trackPendingSubmission(key: string, entry: PendingSubmission): void {
  window.localStorage.setItem(entryKey(key, entry.id), JSON.stringify(entry))
}

// Array.isArray narrows `unknown` to `any[]`; the helpers keep that wire shape.
type RawReceipts = any[]

// An admission only moves forward: queued -> started -> (unknown after an owner restart) ->
// retired (gone from the pending set: terminal). Snapshots can arrive out of order (a resume
// result racing the live fanout, a replayed session.info), so a receipt weaker than the strongest
// state already observed for its admission is stale and never repaints the queue.
const STATUS_RANK: Record<string, number> = { queued: 0, started: 1, unknown: 2, retired: 3 }
// Retired identities are remembered per session only as long as a late snapshot can matter.
const RETIRED_MEMORY = 200

const isStale = (raw: { admission_id: string; status: string }, known: Record<string, PendingSubmission>) =>
  (STATUS_RANK[raw.status] ?? 0) < (STATUS_RANK[known[raw.admission_id]?.status ?? ''] ?? -1)

function collectReceipts(value: RawReceipts, known: Record<string, PendingSubmission>) {
  const receipts = new Map<string, PendingSubmission>()
  const admissionByInput = new Map<string, string>()

  for (const raw of value) {
    if (!raw || typeof raw.admission_id !== 'string' || !['queued', 'started', 'unknown'].includes(raw.status)) {
      continue
    }

    const id = raw.admission_id
    const stale = isStale(raw, known)

    // A stale status is still a listing: the admission is pending at its stronger observed state
    // (a replayed `started` keeps an `unknown` card and its Discard). A retired one stays retired.
    if (stale && known[id]?.status === 'retired') {
      continue
    }

    if (typeof raw.input_id === 'string') {
      admissionByInput.set(raw.input_id, id)
    }

    receipts.set(id, {
      ...known[id],
      id,
      text: typeof raw.user === 'string' ? raw.user : (known[id]?.text ?? ''),
      status: stale ? known[id].status : raw.status
    })
  }

  return { receipts, admissionByInput }
}

// Only a snapshot at least as new as the one that last listed an admission proves it left the
// pending set. An OLDER snapshot of the same numbering (a delayed resume answer, a replayed frame)
// predates it; its absence there says nothing. A different epoch is a later numbering (owner
// restart, evicted ring): sequences cannot be compared, and the current owner is authoritative.
// An unfenced snapshot (legacy shape) keeps the previous absence rule.
const predates = (fence: PendingSnapshotFence | undefined, seen: PendingSnapshotFence | undefined) =>
  Boolean(fence && seen && fence.epoch === seen.epoch && fence.sequence < seen.sequence)

function projectQueue(
  current: QueuedPromptEntry[],
  receipts: Map<string, PendingSubmission>,
  admissionByInput: Map<string, string>,
  stillPending: Set<string>
): QueuedPromptEntry[] {
  const next: QueuedPromptEntry[] = []

  for (const entry of current) {
    const receipt = receipts.get(admissionByInput.get(entry.id) ?? entry.id)

    if (receipt) {
      if (receipt.status !== 'started') {
        next.push({ ...entry, id: receipt.id, serverStatus: receipt.status })
      }

      receipts.delete(receipt.id)
    } else if (!entry.serverStatus || stillPending.has(entry.id)) {
      next.push(entry)
    }
  }

  for (const receipt of receipts.values()) {
    if (receipt.status !== 'started') {
      next.push({
        id: receipt.id,
        text: receipt.text,
        displayText: receipt.displayText,
        attachments: [],
        queuedAt: Date.now(),
        serverStatus: receipt.status
      })
    }
  }

  return next
}

function updateKnownReceipts(known: Record<string, PendingSubmission>, value: RawReceipts, fence?: PendingSnapshotFence): void {
  // Only observed server records may be retired by their later absence. The retirement is kept
  // (text dropped) so a stale snapshot that still lists the admission cannot resurrect its card.
  for (const [id, entry] of Object.entries(known)) {
    if (entry.status && entry.status !== 'retired' && !value.some(raw => raw?.admission_id === id) &&
        !predates(fence, entry.seen)) {
      known[id] = { id, text: '', status: 'retired' }
    }
  }

  const retired = Object.keys(known).filter(id => known[id].status === 'retired')

  for (const id of retired.slice(0, Math.max(0, retired.length - RETIRED_MEMORY))) {
    delete known[id]
  }

  for (const raw of value) {
    if (typeof raw?.admission_id === 'string' && !isStale(raw, known)) {
      const previous = known[raw.admission_id]

      known[raw.admission_id] = {
        ...previous,
        id: raw.admission_id,
        text: raw.user ?? previous?.text ?? '',
        status: raw.status,
        ...(fence && !predates(fence, previous?.seen) ? { seen: fence } : {})
      }
    }
  }
}

/** Project one pending snapshot onto the session's queue. `fence` orders canonical snapshots so a
 *  delayed older one cannot retire an admission a newer one still listed. */
export function reconcilePendingSubmissions(key: string, value: unknown, fence?: PendingSnapshotFence): void {
  if (!Array.isArray(value)) {
    return
  }

  const before = readPendingSubmissions(key)
  const known = { ...before }
  const { receipts, admissionByInput } = collectReceipts(value, known)

  const stillPending = new Set(Object.values(known).filter(entry =>
    entry.status && entry.status !== 'retired' && predates(fence, entry.seen) &&
    !value.some(raw => raw?.admission_id === entry.id)).map(entry => entry.id))

  const current = getQueuedPrompts(key)
  const next = projectQueue(current, receipts, admissionByInput, stillPending)

  updateKnownReceipts(known, value, fence)
  writeEntries(key, before, known)

  if (JSON.stringify(current) !== JSON.stringify(next)) {
    writeSessionQueue(key, next)
  }
}

/** The fence of a canonical frame or snapshot, when it carries a usable one. */
export function pendingSnapshotFence(epoch: unknown, sequence: unknown): PendingSnapshotFence | undefined {
  return typeof epoch === 'string' && epoch && typeof sequence === 'number' && Number.isFinite(sequence)
    ? { epoch, sequence }
    : undefined
}
