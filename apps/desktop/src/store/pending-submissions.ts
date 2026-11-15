import { noteListedSubmissions } from './admitted-submissions'
import { getQueuedPrompts, type QueuedPromptEntry, queuedSubmissionId, writeSessionQueue } from './composer-queue'
import { $sessions, resolveComposerSessionKey } from './session'

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
  /** The snapshot's queued-row controls; carried to the card only, never journaled. */
  controls?: string[]
}

/** A canonical pending snapshot's place in its session's event order: the replay epoch and the
 *  event sequence it reflects (a live `session.info` frame's own seq, or a resume snapshot's
 *  `last_sequence`, both read under the owner's event-stream lock). */
export interface PendingSnapshotFence {
  epoch: string
  sequence: number
}

const entryKey = (session: string, id: string) => `${ENTRY_PREFIX}${JSON.stringify([session, id])}`

// The journal stores cards, not payloads: its text is display-only (queue drains
// never re-send server-owned rows), so a submit's full expanded text (slash-skill
// expansions reach ~280k chars) is capped before it enters localStorage. Uncapped
// pre-submit writes filled the 50 MiB-char origin quota and aborted submits (#ATLAS-125).
const JOURNAL_TEXT_LIMIT = 4096

const boundedEntry = (entry: PendingSubmission): PendingSubmission => {
  const text = boundedText(entry.text)
  const displayText = typeof entry.displayText === 'string' ? boundedText(entry.displayText) : entry.displayText

  return text === entry.text && displayText === entry.displayText ? entry : { ...entry, text, displayText }
}

const boundedText = (text: string): string => (text.length > JOURNAL_TEXT_LIMIT ? text.slice(0, JOURNAL_TEXT_LIMIT) : text)

const isQuotaError = (error: unknown): boolean =>
  error instanceof DOMException && (error.name === 'QuotaExceededError' || error.code === 22)

/** Drop the least informative journal entries first: a pre-track the owner never
 *  gave a status (never replayed by design), then retired records, then settled
 *  cards. localStorage enumerates in insertion order, so ties resolve oldest-first. */
const EVICTION_RANK: Record<string, number> = { '': 0, retired: 1, queued: 2, started: 3, unknown: 4 }

function evictJournalEntry(keepKey: string, maxRank: number): boolean {
  let victim: { key: string; rank: number } | undefined

  for (let index = 0; index < window.localStorage.length; index += 1) {
    const key = window.localStorage.key(index)

    if (!key?.startsWith(ENTRY_PREFIX) || key === keepKey) { continue }
    const entry = parse<null | PendingSubmission>(window.localStorage.getItem(key), null)
    const rank = EVICTION_RANK[entry?.status ?? ''] ?? 5

    if (rank <= maxRank && (!victim || rank < victim.rank)) { victim = { key, rank } }
  }

  if (!victim) { return false }
  window.localStorage.removeItem(victim.key)

  return true
}

/** setItem that cannot take a submit down with it: on a full origin it evicts
 *  journal entries (see evictJournalEntry; a statusless pre-track may evict only
 *  statusless or retired records, never live cards) and retries; an exhausted or
 *  unavailable storage logs and returns false instead of throwing (#ATLAS-125). */
function safeSetEntry(key: string, entry: PendingSubmission): boolean {
  const value = JSON.stringify(boundedEntry(entry))
  const maxRank = entry.status ? 5 : 1

  for (;;) {
    try {
      window.localStorage.setItem(key, value)

      return true
    } catch (error) {
      if (!isQuotaError(error) || !evictJournalEntry(key, maxRank)) {
        console.warn('[pending-submissions] journal write skipped (storage full or unavailable)', { key: key.slice(ENTRY_PREFIX.length, ENTRY_PREFIX.length + 80) })

        return false
      }
    }
  }
}

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

/** Sessions the journal still holds an unsettled admission for (queued, started or unknown when
 *  last observed): the ones whose owner may have moved on while no window was watching. */
export function journaledPendingSessions(): string[] {
  const sessions = new Set<string>()
  const unsettled = (entry: null | PendingSubmission | undefined) => Boolean(entry?.status && entry.status !== 'retired')

  for (const [session, entries] of Object.entries(parse(window.localStorage.getItem(LEGACY_KEY), {} as Record<string, Record<string, PendingSubmission>>))) {
    if (Object.values(entries ?? {}).some(unsettled)) { sessions.add(session) }
  }

  for (let index = 0; index < window.localStorage.length; index += 1) {
    const key = window.localStorage.key(index)

    if (!key?.startsWith(ENTRY_PREFIX)) { continue }
    const [owner] = parse(key.slice(ENTRY_PREFIX.length), [] as unknown[])

    if (typeof owner === 'string' && unsettled(parse<null | PendingSubmission>(window.localStorage.getItem(key), null))) { sessions.add(owner) }
  }

  return [...sessions]
}

/** One admission's record without enumerating the journal: its v2 key, else the legacy blob. */
function readPendingSubmission(session: string, id: string): PendingSubmission | undefined {
  return parse<null | PendingSubmission>(window.localStorage.getItem(entryKey(session, id)), null) ??
    parse(window.localStorage.getItem(LEGACY_KEY), {} as Record<string, Record<string, PendingSubmission>>)[session]?.[id]
}

function writeEntries(session: string, before: Record<string, PendingSubmission>, after: Record<string, PendingSubmission>): boolean {
  for (const id of Object.keys(before)) {
    if (!(id in after)) { window.localStorage.removeItem(entryKey(session, id)) }
  }

  let allWritten = true

  for (const [id, entry] of Object.entries(after)) {
    if (JSON.stringify(before[id]) !== JSON.stringify(boundedEntry(entry))) {
      allWritten = safeSetEntry(entryKey(session, id), entry) && allWritten
    }
  }

  // Upgrade: once this session's entries live in v2 keys, drop its slice of the legacy blob.
  const legacy = parse(window.localStorage.getItem(LEGACY_KEY), {} as Record<string, unknown>)

  if (session in legacy) {
    for (const [id, entry] of Object.entries(after)) {
      allWritten = safeSetEntry(entryKey(session, id), entry) && allWritten
    }

    if (allWritten) {
      delete legacy[session]

      try {
        if (Object.keys(legacy).length) { window.localStorage.setItem(LEGACY_KEY, JSON.stringify(legacy)) }
        else { window.localStorage.removeItem(LEGACY_KEY) }
      } catch {
        // Keep the migrated v2 entries even if rewriting the legacy blob hits quota.
      }
    }
  }

  return allWritten
}

// An admission only moves forward: queued -> started -> (unknown after an owner restart) ->
// retired (gone from the pending set: terminal). Snapshots can arrive out of order (a resume
// result racing the live fanout, a replayed session.info), so a receipt weaker than the strongest
// state already observed for its admission is stale and never repaints the queue.
const STATUS_RANK: Record<string, number> = { queued: 0, started: 1, unknown: 2, retired: 3 }

// The one journal identity for tracking and reconciliation: the key the queue panel reads. After
// compression that is the lineage root, not the tip a submit targets or the runtime reports.
const journalKey = (storedId: string) => resolveComposerSessionKey(storedId, $sessions.get()) ?? storedId

const rank = (entry: PendingSubmission) => STATUS_RANK[entry.status ?? ''] ?? -1

// Within one numbering the later sequence; across epochs they cannot be compared, keep the first.
const laterFence = (a?: PendingSnapshotFence, b?: PendingSnapshotFence) =>
  a && b && a.epoch === b.epoch && b.sequence > a.sequence ? b : (a ?? b)

/** Two records of one admission as one: the stronger status (the newer on a tie), the text and
 *  display text either knows, and the later fence. A retirement stays bare. */
function mergeReceipt(held: PendingSubmission | undefined, next: PendingSubmission): PendingSubmission {
  if (!held) {
    return next
  }

  const [strong, weak] = rank(next) >= rank(held) ? [next, held] : [held, next]

  if (strong.status === 'retired') {
    return strong
  }

  const seen = laterFence(strong.seen, weak.seen)

  return {
    ...weak,
    ...strong,
    text: strong.text || weak.text,
    displayText: strong.displayText ?? weak.displayText,
    ...(seen ? { seen } : {})
  }
}

/** Fold a journal kept under `from` into `to` once `from` resolves there (a tip journaled before
 *  its lineage row loaded), so its status, display text and fences keep guarding the card. */
function adoptJournal(from: string, to: string): boolean {
  const stray = from === to ? {} : readPendingSubmissions(from)

  if (!Object.keys(stray).length) {
    return true
  }

  const before = readPendingSubmissions(to)
  const merged = { ...before }

  for (const [id, entry] of Object.entries(stray)) {
    merged[id] = mergeReceipt(merged[id], entry)
  }

  if (!writeEntries(to, before, merged)) {
    return false
  }

  writeEntries(from, stray, {})

  return true
}

// Tip->root lineages whose tip journal tracking already folded. Adoption enumerates the journal,
// so a send pays for it only when it first observes the lineage; reconciliation still adopts
// whatever another window journals under the tip afterwards.
const adoptedLineages = new Set<string>()

// Not an outbox: an uncertain accepted input must never be replayed automatically.
export function trackPendingSubmission(storedId: string, entry: PendingSubmission): void {
  const key = journalKey(storedId)
  const lineage = JSON.stringify([storedId, key])

  if (storedId !== key && !adoptedLineages.has(lineage) && adoptJournal(storedId, key)) {
    adoptedLineages.add(lineage)
  }

  const held = readPendingSubmission(key, entry.id)
  safeSetEntry(entryKey(key, entry.id), mergeReceipt(held, entry))
}

/** Remove a pre-tracked entry that never became a pending admission: the owner's
 *  ACK named no identity for it (a legacy `hermes serve` contract), or the submit
 *  failed before admission. No card exists, so the pre-track is dropped rather
 *  than leaked into the journal forever (#ATLAS-125). Never touches an entry the
 *  owner has given a status — that one may still be a live pending card. */
export function dropPendingSubmission(storedId: string, id: string): void {
  const key = journalKey(storedId)
  const entry = readPendingSubmission(key, id)

  if (!entry || entry.status) { return }
  window.localStorage.removeItem(entryKey(key, id))
}

/** One-time startup sweep (#ATLAS-125): at module load no submit is in flight, so
 *  a statusless entry can only be a leftover pre-track from a previous run whose
 *  ACK never named an admission identity — its card never existed, and it can
 *  never be upgraded by this renderer's submit path. Drop it; a pending fanout
 *  that still lists the send recreates the card under its admission id. Also
 *  rewrite any oversized text a previous build stored, reclaiming the origin. */
function sweepJournalOnLoad(): void {
  for (let index = window.localStorage.length - 1; index >= 0; index -= 1) {
    const key = window.localStorage.key(index)

    if (!key?.startsWith(ENTRY_PREFIX)) { continue }
    const entry = parse<null | PendingSubmission>(window.localStorage.getItem(key), null)

    if (!entry) { continue }

    if (!entry.status) {
      window.localStorage.removeItem(key)
    } else if (entry.text.length > JOURNAL_TEXT_LIMIT || (typeof entry.displayText === 'string' && entry.displayText.length > JOURNAL_TEXT_LIMIT)) {
      safeSetEntry(key, entry)
    }
  }
}

sweepJournalOnLoad()

// Array.isArray narrows `unknown` to `any[]`; the helpers keep that wire shape.
type RawReceipts = any[]

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

    const inputEntry = typeof raw.input_id === 'string' && raw.input_id !== id ? known[raw.input_id] : undefined
    const previous = inputEntry ? mergeReceipt(inputEntry, known[id] ?? { id, text: '' }) : known[id]

    receipts.set(id, {
      ...previous,
      id,
      text: typeof raw.user === 'string' ? boundedText(raw.user) : (previous?.text ?? ''),
      status: stale ? known[id].status : raw.status,
      ...(Array.isArray(raw.controls) ? { controls: raw.controls.filter((c: unknown) => typeof c === 'string') } : {})
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
  // The owner lists its admissions in FIFO order: a card it lists is shown in that order, so an
  // admission back in its queue place (a steer that never reached the turn) is not shown last.
  const order = new Map([...receipts.keys()].map((id, index) => [id, index]))
  const next: QueuedPromptEntry[] = []

  for (const entry of current) {
    const submissionId = queuedSubmissionId(entry)
    const receipt = receipts.get(admissionByInput.get(submissionId) ?? submissionId)

    if (receipt) {
      if (receipt.status !== 'started') {
        next.push({ ...entry, id: receipt.id, serverStatus: receipt.status, serverControls: receipt.controls })
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
        serverStatus: receipt.status,
        serverControls: receipt.controls
      })
    }
  }

  return inOwnerOrder(next, order)
}

/** Listed cards take the owner's order among the slots they hold; every other card keeps its slot. */
function inOwnerOrder(entries: QueuedPromptEntry[], order: Map<string, number>): QueuedPromptEntry[] {
  const slots = entries.flatMap((entry, index) => (order.has(entry.id) ? [index] : []))
  const listed = slots.map(index => entries[index]).sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0))
  const next = [...entries]

  slots.forEach((slot, index) => {
    next[slot] = listed[index]
  })

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

  // A fenced snapshot lists the owner's pending admissions and knows their input
  // identities. A statusless pre-tracked submit it mentions under neither id was
  // admitted elsewhere (its input key is folded below), refused, or lost — no
  // card exists for it, so the pre-track is dropped instead of leaking per send
  // (#ATLAS-125). Unfenced (legacy) snapshots say nothing and retire nothing.
  if (fence) {
    for (const [id, entry] of Object.entries(known)) {
      if (!entry.status && !value.some(raw => raw?.admission_id === id || raw?.input_id === id)) {
        delete known[id]
      }
    }
  }

  const retired = Object.keys(known).filter(id => known[id].status === 'retired')

  for (const id of retired.slice(0, Math.max(0, retired.length - RETIRED_MEMORY))) {
    delete known[id]
  }

  for (const raw of value) {
    if (typeof raw?.admission_id === 'string' && !isStale(raw, known)) {
      // The pre-track was written under the client submission id; the owner may
      // admit it under a different admission id. Fold the input-keyed record
      // into the admission key so its journal entry is reused, not duplicated.
      const inputEntry = typeof raw.input_id === 'string' && raw.input_id !== raw.admission_id ? known[raw.input_id] : undefined
      const previous = inputEntry ? mergeReceipt(inputEntry, known[raw.admission_id] ?? { id: raw.admission_id, text: '' }) : known[raw.admission_id]

      known[raw.admission_id] = {
        ...previous,
        id: raw.admission_id,
        text: typeof raw.user === 'string' ? boundedText(raw.user) : (previous?.text ?? ''),
        // A steer in transfer reads `started` but can still go back to its queue place: it never
        // raises the admission's rank, so the owner's later `queued` listing repaints its card.
        status: raw.steering === true ? (previous?.status ?? 'queued') : raw.status,
        ...(fence && !predates(fence, previous?.seen) ? { seen: fence } : {})
      }

      if (inputEntry) { delete known[raw.input_id] }
    }
  }
}

/** Project one pending snapshot onto the session's queue. `fence` orders canonical snapshots so a
 *  delayed older one cannot retire an admission a newer one still listed. */
export function reconcilePendingSubmissions(storedId: string, value: unknown, fence?: PendingSnapshotFence): void {
  if (!Array.isArray(value)) {
    return
  }

  // Project onto the key the queue panel reads: only snapshots projected there can retire a card
  // shown there. Receipts journaled under the tip before the lineage was known move along.
  const key = journalKey(storedId)
  adoptJournal(storedId, key)
  noteListedSubmissions(key, value.flatMap(raw => (typeof raw?.input_id === 'string' ? [raw.input_id] : [])))

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
