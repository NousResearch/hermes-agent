import type { ChatMessage } from '@/lib/chat-messages'
import { withUniqueToolCallIdsWithinMessage } from '@/lib/chat-messages'

const PREFIX = 'hermes.transcript-tail.v3:'
const INDEX_KEY = 'hermes.transcript-tail.v3-index'
const LEGACY_ROOTS = ['hermes.transcript-tail.v1', 'hermes.transcript-tail.v2'] as const
const TAIL_MESSAGES = 40
const MAX_ENTRY_BYTES = 256 * 1024
const MAX_ENTRIES = 50

export interface TranscriptTailAuthority {
  connectionId: string
  profile: string
  lineageRootId: string
  resolvedTipId: string
  displayRevision: number
}

interface NormalizedTranscriptTailAuthority extends TranscriptTailAuthority {}

export type TranscriptTailCoverage = 'latest-page' | 'latest-page-tail'

export interface TranscriptTailPagination {
  limit: number
  offset: number
  order: 'latest'
  returned: number
}

export interface TranscriptTailCacheEntry extends TranscriptTailAuthority {
  coverage: TranscriptTailCoverage
  storedSessionId: string
  messages: ChatMessage[]
  pagination?: TranscriptTailPagination
  savedAt: number
}

export interface SaveTranscriptTailOptions {
  pagination?: Omit<TranscriptTailPagination, 'order'> & { order: 'latest' | 'oldest' }
}

let legacyPurged = false

function isLegacyCacheKey(key: string): boolean {
  return LEGACY_ROOTS.some(root => key === root || key === `${root}-index` || key.startsWith(`${root}:`))
}

function purgeLegacyCaches(store: Storage): void {
  if (legacyPurged) {
    return
  }

  try {
    const doomed: string[] = []

    for (let index = 0; index < store.length; index += 1) {
      const key = store.key(index)

      if (key && isLegacyCacheKey(key)) {
        doomed.push(key)
      }
    }

    for (const key of doomed) {
      store.removeItem(key)
    }

    legacyPurged = true
  } catch {
    // Best effort. Retry the sweep on the next valid cache access.
  }
}

function storage(): Storage | null {
  try {
    const store = window.localStorage

    purgeLegacyCaches(store)

    return store
  } catch {
    return null
  }
}

function normalizedStoredSessionId(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null
  }

  return value.trim() || null
}

function normalizedAuthority(value: unknown): NormalizedTranscriptTailAuthority | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const authority = value as Partial<TranscriptTailAuthority>
  const lineageRootId = typeof authority.lineageRootId === 'string' ? authority.lineageRootId.trim() : ''
  const resolvedTipId = typeof authority.resolvedTipId === 'string' ? authority.resolvedTipId.trim() : ''
  const displayRevision = authority.displayRevision

  if (
    !lineageRootId ||
    !resolvedTipId ||
    typeof displayRevision !== 'number' ||
    !Number.isFinite(displayRevision) ||
    !Number.isInteger(displayRevision) ||
    displayRevision < 0
  ) {
    return null
  }

  return {
    connectionId: String(authority.connectionId ?? '').trim(),
    displayRevision,
    lineageRootId,
    profile: String(authority.profile ?? '').trim() || 'default',
    resolvedTipId
  }
}

function normalizedPagination(value: unknown): TranscriptTailPagination | undefined {
  if (!value || typeof value !== 'object') {
    return undefined
  }

  const pagination = value as Partial<TranscriptTailPagination>
  const validInteger = (candidate: unknown): candidate is number =>
    typeof candidate === 'number' && Number.isFinite(candidate) && Number.isInteger(candidate) && candidate >= 0

  if (
    !validInteger(pagination.limit) ||
    !validInteger(pagination.offset) ||
    !validInteger(pagination.returned) ||
    pagination.order !== 'latest'
  ) {
    return undefined
  }

  return {
    limit: pagination.limit,
    offset: pagination.offset,
    order: 'latest',
    returned: pagination.returned
  }
}

function entrySuffix(storedSessionId: string, authority: NormalizedTranscriptTailAuthority): string {
  return JSON.stringify([
    authority.connectionId,
    authority.profile,
    storedSessionId,
    authority.lineageRootId,
    authority.resolvedTipId,
    authority.displayRevision
  ])
}

function readIndexState(store: Storage): { ids: string[]; valid: boolean } {
  try {
    const raw = store.getItem(INDEX_KEY)
    const parsed = raw ? JSON.parse(raw) : null

    return Array.isArray(parsed)
      ? { ids: parsed.filter(id => typeof id === 'string'), valid: true }
      : { ids: [], valid: false }
  } catch {
    return { ids: [], valid: false }
  }
}

function readIndex(store: Storage): string[] {
  return readIndexState(store).ids
}

function writeIndex(store: Storage, ids: string[]): boolean {
  try {
    store.setItem(INDEX_KEY, JSON.stringify(ids))

    return true
  } catch {
    return false
  }
}

function actualV3Keys(store: Storage): string[] {
  const keys: string[] = []

  for (let index = 0; index < store.length; index += 1) {
    const key = store.key(index)

    if (key?.startsWith(PREFIX)) {
      keys.push(key)
    }
  }

  return keys
}

interface V3NamespaceSnapshot {
  entries: Array<[key: string, raw: string]>
  indexRaw: string | null
}

function snapshotV3Namespace(store: Storage): V3NamespaceSnapshot | null {
  try {
    const entries = actualV3Keys(store).flatMap(key => {
      const raw = store.getItem(key)

      return raw === null ? [] : ([[key, raw]] as Array<[string, string]>)
    })

    return { entries, indexRaw: store.getItem(INDEX_KEY) }
  } catch {
    return null
  }
}

function restoreV3Namespace(store: Storage, snapshot: V3NamespaceSnapshot): void {
  try {
    for (const key of actualV3Keys(store)) {
      try {
        store.removeItem(key)
      } catch {
        // best effort
      }
    }

    try {
      store.removeItem(INDEX_KEY)
    } catch {
      // best effort
    }

    const restoredSuffixes = new Set<string>()

    for (const [key, raw] of snapshot.entries) {
      try {
        store.setItem(key, raw)
        restoredSuffixes.add(key.slice(PREFIX.length))
      } catch {
        // Keep restoring independent entries.
      }
    }

    if (snapshot.indexRaw === null) {
      return
    }

    try {
      const parsed = JSON.parse(snapshot.indexRaw)
      const indexedSuffixes = Array.isArray(parsed) ? parsed.filter(value => typeof value === 'string') : []
      const fullyRestored = snapshot.entries.every(([key]) => restoredSuffixes.has(key.slice(PREFIX.length)))
      const restoredIndexRaw = fullyRestored
        ? snapshot.indexRaw
        : JSON.stringify(indexedSuffixes.filter(suffix => restoredSuffixes.has(suffix)))

      store.setItem(INDEX_KEY, restoredIndexRaw)
    } catch {
      try {
        store.removeItem(INDEX_KEY)
      } catch {
        // best effort; never fabricate an index for entries we could not restore
      }
    }
  } catch {
    // best effort recovery only
  }
}

interface DiscoveredEntry {
  entry: TranscriptTailCacheEntry
  key: string
  savedAt: number
  storedSessionId: string
  suffix: string
}

function validatePersistedEntry(raw: string, key: string): DiscoveredEntry | null {
  try {
    const parsed = JSON.parse(raw) as Partial<TranscriptTailCacheEntry>
    const storedSessionId = normalizedStoredSessionId(parsed?.storedSessionId)
    const authority = normalizedAuthority(parsed)
    const savedAt = parsed?.savedAt

    if (
      !parsed ||
      !storedSessionId ||
      !authority ||
      typeof savedAt !== 'number' ||
      !Number.isFinite(savedAt) ||
      savedAt < 0 ||
      !Array.isArray(parsed.messages) ||
      parsed.messages.length === 0 ||
      !entryMatches(parsed as TranscriptTailCacheEntry, storedSessionId, authority)
    ) {
      return null
    }

    const pagination = normalizedPagination(parsed.pagination)
    const declaredCoverage =
      parsed.coverage === 'latest-page' || parsed.coverage === 'latest-page-tail'
        ? parsed.coverage
        : 'latest-page-tail'
    const coverage =
      declaredCoverage === 'latest-page' && pagination && pagination.returned !== parsed.messages.length
        ? 'latest-page-tail'
        : declaredCoverage
    const entry: TranscriptTailCacheEntry = {
      ...(parsed as TranscriptTailCacheEntry),
      coverage,
      ...(pagination ? { pagination } : {})
    }

    const suffix = entrySuffix(storedSessionId, authority)

    return key === PREFIX + suffix ? { entry, key, savedAt, storedSessionId, suffix } : null
  } catch {
    return null
  }
}

function discoverEntry(store: Storage, key: string): DiscoveredEntry | null {
  try {
    const raw = store.getItem(key)

    return raw ? validatePersistedEntry(raw, key) : null
  } catch {
    return null
  }
}

function orderedDiscoveredEntries(store: Storage, discovered: DiscoveredEntry[]): DiscoveredEntry[] {
  const indexPosition = new Map<string, number>()

  readIndex(store).forEach((suffix, position) => {
    if (!indexPosition.has(suffix)) {
      indexPosition.set(suffix, position)
    }
  })

  return [...discovered].sort((left, right) => {
    const bySavedAt = left.savedAt - right.savedAt

    if (bySavedAt !== 0) {
      return bySavedAt
    }

    const leftPosition = indexPosition.get(left.suffix) ?? Number.MAX_SAFE_INTEGER
    const rightPosition = indexPosition.get(right.suffix) ?? Number.MAX_SAFE_INTEGER

    return leftPosition - rightPosition || left.suffix.localeCompare(right.suffix)
  })
}

function persistActualSurvivorIndex(store: Storage): boolean {
  try {
    const survivors = actualV3Keys(store).flatMap(key => {
      const entry = discoverEntry(store, key)

      return entry ? [entry] : []
    })

    return writeIndex(
      store,
      orderedDiscoveredEntries(store, survivors).map(entry => entry.suffix)
    )
  } catch {
    return false
  }
}

function reconcileIndexUnsafe(store: Storage, touchedSuffix: string): boolean {
  const corruptKeys: string[] = []
  const discovered = orderedDiscoveredEntries(
    store,
    actualV3Keys(store).flatMap(key => {
      const entry = discoverEntry(store, key)

      if (!entry) {
        corruptKeys.push(key)

        return []
      }

      return [entry]
    })
  )

  const ordered = discovered.map(entry => entry.suffix).filter(suffix => suffix !== touchedSuffix)
  const touched = discovered.find(entry => entry.suffix === touchedSuffix)

  if (touched) {
    ordered.push(touchedSuffix)
  }

  const evicted = ordered.slice(0, Math.max(0, ordered.length - MAX_ENTRIES))
  const kept = ordered.slice(evicted.length)

  if (!writeIndex(store, kept)) {
    return false
  }

  for (const suffix of evicted) {
    try {
      store.removeItem(PREFIX + suffix)
    } catch {
      // best effort; the next reconciliation sees and re-evicts the orphan
    }
  }

  for (const key of corruptKeys) {
    try {
      store.removeItem(key)
    } catch {
      // best effort
    }
  }

  return true
}

function reconcileIndex(store: Storage, touchedSuffix: string): boolean {
  try {
    return reconcileIndexUnsafe(store, touchedSuffix)
  } catch {
    return false
  }
}

function entryMatches(
  entry: TranscriptTailCacheEntry,
  storedSessionId: string,
  authority: NormalizedTranscriptTailAuthority
): boolean {
  return (
    entry.storedSessionId === storedSessionId &&
    entry.connectionId === authority.connectionId &&
    entry.profile === authority.profile &&
    entry.lineageRootId === authority.lineageRootId &&
    entry.resolvedTipId === authority.resolvedTipId &&
    entry.displayRevision === authority.displayRevision
  )
}

export function saveTranscriptTail(
  storedSessionId: string,
  messages: ChatMessage[],
  proof: TranscriptTailAuthority,
  options: SaveTranscriptTailOptions = {}
): void {
  const id = normalizedStoredSessionId(storedSessionId)
  const authority = normalizedAuthority(proof)

  // Validate before touching storage: invalid proof must not purge legacy
  // state, move the LRU, or alter an existing entry.
  if (!id || !authority || !Array.isArray(messages) || messages.length === 0) {
    return
  }

  const store = storage()

  if (!store) {
    return
  }

  const pagination = normalizedPagination(options.pagination)
  const retainedMessages = messages.slice(-TAIL_MESSAGES)
  let entry: TranscriptTailCacheEntry = {
    ...authority,
    coverage: retainedMessages.length === messages.length ? 'latest-page' : 'latest-page-tail',
    messages: retainedMessages,
    ...(pagination ? { pagination } : {}),
    savedAt: Date.now(),
    storedSessionId: id
  }
  let serialized: string

  try {
    serialized = JSON.stringify(entry)
  } catch {
    return
  }

  if (serialized.length > MAX_ENTRY_BYTES) {
    const shorterMessages = messages.slice(-8)
    entry = {
      ...entry,
      coverage: shorterMessages.length === messages.length ? 'latest-page' : 'latest-page-tail',
      messages: shorterMessages
    }

    try {
      serialized = JSON.stringify(entry)
    } catch {
      return
    }

    if (serialized.length > MAX_ENTRY_BYTES) {
      return
    }
  }

  const suffix = entrySuffix(id, authority)
  const key = PREFIX + suffix
  let previousRaw: string | null

  try {
    previousRaw = store.getItem(key)
  } catch {
    return
  }

  const rollbackEntry = (): void => {
    try {
      if (previousRaw === null) {
        store.removeItem(key)
      } else {
        store.setItem(key, previousRaw)
      }
    } catch {
      // best effort; storage is already unavailable
    }
  }

  try {
    store.setItem(key, serialized)

    if (!reconcileIndex(store, suffix)) {
      rollbackEntry()
    }
  } catch {
    const snapshot = snapshotV3Namespace(store)

    if (!snapshot) {
      return
    }

    try {
      clearTranscriptTails()
      store.setItem(key, serialized)

      if (!reconcileIndex(store, suffix)) {
        restoreV3Namespace(store, snapshot)
      }
    } catch {
      restoreV3Namespace(store, snapshot)
      // Storage unavailable; instant paint remains an optional optimization.
    }
  }
}

export function loadTranscriptTail(
  storedSessionId: string,
  proof: TranscriptTailAuthority
): TranscriptTailCacheEntry | null {
  const id = normalizedStoredSessionId(storedSessionId)
  const authority = normalizedAuthority(proof)

  if (!id || !authority) {
    return null
  }

  const store = storage()

  if (!store) {
    return null
  }

  const suffix = entrySuffix(id, authority)
  let raw: string | null

  try {
    raw = store.getItem(PREFIX + suffix)
  } catch {
    return null
  }

  if (!raw) {
    return null
  }

  try {
    const validated = validatePersistedEntry(raw, PREFIX + suffix)

    if (!validated || !entryMatches(validated.entry, id, authority)) {
      throw new Error('invalid transcript-tail entry')
    }

    return {
      ...validated.entry,
      messages: validated.entry.messages.map(withUniqueToolCallIdsWithinMessage)
    }
  } catch {
    try {
      store.removeItem(PREFIX + suffix)
    } catch {
      // best effort
    }

    writeIndex(
      store,
      readIndex(store).filter(indexedSuffix => indexedSuffix !== suffix)
    )

    return null
  }
}

export function dropTranscriptTail(storedSessionId: string, proof?: TranscriptTailAuthority): void {
  const id = normalizedStoredSessionId(storedSessionId)
  const authority = normalizedAuthority(proof)

  if (!id || !authority) {
    return
  }

  const store = storage()

  if (!store) {
    return
  }

  try {
    const suffix = entrySuffix(id, authority)

    store.removeItem(PREFIX + suffix)
    writeIndex(
      store,
      readIndex(store).filter(entry => entry !== suffix)
    )
  } catch {
    // best effort
  }
}

export function dropTranscriptTailEverywhere(storedSessionId: string): void {
  const id = normalizedStoredSessionId(storedSessionId)

  if (!id) {
    return
  }

  const store = storage()

  if (!store) {
    return
  }

  const storedIdFromSuffix = (key: string): string | null => {
    try {
      const parsed = JSON.parse(key.slice(PREFIX.length))

      return Array.isArray(parsed) && typeof parsed[2] === 'string' ? parsed[2].trim() || null : null
    } catch {
      return null
    }
  }

  const storedIdFromEntry = (key: string): string | null => {
    try {
      const raw = store.getItem(key)
      const parsed = raw ? JSON.parse(raw) : null

      return normalizedStoredSessionId(parsed?.storedSessionId)
    } catch {
      return null
    }
  }

  let keys: string[]

  try {
    keys = actualV3Keys(store)
  } catch {
    return
  }

  for (const key of keys) {
    if (storedIdFromEntry(key) !== id && storedIdFromSuffix(key) !== id) {
      continue
    }

    try {
      store.removeItem(key)
    } catch {
      // One inaccessible entry must not prevent later scopes from being removed.
    }
  }

  persistActualSurvivorIndex(store)
}

export function clearTranscriptTails(): void {
  const store = storage()

  if (!store) {
    return
  }

  for (const key of actualV3Keys(store)) {
    try {
      store.removeItem(key)
    } catch {
      // best effort
    }
  }

  try {
    store.removeItem(INDEX_KEY)
  } catch {
    // best effort
  }
}
