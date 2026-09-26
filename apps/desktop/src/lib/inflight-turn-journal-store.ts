import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages'

/**
 * localStorage persistence for the in-flight turn journal: one bounded v2
 * entry per stored session, its codec and sweep, and the one-shot migration
 * out of the v1 aggregate. Storage failures are swallowed — recovery state
 * must never break chat streaming.
 */

const LEGACY_STORAGE_KEY = 'hermes.desktop.inflightTurnJournal.v1'
const STORAGE_PREFIX = 'hermes.desktop.inflightTurnJournal.v2:'
const LEGACY_MIGRATION_KEY = 'hermes.desktop.inflightTurnJournal.v2.migrated'
const DISCARDED_SNAPSHOT_RAW = '0'
const STORE_VERSION = 1
const MAX_SESSION_STORE_CHARS = 4 * 1024 * 1024
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000
// Keep the worst-case v2 namespace below a conservative localStorage budget
// while retaining the 24 newest session slots for ordinary small snapshots.
const MAX_ENTRY_CHARS = 160 * 1024
const MAX_ENTRIES = Math.min(24, Math.floor(MAX_SESSION_STORE_CHARS / MAX_ENTRY_CHARS))
const MAX_LEGACY_STORE_CHARS = 2 * 1024 * 1024
const MAX_SESSION_KEY_CHARS = 512
const MAX_JOURNALED_MESSAGES = 24
const MAX_TEXT_PART_CHARS = 64 * 1024
const MAX_METADATA_CHARS = 2 * 1024
const MAX_USER_ATTACHMENT_REFS = 256
const MAX_USER_ATTACHMENT_REF_CHARS = 64 * 1024

// A renderer can accumulate one entry per session over its lifetime. Sweep the
// bounded v2 namespace once on first journal access; never scan it on the
// 400ms streaming write path.
let sessionStoreSwept = false

export interface InFlightTurnSnapshot {
  messages: ChatMessage[]
  streamId: null | string
  turnStartedAt: null | number
  updatedAt: number
}

interface JournalStore {
  entries: Record<string, InFlightTurnSnapshot>
  version: typeof STORE_VERSION
}

function storage(): Storage | null {
  try {
    return typeof window === 'undefined' ? null : window.localStorage
  } catch {
    return null
  }
}

function sessionStorageKey(storedSessionId: string): null | string {
  try {
    const encoded = encodeURIComponent(storedSessionId)

    return encoded.length > 0 && encoded.length <= MAX_SESSION_KEY_CHARS ? `${STORAGE_PREFIX}${encoded}` : null
  } catch {
    return null
  }
}

function readRaw(store: Storage, key: string): null | string {
  try {
    return store.getItem(key)
  } catch {
    return null
  }
}

function removeRaw(store: Storage, key: string): void {
  try {
    store.removeItem(key)
  } catch {
    // Best-effort recovery state must not interrupt chat streaming.
  }
}

function writeRaw(store: Storage, key: string, value: string): boolean {
  try {
    store.setItem(key, value)

    return true
  } catch {
    return false
  }
}

function isSnapshot(value: unknown): value is InFlightTurnSnapshot {
  if (!value || typeof value !== 'object') {
    return false
  }

  const snapshot = value as Partial<InFlightTurnSnapshot>

  return (
    Array.isArray(snapshot.messages) &&
    snapshot.messages.every(
      message =>
        Boolean(message) &&
        typeof message === 'object' &&
        typeof message.id === 'string' &&
        ['assistant', 'system', 'tool', 'user'].includes(message.role) &&
        Array.isArray(message.parts) &&
        message.parts.every(
          part =>
            Boolean(part) &&
            typeof part === 'object' &&
            typeof part.type === 'string' &&
            (part.sourceRowId === undefined ||
              (typeof part.sourceRowId === 'number' && Number.isFinite(part.sourceRowId))) &&
            (part.type !== 'text' && part.type !== 'reasoning'
              ? part.type !== 'tool-call' ||
                (typeof part.toolName === 'string' &&
                  (part.toolCallId === undefined || typeof part.toolCallId === 'string') &&
                  (part.isError === undefined || typeof part.isError === 'boolean'))
              : typeof part.text === 'string' && (part.parentId === undefined || typeof part.parentId === 'string'))
        ) &&
        (message.timestamp === undefined ||
          (typeof message.timestamp === 'number' && Number.isFinite(message.timestamp))) &&
        (message.pending === undefined || typeof message.pending === 'boolean') &&
        (message.userOriginated === undefined || typeof message.userOriginated === 'boolean') &&
        (message.runtimeTurnStartedAt === undefined ||
          (typeof message.runtimeTurnStartedAt === 'number' && Number.isFinite(message.runtimeTurnStartedAt))) &&
        (message.error === undefined || typeof message.error === 'string') &&
        (message.branchGroupId === undefined || typeof message.branchGroupId === 'string') &&
        (message.hidden === undefined || typeof message.hidden === 'boolean') &&
        (message.interim === undefined || typeof message.interim === 'boolean') &&
        (message.recovered === undefined || typeof message.recovered === 'boolean') &&
        (message.durableComplete === undefined || typeof message.durableComplete === 'boolean') &&
        (message.attachmentRefs === undefined ||
          (Array.isArray(message.attachmentRefs) && message.attachmentRefs.every(ref => typeof ref === 'string'))) &&
        (message.rowId === undefined || (typeof message.rowId === 'number' && Number.isFinite(message.rowId)))
    ) &&
    (snapshot.streamId === null || typeof snapshot.streamId === 'string') &&
    (snapshot.turnStartedAt === null || typeof snapshot.turnStartedAt === 'number') &&
    typeof snapshot.updatedAt === 'number' &&
    Number.isFinite(snapshot.updatedAt)
  )
}

function parseSnapshot(raw: string): InFlightTurnSnapshot | null {
  if (raw.length > MAX_ENTRY_CHARS) {
    return null
  }

  try {
    const parsed = JSON.parse(raw)

    return isSnapshot(parsed) ? parsed : null
  } catch {
    return null
  }
}

function serializeSnapshot(snapshot: InFlightTurnSnapshot): string | null {
  let messages = snapshot.messages

  while (messages.length > 0) {
    try {
      const raw = JSON.stringify({ ...snapshot, messages })

      if (raw.length <= MAX_ENTRY_CHARS) {
        return raw
      }
    } catch {
      return null
    }

    // Keep the join-key row and newest assistant progress while dropping the
    // oldest sealed rows. If those two rows alone do not fit, the caller must
    // avoid replacing an older recoverable snapshot with a tombstone.
    if (messages.length <= 2) {
      return null
    }

    messages = [messages[0], ...messages.slice(2)]
  }

  return null
}

function sweepSessionStore(store: Storage, reserveSlot = false): void {
  if (sessionStoreSwept) {
    return
  }

  sessionStoreSwept = true

  try {
    const sessionKeys: string[] = []

    for (let index = 0; index < store.length; index += 1) {
      const key = store.key(index)

      if (key?.startsWith(STORAGE_PREFIX)) {
        sessionKeys.push(key)
      }
    }

    const liveEntries: Array<{ key: string; snapshot: InFlightTurnSnapshot }> = []
    const migrated = readRaw(store, LEGACY_MIGRATION_KEY) !== null

    for (const key of sessionKeys) {
      const raw = readRaw(store, key)

      // A tombstone is intentional state. It suppresses the stale v1
      // predecessor until the one-shot migration removes the aggregate.
      if (raw === DISCARDED_SNAPSHOT_RAW) {
        if (migrated) {
          removeRaw(store, key)
        }

        continue
      }

      const snapshot = raw ? parseSnapshot(raw) : null

      if (!snapshot || isExpired(snapshot)) {
        removeRaw(store, key)

        continue
      }

      liveEntries.push({ key, snapshot })
    }

    liveEntries
      .sort((left, right) => right.snapshot.updatedAt - left.snapshot.updatedAt)
      .slice(reserveSlot ? MAX_ENTRIES - 1 : MAX_ENTRIES)
      .forEach(entry => removeRaw(store, entry.key))
  } catch {
    // The journal is best effort; a storage enumeration failure must not
    // interrupt renderer work or turn persistence.
  }
}

function boundedString(value: string, maxChars: number): string {
  return value.length <= maxChars ? value : value.slice(0, maxChars)
}

function boundedPart(part: ChatMessagePart): ChatMessagePart | null {
  if (part.type === 'text') {
    return {
      type: 'text',
      text: boundedString(part.text, MAX_TEXT_PART_CHARS),
      ...(part.sourceRowId === undefined ? {} : { sourceRowId: part.sourceRowId }),
      ...(part.parentId === undefined ? {} : { parentId: boundedString(part.parentId, MAX_METADATA_CHARS) })
    }
  }

  if (part.type === 'reasoning') {
    return {
      type: 'reasoning',
      text: boundedString(part.text, MAX_TEXT_PART_CHARS),
      ...(part.parentId === undefined ? {} : { parentId: boundedString(part.parentId, MAX_METADATA_CHARS) })
    }
  }

  if (part.type === 'tool-call') {
    // Tool payloads can contain multi-megabyte command output. Recovery only
    // needs invocation identity and failure state; args/results are available
    // from the backend transcript when it survives.
    return {
      type: 'tool-call',
      toolName: boundedString(part.toolName, MAX_METADATA_CHARS),
      args: {},
      ...(part.toolCallId === undefined ? {} : { toolCallId: boundedString(part.toolCallId, MAX_METADATA_CHARS) }),
      ...(part.result === undefined ? {} : { result: {} }),
      ...(part.isError === undefined ? {} : { isError: part.isError })
    }
  }

  // Rich file/image/data/source parts can embed large payloads. They are not
  // required for in-flight text/tool recovery and remain backend-owned.
  return null
}

function boundedMessages(messages: ChatMessage[]): ChatMessage[] | null {
  const bounded =
    messages.length <= MAX_JOURNALED_MESSAGES
      ? messages
      : [messages[0], ...messages.slice(-(MAX_JOURNALED_MESSAGES - 1))]

  // User text and attachment refs are the recovery join key. Truncating either
  // could attach a journal tail to the wrong transcript row, so pathological
  // prompts skip journaling instead of weakening the match.
  if (
    bounded.some(message => {
      if (message.role !== 'user') {
        return false
      }

      if (
        message.parts.some(
          part => (part.type === 'text' || part.type === 'reasoning') && part.text.length > MAX_TEXT_PART_CHARS
        )
      ) {
        return true
      }

      const refs = message.attachmentRefs

      if (!refs) {
        return false
      }

      if (refs.length > MAX_USER_ATTACHMENT_REFS) {
        return true
      }

      let chars = 0

      for (const ref of refs) {
        chars += ref.length

        if (chars > MAX_USER_ATTACHMENT_REF_CHARS) {
          return true
        }
      }

      return false
    })
  ) {
    return null
  }

  return bounded.map(message => ({
    id: boundedString(message.id, MAX_METADATA_CHARS),
    role: message.role,
    ...(message.userOriginated === undefined ? {} : { userOriginated: message.userOriginated }),
    ...(message.runtimeTurnStartedAt === undefined ? {} : { runtimeTurnStartedAt: message.runtimeTurnStartedAt }),
    parts: message.parts.map(boundedPart).filter((part): part is ChatMessagePart => part !== null),
    ...(message.timestamp === undefined ? {} : { timestamp: message.timestamp }),
    ...(message.rowId === undefined ? {} : { rowId: message.rowId }),
    ...(message.pending === undefined ? {} : { pending: message.pending }),
    ...(message.error === undefined ? {} : { error: boundedString(message.error, MAX_METADATA_CHARS) }),
    ...(message.branchGroupId === undefined
      ? {}
      : { branchGroupId: boundedString(message.branchGroupId, MAX_METADATA_CHARS) }),
    ...(message.hidden === undefined ? {} : { hidden: message.hidden }),
    ...(message.interim === undefined ? {} : { interim: message.interim }),
    ...(message.recovered === undefined ? {} : { recovered: message.recovered }),
    ...(message.durableComplete === undefined ? {} : { durableComplete: message.durableComplete }),
    ...(message.attachmentRefs === undefined
      ? {}
      : {
          attachmentRefs:
            message.role === 'user'
              ? [...message.attachmentRefs]
              : message.attachmentRefs
                  .slice(0, MAX_USER_ATTACHMENT_REFS)
                  .map(ref => boundedString(ref, MAX_METADATA_CHARS))
        }),
    ...(message.rowId === undefined ? {} : { rowId: message.rowId })
  }))
}

function migrateLegacyStore(store: Storage): void {
  if (readRaw(store, LEGACY_MIGRATION_KEY) !== null) {
    return
  }

  const raw = readRaw(store, LEGACY_STORAGE_KEY)

  if (raw === null) {
    return
  }

  // Claim the migration before touching the aggregate. If storage is failing,
  // skip legacy recovery rather than retrying an expensive parse on every read.
  if (!writeRaw(store, LEGACY_MIGRATION_KEY, '1')) {
    return
  }

  // Release the multi-megabyte aggregate before allocating per-session v2
  // entries. The captured string remains available for this one migration.
  removeRaw(store, LEGACY_STORAGE_KEY)

  if (!raw) {
    return
  }

  if (raw.length > MAX_LEGACY_STORE_CHARS) {
    return
  }

  try {
    const parsed = JSON.parse(raw) as Partial<JournalStore>

    if (
      parsed.version !== STORE_VERSION ||
      !parsed.entries ||
      typeof parsed.entries !== 'object' ||
      Array.isArray(parsed.entries)
    ) {
      return
    }

    const existingV2Keys = new Set<string>()

    for (let index = 0; index < store.length; index += 1) {
      const key = store.key(index)

      if (key?.startsWith(STORAGE_PREFIX)) {
        existingV2Keys.add(key)
      }
    }

    const entries = Object.entries(parsed.entries)
      .filter((entry): entry is [string, InFlightTurnSnapshot] => isSnapshot(entry[1]) && !isExpired(entry[1]))
      .sort((a, b) => b[1].updatedAt - a[1].updatedAt)
      .slice(0, Math.max(0, MAX_ENTRIES - existingV2Keys.size))

    for (const [storedSessionId, snapshot] of entries) {
      const key = sessionStorageKey(storedSessionId)
      const messages = boundedMessages(snapshot.messages)
      const value = messages ? serializeSnapshot({ ...snapshot, messages }) : null

      // A v2 snapshot may have been written before the one-shot migration ran.
      // Never replace newer per-session state with its stale v1 predecessor.
      if (key && value && readRaw(store, key) === null) {
        if (writeRaw(store, key, value)) {
          existingV2Keys.add(key)
        }
      }
    }
  } catch {
    // Malformed legacy data is discarded below.
  }
}

function discardSnapshot(store: Storage, key: string): void {
  // Migrate first so an existing v2 key suppresses its stale v1 predecessor,
  // then remove the current session. This keeps every discard path from
  // resurrecting legacy state on a later read.
  migrateLegacyStore(store)
  removeRaw(store, key)
}

export function readSnapshot(storedSessionId: string): InFlightTurnSnapshot | null {
  const store = storage()
  const key = sessionStorageKey(storedSessionId)

  if (!store || !key) {
    return null
  }

  sweepSessionStore(store)

  let raw = readRaw(store, key)

  if (!raw) {
    migrateLegacyStore(store)
    raw = readRaw(store, key)
  }

  if (!raw) {
    return null
  }

  const snapshot = parseSnapshot(raw)

  if (!snapshot || isExpired(snapshot)) {
    discardSnapshot(store, key)

    return null
  }

  return snapshot
}

export function removeSnapshot(storedSessionId: string): void {
  const store = storage()
  const key = sessionStorageKey(storedSessionId)

  if (store && key) {
    sweepSessionStore(store)

    // Settling a session before the one-shot migration must clear its legacy
    // entry too; otherwise a later read can migrate and resurrect stale state.
    // This aggregate parse is terminal-transition work, never a stream write.
    discardSnapshot(store, key)
  }
}

function isExpired(entry: InFlightTurnSnapshot, now = Date.now()): boolean {
  return now - entry.updatedAt > MAX_AGE_MS
}

export function writeSnapshot(storedSessionId: string, snapshot: Omit<InFlightTurnSnapshot, 'updatedAt'>): void {
  const store = storage()
  const key = sessionStorageKey(storedSessionId)

  if (!store || !key) {
    return
  }

  sweepSessionStore(store, true)

  const messages = boundedMessages(snapshot.messages)

  if (!messages) {
    // Keep the timer write path free of aggregate migration. This tiny invalid
    // v2 value suppresses the stale v1 predecessor until read/settle performs
    // the one-shot migration and removes it.
    tombstoneUnlessRecoverable(store, key)

    return
  }

  const raw = serializeSnapshot({
    messages,
    streamId: snapshot.streamId,
    turnStartedAt: snapshot.turnStartedAt,
    updatedAt: Date.now()
  })

  if (!raw) {
    // Preserve an older bounded snapshot if the newest assistant row alone is
    // too large. A tombstone is only needed when there is no recoverable v2
    // value, so stale v1 state cannot be resurrected on a later read.
    tombstoneUnlessRecoverable(store, key)

    return
  }

  if (!writeRaw(store, key, raw)) {
    // A quota failure must not leave an older, misleading snapshot behind, or
    // let the stale v1 predecessor be resurrected on a later read.
    tombstoneUnlessRecoverable(store, key)
  }
}

function tombstoneUnlessRecoverable(store: Storage, key: string): void {
  const previous = readRaw(store, key)

  if (previous) {
    const snapshot = parseSnapshot(previous)

    if (snapshot && !isExpired(snapshot)) {
      return
    }
  }

  if (!writeRaw(store, key, DISCARDED_SNAPSHOT_RAW)) {
    removeRaw(store, key)
  }
}

/** @internal Test-only reset for the once-per-renderer sweep. */
export function resetSessionStoreSweepForTests(): void {
  sessionStoreSwept = false
}
