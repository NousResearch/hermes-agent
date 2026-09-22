import { type ChatMessage, type ChatMessagePart, chatMessageText } from '@/lib/chat-messages'

/**
 * Crash-survivable in-flight turn journal.
 *
 * While a session is busy, the visible tail of the running turn (user prompt +
 * streamed assistant rows, tool calls included) is persisted to localStorage.
 * If the renderer or the whole app dies mid-turn, session resume folds the
 * journaled tail back onto the restored transcript, so streamed progress is
 * not silently lost. The backend's own `inflight` snapshot (merged by
 * `appendLiveSessionProjection`) covers reconnects while the backend is alive;
 * this journal covers the cases where the backend died too — and it is richer,
 * because the backend snapshot carries text only while the journal keeps the
 * full part structure.
 *
 * Best-effort by design: storage failures must never break chat streaming.
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
/** Streaming repaints arrive every ~33ms; localStorage writes are synchronous.
 *  Trailing-edge throttle keeps the journal off the hot path — a crash costs at
 *  most this much of the newest tail. */
const PERSIST_THROTTLE_MS = 400

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

export interface JournalableSessionState {
  awaitingResponse: boolean
  busy: boolean
  messages: ChatMessage[]
  storedSessionId: null | string
  streamId: null | string
  turnStartedAt: null | number
}

interface JournalStore {
  entries: Record<string, InFlightTurnSnapshot>
  version: typeof STORE_VERSION
}

export interface InFlightRecoveryResult {
  applied: boolean
  /** The base transcript already contains the journaled turn's completed
   *  reply — the journal entry is stale and has been cleared. */
  caughtUp: boolean
  messages: ChatMessage[]
  streamId: null | string
  turnStartedAt: null | number
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
        (message.error === undefined || typeof message.error === 'string') &&
        (message.branchGroupId === undefined || typeof message.branchGroupId === 'string') &&
        (message.hidden === undefined || typeof message.hidden === 'boolean') &&
        (message.interim === undefined || typeof message.interim === 'boolean') &&
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
    parts: message.parts.map(boundedPart).filter((part): part is ChatMessagePart => part !== null),
    ...(message.timestamp === undefined ? {} : { timestamp: message.timestamp }),
    ...(message.pending === undefined ? {} : { pending: message.pending }),
    ...(message.error === undefined ? {} : { error: boundedString(message.error, MAX_METADATA_CHARS) }),
    ...(message.branchGroupId === undefined
      ? {}
      : { branchGroupId: boundedString(message.branchGroupId, MAX_METADATA_CHARS) }),
    ...(message.hidden === undefined ? {} : { hidden: message.hidden }),
    ...(message.interim === undefined ? {} : { interim: message.interim }),
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

function readSnapshot(storedSessionId: string): InFlightTurnSnapshot | null {
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

function removeSnapshot(storedSessionId: string): void {
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

function cloneMessages(messages: ChatMessage[]): ChatMessage[] {
  try {
    return JSON.parse(JSON.stringify(messages)) as ChatMessage[]
  } catch {
    return []
  }
}

function normalizedText(value: string): string {
  return value.replace(/\s+/g, ' ').trim()
}

function attachmentSignature(message: ChatMessage): string {
  return (message.attachmentRefs ?? []).join('\n')
}

function userMessagesMatch(left: ChatMessage, right: ChatMessage): boolean {
  return (
    left.role === 'user' &&
    right.role === 'user' &&
    normalizedText(chatMessageText(left)) === normalizedText(chatMessageText(right)) &&
    attachmentSignature(left) === attachmentSignature(right)
  )
}

function partHasRecoverableContent(part: ChatMessagePart): boolean {
  if (part.type === 'text' || part.type === 'reasoning') {
    return typeof part.text === 'string' && part.text.trim().length > 0
  }

  return part.type === 'tool-call'
}

function assistantHasRecoverableContent(message: ChatMessage): boolean {
  return message.role === 'assistant' && (Boolean(message.error) || message.parts.some(partHasRecoverableContent))
}

/** A live-turn projection row (backend `inflight` via appendLiveSessionProjection,
 *  or a still-streaming local bubble) — as opposed to a completed transcript row. */
function isLiveProjectionRow(message: ChatMessage): boolean {
  return (
    Boolean(message.pending) ||
    message.id.startsWith('assistant-stream-') ||
    message.id.startsWith('inflight-assistant-')
  )
}

/** Visible tail of the running turn: the streaming assistant row (plus any
 *  interim rows sealed after it) back to the user prompt that started it. */
function recoverableTail(messages: ChatMessage[], streamId: null | string): ChatMessage[] {
  const visible = messages.filter(message => !message.hidden)
  let assistantIndex = -1

  if (streamId) {
    assistantIndex = visible.findIndex(message => message.id === streamId && assistantHasRecoverableContent(message))
  }

  if (assistantIndex < 0) {
    for (let index = visible.length - 1; index >= 0; index -= 1) {
      const message = visible[index]

      if (message.role === 'user') {
        break
      }

      if (assistantHasRecoverableContent(message)) {
        assistantIndex = index

        break
      }
    }
  }

  if (assistantIndex < 0) {
    return []
  }

  let start = assistantIndex

  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    if (visible[index].role === 'user') {
      start = index

      // A mid-turn redirect inserts its correction as another user row right
      // before the live reply, so the turn can open with a RUN of user rows.
      // Keep walking back over them: stopping at the nearest one journals the
      // correction alone and loses the prompt that actually started the turn.
      while (start > 0 && visible[start - 1].role === 'user') {
        start -= 1
      }

      break
    }
  }

  return visible.slice(start)
}

function normalizeRecoveredTail(tail: ChatMessage[], keepPending: boolean): ChatMessage[] {
  return cloneMessages(tail).map(message =>
    message.role === 'assistant'
      ? {
          ...message,
          pending: keepPending ? (message.pending ?? true) : false
        }
      : { ...message, pending: false }
  )
}

function assistantTextLength(message: ChatMessage): number {
  return chatMessageText(message).length
}

/** Merge the journal's last assistant row into the base's live projection row.
 *
 * The journal carries structure (tool calls, reasoning) the backend snapshot
 * lacks; the backend text may be newer than the journal's last throttled
 * write. Keep the journal's parts, but let the longer text win — and keep the
 * BASE row's id so live deltas keep appending to the row the stream handler
 * already targets.
 */
function hasStructuralParts(message: ChatMessage): boolean {
  return message.parts.some(part => part.type === 'reasoning' || part.type === 'tool-call')
}

function overlayProjectionRow(projection: ChatMessage, journalRow: ChatMessage): ChatMessage {
  // A projected error (retained failed turn) must survive the overlay.
  const error = journalRow.error ?? projection.error

  const merged: ChatMessage = {
    ...journalRow,
    id: projection.id,
    pending: projection.pending,
    ...(error ? { error } : {})
  }

  if (assistantTextLength(projection) <= assistantTextLength(journalRow)) {
    return merged
  }

  // Backend text is newer than the journal's last throttled write — swap it
  // into the journal's first text part, keeping tool calls and reasoning.
  // When the journal already carries structure, only accept a *strict*
  // extension of the answer text. A longer flat dump that starts with
  // thinking chatter must not overwrite / insert as answer text (#76444).
  const projectionText = chatMessageText(projection)
  const journalText = chatMessageText(journalRow).trim()

  if (hasStructuralParts(journalRow)) {
    const next = projectionText.trim()

    if (!journalText || !next.startsWith(journalText)) {
      return merged
    }
  }

  const parts: ChatMessagePart[] = []
  let textReplaced = false

  for (const part of journalRow.parts) {
    if (part.type !== 'text') {
      parts.push(part)
    } else if (!textReplaced) {
      parts.push({ ...part, text: projectionText })
      textReplaced = true
    }
  }

  if (!textReplaced) {
    parts.push({ type: 'text', text: projectionText })
  }

  return { ...merged, parts }
}

/** Rows the base transcript doesn't already hold by id. The journal and the
 *  base can both carry the same row (a resume that replays a still-journaled
 *  turn), and appending it twice puts a duplicate id in the transcript —
 *  which assistant-ui's MessageRepository rejects by throwing. */
function withoutBaseIds(rows: ChatMessage[], baseMessages: ChatMessage[]): ChatMessage[] {
  const baseIds = new Set(baseMessages.map(message => message.id))

  return rows.filter(row => !baseIds.has(row.id))
}

/** Whether every recoverable assistant row in the journal tail already exists
 *  as committed text in the base transcript. When true, the journal outlived
 *  the turn it recorded and appending it would re-render the same answers at
 *  the end of the transcript (the "scrambled conversation" regression). */
function journalTailAlreadyCommitted(tailAssistants: ChatMessage[], baseMessages: ChatMessage[]): boolean {
  const recoverable = tailAssistants.filter(assistantHasRecoverableContent)

  if (recoverable.length === 0) {
    return false
  }

  const baseTexts = new Set(
    baseMessages
      .filter(message => message.role === 'assistant' && !message.hidden)
      .map(message => normalizedText(chatMessageText(message)))
  )

  return recoverable.every(message => {
    const text = normalizedText(chatMessageText(message))

    // Error-only rows carry no text to verify against — keep the conservative
    // append path rather than risk dropping a recoverable failure.
    return text.length > 0 && baseTexts.has(text)
  })
}

export function mergeInFlightMessages(
  baseMessages: ChatMessage[],
  tailMessages: ChatMessage[],
  options: { keepPending?: boolean } = {}
): InFlightRecoveryResult {
  const noop: InFlightRecoveryResult = {
    applied: false,
    caughtUp: false,
    messages: baseMessages,
    streamId: null,
    turnStartedAt: null
  }

  const tail = normalizeRecoveredTail(tailMessages, Boolean(options.keepPending))

  if (!tail.some(assistantHasRecoverableContent)) {
    return noop
  }

  const tailUserIndex = tail.findIndex(message => message.role === 'user')
  const tailUser = tailUserIndex >= 0 ? tail[tailUserIndex] : null
  const tailAssistants = tail.slice(tailUserIndex + 1)
  const lastJournalRow = tailAssistants.findLast(assistantHasRecoverableContent) ?? null
  const matchingUserIndex = tailUser ? baseMessages.findLastIndex(message => userMessagesMatch(message, tailUser)) : -1

  if (matchingUserIndex < 0) {
    // No base user matches the tail's user row (a projected user-inflight row
    // that never persisted, or a tail captured without its user prompt). If the
    // tail's answers are already committed in the transcript, the journal is
    // stale — appending it would re-render the same replies at the end of the
    // conversation. Otherwise, the base never saw this turn at all: append the
    // whole tail (the crash-recovery path the journal exists for).
    if (journalTailAlreadyCommitted(tailAssistants, baseMessages)) {
      return { ...noop, caughtUp: true }
    }

    const streamId = lastJournalRow?.id ?? null

    return {
      applied: true,
      caughtUp: false,
      messages: [...baseMessages, ...withoutBaseIds(tail, baseMessages)],
      // Only a genuinely running turn keeps a live stream target. On an idle
      // resume, carrying the stale streamId would keep the journal entry alive
      // (persistInFlightTurnState only clears when streamId is null) and the
      // same tail would be folded again on every open.
      streamId: options.keepPending ? streamId : null,
      turnStartedAt: null
    }
  }

  const afterUser = baseMessages.slice(matchingUserIndex + 1)

  const completedReply = afterUser.find(
    message => assistantHasRecoverableContent(message) && !isLiveProjectionRow(message)
  )

  if (completedReply) {
    // The transcript already holds this turn's committed reply — the journal
    // entry is stale.
    return { ...noop, caughtUp: true }
  }

  const projectionIndex = baseMessages.findIndex(
    (message, index) => index > matchingUserIndex && message.role === 'assistant' && isLiveProjectionRow(message)
  )

  if (projectionIndex < 0) {
    if (tailAssistants.length === 0) {
      return noop
    }

    const streamId = lastJournalRow?.id ?? null

    return {
      applied: true,
      caughtUp: false,
      messages: [...baseMessages, ...withoutBaseIds(tailAssistants, baseMessages)],
      // Same idle-resume rule as the other exit paths: only a running turn
      // keeps the stream target alive. Carrying the stale streamId here kept
      // the journal entry alive (persistInFlightTurnState only clears when
      // streamId is null), so the same tail was folded again on every open.
      streamId: options.keepPending ? streamId : null,
      turnStartedAt: null
    }
  }

  // Backend projection row present (text-only): overlay the journal's
  // structure onto it instead of treating it as "caught up" — that is how
  // locally recorded tool progress used to get dropped.
  const projection = baseMessages[projectionIndex]
  const merged = lastJournalRow ? overlayProjectionRow(projection, lastJournalRow) : projection

  const sealedRows = tailAssistants.filter(
    message => message !== lastJournalRow && assistantHasRecoverableContent(message)
  )

  const messages = [
    ...baseMessages.slice(0, projectionIndex),
    ...sealedRows,
    merged,
    ...baseMessages.slice(projectionIndex + 1)
  ]

  return {
    applied: true,
    caughtUp: false,
    messages,
    // Same idle-resume rule as the append path: only a running turn keeps the
    // stream target alive, so an idle resume clears the journal instead of
    // re-folding the same tail on every open.
    streamId: options.keepPending ? merged.id : null,
    turnStartedAt: null
  }
}

const persistTimers = new Map<string, ReturnType<typeof setTimeout>>()
const persistLatest = new Map<string, JournalableSessionState>()

/** @internal Test-only reset for module-scoped throttles and sweep state. */
export function resetInFlightTurnJournalStateForTests(): void {
  for (const timer of persistTimers.values()) {
    clearTimeout(timer)
  }

  persistTimers.clear()
  persistLatest.clear()
  sessionStoreSwept = false
}

function writeSnapshot(storedSessionId: string, state: JournalableSessionState): void {
  const tail = recoverableTail(state.messages, state.streamId)

  if (tail.length === 0) {
    return
  }

  const store = storage()
  const key = sessionStorageKey(storedSessionId)

  if (!store || !key) {
    return
  }

  sweepSessionStore(store, true)

  const messages = boundedMessages(tail)

  if (!messages) {
    // Keep the timer write path free of aggregate migration. This tiny invalid
    // v2 value suppresses the stale v1 predecessor until read/settle performs
    // the one-shot migration and removes it.
    tombstoneUnlessRecoverable(store, key)

    return
  }

  const raw = serializeSnapshot({
    messages,
    streamId: state.streamId,
    turnStartedAt: state.turnStartedAt,
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

/** Persist the running turn's visible tail (throttled), or clear the entry the
 *  moment the turn settles. Call on every session-state commit. */
export function persistInFlightTurnState(state: JournalableSessionState): void {
  const storedSessionId = state.storedSessionId

  if (!storedSessionId) {
    return
  }

  if (!state.busy && !state.awaitingResponse && !state.streamId) {
    clearInFlightTurnJournal(storedSessionId)

    return
  }

  persistLatest.set(storedSessionId, state)

  if (persistTimers.has(storedSessionId)) {
    return
  }

  persistTimers.set(
    storedSessionId,
    setTimeout(() => {
      persistTimers.delete(storedSessionId)
      const latest = persistLatest.get(storedSessionId)

      persistLatest.delete(storedSessionId)

      if (latest) {
        writeSnapshot(storedSessionId, latest)
      }
    }, PERSIST_THROTTLE_MS)
  )
}

export function readInFlightTurnJournal(storedSessionId: null | string): InFlightTurnSnapshot | null {
  if (!storedSessionId) {
    return null
  }

  return readSnapshot(storedSessionId)
}

/** Fold a journaled in-flight tail back onto a restored transcript. A no-op
 *  returns `baseMessages` by reference so callers keep their fast-path ref. */
export function recoverInFlightTurnJournal(
  storedSessionId: null | string,
  baseMessages: ChatMessage[],
  options: { keepPending?: boolean } = {}
): InFlightRecoveryResult {
  const snapshot = readInFlightTurnJournal(storedSessionId)

  if (!snapshot) {
    return {
      applied: false,
      caughtUp: false,
      messages: baseMessages,
      streamId: null,
      turnStartedAt: null
    }
  }

  const recovered = mergeInFlightMessages(baseMessages, snapshot.messages, options)

  if (recovered.caughtUp) {
    clearInFlightTurnJournal(storedSessionId)
  }

  return {
    ...recovered,
    // Never resurrect a stale stream target on an idle resume: with
    // keepPending=false the session is not running, so the recovered rows are
    // settled history and the journal must clear on the next state update —
    // otherwise the same stale tail is folded again on every open.
    streamId: recovered.applied ? (recovered.streamId ?? (options.keepPending ? snapshot.streamId : null)) : null,
    turnStartedAt: recovered.applied ? snapshot.turnStartedAt : null
  }
}

export function clearInFlightTurnJournal(storedSessionId: null | string): void {
  if (!storedSessionId) {
    return
  }

  const timer = persistTimers.get(storedSessionId)

  if (timer) {
    clearTimeout(timer)
    persistTimers.delete(storedSessionId)
  }

  persistLatest.delete(storedSessionId)

  removeSnapshot(storedSessionId)
}
