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

/** One localStorage key PER SESSION. The v1 single-key store meant every
 *  throttled write re-parsed and re-stringified EVERY busy session's tail —
 *  with a grid of concurrent streams that was a whole-store JSON round-trip
 *  dozens of times a second, all on the main thread. Per-session keys make a
 *  write O(own tail) regardless of how many other sessions are streaming. */
const STORAGE_PREFIX = 'hermes.desktop.inflightTurnJournal.v2:'
const LEGACY_STORAGE_KEY = 'hermes.desktop.inflightTurnJournal.v1'
const MAX_ENTRIES = 24
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000
/** Streaming repaints arrive every ~33ms; localStorage writes are synchronous.
 *  Trailing-edge throttle keeps the journal off the hot path — a crash costs at
 *  most this much of the newest tail. */
const PERSIST_THROTTLE_MS = 400

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

const entryKey = (storedSessionId: string) => `${STORAGE_PREFIX}${storedSessionId}`

function isExpired(entry: InFlightTurnSnapshot, now = Date.now()): boolean {
  return now - entry.updatedAt > MAX_AGE_MS
}

function loadEntry(storedSessionId: string): InFlightTurnSnapshot | null {
  const store = storage()

  if (!store) {
    return null
  }

  try {
    const raw = store.getItem(entryKey(storedSessionId))
    const parsed = raw ? (JSON.parse(raw) as InFlightTurnSnapshot) : null

    return parsed && typeof parsed.updatedAt === 'number' && Array.isArray(parsed.messages) ? parsed : null
  } catch {
    return null
  }
}

function saveEntry(storedSessionId: string, entry: InFlightTurnSnapshot): void {
  try {
    storage()?.setItem(entryKey(storedSessionId), JSON.stringify(entry))
  } catch {
    // Quota/private-mode failures: the journal is a recovery aid, not truth.
  }
}

function removeEntry(storedSessionId: string): void {
  try {
    storage()?.removeItem(entryKey(storedSessionId))
  } catch {
    // Same best-effort stance as saveEntry.
  }
}

// Split a v1 single-key store into per-session entries. Checked on every
// journal touch (a null getItem is free); a populated v1 store exists at most
// once, right after the upgrade.
function migrateLegacyStore(store: Storage): void {
  try {
    const legacy = store.getItem(LEGACY_STORAGE_KEY)

    if (!legacy) {
      return
    }

    const parsed = JSON.parse(legacy)

    if (parsed && typeof parsed.entries === 'object' && !Array.isArray(parsed.entries)) {
      for (const [id, entry] of Object.entries(parsed.entries as Record<string, InFlightTurnSnapshot>)) {
        saveEntry(id, entry)
      }
    }
  } catch {
    // A corrupt v1 store has nothing worth carrying over.
  }

  try {
    store.removeItem(LEGACY_STORAGE_KEY)
  } catch {
    // Best-effort, like every other journal write.
  }
}

// One-time prune per renderer: drop expired/overflow entries. Startup-only on
// purpose — entries clear on settle, so anything left over is crash residue,
// and enumerating localStorage on the write path would defeat the point.
let housekeepingDone = false

function ensureHousekeeping(): void {
  const store = storage()

  if (!store) {
    return
  }

  migrateLegacyStore(store)

  if (housekeepingDone) {
    return
  }

  housekeepingDone = true

  try {
    const keys: string[] = []

    for (let index = 0; index < store.length; index += 1) {
      const key = store.key(index)

      if (key?.startsWith(STORAGE_PREFIX)) {
        keys.push(key)
      }
    }

    const live: { key: string; updatedAt: number }[] = []

    for (const key of keys) {
      let entry: InFlightTurnSnapshot | null = null

      try {
        entry = JSON.parse(store.getItem(key) ?? '') as InFlightTurnSnapshot
      } catch {
        // Unparseable — prune below.
      }

      if (!entry || typeof entry.updatedAt !== 'number' || isExpired(entry)) {
        store.removeItem(key)
      } else {
        live.push({ key, updatedAt: entry.updatedAt })
      }
    }

    live.sort((a, b) => b.updatedAt - a.updatedAt)

    for (const { key } of live.slice(MAX_ENTRIES)) {
      store.removeItem(key)
    }
  } catch {
    // Best-effort, like every other journal write.
  }
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

  return cloneMessages(visible.slice(start))
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
    // Base doesn't know this turn at all (user row was never persisted):
    // append the whole tail.
    const streamId = lastJournalRow?.id ?? null

    return {
      applied: true,
      caughtUp: false,
      messages: [...baseMessages, ...withoutBaseIds(tail, baseMessages)],
      streamId,
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
      streamId,
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

  return { applied: true, caughtUp: false, messages, streamId: merged.id, turnStartedAt: null }
}

const persistTimers = new Map<string, ReturnType<typeof setTimeout>>()
const persistLatest = new Map<string, JournalableSessionState>()

function writeSnapshot(storedSessionId: string, state: JournalableSessionState): void {
  const tail = recoverableTail(state.messages, state.streamId)

  if (tail.length === 0) {
    return
  }

  ensureHousekeeping()
  saveEntry(storedSessionId, {
    messages: tail,
    streamId: state.streamId,
    turnStartedAt: state.turnStartedAt,
    updatedAt: Date.now()
  })
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

  ensureHousekeeping()
  const entry = loadEntry(storedSessionId)

  if (!entry) {
    return null
  }

  if (isExpired(entry)) {
    removeEntry(storedSessionId)

    return null
  }

  return entry
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
    streamId: recovered.applied ? (recovered.streamId ?? snapshot.streamId) : null,
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
  ensureHousekeeping()
  removeEntry(storedSessionId)
}
