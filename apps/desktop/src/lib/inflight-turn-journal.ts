import { recordJournalWrite } from '@/diagnostics/capture'
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
 * this journal covers the cases where the backend died too.
 *
 * Storage layout (v2): one localStorage key PER SESSION
 * (`hermes.desktop.inflightTurnJournal.v2.<storedSessionId>`), each holding a
 * BOUNDED recovery projection of the turn tail. The v1 layout — every
 * session's full tail in one aggregate key — grew to many megabytes, and its
 * per-tick read-clone-stringify-write of the whole store blocked the renderer
 * main thread for hundreds of ms (the 2026-08-06 hitching root cause). The
 * hot path is now write-only: build a small projection, stringify it once,
 * `setItem` one key. No journal key is read before writing, and there is
 * deliberately NO in-memory aggregate cache — secondary/peer windows share
 * the storage partition, and a renderer-local cache could clobber or
 * resurrect entries another window owns.
 *
 * The projection keeps what recovery actually needs — the user prompt in full
 * (recovery matching compares its text), message ids/roles/order, assistant
 * text, bounded reasoning, and tool-call identity (id/name/error) with short
 * arg/result previews — and drops what it does not: multi-megabyte tool
 * results, terminal dumps, inline diffs, embedded data.
 *
 * Best-effort by design: storage failures must never break chat streaming.
 * Kill switch for A/B diagnosis: set localStorage key
 * `hermes.desktop.inflightTurnJournal.disabled` to `'1'` and reload.
 */

const LEGACY_STORAGE_KEY = 'hermes.desktop.inflightTurnJournal.v1'
const KEY_PREFIX = 'hermes.desktop.inflightTurnJournal.v2.'
const KILL_SWITCH_KEY = 'hermes.desktop.inflightTurnJournal.disabled'
const STORE_VERSION = 2
const MAX_ENTRIES = 24
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000
/** Streaming repaints arrive every ~33ms; localStorage writes are synchronous.
 *  Trailing-edge throttle keeps the journal off the hot path — a crash costs at
 *  most this much of the newest tail. */
const PERSIST_THROTTLE_MS = 400

// Projection bounds, in UTF-16 code units (what localStorage quota counts).
// Normal caps first; the `tight` pass applies when a projected entry still
// exceeds the serialized budget (e.g. an enormous streamed answer).
const ASSISTANT_TEXT_CAP = 64 * 1024
const ASSISTANT_TEXT_CAP_TIGHT = 8 * 1024
const REASONING_CAP = 16 * 1024
const TOOL_PREVIEW_CAP = 2 * 1024
/** Fail-safe for one serialized entry. An entry can exceed it only when the
 *  user prompt alone does (kept in full — recovery matching needs it). */
const ENTRY_BUDGET = 256 * 1024
/** Hard ceiling after the tight re-projection. Above this the write is
 *  SKIPPED: a pathological prompt's crash recovery is worth less than paying
 *  a multi-megabyte synchronous setItem every tick — the exact stall this
 *  file exists to prevent. */
const ENTRY_HARD_CEILING = 2 * ENTRY_BUDGET
/** Preview walker depth limit — deeper structure collapses to a placeholder. */
const PREVIEW_MAX_DEPTH = 6

export interface InFlightTurnSnapshot {
  messages: ChatMessage[]
  streamId: null | string
  turnStartedAt: null | number
  updatedAt: number
}

interface StoredJournalEntry extends InFlightTurnSnapshot {
  version: typeof STORE_VERSION
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

function sessionKey(storedSessionId: string): string {
  return `${KEY_PREFIX}${storedSessionId}`
}

function isExpired(entry: InFlightTurnSnapshot, now = Date.now()): boolean {
  return now - entry.updatedAt > MAX_AGE_MS
}

// --- Bounded recovery projection --------------------------------------------

function truncated(text: string, cap: number): string {
  return text.length > cap ? `${text.slice(0, cap)}…[truncated]` : text
}

/** JSON-ish preview of an arbitrary value that never does more work than the
 *  cap allows: emission and traversal are the same loop, so a multi-megabyte
 *  tool result costs ~cap code units of walking, not a full `JSON.stringify`
 *  of the whole structure first. Output is display text, not guaranteed
 *  parseable JSON — every `argsText` consumer already JSON.parses inside a
 *  try/catch with an `{}` fallback, and the pre-existing `…[truncated]`
 *  marker broke parseability anyway. */
function boundedJsonPreview(value: unknown, cap: number): string {
  const out: string[] = []
  let used = 0
  let full = false

  const emit = (text: string): void => {
    if (full) {
      return
    }

    if (used + text.length >= cap) {
      out.push(`${text.slice(0, Math.max(0, cap - used))}…[truncated]`)
      full = true

      return
    }

    out.push(text)
    used += text.length
  }

  const seen = new Set<object>()

  const walk = (node: unknown, depth: number): void => {
    if (full) {
      return
    }

    if (node === null || node === undefined) {
      emit('null')

      return
    }

    if (typeof node === 'string') {
      // Slice BEFORE stringify so a giant string never gets fully escaped.
      emit(JSON.stringify(node.length > cap ? node.slice(0, cap) : node))

      return
    }

    if (typeof node === 'number' || typeof node === 'boolean' || typeof node === 'bigint') {
      emit(String(node))

      return
    }

    if (typeof node !== 'object') {
      emit('null')

      return
    }

    if (seen.has(node) || depth >= PREVIEW_MAX_DEPTH) {
      emit(Array.isArray(node) ? '[…]' : '{…}')

      return
    }

    seen.add(node)

    if (Array.isArray(node)) {
      emit('[')

      for (let index = 0; index < node.length && !full; index += 1) {
        if (index > 0) {
          emit(',')
        }

        walk(node[index], depth + 1)
      }

      emit(']')

      return
    }

    emit('{')

    let first = true

    // `for…in` walks keys lazily — `Object.keys` would allocate the full key
    // array of a huge object even when the budget stops us after a few.
    for (const key in node) {
      if (full) {
        break
      }

      if (!Object.prototype.hasOwnProperty.call(node, key)) {
        continue
      }

      if (!first) {
        emit(',')
      }

      first = false
      emit(`${JSON.stringify(key)}:`)
      walk((node as Record<string, unknown>)[key], depth + 1)
    }

    emit('}')
  }

  try {
    walk(value, 0)
  } catch {
    // A throwing getter mid-walk: keep whatever was emitted so far.
  }

  return out.join('')
}

function previewOf(value: unknown, cap: number): null | string {
  if (value === undefined || value === null) {
    return null
  }

  if (typeof value === 'string') {
    return truncated(value, cap)
  }

  return boundedJsonPreview(value, cap)
}

/** Project one part into its bounded journal form, or null to drop it.
 *  Builds fresh small objects — this replaces the old JSON deep clone, so
 *  multi-megabyte payloads are never copied at all. */
function projectPart(part: ChatMessagePart, role: ChatMessage['role'], tight: boolean): ChatMessagePart | null {
  if (part.type === 'text') {
    const text = typeof part.text === 'string' ? part.text : ''

    // The user prompt stays IN FULL: recovery matching (`userMessagesMatch`)
    // normalizes and compares the whole text, and truncating it would orphan
    // the journaled turn from its transcript row.
    return {
      type: 'text',
      text: role === 'user' ? text : truncated(text, tight ? ASSISTANT_TEXT_CAP_TIGHT : ASSISTANT_TEXT_CAP)
    }
  }

  if (part.type === 'reasoning') {
    if (tight) {
      return null
    }

    const text = typeof part.text === 'string' ? part.text : ''

    return { type: 'reasoning', text: truncated(text, REASONING_CAP) }
  }

  if (part.type === 'tool-call') {
    // Identity + status survive; large bodies (args, results, terminal dumps,
    // inline diffs) shrink to short previews or are dropped outright.
    const argsPreview = tight ? null : previewOf(part.argsText ?? part.args, TOOL_PREVIEW_CAP)
    const resultPreview = tight ? undefined : (previewOf(part.result, TOOL_PREVIEW_CAP) ?? undefined)

    return {
      // Renderers read `argsText`; an empty args object keeps the part shape
      // valid without copying a potentially huge original.
      args: {} as never,
      argsText: argsPreview ?? '',
      toolCallId: part.toolCallId,
      toolName: part.toolName,
      type: 'tool-call',
      ...(resultPreview !== undefined && { result: resultPreview }),
      ...('isError' in part && { isError: Boolean(part.isError) })
    } as ChatMessagePart
  }

  // Anything else (images, files, attachments-as-parts) is bulk the journal
  // does not need — recovery only inspects text/reasoning/tool-call parts.
  return null
}

function projectMessage(message: ChatMessage, tight: boolean): ChatMessage {
  const parts = message.parts
    .map(part => projectPart(part, message.role, tight))
    .filter((part): part is ChatMessagePart => part !== null)

  return {
    id: message.id,
    parts,
    role: message.role,
    ...(message.timestamp !== undefined && { timestamp: message.timestamp }),
    ...(message.pending !== undefined && { pending: message.pending }),
    ...(message.error !== undefined && { error: message.error }),
    ...(message.interim !== undefined && { interim: message.interim }),
    ...(message.attachmentRefs !== undefined && { attachmentRefs: [...message.attachmentRefs] })
  }
}

function projectTail(messages: ChatMessage[], tight: boolean): ChatMessage[] {
  return messages.map(message => projectMessage(message, tight))
}

// --- Recovery-tail extraction and merge (semantics unchanged) ----------------

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
 *  interim rows sealed after it) back to the user prompt that started it.
 *  Returns the live rows BY REFERENCE — the caller projects them into the
 *  bounded journal form, so no deep clone happens here. */
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

// --- Boot: kill switch, legacy migration, global sweep -----------------------

let booted = false
let journalDisabled = false

function parseStoredEntry(raw: null | string): null | StoredJournalEntry {
  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw)

    if (
      !parsed ||
      parsed.version !== STORE_VERSION ||
      !Array.isArray(parsed.messages) ||
      typeof parsed.updatedAt !== 'number'
    ) {
      return null
    }

    return parsed as StoredJournalEntry
  } catch {
    return null
  }
}

/** One-shot v1 → v2 migration ("just boot and migrate"): every non-expired
 *  legacy entry runs through the SAME bounded projection — so a multi-MB
 *  legacy blob comes out KB-class — and lands on its per-session key
 *  (new-key-wins). The v1 key is then deleted unconditionally: this is a
 *  best-effort journal, and an unparseable or quota-broken blob is worth less
 *  than an unblocked main thread. Idempotent by construction — the v1 key is
 *  gone after the first run. */
function migrateLegacyStore(store: Storage): void {
  let raw: null | string = null

  try {
    raw = store.getItem(LEGACY_STORAGE_KEY)

    if (raw) {
      const parsed = JSON.parse(raw) as { entries?: Record<string, InFlightTurnSnapshot>; version?: number }

      if (parsed && parsed.version === 1 && parsed.entries && typeof parsed.entries === 'object') {
        for (const [storedSessionId, entry] of Object.entries(parsed.entries)) {
          if (!entry || !Array.isArray(entry.messages) || typeof entry.updatedAt !== 'number' || isExpired(entry)) {
            continue
          }

          const key = sessionKey(storedSessionId)

          if (store.getItem(key) !== null) {
            continue
          }

          const migrated: StoredJournalEntry = {
            messages: projectTail(entry.messages, false),
            streamId: entry.streamId ?? null,
            turnStartedAt: entry.turnStartedAt ?? null,
            updatedAt: entry.updatedAt,
            version: STORE_VERSION
          }

          let serialized = JSON.stringify(migrated)

          if (serialized.length > ENTRY_BUDGET) {
            serialized = JSON.stringify({ ...migrated, messages: projectTail(entry.messages, true) })
          }

          try {
            store.setItem(key, serialized)
          } catch {
            // Quota mid-migration: keep going — later entries may be smaller.
          }
        }
      }
    }
  } catch {
    // Unparseable legacy blob: fall through to the delete.
  }

  if (raw !== null) {
    try {
      store.removeItem(LEGACY_STORAGE_KEY)
    } catch {
      // Nothing left to do.
    }
  }
}

/** Startup sweep over v2 keys: drop expired/malformed entries and enforce the
 *  global entry cap. This replaces the v1 per-write filter/sort/slice — which
 *  was exactly the work that made every 400ms tick pay for all sessions. */
function sweepJournalKeys(store: Storage): void {
  const keys: string[] = []

  for (let index = 0; index < store.length; index += 1) {
    const key = store.key(index)

    if (key && key.startsWith(KEY_PREFIX)) {
      keys.push(key)
    }
  }

  const alive: { key: string; updatedAt: number }[] = []

  for (const key of keys) {
    const entry = parseStoredEntry(store.getItem(key))

    if (!entry || isExpired(entry)) {
      try {
        store.removeItem(key)
      } catch {
        // Best effort.
      }

      continue
    }

    alive.push({ key, updatedAt: entry.updatedAt })
  }

  if (alive.length > MAX_ENTRIES) {
    alive.sort((a, b) => b.updatedAt - a.updatedAt)

    for (const { key } of alive.slice(MAX_ENTRIES)) {
      try {
        store.removeItem(key)
      } catch {
        // Best effort.
      }
    }
  }
}

function ensureBooted(): void {
  if (booted) {
    return
  }

  booted = true

  const store = storage()

  if (!store) {
    return
  }

  try {
    journalDisabled = store.getItem(KILL_SWITCH_KEY) === '1'
  } catch {
    // Unreadable kill switch: treat as enabled.
  }

  if (journalDisabled) {
    // A/B isolation: when disabled, the journal subsystem does NOTHING —
    // no migration (the legacy blob stays put for the revert case), no
    // sweep, no writes. The kill-switch read above is the only storage
    // access the whole subsystem performs.
    return
  }

  try {
    migrateLegacyStore(store)
    sweepJournalKeys(store)
  } catch {
    // Boot housekeeping is best-effort; the write path guards itself.
  }
}

// --- Hot path ----------------------------------------------------------------

const persistTimers = new Map<string, ReturnType<typeof setTimeout>>()
const persistLatest = new Map<string, JournalableSessionState>()
const quotaWarned = new Set<string>()
const oversizeWarned = new Set<string>()
/** Distinct sessions this renderer has journaled since boot — when a NEW one
 *  pushes the count past the global cap, the sweep runs once more so a
 *  long-lived renderer cannot accumulate unbounded keys between reloads. */
const sessionsWrittenSinceBoot = new Set<string>()

/** Test hook: re-run boot (kill switch, migration, sweep) on next use and
 *  drop all pending timers/state. */
export function resetInFlightTurnJournalForTests(): void {
  booted = false
  journalDisabled = false

  for (const timer of persistTimers.values()) {
    clearTimeout(timer)
  }

  persistTimers.clear()
  persistLatest.clear()
  quotaWarned.clear()
  oversizeWarned.clear()
  sessionsWrittenSinceBoot.clear()
}

function isQuotaError(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    (error.name === 'QuotaExceededError' || error.name === 'NS_ERROR_DOM_QUOTA_REACHED' || error.code === 22)
  )
}

function writeSnapshot(storedSessionId: string, state: JournalableSessionState): void {
  const store = storage()

  if (!store) {
    return
  }

  const started = performance.now()
  const tail = recoverableTail(state.messages, state.streamId)

  // `persistTimers` no longer holds this session (its timer just fired), so
  // `size + 1` is the number of sessions currently journaling — the journal's
  // own busy-session signal, mirroring StreamDeltaAppliedEvent.busySessions.
  const busySessions = persistTimers.size + 1

  if (tail.length === 0) {
    recordJournalWrite({
      busySessions,
      bytes: 0,
      durationMs: performance.now() - started,
      outcome: 'skipped'
    })

    return
  }

  const entry: StoredJournalEntry = {
    messages: projectTail(tail, false),
    streamId: state.streamId,
    turnStartedAt: state.turnStartedAt,
    updatedAt: Date.now(),
    version: STORE_VERSION
  }

  let serialized = JSON.stringify(entry)

  if (serialized.length > ENTRY_BUDGET) {
    // Structural reduction, never byte-slicing the JSON: re-project with the
    // tight caps (drops reasoning and tool previews, shrinks assistant text).
    serialized = JSON.stringify({ ...entry, messages: projectTail(tail, true) })

    if (serialized.length > ENTRY_HARD_CEILING) {
      // Still enormous after the tight pass — only an un-truncatable user
      // prompt can do this. Skip the write outright: recovery for a
      // pathological prompt is worth less than a multi-MB setItem per tick.
      if (!oversizeWarned.has(storedSessionId)) {
        oversizeWarned.add(storedSessionId)
        console.warn(
          `[hermes] in-flight turn journal entry exceeds the hard size ceiling for session ${storedSessionId}; ` +
            'skipping journal writes for this turn (crash recovery degraded).'
        )
      }

      recordJournalWrite({
        busySessions,
        bytes: serialized.length,
        durationMs: performance.now() - started,
        outcome: 'oversize'
      })

      return
    }
  }

  let outcome: 'error' | 'ok' | 'quota' = 'ok'

  try {
    store.setItem(sessionKey(storedSessionId), serialized)

    // Runtime enforcement of the global entry cap: only when a NEW session
    // key appears while over the cap (rare, and each surviving entry is
    // budget-bounded), so the hot path stays a single setItem.
    if (!sessionsWrittenSinceBoot.has(storedSessionId)) {
      sessionsWrittenSinceBoot.add(storedSessionId)

      if (sessionsWrittenSinceBoot.size > MAX_ENTRIES) {
        sweepJournalKeys(store)
      }
    }
  } catch (error) {
    outcome = isQuotaError(error) ? 'quota' : 'error'

    if (!quotaWarned.has(storedSessionId)) {
      quotaWarned.add(storedSessionId)
      console.warn(
        `[hermes] in-flight turn journal write failed (${outcome}) for session ${storedSessionId}; ` +
          'crash recovery for this turn is degraded.',
        error
      )
    }
  }

  recordJournalWrite({
    busySessions,
    bytes: serialized.length,
    durationMs: performance.now() - started,
    outcome
  })
}

/** Persist the running turn's visible tail (throttled), or clear the entry the
 *  moment the turn settles. Call on every session-state commit. */
export function persistInFlightTurnState(state: JournalableSessionState): void {
  ensureBooted()

  if (journalDisabled) {
    return
  }

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
  ensureBooted()

  if (!storedSessionId) {
    return null
  }

  const store = storage()

  if (!store) {
    return null
  }

  const key = sessionKey(storedSessionId)
  let raw: null | string = null

  try {
    raw = store.getItem(key)
  } catch {
    // A Web Storage read failure degrades to "no journal" — recovery is
    // optional and must never break session resume.
    return null
  }

  const entry = parseStoredEntry(raw)

  if (!entry) {
    return null
  }

  if (isExpired(entry)) {
    try {
      store.removeItem(key)
    } catch {
      // Best effort.
    }

    return null
  }

  return {
    messages: entry.messages,
    streamId: entry.streamId,
    turnStartedAt: entry.turnStartedAt,
    updatedAt: entry.updatedAt
  }
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
  ensureBooted()

  if (!storedSessionId) {
    return
  }

  const timer = persistTimers.get(storedSessionId)

  if (timer) {
    clearTimeout(timer)
    persistTimers.delete(storedSessionId)
  }

  persistLatest.delete(storedSessionId)

  const store = storage()

  if (!store) {
    return
  }

  try {
    store.removeItem(sessionKey(storedSessionId))
  } catch {
    // Best effort.
  }
}
