import type { ChatMessage } from '@/lib/chat-messages'
import {
  type InFlightTurnSnapshot,
  readSnapshot,
  removeSnapshot,
  resetSessionStoreSweepForTests,
  writeSnapshot
} from '@/lib/inflight-turn-journal-store'
import { mergeInFlightMessages } from '@/lib/inflight-turn-merge'
import {
  assistantHasRecoverableContent,
  type InFlightRecoveryResult,
  journaledRuntimeStartedAt
} from '@/lib/inflight-turn-rows'

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

/** Streaming repaints arrive every ~33ms; localStorage writes are synchronous.
 *  Trailing-edge throttle keeps the journal off the hot path — a crash costs at
 *  most this much of the newest tail. */
const PERSIST_THROTTLE_MS = 400

export interface JournalableSessionState {
  awaitingResponse: boolean
  busy: boolean
  messages: ChatMessage[]
  storedSessionId: null | string
  streamId: null | string
  turnStartedAt: null | number
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

  // Only explicitly marked runtime projections use this boundary. Human and
  // legacy turns keep their existing user-run scan below. A later stream row
  // may follow a still-pending hydrated runtime reply without carrying its
  // metadata yet; walk over that open assistant tail, never a completed reply.
  let runtimeStartedAt = visible[assistantIndex].runtimeTurnStartedAt

  for (let index = assistantIndex - 1; runtimeStartedAt === undefined && index >= 0; index -= 1) {
    const row = visible[index]

    if (row.role === 'assistant' && row.pending !== true && row.interim !== true) {
      break
    }

    runtimeStartedAt = row.runtimeTurnStartedAt

    if (row.role === 'user') {
      break
    }
  }

  if (runtimeStartedAt !== undefined) {
    const start = visible.findIndex(message => message.runtimeTurnStartedAt === runtimeStartedAt)

    const tail = visible.slice(start).map(message => message.role === 'assistant' && message.runtimeTurnStartedAt === undefined
      ? { ...message, runtimeTurnStartedAt: runtimeStartedAt }
      : message)

    // One preceding durable row is an optional catch-up anchor. It stays
    // unmarked, is never recovered as runtime content, and lets an idle REST
    // transcript prove the suffix without comparing clocks or all-history text.
    const anchor = visible[start - 1]

    if (anchor && (anchor.rowId !== undefined || anchor.timestamp !== undefined)) {
      const durableAnchor = { ...anchor }
      delete durableAnchor.runtimeTurnStartedAt

      return [durableAnchor, ...tail]
    }

    return tail
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

const persistTimers = new Map<string, ReturnType<typeof setTimeout>>()
const persistLatest = new Map<string, JournalableSessionState>()

/** @internal Test-only reset for module-scoped throttles and sweep state. */
export function resetInFlightTurnJournalStateForTests(): void {
  for (const timer of persistTimers.values()) {
    clearTimeout(timer)
  }

  persistTimers.clear()
  persistLatest.clear()
  resetSessionStoreSweepForTests()
}

function writeTurnTail(storedSessionId: string, state: JournalableSessionState): void {
  const tail = recoverableTail(state.messages, state.streamId)

  if (tail.length === 0) {
    return
  }

  writeSnapshot(storedSessionId, { messages: tail, streamId: state.streamId, turnStartedAt: state.turnStartedAt })
}

/** Persist the running turn's visible tail (throttled), or clear the entry the
 *  moment the turn settles. Call on every session-state commit. */
export function persistInFlightTurnState(state: JournalableSessionState): void {
  const storedSessionId = state.storedSessionId

  if (!storedSessionId) {
    return
  }

  if (!state.busy && !state.awaitingResponse && !state.streamId) {
    // Recovery can leave only human rows uncommitted after the assistant/tool
    // prefix persisted. Keep the original journal (and its durable anchor),
    // rather than rewriting it from that smaller display tail on idle commits.
    if (state.messages.some(message => message.recovered)) {
      const snapshot = readInFlightTurnJournal(storedSessionId)

      const runtimeStartedAt = journaledRuntimeStartedAt(snapshot?.messages ?? [])

      const recoveredRuntime = runtimeStartedAt !== undefined && state.messages.some(message =>
        message.recovered && (message.runtimeTurnStartedAt === runtimeStartedAt ||
          snapshot?.messages.some(journaled => journaled.id === message.id)))

      if (recoveredRuntime) {
        cancelPendingPersist(storedSessionId)

        return
      }
    }

    if (!(state.messages.some(message => message.recovered) &&
      recoverableTail(state.messages, null).some(message => message.recovered))) {
      clearInFlightTurnJournal(storedSessionId)

      return
    }
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
        writeTurnTail(storedSessionId, latest)
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
  const runtimeStartedAt = journaledRuntimeStartedAt(snapshot.messages)

  const recoveredRuntimeIsActive = runtimeStartedAt !== undefined && recovered.messages.some(message =>
    message.id === recovered.streamId && message.runtimeTurnStartedAt === runtimeStartedAt
  )

  if (recovered.caughtUp) {
    clearInFlightTurnJournal(storedSessionId)
  }

  return {
    ...recovered,
    // Never resurrect a stale stream target on an idle resume: with
    // keepPending=false the session is not running. The recovered marker
    // retains uncommitted progress independently until durable catch-up.
    streamId: recovered.applied
      ? (runtimeStartedAt !== undefined ? recovered.streamId : recovered.streamId ?? (options.keepPending ? snapshot.streamId : null))
      : null,
    turnStartedAt: recovered.applied && (runtimeStartedAt === undefined || recoveredRuntimeIsActive)
      ? snapshot.turnStartedAt
      : null
  }
}

export function clearInFlightTurnJournal(storedSessionId: null | string): void {
  if (!storedSessionId) {
    return
  }

  cancelPendingPersist(storedSessionId)
  removeSnapshot(storedSessionId)
}

function cancelPendingPersist(storedSessionId: string): void {
  const timer = persistTimers.get(storedSessionId)

  if (timer) {
    clearTimeout(timer)
    persistTimers.delete(storedSessionId)
  }

  persistLatest.delete(storedSessionId)
}
