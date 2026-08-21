/**
 * Shared rewind/interrupt core for the prompt verbs — the ONE implementation
 * of the submit primitive + the pure message math behind cancel / reload /
 * restore / edit / branch-visibility. Both the primary chat (`index.ts`) and
 * session tiles (`session-tile-actions.ts`) build on these so the two surfaces
 * can't silently diverge (the tile's "sends only once" busy-ref bug was exactly
 * that class of drift). The functions here are PURE — planners compute from a
 * `ChatMessage[]`, optimistic transforms map a `ClientSessionState` to the next
 * — so each caller keeps its own state-write + error-handling wiring.
 */

import type { AppendMessage, ThreadMessage } from '@assistant-ui/react'

import type { ClientSessionState } from '@/app/types'
import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
import {
  branchGroupForUser,
  type ChatMessage,
  chatMessageText,
  completeOpenTimelineParts,
  textPart
} from '@/lib/chat-messages'

import {
  appendText,
  isFailedUserTurn,
  isSessionBusyError,
  visibleUserIndexAtOrdinal,
  visibleUserMessageIndices,
  visibleUserOrdinal,
  withSessionBusyRetry,
  withSessionNotFoundResume
} from './utils'

type RequestGateway = <T = unknown>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>

/**
 * Post-rewrite durable ids of the surviving visible user turns, in visible-user
 * ordinal order — the gateway's `survivor_user_row_ids` on a truncating
 * `prompt.submit`. A rewind's `replace_messages` re-inserts the kept prefix as
 * NEW SQLite rows, so every pre-rewind `ChatMessage.rowId` on a surviving
 * bubble is stale the moment the rewind lands; targeting one on the next
 * rewind/edit/regenerate gets a fail-closed 4018 from the gateway. `null`
 * means that turn has no durable id (drop the cached one, don't keep a stale
 * one). Absent entirely = the submit didn't truncate a durable session (or an
 * older gateway) — leave state untouched.
 */
export type SurvivorUserRowIds = readonly (null | number)[]

interface PromptSubmitResult {
  status?: string
  survivor_user_row_ids?: unknown
}

export function survivorRowIdsFrom(result: PromptSubmitResult | undefined): SurvivorUserRowIds | undefined {
  const raw = result?.survivor_user_row_ids

  if (!Array.isArray(raw)) {
    return undefined
  }

  return raw.map(entry => (typeof entry === 'number' && Number.isInteger(entry) ? entry : null))
}

/**
 * Rebind the surviving visible user turns to their authoritative post-rewind
 * row ids (positional, same visible-user filter `visibleUserOrdinal` uses —
 * the exact parity truncate ordinals already rely on). Turns past the end of
 * the survivor list — the resubmitted turn itself, whose durable id doesn't
 * exist yet — and `null` entries get their cached rowId cleared instead: a
 * stale id now addresses an archived row and would be refused with 4018.
 */
export function rebindSurvivorRowIds(messages: ChatMessage[], survivorRowIds: SurvivorUserRowIds): ChatMessage[] {
  // Same ordinal space as the truncate math: visible AND persisted (failed
  // turns never reached the gateway, so they hold no survivor slot).
  const indices = new Set(visibleUserMessageIndices(messages))
  let ordinal = 0

  return messages.map((message, index) => {
    if (!indices.has(index)) {
      return message
    }

    const next = ordinal < survivorRowIds.length ? survivorRowIds[ordinal] : null
    ordinal += 1

    if (typeof next === 'number') {
      return message.rowId === next ? message : { ...message, rowId: next }
    }

    return message.rowId === undefined ? message : { ...message, rowId: undefined }
  })
}

/**
 * Renderer-synthetic message ids (`${timestamp}-${index}-${role}` from
 * chat-messages/hydration.ts, plus older `user-…` / `assistant-…` shapes). Gateway
 * history never carries them — only durable `row_id` / platform message_id.
 */
export function isSyntheticRendererId(messageId: string | undefined): boolean {
  return (
    typeof messageId === 'string' &&
    (messageId.startsWith('user-') ||
      messageId.startsWith('assistant-') ||
      messageId.includes('-synthetic-') ||
      /^\d+-\d+-(user|assistant|tools)\b/.test(messageId))
  )
}

/**
 * Build `prompt.submit` truncation params. `confirm_truncate` states that this
 * submit really is a rewind/edit/regenerate: the gateway drops history only for
 * a submit that says so, so a leftover ordinal riding along on an ordinary send
 * cannot delete the transcript. Ordinal 0 additionally truncates to an empty
 * transcript (restore/regenerate the first user turn), which the gateway gates
 * behind `confirm_empty_truncate` on top of that.
 */
export function truncateSubmitParams(
  truncateOrdinal: number | undefined,
  truncateMessageId?: string,
  truncateRowId?: number
): Record<string, unknown> {
  const hasOrdinal = typeof truncateOrdinal === 'number' && Number.isInteger(truncateOrdinal) && truncateOrdinal >= 0
  const hasRowId = typeof truncateRowId === 'number' && Number.isInteger(truncateRowId)

  const hasMessageId =
    typeof truncateMessageId === 'string' && truncateMessageId.length > 0 && !isSyntheticRendererId(truncateMessageId)

  if (!hasOrdinal && !hasMessageId && !hasRowId) {
    return {}
  }

  return {
    confirm_truncate: true,
    ...(hasOrdinal ? { truncate_before_user_ordinal: truncateOrdinal } : {}),
    ...(hasMessageId ? { truncate_before_message_id: truncateMessageId } : {}),
    ...(hasRowId ? { truncate_before_row_id: truncateRowId } : {}),
    ...(truncateOrdinal === 0 ? { confirm_empty_truncate: true } : {})
  }
}

interface DurableHistoryMessage {
  display_kind?: string
  role?: string
  row_id?: unknown
  text?: string
}

/**
 * Resolve the durable row id of a user turn by CONTENT against the gateway's
 * stamped transcript (`session.history` ships `row_id` per persisted row).
 *
 * The edit-after-interrupt bubble has no bound rowId (the durable row exists —
 * the client just never learned its id), and renderer/gateway ordinal spaces
 * diverge, so ordinal math cannot substitute (#87059: a 12-turn divergence cut
 * 78 messages). Content matching is exact-or-nothing: a unique text match wins;
 * ambiguity prefers the LAST match only when `expectedOrdinal` says the target
 * is the latest persisted turn (the edit-after-interrupt shape — the just-sent
 * message is by definition the newest). Anything else returns undefined and the
 * caller degrades to a plain resubmit, never a guessed cut.
 */
export async function resolveDurableRowId(
  requestGateway: RequestGateway,
  sessionId: string,
  sourceText: string,
  expectedOrdinal: number | undefined
): Promise<number | undefined> {
  const wanted = sourceText.trim()

  if (!wanted) {
    return undefined
  }

  let messages: DurableHistoryMessage[]

  try {
    const result = await requestGateway<{ messages?: unknown }>('session.history', { session_id: sessionId })

    messages = Array.isArray(result?.messages) ? (result.messages as DurableHistoryMessage[]) : []
  } catch {
    return undefined
  }

  const durableUsers = messages.filter(
    message =>
      message.role === 'user' &&
      !message.display_kind &&
      typeof message.row_id === 'number' &&
      Number.isInteger(message.row_id)
  )

  const matches = durableUsers.filter(message => (message.text ?? '').trim() === wanted)

  if (matches.length === 1) {
    return matches[0].row_id as number
  }

  if (matches.length > 1 && typeof expectedOrdinal === 'number' && expectedOrdinal >= durableUsers.length - 1) {
    const last = matches[matches.length - 1]

    return durableUsers[durableUsers.length - 1] === last ? (last.row_id as number) : undefined
  }

  return undefined
}

/**
 * Rewind a turn: `prompt.submit` with an optional `truncate_before_user_ordinal`
 * / `truncate_before_message_id` / `truncate_before_row_id` (drops that user turn + everything after).
 * Idle rewinds submit directly; live/stuck turns interrupt first, and a raced
 * "session busy" response interrupts + retries through the shared busy gate.
 *
 * Resolves with the gateway's post-rewrite survivor row ids (see
 * `SurvivorUserRowIds`) so the caller can rebind surviving bubbles, or
 * undefined when the submit didn't truncate a durable transcript.
 */
export async function runRewindSubmit(
  requestGateway: RequestGateway,
  sessionId: string,
  text: string,
  truncateOrdinal: number | undefined,
  truncateMessageId: string | undefined,
  interruptFirst: boolean,
  recovery?: { storedSessionId?: null | string; onSessionRecovered?: (sessionId: string) => void },
  truncateRowId?: number,
  sourceText?: string
): Promise<SurvivorUserRowIds | undefined> {
  // Recovery may rebind the live id mid-flight; interrupt/submit must both
  // follow it rather than pinning the dead one.
  let liveSessionId = sessionId

  // A truncation without a durable address is the #87059 shape: the gateway
  // fails it closed (4004) for any persisted session, so sending it can only
  // produce an error. Resolve the row id by content first (the durable row
  // usually exists — the bubble just never learned its id, e.g. edit after an
  // interrupted turn). When resolution fails too, degrade to a PLAIN resubmit:
  // append the corrected text without dropping anything, never guess a cut.
  let resolvedRowId = truncateRowId
  let resolvedOrdinal = truncateOrdinal
  let resolvedMessageId = truncateMessageId

  const wantsTruncation =
    typeof truncateOrdinal === 'number' ||
    typeof truncateRowId === 'number' ||
    (typeof truncateMessageId === 'string' && truncateMessageId.length > 0 && !isSyntheticRendererId(truncateMessageId))

  const hasDurableAddress =
    typeof truncateRowId === 'number' ||
    (typeof truncateMessageId === 'string' && truncateMessageId.length > 0 && !isSyntheticRendererId(truncateMessageId))

  if (wantsTruncation && !hasDurableAddress) {
    resolvedRowId =
      sourceText === undefined
        ? undefined
        : await resolveDurableRowId(requestGateway, liveSessionId, sourceText, truncateOrdinal)

    // Either way the client-side ordinal is untrustworthy here (its space can
    // diverge from the gateway's — the #87059 root). Resolved: the row id alone
    // is the address; sending the divergent ordinal too would trip the
    // gateway's 4030 cross-check. Unresolved: plain resubmit, no truncation.
    resolvedOrdinal = undefined
    resolvedMessageId = undefined
  }

  // Durable id present (#88082, #89244): drop the client ordinal. Renderer
  // ordinals and gateway tip ordinals are different spaces — tail-only
  // prefetch (window-relative), in-place compact (display-lineage scrollback
  // vs tip, prefix_user_count structurally 0), and any later visibility
  // skew. The cut is aimed by the resolved durable id; sending a divergent
  // ordinal trips the gateway's 4030 cross-check. Unknown ids still fail
  // closed at 4018. The ordinal is only a tripwire, and it is calibrated in
  // a space the gateway cannot reliably compute.
  if (wantsTruncation && hasDurableAddress) {
    resolvedOrdinal = undefined
  }

  const interrupt = async () => {
    try {
      await requestGateway('session.interrupt', { session_id: liveSessionId })
    } catch {
      // Best-effort. The submit path still gates on the gateway state.
    }
  }

  const submitFor = (targetId: string) =>
    requestGateway<PromptSubmitResult>(
      'prompt.submit',
      {
        session_id: targetId,
        text,
        ...truncateSubmitParams(resolvedOrdinal, resolvedMessageId, resolvedRowId),
        // A first-turn rewind resolves to an empty transcript, which the
        // gateway additionally gates behind confirm_empty_truncate. When the
        // client ordinal is dropped (durable-id path), carry the flag from
        // the caller's ordinal-0 belief: required when right, ignored by the
        // gateway when the cut isn't actually empty.
        ...(resolvedOrdinal === undefined && truncateOrdinal === 0 ? { confirm_empty_truncate: true } : {})
      },
      PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
    )

  const submit = async () => {
    const { result, sessionId: usedId } = await withSessionNotFoundResume(
      liveSessionId,
      recovery?.storedSessionId,
      submitFor,
      {
        requestGateway,
        onRecovered: recoveredId => {
          liveSessionId = recoveredId
          recovery?.onSessionRecovered?.(recoveredId)
        }
      }
    )

    liveSessionId = usedId

    return survivorRowIdsFrom(result)
  }

  if (interruptFirst) {
    await interrupt()
  }

  try {
    return await submit()
  } catch (err) {
    if (!isSessionBusyError(err)) {
      throw err
    }

    await interrupt()

    return await withSessionBusyRetry(submit)
  }
}

/** Cancel/stop finalize: drop empty pending/stream placeholders, un-pend the rest. */
export function finalizeInterruptedMessages(
  messages: ChatMessage[],
  streamId?: null | string,
  occurredAt = Date.now() / 1000
): ChatMessage[] {
  return messages
    .filter(
      message =>
        !(
          (message.pending || message.id === streamId) &&
          message.parts.length === 0 &&
          !chatMessageText(message).trim()
        )
    )
    .map(message =>
      message.pending || message.id === streamId
        ? {
            ...message,
            completedAt: occurredAt,
            parts: completeOpenTimelineParts(message.parts, occurredAt),
            pending: false
          }
        : message
    )
}

/**
 * Arrival-ordered mid-turn user insert (#73793, #83151).
 *
 * A message typed while a turn streams must land AFTER every assistant row the
 * user had already watched arrive — never spliced above it. Seal the live
 * stream bubble in place (marked interim so the terminal completion settles
 * onto it or follows it instead of duplicating), append the new user bubble at
 * the live tail, and clear `streamId` so the turn's next delta seeds a fresh
 * assistant bubble BELOW the correction rather than mutating the sealed one
 * above it. Also retires the old insert-before-the-active-reply contract whose
 * `lastAssistantIndex` fallback could splice the bubble mid-thread when the
 * stream id was missing or stale (#83151).
 */
export function appendMidTurnUserMessage<
  State extends { interimBoundaryPending: boolean; messages: ChatMessage[]; streamId: null | string }
>(state: State, message: ChatMessage): State {
  const liveId = state.streamId
  const sealed = finalizeInterruptedMessages(state.messages, liveId)
  const sealedLiveKept = liveId !== null && sealed.some(row => row.id === liveId)

  const messages = [
    ...(sealedLiveKept ? sealed.map(row => (row.id === liveId ? { ...row, interim: true } : row)) : sealed),
    message
  ]

  return {
    ...state,
    messages,
    streamId: null,
    interimBoundaryPending: state.interimBoundaryPending || sealedLiveKept
  }
}

// ---------------------------------------------------------------------------
// Reload (regenerate)
// ---------------------------------------------------------------------------

export interface ReloadPlan {
  branchGroupId: string
  /** Original persisted text of the turn — the durable-row-id content key. */
  sourceText: string
  text: string
  truncateOrdinal: number | undefined
  truncateMessageId?: string
  truncateRowId?: number
  userIndex: number
}

/** The user turn to re-run for a reload from `parentId` (or the last turn). */
export function planReload(messages: ChatMessage[], parentId: null | string): null | ReloadPlan {
  const parentIndex = parentId ? messages.findIndex(m => m.id === parentId) : messages.length - 1

  const userBack =
    parentIndex >= 0 ? [...messages.slice(0, parentIndex + 1)].reverse().findIndex(m => m.role === 'user') : -1

  if (userBack < 0) {
    return null
  }

  const userIndex = parentIndex - userBack
  const userMessage = messages[userIndex]
  const text = userMessage ? chatMessageText(userMessage).trim() : ''

  if (!userMessage || !text) {
    return null
  }

  const targetAssistant =
    parentId && messages[parentIndex]?.role === 'assistant'
      ? messages[parentIndex]
      : messages.slice(userIndex + 1).find(m => m.role === 'assistant')

  // Failed turn: the user msg never reached the gateway, so any truncation
  // address would mis-aim (#86573/#86623) — resubmit plainly instead.
  const isFailedTurn = isFailedUserTurn(messages, userIndex)

  return {
    branchGroupId: targetAssistant?.branchGroupId ?? branchGroupForUser(userMessage),
    sourceText: text,
    text,
    truncateOrdinal: isFailedTurn ? undefined : visibleUserOrdinal(messages, userIndex),
    truncateMessageId: isFailedTurn ? undefined : userMessage.id,
    truncateRowId: isFailedTurn ? undefined : userMessage.rowId,
    userIndex
  }
}

/** Optimistic reload state: keep the user turn, hide the branch's assistants. */
export function applyReloadOptimistic(state: ClientSessionState, plan: ReloadPlan): ClientSessionState {
  const nextUserIndex = state.messages.findIndex((m, i) => i > plan.userIndex && m.role === 'user')
  const end = nextUserIndex < 0 ? state.messages.length : nextUserIndex

  return {
    ...state,
    awaitingResponse: true,
    busy: true,
    interrupted: false,
    messages: [
      ...state.messages.slice(0, plan.userIndex + 1),
      ...state.messages
        .slice(plan.userIndex + 1, end)
        .map(m => (m.role === 'assistant' ? { ...m, branchGroupId: plan.branchGroupId, hidden: true } : m))
    ],
    pendingBranchGroup: plan.branchGroupId,
    sawAssistantPayload: false,
    // Arm the turn clock with the optimistic busy. The clock is what bounds
    // the no-payload settle gate's pre-start hold (#86795): an armed turn
    // with no clock is settled by the first running=false heartbeat, so a
    // regenerate racing a heartbeat would lose its spinner mid-flight.
    turnLive: false,
    turnStartedAt: Date.now()
  }
}

// ---------------------------------------------------------------------------
// Restore (rewind checkpoint)
// ---------------------------------------------------------------------------

export interface RestoreTarget {
  text?: string
  userOrdinal?: null | number
}

export interface RestorePlan {
  sourceIndex: number
  /** Original persisted text of the turn — the durable-row-id content key. */
  sourceText: string
  text: string
  truncateOrdinal: number | undefined
  truncateMessageId?: string
  truncateRowId?: number
}

/** Resolve the user turn to rewind to; throws with a user-facing reason. */
export function planRestore(messages: ChatMessage[], messageId: string, target?: RestoreTarget): RestorePlan {
  const idIndex = messages.findIndex(m => m.id === messageId && m.role === 'user')

  const fallbackIndex =
    target?.userOrdinal === null || target?.userOrdinal === undefined
      ? -1
      : visibleUserIndexAtOrdinal(messages, target.userOrdinal)

  const sourceIndex = idIndex >= 0 ? idIndex : fallbackIndex
  const source = messages[sourceIndex]

  if (!source || source.role !== 'user') {
    throw new Error('Could not find the message to restore.')
  }

  const sourceText = chatMessageText(source).trim()
  const text = (sourceText || target?.text?.trim() || '').trim()

  if (!text) {
    throw new Error('Cannot restore an empty message.')
  }

  // Failed turn: the target user msg never reached the gateway, so any
  // truncation address would mis-aim (#86573/#86623) — resubmit plainly.
  const isFailedTurn = isFailedUserTurn(messages, sourceIndex)

  const truncateOrdinal =
    target?.userOrdinal === null || target?.userOrdinal === undefined
      ? visibleUserOrdinal(messages, sourceIndex)
      : target.userOrdinal

  return {
    sourceIndex,
    sourceText: sourceText || text,
    text,
    truncateOrdinal: isFailedTurn ? undefined : truncateOrdinal,
    truncateMessageId: isFailedTurn ? undefined : source.id,
    truncateRowId: isFailedTurn ? undefined : source.rowId
  }
}

// ---------------------------------------------------------------------------
// Edit (revert + resubmit with new text)
// ---------------------------------------------------------------------------

export interface EditPlan {
  editedMessage: ChatMessage
  isFailedTurn: boolean
  sourceIndex: number
  /** Original persisted text of the edited turn — the durable-row-id content key. */
  sourceText: string
  text: string
  truncateOrdinal: number | undefined
  truncateMessageId?: string
  truncateRowId?: number
}

/** Resolve the edited user turn, or null when nothing changed / invalid. */
export function planEdit(messages: ChatMessage[], edited: AppendMessage): EditPlan | null {
  const sourceId = edited.sourceId || edited.parentId
  const text = appendText(edited)

  if (!sourceId || !text || edited.role !== 'user') {
    return null
  }

  const sourceIndex = messages.findIndex(m => m.id === sourceId)
  const source = messages[sourceIndex]

  if (!source || source.role !== 'user' || chatMessageText(source).trim() === text) {
    return null
  }

  // Failed turn: the optimistic user msg never reached the gateway, so a
  // truncate-by-ordinal would 422 — resubmit plainly instead.
  const isFailedTurn = isFailedUserTurn(messages, sourceIndex)

  return {
    editedMessage: { ...source, parts: [textPart(text)] },
    isFailedTurn,
    sourceIndex,
    sourceText: chatMessageText(source).trim(),
    text,
    truncateOrdinal: isFailedTurn ? undefined : visibleUserOrdinal(messages, sourceIndex),
    truncateMessageId: isFailedTurn ? undefined : source.id,
    truncateRowId: isFailedTurn ? undefined : source.rowId
  }
}

/** Optimistic rewind-to state for restore/edit: drop everything after the
 *  source turn (edit swaps in the edited message; restore keeps the original). */
export function applyRewindOptimistic(
  state: ClientSessionState,
  sourceIndex: number,
  editedMessage?: ChatMessage
): ClientSessionState {
  return {
    ...state,
    awaitingResponse: true,
    busy: true,
    interrupted: false,
    messages: editedMessage
      ? [...state.messages.slice(0, sourceIndex), editedMessage]
      : state.messages.slice(0, sourceIndex + 1),
    pendingBranchGroup: null,
    sawAssistantPayload: false,
    // Same as applyReloadOptimistic: seed the clock so the no-payload settle
    // gate holds through the submit round trip but never latches (#86795).
    turnLive: false,
    turnStartedAt: Date.now()
  }
}

// ---------------------------------------------------------------------------
// Branch visibility (assistant-ui hides non-active branches)
// ---------------------------------------------------------------------------

/** Sync each assistant branch message's `hidden` to what the thread renders. */
export function applyBranchVisibility(state: ClientSessionState, next: readonly ThreadMessage[]): ClientSessionState {
  const visibleIds = new Set(next.map(m => m.id))
  let changed = false

  const messages = state.messages.map(message => {
    if (message.role !== 'assistant' || !message.branchGroupId) {
      return message
    }

    const hidden = !visibleIds.has(message.id)

    if (message.hidden === hidden) {
      return message
    }

    changed = true

    return { ...message, hidden }
  })

  return changed ? { ...state, messages } : state
}
