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
import { branchGroupForUser, type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'

import {
  appendText,
  isSessionBusyError,
  visibleUserIndexAtOrdinal,
  visibleUserOrdinal,
  withSessionBusyRetry,
  withSessionNotFoundResume
} from './utils'

type RequestGateway = <T = unknown>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>

/**
 * Build `prompt.submit` truncation params. `confirm_truncate` states that this
 * submit really is a rewind/edit/regenerate: the gateway drops history only for
 * a submit that says so, so a leftover ordinal riding along on an ordinary send
 * cannot delete the transcript. Ordinal 0 additionally truncates to an empty
 * transcript (restore/regenerate the first user turn), which the gateway gates
 * behind `confirm_empty_truncate` on top of that.
 */
export function truncateSubmitParams(truncateOrdinal: number | undefined): Record<string, unknown> {
  if (truncateOrdinal === undefined) {
    return {}
  }

  return {
    confirm_truncate: true,
    truncate_before_user_ordinal: truncateOrdinal,
    ...(truncateOrdinal === 0 ? { confirm_empty_truncate: true } : {})
  }
}

/**
 * Rewind a turn: `prompt.submit` with an optional `truncate_before_user_ordinal`
 * (drops that user turn + everything after). Idle rewinds submit directly
 * (interrupting an idle agent can leave a stale interrupt flag that cancels the
 * fresh turn); live/stuck turns interrupt first, and a raced "session busy"
 * response interrupts + retries through the shared busy gate.
 *
 * A rewind runs right after `cancelRun`, and interrupting can drop the
 * gateway's in-memory session — so the follow-up submit hits a dead runtime id
 * and used to surface a bare "session not found" ("Restore checkpoint" failing
 * after a stop). Pass `recovery` so a stale id re-registers and retries like
 * every other session-scoped RPC.
 */
export async function runRewindSubmit(
  requestGateway: RequestGateway,
  sessionId: string,
  text: string,
  truncateOrdinal: number | undefined,
  interruptFirst: boolean,
  recovery?: { storedSessionId?: null | string; onSessionRecovered?: (sessionId: string) => void }
): Promise<void> {
  // Recovery may rebind the live id mid-flight; interrupt/submit must both
  // follow it rather than pinning the dead one.
  let liveSessionId = sessionId

  const interrupt = async () => {
    try {
      await requestGateway('session.interrupt', { session_id: liveSessionId })
    } catch {
      // Best-effort. The submit path still gates on the gateway state.
    }
  }

  const submitFor = (targetId: string) =>
    requestGateway(
      'prompt.submit',
      {
        session_id: targetId,
        text,
        ...truncateSubmitParams(truncateOrdinal)
      },
      PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
    )

  const submit = async () => {
    const { sessionId: usedId } = await withSessionNotFoundResume(liveSessionId, recovery?.storedSessionId, submitFor, {
      requestGateway,
      onRecovered: recoveredId => {
        liveSessionId = recoveredId
        recovery?.onSessionRecovered?.(recoveredId)
      }
    })

    liveSessionId = usedId
  }

  if (interruptFirst) {
    await interrupt()
  }

  try {
    await submit()
  } catch (err) {
    if (!isSessionBusyError(err)) {
      throw err
    }

    await interrupt()
    await withSessionBusyRetry(submit)
  }
}

/** Cancel/stop finalize: drop empty pending/stream placeholders, un-pend the rest. */
export function finalizeInterruptedMessages(messages: ChatMessage[], streamId?: null | string): ChatMessage[] {
  return messages
    .filter(message => !((message.pending || message.id === streamId) && !chatMessageText(message).trim()))
    .map(message => (message.pending || message.id === streamId ? { ...message, pending: false } : message))
}

// ---------------------------------------------------------------------------
// Reload (regenerate)
// ---------------------------------------------------------------------------

export interface ReloadPlan {
  branchGroupId: string
  text: string
  truncateOrdinal: number
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

  return {
    branchGroupId: targetAssistant?.branchGroupId ?? branchGroupForUser(userMessage),
    text,
    truncateOrdinal: visibleUserOrdinal(messages, userIndex),
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
    sawAssistantPayload: false
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
  text: string
  truncateOrdinal: number
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

  const text = (chatMessageText(source).trim() || target?.text?.trim() || '').trim()

  if (!text) {
    throw new Error('Cannot restore an empty message.')
  }

  const truncateOrdinal =
    target?.userOrdinal === null || target?.userOrdinal === undefined
      ? visibleUserOrdinal(messages, sourceIndex)
      : target.userOrdinal

  return { sourceIndex, text, truncateOrdinal }
}

// ---------------------------------------------------------------------------
// Edit (revert + resubmit with new text)
// ---------------------------------------------------------------------------

export interface EditPlan {
  editedMessage: ChatMessage
  isFailedTurn: boolean
  sourceIndex: number
  text: string
  truncateOrdinal: number | undefined
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
  const nextMessage = messages[sourceIndex + 1]
  const isFailedTurn = nextMessage?.role === 'assistant' && Boolean(nextMessage.error)

  return {
    editedMessage: { ...source, parts: [textPart(text)] },
    isFailedTurn,
    sourceIndex,
    text,
    truncateOrdinal: isFailedTurn ? undefined : visibleUserOrdinal(messages, sourceIndex)
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
    sawAssistantPayload: false
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
