import type { ClientSessionState } from '@/app/types'
import type { ChatMessage } from '@/lib/chat-messages'
import { $messages } from '@/store/session'

// Renderer-local user rows painted by primary submit and tile steer share
// this id shape: `user-<ms>-<base36 nonce>`.
const OPTIMISTIC_USER_ID = /^user-\d+-[a-z0-9]{0,6}$/

function isLocalOptimisticUserRow(message: ChatMessage): boolean {
  return message.role === 'user' && message.rowId === undefined && OPTIMISTIC_USER_ID.test(message.id)
}

/** Dismiss a failed turn: renderer-local rows leave; a saved reply stays,
 * cleared of the grafted error. Durable history is never removed. */
export function clearDismissedErrorRows(messages: ChatMessage[], messageId: string): ChatMessage[] {
  const assistantIndex = messages.findIndex(
    message => message.id === messageId && message.role === 'assistant' && Boolean(message.error)
  )

  if (assistantIndex < 0) {
    return messages
  }

  const assistant = messages[assistantIndex]

  // A saved reply (rowId present): `mergeStoredAssistantErrors` grafted this
  // error onto durable content (#132362) — clear the error markers, keep the
  // row and its parts. Splicing it out hid the saved reply until the next
  // refresh re-grafted the same error onto it.
  if (assistant.rowId !== undefined) {
    const cleared = {
      ...assistant,
      error: undefined,
      errorSurface: undefined,
      pending: false
    }

    return [...messages.slice(0, assistantIndex), cleared, ...messages.slice(assistantIndex + 1)]
  }

  const startIndex =
    assistantIndex > 0 && isLocalOptimisticUserRow(messages[assistantIndex - 1]) ? assistantIndex - 1 : assistantIndex

  return [...messages.slice(0, startIndex), ...messages.slice(assistantIndex + 1)]
}

/** The dismissal path ContribWiring runs for a renderer-local failed turn:
 * clear BOTH the live view ($messages) and the warm runtime cache, so the
 * dismissed turn survives neither re-sync nor a warm session switch. The
 * view goes first — the cache update below re-syncs and reads $messages as
 * the error-preservation baseline. */
export function dismissFailedTurn(
  runtimeSessionId: string,
  messageId: string,
  updateSessionState: (sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => void
): void {
  $messages.set(clearDismissedErrorRows($messages.get(), messageId))

  updateSessionState(runtimeSessionId, state => ({
    ...state,
    messages: clearDismissedErrorRows(state.messages, messageId)
  }))
}
