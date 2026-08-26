import type { ChatMessage } from '@/lib/chat-messages'

// Renderer-local user rows painted by primary submit and tile steer share
// this id shape: `user-<ms>-<base36 nonce>`.
const OPTIMISTIC_USER_ID = /^user-\d+-[a-z0-9]{0,6}$/

function isLocalOptimisticUserRow(message: ChatMessage): boolean {
  return message.role === 'user' && message.rowId === undefined && OPTIMISTIC_USER_ID.test(message.id)
}

/** Remove a renderer-local failed turn without touching durable history. */
export function clearDismissedErrorRows(messages: ChatMessage[], messageId: string): ChatMessage[] {
  const assistantIndex = messages.findIndex(
    message => message.id === messageId && message.role === 'assistant' && Boolean(message.error)
  )

  if (assistantIndex < 0) {
    return messages
  }

  const startIndex =
    assistantIndex > 0 && isLocalOptimisticUserRow(messages[assistantIndex - 1]) ? assistantIndex - 1 : assistantIndex

  return [...messages.slice(0, startIndex), ...messages.slice(assistantIndex + 1)]
}
