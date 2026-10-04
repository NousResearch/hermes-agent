import type { ChatMessage } from '@/lib/chat-messages'

/** A folded assistant bubble starts at rowId but can end several stored rows later. */
export function branchCutoffRowId(message: ChatMessage | undefined): number | undefined {
  if (!message) {
    return undefined
  }

  const lastText = message.parts.findLast(part => part.type === 'text' && part.text.trim())
  const rowId = lastText?.sourceRowId ?? message.rowId

  return typeof rowId === 'number' && Number.isSafeInteger(rowId) && rowId > 0 ? rowId : undefined
}
