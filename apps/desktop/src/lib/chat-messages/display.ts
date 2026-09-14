import type { ChatMessage, ChatMessagePart } from './types'

const SILENT_RESPONSE = '[[SILENT]]'

/** Presentation only: retain raw messages and their turn/branch boundaries.
 * Buffer a possible control token until it diverges or the stream settles. */
export function assistantDisplayParts(message: ChatMessage): ChatMessagePart[] {
  if (message.role !== 'assistant' || message.error) {
    return message.parts
  }

  // Reasoning events can split one response into several text parts. Use the
  // streamed concatenation here, not chatMessageText's paragraph separators.
  const response = message.parts.filter(part => part.type === 'text').map(part => part.text).join('').trim()
  const wholeResponse = response === SILENT_RESPONSE || Boolean(message.pending && response && SILENT_RESPONSE.startsWith(response))
  const parts = message.parts.filter((part, index) => {
    if (part.type !== 'text') {
      return true
    }

    const text = part.text.trim()
    const pendingPrefix = message.pending && index === message.parts.length - 1 && text && SILENT_RESPONSE.startsWith(text)

    return !wholeResponse && text !== SILENT_RESPONSE && !pendingPrefix
  })

  if (parts.length === message.parts.length) {
    return message.parts
  }

  // Keep non-text evidence accessible through the existing collapsed groups.
  return parts
}
