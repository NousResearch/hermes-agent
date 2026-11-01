import type { ChatMessage, ChatMessagePart } from './types'

const SILENT_RESPONSE = '[[SILENT]]'

/**
 * Project the model's control-only reply out of the transcript without
 * changing the stored message or its branch identity. A streaming prefix is
 * buffered until it either becomes the control token or diverges from it.
 */
export function assistantDisplayParts(message: ChatMessage): ChatMessagePart[] {
  if (message.role !== 'assistant' || message.error) {
    return message.parts
  }

  // Reasoning events can split one response into several text parts. The wire
  // response is their direct concatenation, not the paragraph-joined display.
  const response = message.parts
    .filter((part): part is Extract<ChatMessagePart, { type: 'text' }> => part.type === 'text')
    .map(part => part.text)
    .join('')
    .trim()

  const controlOnly =
    response === SILENT_RESPONSE || Boolean(message.pending && response && SILENT_RESPONSE.startsWith(response))

  return controlOnly ? message.parts.filter(part => part.type !== 'text') : message.parts
}
