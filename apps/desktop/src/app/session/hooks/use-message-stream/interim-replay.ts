import { type ChatMessage, chatMessageText, normalizeWs } from '@/lib/chat-messages'

/** A late seal must not turn a tool-only continuation into a second commentary. */
export function isToolOnlyInterimReplay(
  messages: ChatMessage[],
  streamId: string | null,
  text: string,
  interimBoundaryPending: boolean
): boolean {
  if (!streamId || !interimBoundaryPending) {
    return false
  }

  const streamIndex = messages.findIndex(message => message.id === streamId)
  const live = messages[streamIndex]

  if (!live?.parts.some(part => part.type === 'tool-call')) {
    return false
  }

  // Any new response in this bubble is a separate occurrence. A tool may have
  // started after its text but before the legacy/delayed interim seal arrived.
  if (live.parts.some(part => (part.type === 'text' || part.type === 'reasoning') && part.text.trim())) {
    return false
  }

  const lastUserIndex = messages.findLastIndex(message => message.role === 'user')
  const normalized = normalizeWs(text)

  return messages.some(
    (message, index) =>
      index > lastUserIndex &&
      index < streamIndex &&
      message.role === 'assistant' &&
      message.interim === true &&
      !message.hidden &&
      normalizeWs(chatMessageText(message)) === normalized
  )
}
