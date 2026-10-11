import {
  assistantTextPart,
  type ChatMessage,
  chatMessageText,
  mergeFinalAssistantText,
  renderMediaTags
} from '@/lib/chat-messages'
import { generatedImageEchoSources, stripGeneratedImageEchoes } from '@/lib/generated-images'

const flat = (text: string) => text.replace(/\s+/g, ' ').trim()

/**
 * Stop seals the live bubble at the click and drops every later delta, but the
 * agent keeps streaming until it honours the interrupt and persists all of it
 * (state.db, the next turn's context). Its interrupted `message.complete`
 * carries that persisted partial (#121594). When it extends what the turn's
 * bubble shows, the bubble takes it; when Stop landed before anything was
 * painted, a bubble is added. Extend-only: a shorter or different text never
 * replaces what the user saw.
 */
export function extendInterruptedReply(messages: ChatMessage[], rawText: string, occurredAt: number): ChatMessage[] {
  const text = renderMediaTags(rawText).trim()

  if (!text) {
    return messages
  }

  const lastUserIndex = messages.findLastIndex(message => message.role === 'user')

  const turn = messages.filter(
    (message, index) => index > lastUserIndex && message.role === 'assistant' && !message.hidden
  )

  // Stop keeps a painted live bubble as a settled, non-interim row; an
  // interim tail means the live segment had nothing painted and was dropped.
  const target = turn.at(-1)

  if (target && !target.interim) {
    const visible = stripGeneratedImageEchoes(text, generatedImageEchoSources(target.parts)).trim()
    const parts = mergeFinalAssistantText(target.parts, visible, occurredAt)
    const shown = flat(chatMessageText(target))
    const persisted = flat(chatMessageText({ ...target, parts }))

    if (persisted.length <= shown.length || !persisted.startsWith(shown)) {
      return messages
    }

    return messages.map(message => (message === target ? { ...message, parts } : message))
  }

  if (turn.some(message => flat(chatMessageText(message)).includes(flat(text)))) {
    return messages
  }

  return [
    ...messages,
    {
      id: `assistant-interrupted-${Date.now()}`,
      role: 'assistant',
      parts: [{ ...assistantTextPart(text, occurredAt), completedAt: occurredAt }],
      timestamp: occurredAt,
      completedAt: occurredAt,
      pending: false
    }
  ]
}
