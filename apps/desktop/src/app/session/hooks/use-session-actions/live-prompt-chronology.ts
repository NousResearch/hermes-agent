import type { ChatMessage } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

/** User rows belonging to the live turn, not just the last uninterrupted user run. */
export function liveTurnUserMessages(messages: ChatMessage[], turnStartedAt?: number | null): ChatMessage[] {
  const users: ChatMessage[] = []
  const timed = typeof turnStartedAt === 'number' && Number.isFinite(turnStartedAt) && turnStartedAt > 0
  const latestUserIndex = messages.findLastIndex(message => message.role === 'user')
  let crossedDurableOutput = false

  for (let index = latestUserIndex; index >= 0; index--) {
    const message = messages[index]
    const timestamp = message.timestamp
    const dated = timed && typeof timestamp === 'number' && Number.isFinite(timestamp)

    // The immediate user/live-tail run can predate acceptance (optimistic send
    // time or queue drain). Only expansion across durable output needs dates.
    if (crossedDurableOutput && (!dated || timestamp < turnStartedAt)) {
      break
    }

    if (message.role === 'user') {
      users.unshift(message)

      continue
    }

    if (
      message.role === 'assistant' &&
      (message.pending === true || message.interim === true || isLiveTailReplyId(message.id))
    ) {
      continue
    }

    // Tool output is persisted during a turn. Crossing it requires dates on
    // both the output and the prompt; undated historical repeats are ambiguous.
    if (dated && timestamp >= turnStartedAt) {
      crossedDurableOutput = true

      continue
    }

    break
  }

  return users
}

/** A truncated tail can omit the prompt while retaining its dated output. */
export function insertLivePrompt(
  messages: ChatMessage[],
  prompt: ChatMessage[],
  turnStartedAt?: number | null
): ChatMessage[] {
  if (!prompt.length) {
    return messages
  }

  const timed = typeof turnStartedAt === 'number' && Number.isFinite(turnStartedAt) && turnStartedAt > 0

  const firstOutput = timed
    ? messages.findIndex(message => typeof message.timestamp === 'number' && message.timestamp >= turnStartedAt)
    : -1

  const at = firstOutput < 0 ? messages.length : firstOutput

  // Retain the gateway's acceptance time so another activation recognizes this
  // projection across durable output, without trusting an undated old repeat.
  const datedPrompt = timed ? prompt.map(message => ({ ...message, timestamp: turnStartedAt })) : prompt

  return [...messages.slice(0, at), ...datedPrompt, ...messages.slice(at)]
}
