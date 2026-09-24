import { resolveSpokenReply, type SpokenReplyMessage } from './spoken-reply'

export interface AutoSpeakMessage extends SpokenReplyMessage {
  interim?: boolean
  pending?: boolean
  text: string
}

export interface AutoSpeakReply {
  id: string
  pending: boolean
  text: string
}

function lastVisibleAssistant(messages: readonly AutoSpeakMessage[]): AutoSpeakMessage | undefined {
  return messages.findLast(message => message.role === 'assistant' && !message.hidden)
}

/** True when a user bubble sits after the spoken assistant slot. */
function userFollowsSpokenOrdinal(messages: readonly AutoSpeakMessage[], spokenOrdinal: number): boolean {
  let ordinal = -1

  for (const message of messages) {
    if (message.hidden) {
      continue
    }

    if (message.role === 'assistant') {
      ordinal += 1
      continue
    }

    if (message.role === 'user' && ordinal >= spokenOrdinal) {
      return true
    }
  }

  return false
}

/**
 * Next reply Desktop auto-speak should read. Skips live-tail rewrites,
 * hydrate duplicates of the same text, and sealed interim narration.
 */
export function selectAutoSpeakReply(
  sessionId: string | null | undefined,
  messages: readonly AutoSpeakMessage[]
): AutoSpeakReply | null {
  const last = lastVisibleAssistant(messages)

  if (!last) {
    return null
  }

  const spoken = resolveSpokenReply(sessionId, messages)

  if (last.id === spoken?.id) {
    return null
  }

  const text = last.text.trim()

  if (!text) {
    return null
  }

  if (last.pending) {
    return { id: last.id, pending: true, text }
  }

  // Narration sealed mid-turn — wait for the final bubble.
  if (last.interim) {
    return null
  }

  // Already voiced this user-turn. A hydrate/history copy can change id AND
  // wording (markdown, trailing delta) — do not treat that as a new reply.
  if (spoken && !userFollowsSpokenOrdinal(messages, spoken.ordinal)) {
    return null
  }

  return { id: last.id, pending: false, text }
}
