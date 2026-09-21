import { assistantTextPart, type ChatMessage, chatMessageText } from '@/lib/chat-messages'

/** Whitespace normalization is for comparison only; cuts always address the original text. */
function comparable(text: string): { text: string; ends: number[] } {
  let normalized = ''
  const ends: number[] = []

  for (const match of text.matchAll(/\s+|\S/g)) {
    const value = /^\s/.test(match[0]) ? ' ' : match[0]

    if (!normalized && value === ' ') {
      continue
    }

    normalized += value
    ends.push(match.index + match[0].length)
  }

  return { text: normalized.trimEnd(), ends }
}

function textSpans(messages: ChatMessage[]) {
  let text = ''

  for (const message of messages) {
    for (const part of message.parts) {
      if (part.type !== 'text' || !part.text.trim()) {
        continue
      }

      if (text) {
        text += '\n\n'
      }

      text += part.text
    }
  }

  return { text }
}

function hasContent(message: ChatMessage): boolean {
  return Boolean(chatMessageText(message).trim() || message.parts.some(part => part.type !== 'text') || message.error)
}

/** Merge only the assistant run between two already-paired user occurrences. */
export function mergeLiveAssistantRun(projected: ChatMessage[], cached: ChatMessage[]): ChatMessage[] {
  const local = cached.filter(hasContent)

  if (!local.length) {
    return projected
  }

  if (!projected.length) {
    return local
  }

  const live = comparable(textSpans(local).text).text
  const remoteSpans = textSpans(projected)
  const remote = comparable(remoteSpans.text)
  const terminal = projected.at(-1)!

  if (terminal.error) {
    return [...local, terminal]
  }

  if (!remote.text || live.startsWith(remote.text)) {
    return local.map((message, index) =>
      index === local.length - 1 ? { ...message, pending: terminal.pending === true } : message
    )
  }

  if (remote.text.startsWith(live)) {
    const suffix = live ? remoteSpans.text.slice(remote.ends[live.length - 1]) : remoteSpans.text
    const last = local.at(-1)!

    return [
      ...local.slice(0, -1),
      {
        ...last,
        pending: terminal.pending === true,
        interim: terminal.interim,
        parts: [...last.parts, assistantTextPart(suffix)]
      }
    ]
  }

  // No evidence that one is a prefix of the other: retain both. A stale or
  // transformed snapshot is not permission to erase unseen live output.
  return [...local, ...projected]
}
