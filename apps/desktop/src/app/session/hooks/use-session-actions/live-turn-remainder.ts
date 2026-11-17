import { appendAssistantTextPart, type ChatMessage, chatMessageText } from '@/lib/chat-messages'

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
  const spans: string[] = []

  for (const message of messages) {
    let text = ''

    for (const part of message.parts) {
      if (part.type === 'text') {
        text += part.text
      } else if (text) {
        spans.push(text)
        text = ''
      }
    }

    if (text) {
      spans.push(text)
    }
  }

  // Adjacent chunks are one text occurrence, even if an earlier activation
  // split them. Only a message/tool/channel boundary inserts a separator.
  return { text: spans.filter(text => text.trim()).join('\n\n') }
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

  const terminal = projected.at(-1)!

  if (terminal.error) {
    // A previous activation may already have appended this retained failure
    // beside a distinct local response. Reconcile that occurrence, not the run.
    const replayIndex = local.findIndex(message => message.id === terminal.id)

    if (replayIndex >= 0 && local.length > 1) {
      return [
        ...local.slice(0, replayIndex),
        ...mergeLiveAssistantRun(projected, [local[replayIndex]]),
        ...local.slice(replayIndex + 1)
      ]
    }

    if (local.some(message => message.error && message.error !== terminal.error && message.id !== terminal.id)) {
      return [...local, ...projected]
    }
  }

  const live = comparable(textSpans(local).text).text
  const remoteSpans = textSpans(projected)
  const remote = comparable(remoteSpans.text)

  const settled = (message: ChatMessage): ChatMessage => ({
    ...message,
    pending: terminal.pending === true,
    interim: terminal.interim,
    ...(terminal.error ? { error: terminal.error, errorSurface: terminal.errorSurface ?? message.errorSurface } : {})
  })

  if (!remote.text || live.startsWith(remote.text)) {
    return local.map((message, index) => (index === local.length - 1 ? settled(message) : message))
  }

  if (remote.text.startsWith(live)) {
    let suffix = live ? remoteSpans.text.slice(remote.ends[live.length - 1]) : remoteSpans.text
    const last = local.at(-1)!
    const lastPart = last.parts.at(-1)

    // This is the same text occurrence, not another paragraph. Keeping a
    // mid-word suffix in its own part invents a boundary on the next resume.
    if (lastPart?.type === 'text' && /\s$/.test(lastPart.text)) {
      suffix = suffix.trimStart()
    }

    return [
      ...local.slice(0, -1),
      {
        ...settled(last),
        parts: appendAssistantTextPart(last.parts, suffix)
      }
    ]
  }

  // No evidence that one is a prefix of the other: retain both. A stale or
  // transformed snapshot is not permission to erase unseen live output.
  return [...local, ...projected]
}
