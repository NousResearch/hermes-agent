export const AUTO_CONTINUE_ON_MAX_ITERATIONS_MARKER =
  '[Continuing after max-iteration exhaustion]'

type MessageLike = {
  id?: string
  role?: string
  content?: unknown
}

function contentText(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }

  if (!Array.isArray(content)) {
    return ''
  }

  return content
    .map(part => {
      if (!part || typeof part !== 'object') {
        return ''
      }

      const text = (part as { text?: unknown }).text

      return typeof text === 'string' ? text : ''
    })
    .join('')
}

export function isAutoContinueUserContent(content: unknown): boolean {
  return contentText(content).startsWith(AUTO_CONTINUE_ON_MAX_ITERATIONS_MARKER)
}

export function realUserMessageOrdinal(
  messages: readonly MessageLike[],
  currentMessageId: string
): number | null {
  let ordinal = 0

  for (const message of messages) {
    if (message.role !== 'user' || isAutoContinueUserContent(message.content)) {
      continue
    }

    if (message.id === currentMessageId) {
      return ordinal
    }

    ordinal += 1
  }

  return null
}
