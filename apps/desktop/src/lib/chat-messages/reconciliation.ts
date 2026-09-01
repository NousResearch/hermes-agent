import { chatMessageText } from './parts'
import type { ChatMessage, ChatMessagePart } from './types'

const validTimelineBoundary = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

const earliestBoundary = (...values: (number | undefined)[]) => {
  const valid = values.filter(validTimelineBoundary)

  return valid.length ? Math.min(...valid) : undefined
}

const latestBoundary = (...values: (number | undefined)[]) => {
  const valid = values.filter(validTimelineBoundary)

  return valid.length ? Math.max(...valid) : undefined
}

const normalizedTimelineText = (message: ChatMessage) => chatMessageText(message).replace(/\s+/g, ' ').trim()

const assistantTimelineMatch = (stored: ChatMessage, local: ChatMessage) => {
  if (stored.id === local.id) {
    return true
  }

  const localToolIds = new Set(
    local.parts
      .filter(part => part.type === 'tool-call')
      .map(part => (part.type === 'tool-call' ? part.toolCallId : ''))
  )

  const toolMatch = stored.parts.some(part => part.type === 'tool-call' && localToolIds.has(part.toolCallId))

  if (toolMatch) {
    return true
  }

  const storedText = normalizedTimelineText(stored)

  return Boolean(storedText) && storedText === normalizedTimelineText(local)
}

const userTurnMatch = (stored: ChatMessage, local: ChatMessage) =>
  stored.role === 'user' &&
  local.role === 'user' &&
  normalizedTimelineText(stored) === normalizedTimelineText(local) &&
  (stored.attachmentRefs ?? []).join('\n') === (local.attachmentRefs ?? []).join('\n')

/**
 * Find the hydrated assistant representing a local failed tail turn.
 *
 * Text and provider tool-call ids are not globally unique, so the match is
 * deliberately anchored to the last visible user turn on both timelines.
 */
const tailTurnAssistantMatchIndex = (
  storedMessages: ChatMessage[],
  localMessages: ChatMessage[],
  localAssistantIndex: number
) => {
  if (
    localMessages
      .slice(localAssistantIndex + 1)
      .some(message => message.role === 'user' && !message.hidden)
  ) {
    return -1
  }

  let localUserIndex = -1

  for (let index = localAssistantIndex - 1; index >= 0; index -= 1) {
    const message = localMessages[index]

    if (message.role === 'user' && !message.hidden) {
      localUserIndex = index

      break
    }
  }

  let storedUserIndex = -1

  for (let index = storedMessages.length - 1; index >= 0; index -= 1) {
    const message = storedMessages[index]

    if (message.role === 'user' && !message.hidden) {
      storedUserIndex = index

      break
    }
  }

  const localUserCount = localMessages
    .slice(0, localAssistantIndex)
    .filter(message => message.role === 'user' && !message.hidden).length

  const storedUserCount = storedMessages.filter(message => message.role === 'user' && !message.hidden).length

  if (
    localUserIndex === -1 ||
    storedUserIndex === -1 ||
    localUserCount !== storedUserCount ||
    !userTurnMatch(storedMessages[storedUserIndex], localMessages[localUserIndex])
  ) {
    return -1
  }

  const localAssistant = localMessages[localAssistantIndex]

  for (let index = storedMessages.length - 1; index > storedUserIndex; index -= 1) {
    const message = storedMessages[index]

    if (message.role === 'assistant' && !message.hidden && assistantTimelineMatch(message, localAssistant)) {
      return index
    }
  }

  return -1
}

const timelinePartMatch = (stored: ChatMessagePart, local: ChatMessagePart) => {
  if (stored.type !== local.type) {
    return false
  }

  if (stored.type === 'tool-call' && local.type === 'tool-call') {
    return stored.toolCallId === local.toolCallId
  }

  if ((stored.type === 'text' || stored.type === 'reasoning') && local.type === stored.type) {
    return stored.text.replace(/\s+/g, ' ').trim() === local.text.replace(/\s+/g, ' ').trim()
  }

  return false
}

/** Keep richer live timing when durable hydration has only one timestamp per row. */
function reconcileLocalAssistantTimeline(nextMessages: ChatMessage[], currentMessages: ChatMessage[]): ChatMessage[] {
  const localAssistants = currentMessages.filter(message => message.role === 'assistant' && !message.hidden)
  const matches = new Map<number, ChatMessage>()
  let localCursor = localAssistants.length - 1

  for (let nextIndex = nextMessages.length - 1; nextIndex >= 0; nextIndex -= 1) {
    const message = nextMessages[nextIndex]

    if (message.role !== 'assistant' || message.hidden) {
      continue
    }

    for (let localIndex = localCursor; localIndex >= 0; localIndex -= 1) {
      const local = localAssistants[localIndex]

      if (assistantTimelineMatch(message, local)) {
        matches.set(nextIndex, local)
        localCursor = localIndex - 1

        break
      }
    }
  }

  return nextMessages.map((message, messageIndex) => {
    const local = matches.get(messageIndex)

    if (!local) {
      return message
    }

    const unusedLocalParts = new Set(local.parts.map((_, index) => index))

    const parts = message.parts.map(part => {
      const localIndex = local.parts.findIndex(
        (candidate, index) => unusedLocalParts.has(index) && timelinePartMatch(part, candidate)
      )

      if (localIndex === -1) {
        return part
      }

      unusedLocalParts.delete(localIndex)
      const localPart = local.parts[localIndex]

      return {
        ...part,
        completedAt: latestBoundary(part.completedAt, localPart.completedAt),
        timestamp: earliestBoundary(part.timestamp, localPart.timestamp)
      } as ChatMessagePart
    })

    return {
      ...message,
      completedAt: latestBoundary(message.completedAt, local.completedAt, ...parts.map(part => part.completedAt)),
      parts,
      timestamp: earliestBoundary(message.timestamp, local.timestamp, ...parts.map(part => part.timestamp))
    }
  })
}

export function preserveLocalAssistantErrors(
  nextMessages: ChatMessage[],
  currentMessages: ChatMessage[]
): ChatMessage[] {
  nextMessages = reconcileLocalAssistantTimeline(nextMessages, currentMessages)
  const localById = new Map(currentMessages.map(message => [message.id, message]))

  const mergedNextMessages = nextMessages.map(message => {
    if (message.role !== 'assistant' || message.error || message.hidden) {
      return message
    }

    const local = localById.get(message.id)

    if (!local || local.role !== 'assistant' || !local.error || local.hidden) {
      return message
    }

    return {
      ...message,
      error: local.error,
      pending: false
    }
  })

  const existingIds = new Set(mergedNextMessages.map(message => message.id))
  const preserveIds = new Set<string>()
  const normalize = (value: string) => value.replace(/\s+/g, ' ').trim()
  const tailUserInNext = [...mergedNextMessages].reverse().find(message => message.role === 'user' && !message.hidden)
  const tailUserText = tailUserInNext ? normalize(chatMessageText(tailUserInNext)) : ''
  const tailUserRefs = tailUserInNext ? (tailUserInNext.attachmentRefs ?? []).join('\n') : ''

  const matchesTailUserInNext = (candidate: ChatMessage) =>
    Boolean(tailUserInNext) &&
    normalize(chatMessageText(candidate)) === tailUserText &&
    (candidate.attachmentRefs ?? []).join('\n') === tailUserRefs

  for (let index = 0; index < currentMessages.length; index += 1) {
    const message = currentMessages[index]

    if (message.role !== 'assistant' || !message.error || message.hidden || existingIds.has(message.id)) {
      continue
    }

    const hydratedAssistantIndex = tailTurnAssistantMatchIndex(mergedNextMessages, currentMessages, index)

    if (hydratedAssistantIndex !== -1) {
      mergedNextMessages[hydratedAssistantIndex] = {
        ...mergedNextMessages[hydratedAssistantIndex],
        error: message.error,
        pending: false
      }

      continue
    }

    preserveIds.add(message.id)

    for (let probe = index - 1; probe >= 0; probe -= 1) {
      const candidate = currentMessages[probe]

      if (candidate.hidden) {
        continue
      }

      if (candidate.role === 'user' && !existingIds.has(candidate.id) && !matchesTailUserInNext(candidate)) {
        preserveIds.add(candidate.id)
      }

      break
    }
  }

  if (preserveIds.size === 0) {
    return mergedNextMessages
  }

  const preserved = currentMessages
    .filter(message => preserveIds.has(message.id))
    .map(message => ({ ...message, pending: false }))

  return [...mergedNextMessages, ...preserved]
}

export function branchGroupForUser(userMessage: ChatMessage): string {
  return `branch:${userMessage.id}`
}
