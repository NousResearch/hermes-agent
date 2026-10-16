import { type ChatMessage, chatMessageText, normalizeWs as normalizedText } from '@/lib/chat-messages'
import { withoutCoveredAssistantPrefix } from '@/lib/chat-messages/coverage'
import {
  assistantHasRecoverableContent,
  type InFlightRecoveryResult,
  isCommittedRow,
  isLiveProjectionRow,
  journaledRuntimeStartedAt,
  overlayProjectionRow,
  userMessagesMatch
} from '@/lib/inflight-turn-rows'
import { mergeRuntimeTurn } from '@/lib/runtime-turn-merge'

function cloneMessages(messages: ChatMessage[]): ChatMessage[] {
  try {
    return JSON.parse(JSON.stringify(messages)) as ChatMessage[]
  } catch {
    return []
  }
}

function normalizeRecoveredTail(tail: ChatMessage[], keepPending: boolean): ChatMessage[] {
  return cloneMessages(tail).map(message =>
    message.role === 'assistant'
      ? {
          ...message,
          pending: keepPending ? (message.pending ?? true) : false,
          ...(keepPending ? {} : { recovered: true })
        }
      : { ...message, pending: false }
  )
}

/** Rows the base transcript doesn't already hold by id. The journal and the
 *  base can both carry the same row (a resume that replays a still-journaled
 *  turn), and appending it twice puts a duplicate id in the transcript —
 *  which assistant-ui's MessageRepository rejects by throwing. */
function withoutBaseIds(rows: ChatMessage[], baseMessages: ChatMessage[]): ChatMessage[] {
  const baseIds = new Set(baseMessages.map(message => message.id))

  return rows.filter(row => !baseIds.has(row.id))
}

/** Without a matched user interval, a journal can only be stale relative to
 *  the turn that most recently committed. A live row (no durable id yet) is
 *  covered by an equal reply in that last turn; a durable row needs identity.
 *  An older turn saying the same thing does not retire this journal. */
function journalTailAlreadyCommitted(tailAssistants: ChatMessage[], baseMessages: ChatMessage[]): boolean {
  const recoverable = tailAssistants.filter(assistantHasRecoverableContent)

  if (recoverable.length === 0) {
    return false
  }

  // Hidden prompts (bots mode, slash commands) start turns too.
  const lastTurnStart = baseMessages.findLastIndex(message => message.role === 'user')

  const identityCovers = (base: ChatMessage, index: number, journaled: ChatMessage) =>
    base.id === journaled.id || (journaled.rowId === undefined ? index > lastTurnStart : base.rowId === journaled.rowId)

  return recoverable.every(message =>
    baseMessages.some(
      (base, index) =>
        base.role === 'assistant' &&
        !base.hidden &&
        isCommittedRow(base) &&
        identityCovers(base, index, message) &&
        base.error === message.error &&
        normalizedText(chatMessageText(base)) === normalizedText(chatMessageText(message)) &&
        message.parts.every(part => {
          if (part.type === 'tool-call') {
            return base.parts.some(
              candidate =>
                candidate.type === 'tool-call' && Boolean(part.toolCallId) && candidate.toolCallId === part.toolCallId
            )
          }

          return (
            part.type !== 'reasoning' ||
            base.parts.some(
              candidate =>
                candidate.type === 'reasoning' && normalizedText(candidate.text) === normalizedText(part.text)
            )
          )
        })
    )
  )
}

export function mergeInFlightMessages(
  baseMessages: ChatMessage[],
  tailMessages: ChatMessage[],
  options: { keepPending?: boolean } = {}
): InFlightRecoveryResult {
  const noop: InFlightRecoveryResult = {
    applied: false,
    caughtUp: false,
    messages: baseMessages,
    streamId: null,
    turnStartedAt: null
  }

  const tail = normalizeRecoveredTail(tailMessages, Boolean(options.keepPending))

  if (!tail.some(assistantHasRecoverableContent)) {
    return noop
  }

  const runtimeStartedAt = journaledRuntimeStartedAt(tail)

  if (runtimeStartedAt !== undefined) {
    return mergeRuntimeTurn(baseMessages, tail, runtimeStartedAt, Boolean(options.keepPending))
  }

  // Anchor on the latest tail user the base already holds. A steer typed
  // before the first token rides in the tail behind its prompt; when only the
  // prompt persisted, the prompt is the anchor and the steer is recovered
  // output, not a reason to append the prompt a second time.
  let tailUserIndex = tail.findLastIndex(message => message.role === 'user')
  let matchingUserIndex = -1

  for (let index = tailUserIndex; index >= 0; index -= 1) {
    const tailUser = tail[index]

    if (tailUser.role !== 'user') {
      continue
    }

    matchingUserIndex = baseMessages.findLastIndex(message => userMessagesMatch(message, tailUser))

    if (matchingUserIndex >= 0) {
      tailUserIndex = index

      break
    }
  }

  let tailAssistants = tail.slice(tailUserIndex + 1)
  let lastJournalRow = tailAssistants.findLast(assistantHasRecoverableContent) ?? null

  if (matchingUserIndex < 0) {
    // No base user matches the tail's user row (a projected user-inflight row
    // that never persisted, or a tail captured without its user prompt). If the
    // tail's answers are already committed in the transcript, the journal is
    // stale — appending it would re-render the same replies at the end of the
    // conversation. Otherwise, the base never saw this turn at all: append the
    // whole tail (the crash-recovery path the journal exists for).
    if (journalTailAlreadyCommitted(tailAssistants, baseMessages)) {
      return { ...noop, caughtUp: true }
    }

    const streamId = lastJournalRow?.id ?? null

    return {
      applied: true,
      caughtUp: false,
      messages: [...baseMessages, ...withoutBaseIds(tail, baseMessages)],
      // Recovered output stays journaled without pretending the backend is busy.
      streamId: options.keepPending ? streamId : null,
      turnStartedAt: null
    }
  }

  const nextUserIndex = baseMessages.findIndex((message, index) => index > matchingUserIndex && message.role === 'user')
  const end = nextUserIndex < 0 ? baseMessages.length : nextUserIndex
  const afterUser = baseMessages.slice(matchingUserIndex + 1, end)

  const completedReply = afterUser.find(
    message =>
      assistantHasRecoverableContent(message) &&
      isCommittedRow(message) &&
      (message.durableComplete === true ||
        (message.durableComplete === undefined &&
          !message.interim &&
          !message.parts.some(part => part.type === 'tool-call')))
  )

  if (completedReply) {
    // A final reply, not merely a persisted tool round, supersedes this tail.
    return { ...noop, caughtUp: true }
  }

  tailAssistants = withoutCoveredAssistantPrefix(afterUser.filter(isCommittedRow), tailAssistants)
  lastJournalRow = tailAssistants.findLast(assistantHasRecoverableContent) ?? null

  const projectionIndex = baseMessages.findLastIndex(
    (message, index) =>
      index > matchingUserIndex &&
      index < end &&
      message.role === 'assistant' &&
      !message.interim &&
      isLiveProjectionRow(message)
  )

  if (projectionIndex < 0) {
    if (tailAssistants.length === 0) {
      return noop
    }

    const streamId = lastJournalRow?.id ?? null

    return {
      applied: true,
      caughtUp: false,
      messages: [
        ...baseMessages.slice(0, end),
        ...withoutBaseIds(tailAssistants, baseMessages),
        ...baseMessages.slice(end)
      ],
      // Only a running turn keeps a stream target; recovered metadata retains
      // the journal independently until durable completion is observed.
      streamId: options.keepPending ? streamId : null,
      turnStartedAt: null
    }
  }

  // Backend projection row present (text-only): overlay the journal's
  // structure onto it instead of treating it as "caught up" — that is how
  // locally recorded tool progress used to get dropped.
  const projection = baseMessages[projectionIndex]
  const merged = lastJournalRow ? overlayProjectionRow(projection, lastJournalRow) : projection

  const sealedRows = tailAssistants.filter(
    message => message !== lastJournalRow && assistantHasRecoverableContent(message)
  )

  const messages = [
    ...baseMessages.slice(0, projectionIndex),
    ...withoutBaseIds(sealedRows, baseMessages),
    merged,
    ...baseMessages.slice(projectionIndex + 1)
  ]

  return {
    applied: true,
    caughtUp: false,
    messages,
    // Idle recovered output must not reopen a streaming bubble.
    streamId: options.keepPending ? merged.id : null,
    turnStartedAt: null
  }
}
