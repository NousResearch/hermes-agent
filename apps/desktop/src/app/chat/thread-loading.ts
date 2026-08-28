import type { ChatMessage } from '@/lib/chat-messages'

export type ThreadLoadingState = 'response' | 'session'

export function lastVisibleMessageIsUser(messages: ChatMessage[]): boolean {
  // Allocation-free reverse scan — runs in a hot $messages computed.
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (!messages[i].hidden) {
      return messages[i].role === 'user'
    }
  }

  return false
}

export function threadLoadingState(
  loadingSession: boolean,
  busy: boolean,
  awaitingResponse: boolean,
  lastVisibleIsUser: boolean
): ThreadLoadingState | undefined {
  if (loadingSession) {
    return 'session'
  }

  if (busy && awaitingResponse && lastVisibleIsUser) {
    return 'response'
  }

  return undefined
}

export function routedSessionIsLoading({
  activeSessionId,
  knownHistory,
  messagesEmpty,
  resumeExhausted,
  routeSessionMismatch,
  routedSessionView
}: {
  activeSessionId: string | null
  knownHistory: boolean
  messagesEmpty: boolean
  resumeExhausted: boolean
  routeSessionMismatch: boolean
  routedSessionView: boolean
}): boolean {
  if (resumeExhausted || !routedSessionView) {
    return false
  }

  if (routeSessionMismatch) {
    return true
  }

  if (!messagesEmpty) {
    return false
  }

  // Brand-new routed drafts are empty on purpose. A session the list already
  // knows has history must keep the loader up until a display-authoritative
  // transcript arrives — including the unproven warm-cache hold, where the
  // runtime is bound but messages are still suppressed.
  return !activeSessionId || knownHistory
}

