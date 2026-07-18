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
