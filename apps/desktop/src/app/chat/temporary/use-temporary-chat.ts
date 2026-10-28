import { useStore } from '@nanostores/react'

import { $temporaryDraft, $temporarySessionIds } from '@/store/temporary-chat'

/**
 * Is the chat behind *sessionId* temporary? A live chat answers from the ids this window created
 * temporary (stored or runtime id); a fresh draft (no id yet) answers from the draft toggle.
 */
export function useTemporaryChat(sessionId: null | string | undefined): boolean {
  const draft = useStore($temporaryDraft)
  const ids = useStore($temporarySessionIds)

  return sessionId ? ids.has(sessionId) : draft
}
