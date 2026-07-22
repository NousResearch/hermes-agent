import { useEffect } from 'react'

import { $freshSessionRequest } from '@/store/profile'

/**
 * Reset the primary chat when a profile/project action requests a fresh draft.
 *
 * Kept as a hook so the profile store remains router-agnostic while the chat
 * controller owns the actual route/session reset. The nanostore listener is
 * intentionally synchronous: profile selection must clear the old route and
 * session refs before a same-stack Enter can submit through the previous
 * profile's conversation.
 */
export function useFreshSessionRequests(startFreshSessionDraft: () => void): void {
  useEffect(() => $freshSessionRequest.listen(() => startFreshSessionDraft()), [startFreshSessionDraft])
}
