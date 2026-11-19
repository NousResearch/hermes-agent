import { createContext, useContext } from 'react'

import type { CreateBackendSessionForSend } from '@/app/session/hooks/use-session-actions/create-overrides'

/**
 * Where a GPT-Live call started from this composer records. The chat view provides
 * it; the composer's voice hook reads it, so the composer itself passes nothing on.
 */
export interface VoiceCallTarget {
  /** Creates the New Chat draft as the call's own chat (only while it may). */
  createDraftSession?: CreateBackendSessionForSend
  /** The New Chat draft this call would create. */
  draftKey?: string
  /** The session the route points at, to refuse a call while selection catches up. */
  routeSessionId: null | string
  /** The selected stored chat, once selection, route and transcript agree. */
  storedSessionId: null | string
}

const NO_TARGET: VoiceCallTarget = { routeSessionId: null, storedSessionId: null }

const VoiceCallTargetContext = createContext<VoiceCallTarget>(NO_TARGET)

export const VoiceCallTargetProvider = VoiceCallTargetContext.Provider

export const useVoiceCallTarget = (): VoiceCallTarget => useContext(VoiceCallTargetContext)
