import { pageOutranksRenderedTranscript } from '@/app/chat/rewind-generation'
import { $sessionStates } from '@/store/session-states'
import type { SessionMessagesResponse } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

type RewindPage = Pick<SessionMessagesResponse, 'rewind_generation'> | null | undefined
type RenderedTranscript = Pick<ClientSessionState, 'rewindGeneration'> | undefined

interface RewindGate {
  /** True when the page is the authority and the graft must not "correct" it
   *  back: an intentional rewind landed between the two reads, or generation
   *  evidence is unavailable (older backend) and the pre-guard
   *  refresh-authoritative behavior applies. */
  pageNewerThanRendered: boolean
  /** The page's own generation, for the state publish's stamp. */
  rewindGeneration: number | undefined
}

/**
 * Rewind-generation gate for a resume-path persisted page read against the
 * rendered transcript (#119819): a page read AFTER an /undo or /retry
 * legitimately holds fewer rows — it is the authority.
 */
export function resumeRewindGate(page: RewindPage, rendered: RenderedTranscript): RewindGate {
  return {
    pageNewerThanRendered: pageOutranksRenderedTranscript(page, rendered),
    rewindGeneration: page?.rewind_generation
  }
}

/** The same gate for the prefetch graft: the rendered transcript is the
 *  active session's published state. */
export function prefetchRewindGate(page: RewindPage, activeSessionId: string | null): RewindGate {
  return resumeRewindGate(page, $sessionStates.get()[activeSessionId ?? ''])
}
