import { atom, computed, type ReadableAtom } from 'nanostores'

import { $onboardingPhase } from './onboarding-phase'
import { $sessionTiles, storedSessionIdForRuntimeId } from './session-states'

export interface OnboardingSession {
  /** The computer the run lives on (`activeGatewayConnectionId()` at its start); only its backend knows the run. */
  connectionId: null | string
  /** Names the run in `onboarding.ended` and `onboarding.state`. */
  runId: string
  runtimeId: string
  storedId: string
}

/** The onboarding chat while a run is in progress in this window. */
export const $onboardingSession = atom<null | OnboardingSession>(null)

/** Whether runtime id `sessionId` shows the onboarding chat in progress. Matched by the stored id too:
 *  a resume after a reconnect or a backend restart gives the chat a new runtime id. Only while the run
 *  is shown: an ended run is never in progress, whatever the session atom still holds. */
export function isOnboardingChat(sessionId: null | string | undefined): boolean {
  const session = $onboardingSession.get()

  return Boolean(
    sessionId &&
    session &&
    $onboardingPhase.get() === 'shown' &&
    (session.runtimeId === sessionId || storedSessionIdForRuntimeId(sessionId) === session.storedId)
  )
}

/** `isOnboardingChat` as a store, for a render. The React Compiler memoizes a plain call on its
 *  arguments, so a component that calls `isOnboardingChat(sessionId)` keeps its first answer while the
 *  session id stays the same (Skip setup outlived the run that way). */
export const onboardingChatStore = (sessionId: null | string): ReadableAtom<boolean> =>
  computed([$onboardingSession, $onboardingPhase, $sessionTiles], () => isOnboardingChat(sessionId))
