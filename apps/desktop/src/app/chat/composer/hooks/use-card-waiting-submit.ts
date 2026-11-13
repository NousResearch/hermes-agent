import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { sessionOnboardingCardWaiting } from '@/store/parked-card-answers'

/** In the onboarding chat typed text answers a waiting card through Send; an empty draft still offers Stop. */
export function useCardWaitingSubmit(sessionId: null | string, busy: boolean, hasPayload: boolean) {
  const cardWaiting = useStore(useMemo(() => sessionOnboardingCardWaiting(sessionId), [sessionId]))

  return { canSubmit: busy || hasPayload, cardWaiting }
}
