import { useMemo } from 'react'

import { Slot as ContribSlot } from '@/contrib/react/slot'
import { useStoresSelector } from '@/lib/use-session-slice'
import { $sessionStates } from '@/store/session-states'

import { COMPOSER_AREAS } from './contrib'
import { ComposerSessionProvider } from './scope'

/** The `composer.top` banner seam, told which session's composer it sits in: contributions read
 *  it through `useComposerSession` (the SDK door), the runtime id mapped to its stored id. */
export function ComposerTopSlot({ sessionId: id }: { sessionId?: null | string }) {
  const sessionId = id ?? null

  const storedSessionId = useStoresSelector([$sessionStates], () =>
    sessionId ? ($sessionStates.get()[sessionId]?.storedSessionId ?? sessionId) : null
  )

  const value = useMemo(() => ({ sessionId, storedSessionId }), [sessionId, storedSessionId])

  return (
    <ComposerSessionProvider value={value}>
      <ContribSlot area={COMPOSER_AREAS.top} />
    </ComposerSessionProvider>
  )
}
