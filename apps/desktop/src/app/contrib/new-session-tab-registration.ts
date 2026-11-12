import { useEffect, useLayoutEffect, useRef } from 'react'

import { $newSessionTabAction } from '@/components/pane-shell/tree/store'

/** Register the tab-strip "+" action (the generic renderer stays
 *  session-agnostic; null until wired hides the glyph).
 *
 *  The registered wrapper reads the LATEST action through a ref instead of
 *  re-registering on identity churn: the action changes whenever
 *  openNewSessionTile does (i18n copy, gateway props), and the old
 *  teardown/re-register window left the atom null — host.newChat's bots door
 *  refuses while it is null, so an up-to-date desktop was told to update
 *  itself mid-churn (#154651). Set exactly once per mount; cleared on
 *  unmount, where the strip hides its glyph and the refusal is meaningful. */
export function useNewSessionTabActionRegistration(openNewSessionTab: () => void): void {
  const latest = useRef(openNewSessionTab)

  useLayoutEffect(() => {
    latest.current = openNewSessionTab
  }, [openNewSessionTab])

  useEffect(() => {
    $newSessionTabAction.set(() => latest.current())

    return () => $newSessionTabAction.set(null)
  }, [])
}
