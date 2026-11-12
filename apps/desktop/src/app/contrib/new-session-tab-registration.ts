import { useEffect, useLayoutEffect, useRef } from 'react'

import { $newSessionTabAction } from '@/components/pane-shell/tree/store'

/** Register the tab-strip "+" action (the generic renderer stays
 *  session-agnostic; null until wired hides the glyph).
 *
 *  The registered wrapper reads the LATEST action through a ref instead of
 *  re-registering on identity churn: the action changes whenever
 *  openNewSessionTile does (i18n copy, gateway props), and the old
 *  identity-keyed effect wrote null on every churn — the strip glyph reads
 *  this atom and cycled hidden→visible with it, and correctness leaned on the
 *  teardown and re-register staying one synchronous step (StrictMode's dev
 *  replay writes null once per mount regardless). Set exactly once per mount;
 *  cleared on unmount, where the strip hides its glyph and the sdk's
 *  host.newChat gate — which refuses while the atom is null — is meaningful. */
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
