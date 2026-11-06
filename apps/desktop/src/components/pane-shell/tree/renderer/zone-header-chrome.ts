import type { Contribution } from '@/contrib/types'

import { paneChrome } from './track-model'

/** Standing navigation uses the titlebar toggle; tool panels keep a local collapse action. */
export function zoneHeaderChrome(panes: readonly (Contribution | undefined)[]) {
  const sidebarTabs = panes.length > 0 && panes.every(pane => paneChrome(pane).hideOnly)
  // Collapsing the main workspace would strand the app behind a strip.
  const minimizable = !panes.some(pane => paneChrome(pane).uncloseable)

  return {
    minimizable,
    showMinimize: minimizable && !sidebarTabs,
    stripVariant: sidebarTabs ? ('sidebar' as const) : ('default' as const)
  }
}
