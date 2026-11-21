import { revealTreePane } from '@/components/pane-shell/tree/store'
import type { SessionOwnerRoute } from '@/store/session-request-router'
import { openSessionTile, patchSessionTile } from '@/store/session-states'
import { broadcastSessionsChanged } from '@/store/session-sync'

/** Open a `/btw --open` side chat the owning backend already created as a tab docked to the
 *  RIGHT of the conversation. The main chat never moves; the tile carries the owner route so the
 *  owner's socket stays pinned while the tab is open (the branch tile path's rule). On the shared
 *  owner a session's runtime id is its stored id. */
export function openSideChatTab(storedSessionId: string, ownerRoute?: SessionOwnerRoute): true {
  openSessionTile(storedSessionId, 'right', undefined, null, { ownerRoute, workspaceMode: 'sessions' })
  patchSessionTile(storedSessionId, { runtimeId: storedSessionId })
  revealTreePane(`session-tile:${storedSessionId}`)
  broadcastSessionsChanged()

  return true
}
