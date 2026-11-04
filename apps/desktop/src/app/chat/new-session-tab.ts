import type { useSessionActions } from '@/app/session/hooks/use-session-actions'
import { $workspaceMode, $workspaceNewSessionTarget, $workspaceOwnerKey } from '@/components/pane-shell/workspace-scope'

import { frontUntouchedDraftTab } from './reuse-draft-tab'

type OpenNewSessionTile = ReturnType<typeof useSessionActions>['openNewSessionTile']

// The draft tab only exists once session.create answers, so until then the
// reuse check cannot see it. Clicks meanwhile belong to that pending draft
// (the create fronts it when it lands) rather than each starting a twin. The
// latch is per window, like the tab strip, and clears on success or failure.
let pendingDraftCreate: null | Promise<void> = null

/**
 * The tab-strip "+" and ⌘T: open a new session as its own tab (stacked into
 * the workspace zone) WITHOUT polluting the session list. Created
 * `listed: false`, so each new tab's in-memory session stays out of the
 * sidebar until its first message persists a turn and a refresh surfaces it —
 * Cursor-style. Several may be open, but only one blank: an untouched draft
 * (main or a tab) of the default owner is fronted instead of minting a
 * twin, and so is a draft still being created (#145594).
 *
 * Bot Mode aims the "+" at the selected bot's own profile. Anything short of
 * an exact route — a group chat's `blocked` target, an orphaned roster row —
 * falls THROUGH to the ordinary session rather than refusing: the main strip
 * carries plain session tabs alongside bot chats, so a "+" there must never
 * be dead just because the sidebar's current selection has nowhere to route.
 */
export function openNewSessionTab(openNewSessionTile: OpenNewSessionTile): void {
  const workspaceOwnerKey = $workspaceOwnerKey.get()
  const workspaceNewSessionTarget = $workspaceNewSessionTarget.get()

  if ($workspaceMode.get() === 'bots' && workspaceNewSessionTarget?.kind === 'route' && workspaceOwnerKey) {
    void openNewSessionTile('center', {
      listed: false,
      route: workspaceNewSessionTarget.route,
      workspaceScope: {
        ownerRoute: workspaceNewSessionTarget.route,
        workspaceMode: 'bots',
        workspaceOwnerKey
      }
    })

    return
  }

  if (pendingDraftCreate || frontUntouchedDraftTab()) {
    return
  }

  const create = openNewSessionTile('center', { listed: false }).finally(() => {
    if (pendingDraftCreate === create) {
      pendingDraftCreate = null
    }
  })

  pendingDraftCreate = create
}
