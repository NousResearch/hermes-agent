import type { NavigateFunction } from 'react-router'

import { $workspaceIsPage, NEW_CHAT_ROUTE, sessionRoute } from '@/app/routes'
import type { useSessionActions } from '@/app/session/hooks/use-session-actions'
import { $newSessionTabAction } from '@/components/pane-shell/tree/store'
import { $workspaceMode, $workspaceNewSessionTarget, $workspaceOwnerKey } from '@/components/pane-shell/workspace-scope'

import { frontUntouchedDraftTab } from './reuse-draft-tab'

type OpenNewSessionTile = ReturnType<typeof useSessionActions>['openNewSessionTile']

// The draft tab only exists once session.create answers, so until then the
// reuse check cannot see it. Clicks meanwhile belong to that pending draft
// (the create fronts it when it lands) rather than each starting a twin. The
// latch is per window, like the tab strip, and clears on success or failure.
let pendingDraftCreate: null | Promise<void> = null

/**
 * Leave a workspace PAGE (Settings, Profiles, …) for the chat surface. The
 * new tab fronts the workspace PANE, but a page route covers the pane, so a
 * new chat minted from a page stays invisible behind it. The route keeps the
 * primary chat intact when one is selected (its own session route), and only
 * falls back to the new-chat route when the primary holds no session.
 */
export function leaveWorkspacePageForChat(
  navigate: NavigateFunction,
  resolveStoredId: () => null | string
): void {
  if (!$workspaceIsPage.get()) {
    return
  }

  const storedId = resolveStoredId()

  navigate(storedId ? sessionRoute(storedId) : NEW_CHAT_ROUTE)
}

/**
 * The shared new-session action as a palette-row `run`: the tab-strip "+",
 * ⌘T, ⌘N and the sidebar all funnel into the callback wiring publishes
 * ($newSessionTabAction, route-leave included). The fallback covers an
 * unwired surface (e.g. a race before wiring mounts) so the row is never
 * dead — it keeps the legacy new-chat navigation.
 */
export function runSharedNewSessionOrLegacy(fallback: () => void): void {
  const shared = $newSessionTabAction.get()

  if (shared) {
    shared()

    return
  }

  fallback()
}

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
export function openNewSessionTab(
  openNewSessionTile: OpenNewSessionTile,
  options?: { leaveWorkspacePage?: () => void }
): void {
  // A minted tab is invisible behind a workspace page; the caller supplies
  // the route-leave (it owns the router navigate + the primary selection).
  options?.leaveWorkspacePage?.()

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
