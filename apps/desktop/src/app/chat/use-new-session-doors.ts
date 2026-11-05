import { useCallback, useEffect } from 'react'

import type { useSessionActions } from '@/app/session/hooks/use-session-actions'
import { prepareGenericNewSession } from '@/app/session/new-session-route'
import type { SidebarNavItem } from '@/app/types'
import { $newSessionTabAction } from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'

import { startNewSessionBesideFocusedTab } from './focused-tab-open'
import { openNewSessionTab as openNewSessionTabAction } from './new-session-tab'

type SessionActions = ReturnType<typeof useSessionActions>

/**
 * The window's "new session" doors, in one place:
 * - `openNewSessionTab`: the tab-strip "+" and ⌘T always stack a new tab (see `openNewSessionTab`).
 *   Registered here as the strip's "+" action (the generic renderer stays session-agnostic; null
 *   until wired hides the glyph).
 * - `startNewSession` / `navigateSidebar`: the sidebar "New session" row and ⌘N open a new tab
 *   beside the focused session tab, never over it; main only when main is the tab the user is in
 *   (see `startNewSessionBesideFocusedTab`). Both open a GENERIC session: they drop any owner an
 *   earlier bot chat pinned and target the default-profile preference, else the root `default`
 *   profile (`prepareGenericNewSession`); only "+"/⌘T aim at the selected bot.
 *   Every other sidebar row navigates as before.
 */
export function useNewSessionDoors(
  openNewSessionTile: SessionActions['openNewSessionTile'],
  startFreshSessionDraft: () => void,
  selectSidebarItem: (item: SidebarNavItem) => void
) {
  const openNewSessionTab = useCallback(() => openNewSessionTabAction(openNewSessionTile), [openNewSessionTile])

  useEffect(() => {
    $newSessionTabAction.set(openNewSessionTab)

    return () => $newSessionTabAction.set(null)
  }, [openNewSessionTab])

  const startNewSession = useCallback(() => {
    prepareGenericNewSession()

    if (!startNewSessionBesideFocusedTab(openNewSessionTile)) {
      startFreshSessionDraft()
    }
  }, [openNewSessionTile, startFreshSessionDraft])

  const navigateSidebar = useCallback(
    (item: SidebarNavItem) => {
      if (item.action === 'new-session') {
        prepareGenericNewSession()
      }

      if (item.action === 'new-session' && startNewSessionBesideFocusedTab(openNewSessionTile)) {
        setWorkspaceScope('sessions')

        return
      }

      selectSidebarItem(item)
    },
    [openNewSessionTile, selectSidebarItem]
  )

  return { navigateSidebar, openNewSessionTab, startNewSession }
}
