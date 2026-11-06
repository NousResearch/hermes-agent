import type { MutableRefObject } from 'react'
import type { NavigateFunction } from 'react-router'

import { leaveWorkspacePageForChat, openNewSessionTab as openNewSessionTabAction } from '@/app/chat/new-session-tab'
import { navigateToWorkspacePage } from '@/app/routes'
import { prepareDefaultNewSession } from '@/app/session/new-session-route'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'

import type { SidebarNavItem } from '../../../types'

import type { useSessionActions } from './index'

type OpenNewSessionTile = ReturnType<typeof useSessionActions>['openNewSessionTile']

/**
 * The sidebar's item-click handler — topical sibling of use-session-actions'
 * index.ts (that file sits on the FILE_LINES ratchet and may only go down;
 * moved code keeps its cap).
 *
 * The New-session row shares the ONE creation path with the tab-strip "+",
 * ⌘T and ⌘N (openNewSessionTab): a real, unlisted session that fronts the
 * workspace as its own tab, leaving the occupied primary chat alive as its
 * own tab. Leaving a workspace page (Settings, Profiles, …) is supplied by
 * the caller, which owns the router navigate and the primary selection.
 */
export function selectSidebarItemAction(
  item: SidebarNavItem,
  input: {
    navigate: NavigateFunction
    openNewSessionTile: OpenNewSessionTile
    selectedStoredSessionIdRef: MutableRefObject<string | null>
  }
): void {
  const { navigate, openNewSessionTile, selectedStoredSessionIdRef } = input

  if (item.action === 'new-session') {
    prepareDefaultNewSession()
    setWorkspaceScope('sessions')
    openNewSessionTabAction(openNewSessionTile, {
      leaveWorkspacePage: () => leaveWorkspacePageForChat(navigate, () => selectedStoredSessionIdRef.current)
    })

    return
  }

  if (item.route) {
    navigateToWorkspacePage(navigate, item.route)
  }
}
