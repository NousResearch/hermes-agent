import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { $inboxAvailable } from '@/store/inbox'
import { type InterfaceMode, shownInMode } from '@/store/interface-mode'
import { applySidebarNavPrefs } from '@/store/sidebar-nav'

import { ARTIFACTS_ROUTE, CAPABILITIES_ROUTE, CRON_ROUTE, MESSAGING_ROUTE } from '../../routes'
import type { AppView } from '../../routes'
import type { SidebarNavItem } from '../../types'

import { INBOX_NAV_ITEM, InboxNavBadge, inboxNavShown } from './inbox-nav-badge'

// A row's `tier` is the one mode it belongs to (Simple keeps the setup rows,
// Advanced adds the readouts); the list filters once, nothing is passed down.
export const SIDEBAR_NAV: SidebarNavItem[] = [
  {
    id: 'new-session',
    label: '',
    icon: props => <Codicon name="robot" {...props} />,
    action: 'new-session',
    keybindActionId: 'session.new'
  },
  {
    id: 'capabilities',
    label: '',
    icon: props => <Codicon name="symbol-misc" {...props} />,
    route: CAPABILITIES_ROUTE,
    keybindActionId: 'nav.capabilities'
  },
  {
    id: 'messaging',
    label: '',
    icon: props => <Codicon name="comment" {...props} />,
    route: MESSAGING_ROUTE,
    keybindActionId: 'nav.messaging'
  },
  INBOX_NAV_ITEM,
  // Artifacts and Scheduled jobs are outputs of running Hermes the developer
  // way; Capabilities and Messaging are how anyone sets it up.
  {
    id: 'artifacts',
    label: '',
    icon: props => <Codicon name="files" {...props} />,
    route: ARTIFACTS_ROUTE,
    keybindActionId: 'nav.artifacts',
    tier: 'advanced'
  },
  {
    id: 'cron',
    label: '',
    icon: props => <Codicon name="watch" {...props} />,
    route: CRON_ROUTE,
    keybindActionId: 'nav.cron',
    tier: 'advanced'
  }
]

// Built-in rows light up on their own view; contributed rows at their own route.
const NAV_VIEWS: Partial<Record<string, AppView>> = {
  artifacts: 'artifacts',
  capabilities: 'capabilities',
  cron: 'cron',
  inbox: 'inbox',
  messaging: 'messaging'
}

export function navItemActive(item: SidebarNavItem, currentView: AppView, pathname: string): boolean {
  if (currentView === 'extension') {
    return Boolean(item.route) && pathname === item.route
  }

  return NAV_VIEWS[item.id] === currentView
}

/** The rows the sidebar shows: built-in + contributed, for this mode, under the user's nav prefs. */
export function useSidebarNavItems(
  contributedNav: SidebarNavItem[],
  interfaceMode: InterfaceMode,
  navPrefs: Parameters<typeof applySidebarNavPrefs>[1]
): SidebarNavItem[] {
  const inboxAvailable = useStore($inboxAvailable)

  return useMemo(
    () =>
      applySidebarNavPrefs(
        [...SIDEBAR_NAV, ...contributedNav].filter(shownInMode(interfaceMode)).filter(inboxNavShown(inboxAvailable)),
        navPrefs
      ),
    [contributedNav, inboxAvailable, interfaceMode, navPrefs]
  )
}

/** What a row carries after its label (the Inbox's unread count). */
export function NavItemTrailing({ item }: { item: SidebarNavItem }) {
  return item.id === INBOX_NAV_ITEM.id ? <InboxNavBadge /> : null
}
