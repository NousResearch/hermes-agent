import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { Contribution } from '@/contrib/types'
import { useI18n } from '@/i18n'
import { $interfaceMode, shownInMode } from '@/store/interface-mode'
import { $sidebarProfileFilter, toggleSidebarProfileFilter } from '@/store/layout'
import { $profiles } from '@/store/profile'
import { applySidebarNavPrefs } from '@/store/sidebar-nav'

import type { SidebarNavItem } from '../../types'

import { SidebarFilterMenu } from './filter-menu'
import { SimpleWorkspaceMenu } from './simple-workspace-menu'

const NewChatIcon: SidebarNavItem['icon'] = props => <Codicon name="add" {...props} />

/**
 * What the sidebar shows per interface mode. Advanced lists its destinations
 * as rows. Simple keeps New chat and Search at the top, moves every other
 * destination into More, and names things as chats.
 */
export function useSidebarNav(
  builtIn: SidebarNavItem[],
  contributed: SidebarNavItem[],
  prefs: readonly Contribution[]
) {
  const { t } = useI18n()
  const s = t.sidebar
  const mode = useStore($interfaceMode)
  const simple = mode === 'simple'

  return useMemo(() => {
    const items = [...builtIn, ...contributed]
    const available = applySidebarNavPrefs(simple ? items : items.filter(shownInMode(mode)), prefs)

    return {
      homeLabel: simple ? t.rightSidebar.noProjectTitle : s.projects.home,
      icon: (item: SidebarNavItem) => (simple && item.id === 'new-session' ? NewChatIcon : item.icon),
      label: (item: SidebarNavItem) =>
        simple && item.id === 'new-session' ? t.simpleWorkspace.newChat : (s.nav[item.id] ?? item.label),
      moreItems: available.filter(item => item.id !== 'new-session'),
      navItems: simple ? available.filter(item => item.id === 'new-session') : available,
      searchAria: simple ? t.simpleWorkspace.searchChats : s.searchAria,
      searchPlaceholder: simple ? t.simpleWorkspace.searchChats : s.searchPlaceholder,
      sessionsLabel: simple ? t.simpleWorkspace.chats : s.sessions,
      // Simple drops the Pinned section until something is pinned.
      showsPinned: (query: string, pinnedCount: number) => !query && (!simple || pinnedCount > 0),
      simple
    }
  }, [builtIn, contributed, mode, prefs, s, simple, t])
}

/** Simple's Organize chats control, with a removable chip per filtered profile. */
export function OrganizeChatsBar() {
  const { t } = useI18n()
  const simple = useStore($interfaceMode) === 'simple'
  const profileFilter = useStore($sidebarProfileFilter)
  const profiles = useStore($profiles)

  if (!simple) {
    return null
  }

  return (
    <div className="flex flex-wrap items-center gap-1 pb-2 [-webkit-app-region:no-drag]">
      <SidebarFilterMenu label={t.simpleWorkspace.organizeChats} showGrouping side="bottom" />
      {profileFilter.map(profile => (
        <Button key={profile} onClick={() => toggleSidebarProfileFilter(profile)} size="sm" variant="ghost">
          {profiles.find(candidate => candidate.name === profile)?.display_name || profile}
          <Codicon name="close" size="0.75rem" />
        </Button>
      ))}
    </div>
  )
}

/** Advanced keeps the filter menu in the section header; Simple has Organize chats above the list. */
export function HeaderFilterButton({ className }: { className?: string }) {
  const simple = useStore($interfaceMode) === 'simple'

  return simple ? null : (
    <div className="grid size-6 place-items-center">
      <SidebarFilterMenu className={className} />
    </div>
  )
}

interface SimpleSidebarFooterProps {
  items: SidebarNavItem[]
  onNavigate: (item: SidebarNavItem) => void
  pathname: string
  view: string
}

/** Simple's footer row: More, Layout and Settings. */
export function SimpleSidebarFooter({ items, onNavigate, pathname, view }: SimpleSidebarFooterProps) {
  const simple = useStore($interfaceMode) === 'simple'

  return simple ? (
    <SimpleWorkspaceMenu activeRoute={view === 'chat' ? undefined : pathname} items={items} onNavigate={onNavigate} />
  ) : null
}
