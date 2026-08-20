import type { AppView } from '../../routes'
import type { SidebarNavItem } from '../../types'

interface SidebarNavActiveInput {
  contributed: boolean
  currentView: AppView
  item: SidebarNavItem
  pathname: string
}

export function sidebarVisibleView(currentView: AppView, focusedSessionIsTile: boolean): AppView {
  return focusedSessionIsTile ? 'chat' : currentView
}

export function sidebarNavItemIsActive({ contributed, currentView, item, pathname }: SidebarNavActiveInput): boolean {
  if (contributed) {
    return currentView === 'extension' && Boolean(item.route) && pathname === item.route
  }

  return (
    (item.id === 'skills' && currentView === 'skills') ||
    (item.id === 'messaging' && currentView === 'messaging') ||
    (item.id === 'artifacts' && currentView === 'artifacts') ||
    (item.id === 'cron' && currentView === 'cron') ||
    (currentView !== 'chat' && Boolean(item.route) && pathname === item.route)
  )
}
