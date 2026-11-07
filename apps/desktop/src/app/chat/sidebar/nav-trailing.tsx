/**
 * `sidebar.nav.trailing` — decorations attached to a contributed nav row.
 *
 * A plugin that pairs a page with a sidebar entry (e.g. Kanban) often needs to show live
 * state on that row: a badge, a count, a "paused" chip. Re-registering the nav row to
 * change its label would tear the sidebar list down mid-session, so the row's chrome
 * carries a render slot instead, and the plugin's decorator re-renders on its own state.
 *
 * Kept beside `index.tsx` rather than inside it: the sidebar module is a facade already
 * over its file-length cap, and this is a self-contained seam.
 */

import type { ReactNode } from 'react'

import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'

import { SIDEBAR_NAV_TRAILING_AREA, type SidebarNavTrailingContribution } from '../../routes'
import type { SidebarNavItem } from '../../types'

/**
 * Does this `sidebar.nav.trailing` decoration belong on this row?
 *
 * Contribution ids are namespaced by the registry (`kanban:nav`), so a plugin that declares
 * `{ navId: 'nav' }` would never match its own row: match the route it asked for, or the id
 * exactly / by its unqualified suffix.
 */
export function rowOwnsTrailing(
  item: SidebarNavItem,
  data: Partial<SidebarNavTrailingContribution> | undefined
): boolean {
  if (!data) {
    return false
  }

  if (data.navPath) {
    return data.navPath === item.route
  }

  return Boolean(data.navId) && (item.id === data.navId || item.id.endsWith(`:${data.navId}`))
}

/**
 * The chips this row owns, rendered after its label (`ml-auto` pushes them to the edge).
 *
 * Each one sits behind its own boundary: a decorator that throws must cost its chip, not
 * the sidebar. Returns null when the plugin declares nothing, so a plain row adds no node.
 */
export function NavTrailingChips({ item }: { item: SidebarNavItem }): ReactNode {
  const contributions = useContributions(SIDEBAR_NAV_TRAILING_AREA)
  const mine = contributions.filter(c =>
    rowOwnsTrailing(item, c.data as Partial<SidebarNavTrailingContribution> | undefined)
  )
  if (!mine.length) {
    return null
  }

  return mine.map(c => (
    <ContribBoundary id={c.id} key={c.id} variant="chip">
      <ContribRender render={c.render!} />
    </ContribBoundary>
  ))
}
