import type { Contribution } from '@/contrib/types'

// Sidebar-nav preferences — the `sidebarNav.prefs` registry area. A plugin
// (the sidebar manager) hides nav rows or re-orders them by CONTRIBUTING a
// preference; core still owns rendering and merges every contribution at
// render, so a preference only ever moves or drops a row that would otherwise
// render, and an id naming a row that does not exist is inert.
//
// Why a contribution and not a persisted `host.sidebar.hide()` store: the
// `host` singleton cannot tell which plugin wrote a preference, so a persisted
// write would outlive the plugin that made it (a hidden row with nothing left
// to restore it) and two plugins would overwrite each other's order. A
// contribution is attributed, merged with a stated rule, and dropped by the
// loader's per-plugin disposer on disable/reload — the rows come back on their
// own. The USER's choices persist in the plugin's own `ctx.storage`; the plugin
// re-contributes them on register.

export const SIDEBAR_NAV_PREFS_AREA = 'sidebarNav.prefs'

/** Payload (`data`) of a `sidebarNav.prefs` contribution. Ids are the nav rows'
 *  own ids: the core rows `'new-session' | 'capabilities' | 'messaging' |
 *  'artifacts' | 'cron'` (see `SidebarNavId`) or a `sidebar.nav` contribution's id. */
export interface SidebarNavPrefsContribution {
  /** Rows to drop. Merged as the UNION across contributions. */
  hide?: string[]
  /** Rows to place first, in this order. The first-registered contribution's
   *  order wins; later ones place only ids not yet placed. */
  order?: string[]
}

const cleanIds = (ids: unknown): string[] =>
  Array.isArray(ids) ? ids.filter((id): id is string => typeof id === 'string' && id.trim() !== '') : []

/** Apply every `sidebarNav.prefs` contribution to the nav rows. Pure so the
 *  arbitration is testable without a DOM: hidden = union of every `hide`
 *  (hide beats order); `order` = first-registered contribution first, later
 *  contributions place only ids not yet placed; rows no order names keep their
 *  default relative order after the named ones; unknown ids are inert. */
export function applySidebarNavPrefs<T extends { id: string }>(
  items: readonly T[],
  contributions: readonly Contribution[]
): T[] {
  const hidden = new Set<string>()
  const order: string[] = []

  for (const c of contributions) {
    const prefs = c.data as SidebarNavPrefsContribution | undefined

    cleanIds(prefs?.hide).forEach(id => hidden.add(id))
    order.push(...cleanIds(prefs?.order))
  }

  const byId = new Map(items.map(item => [item.id, item]))
  const placed = new Set<string>()
  const ordered: T[] = []

  for (const id of order) {
    const item = byId.get(id)

    if (item && !hidden.has(id) && !placed.has(id)) {
      ordered.push(item)
      placed.add(id)
    }
  }

  for (const item of items) {
    if (!hidden.has(item.id) && !placed.has(item.id)) {
      ordered.push(item)
    }
  }

  return ordered
}
