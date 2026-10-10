import type { Contribution } from '@/contrib/types'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { arraysEqual } from '@/lib/storage'

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
//
// That is the PLUGIN half. The user's own per-row hide (#119965) is not a
// plugin's to hold: it is renderer-owned state like every other sidebar
// preference, so it lives in the core-side `$sidebarNavHidden` atom below. At
// render the user's hidden set filters first and the contribution arbitration
// applies to what remains — a row the user hid is simply not there for a
// contribution to revive.

/** The core rows' ids — the one canonical list the docs above already
 *  enumerate. The Settings row list consumes this, and the rendered sidebar is
 *  pinned to it by test, so adding a core row without listing it here breaks a
 *  test instead of silently desyncing Settings from the sidebar. */
export const SIDEBAR_NAV_IDS = ['new-session', 'capabilities', 'messaging', 'artifacts', 'cron'] as const

export const SIDEBAR_NAV_PREFS_AREA = 'sidebarNav.prefs'

/** Payload (`data`) of a `sidebarNav.prefs` contribution. Ids are the nav rows'
 *  own ids: the core rows `'new-session' | 'capabilities' | 'messaging' |
 *  'artifacts' | 'cron'` (see `SidebarNavId`) or a `sidebar.nav` contribution's
 *  REGISTERED id — `ctx.register` namespaces it to `${pluginId}:${id}`. */
export interface SidebarNavPrefsContribution {
  /** Rows to drop. Merged as the UNION across contributions; `capabilities`
   *  (the row that hosts the Plugins tab) is never dropped. */
  hide?: string[]
  /** Rows to place first, in this order. Contributions apply in the registry's
   *  area order (lowest `Contribution.order`, then registration); the first
   *  order wins, later ones place only ids not yet placed. */
  order?: string[]
}

/** Rows a preference may move but never hide: `capabilities` hosts the Plugins
 *  tab, the user's only path to a plugin's own off-switch. */
const NEVER_HIDDEN: ReadonlySet<string> = new Set(['capabilities'])

const cleanIds = (ids: unknown): string[] =>
  Array.isArray(ids) ? ids.filter((id): id is string => typeof id === 'string' && id.trim() !== '') : []

/** Apply every `sidebarNav.prefs` contribution to the nav rows, in the order
 *  given (the caller passes `registry.getArea`, so lowest `order` first, then
 *  registration). Pure so the arbitration is testable without a DOM:
 *  hidden = union of every `hide` minus `NEVER_HIDDEN` (hide beats order);
 *  `order` = first contribution first, later contributions place only ids not
 *  yet placed; rows no order names keep their default relative order after
 *  the named ones; unknown ids are inert. */
export function applySidebarNavPrefs<T extends { id: string }>(
  items: readonly T[],
  contributions: readonly Contribution[]
): T[] {
  const hidden = new Set<string>()
  const order: string[] = []

  for (const c of contributions) {
    const prefs = c.data as SidebarNavPrefsContribution | undefined

    cleanIds(prefs?.hide).forEach(id => {
      if (!NEVER_HIDDEN.has(id)) {
        hidden.add(id)
      }
    })
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

// ---------------------------------------------------------------------------
// The user's own per-row hide (#119965) — a core-side preference, deliberately
// independent of Interface mode (which rows a MODE rests is policy; which rows
// THIS user never wants is a choice) and of the contribution arbitration above.
// Global window-presentation state like `hermes.desktop.interfaceMode.v1`, not
// connection/profile-scoped.
// ---------------------------------------------------------------------------

const NAV_HIDDEN_STORAGE_KEY = 'hermes.desktop.sidebarNavHidden.v1'

/** Stored lists are hand-editable: same defense as `cleanIds`, plus trim and
 *  dedupe so a polluted record reads as the clean set the setter would write. */
const cleanIdList = (ids: unknown): string[] => [
  ...new Set(
    Array.isArray(ids)
      ? ids.filter((id): id is string => typeof id === 'string').map(id => id.trim()).filter(id => id !== '')
      : []
  )
]

export const $sidebarNavHidden = persistentAtom<string[]>(
  NAV_HIDDEN_STORAGE_KEY,
  [],
  Codecs.json<string[]>(cleanIdList)
)

/** Replace the hidden set. Deduped/trimmed; idempotent — re-setting the same
 *  content never re-writes storage, matching every other sidebar setter. */
export function setSidebarNavHidden(ids: readonly string[]): void {
  const next = cleanIdList(ids)

  if (!arraysEqual($sidebarNavHidden.get(), next)) {
    $sidebarNavHidden.set(next)
  }
}

/** Drop the user's hidden rows from a nav list. Pure like `applySidebarNavPrefs`
 *  so the composition is testable without a DOM: survivors keep order and object
 *  identity; unknown or non-string ids in `hidden` are inert. The user's hide
 *  does NOT respect `NEVER_HIDDEN` — that rule guards PLUGIN contributions (a
 *  plugin must not remove the row hosting the Plugins tab, the user's path to
 *  that plugin's off-switch). The user hiding `capabilities` themselves is a
 *  legitimate choice (#119965 lists it) and is recoverable through the same
 *  Settings toggle, the ⌘K palette, and the `nav.capabilities` keybind, all of
 *  which work independently of the row. */
export function applyUserNavHidden<T extends { id: string }>(items: readonly T[], hidden: readonly string[]): T[] {
  const hiddenIds = new Set(cleanIdList(hidden))

  return items.filter(item => !hiddenIds.has(item.id))
}
