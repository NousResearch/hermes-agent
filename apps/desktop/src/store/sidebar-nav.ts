import { Codecs, persistentAtom } from '@/lib/persisted'

// Sidebar-nav preferences — the store behind `host.sidebar`. A plugin (the
// sidebar manager) hides rows or re-orders them; core still owns rendering, so
// a preference only ever moves or drops a row that would otherwise render, and
// an id naming a row that no longer exists is inert.
//
// These PERSIST, deliberately: a disable → enable cycle must not scramble the
// layout the user chose. The consequence is that they outlive the plugin that
// wrote them — a row hidden by a plugin stays hidden after that plugin is
// removed, and there is no core UI that restores it. The escape hatch is the
// API that wrote it (`host.sidebar.hide(id, false)`, or clearing
// `hermes.desktop.sidebarNavHidden.v1` / `...NavOrder.v1`), which means the
// plugin that owns the preference is the one that can hand it back. Scoping a
// preference to its writer needs the plugin's identity in this store, which the
// host API does not carry yet.
export const $sidebarHiddenNavIds = persistentAtom<string[]>(
  'hermes.desktop.sidebarNavHidden.v1',
  [],
  Codecs.stringArray
)

export const $sidebarNavOrderIds = persistentAtom<string[]>('hermes.desktop.sidebarNavOrder.v1', [], Codecs.stringArray)

/** Hide or show one nav row (idempotent; empty ids are ignored). */
export function setSidebarNavHidden(navId: string, hidden = true): void {
  const id = navId.trim()

  if (!id) {
    return
  }

  const prev = $sidebarHiddenNavIds.get()
  const present = prev.includes(id)

  if (present === hidden) {
    return
  }

  $sidebarHiddenNavIds.set(hidden ? [...prev, id] : prev.filter(existing => existing !== id))
}

/** Replace the manual nav order with `ids` (deduped, blanks dropped). Rows the
 *  order does not name keep their default relative order after the named ones. */
export function setSidebarNavOrder(ids: string[]): void {
  const seen = new Set<string>()
  const next: string[] = []

  for (const raw of ids) {
    const id = (raw || '').trim()

    if (id && !seen.has(id)) {
      seen.add(id)
      next.push(id)
    }
  }

  $sidebarNavOrderIds.set(next)
}

/** Apply hidden + order to the nav rows. Pure so the semantics are testable
 *  without a DOM: hidden ids drop out; ids named in `order` come first in that
 *  exact order; rows the order does not name keep their default relative order
 *  after them. Unknown ids in either list are inert. */
export function orderSidebarNav<T extends { id: string }>(
  items: readonly T[],
  hidden: readonly string[],
  order: readonly string[]
): T[] {
  const hiddenSet = new Set(hidden)
  const byId = new Map(items.map(item => [item.id, item]))
  const placed = new Set<string>()
  const ordered: T[] = []

  for (const id of order) {
    const item = byId.get(id)

    if (item && !hiddenSet.has(id) && !placed.has(id)) {
      ordered.push(item)
      placed.add(id)
    }
  }

  for (const item of items) {
    if (!hiddenSet.has(item.id) && !placed.has(item.id)) {
      ordered.push(item)
    }
  }

  return ordered
}
