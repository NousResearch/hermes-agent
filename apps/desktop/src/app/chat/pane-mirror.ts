/**
 * Mirror a reactive list of "tiles" into layout-tree pane contributions:
 * register a pane per tile, refresh its title in place, and dispose panes whose
 * tile is gone. This is the shared bookkeeping — a keyed registry, a wanted-set
 * diff, a one-time pane closer — behind BOTH session tiles and route (page)
 * tiles; each supplies only what differs (key, title, render, close, edge).
 */

import type { ReadableAtom } from 'nanostores'
import type { ReactElement, ReactNode, PointerEvent as ReactPointerEvent } from 'react'

import { registerPaneCloser, removeTreePane, treePanesWithPrefix } from '@/components/pane-shell/tree/store'
import type { MenuKit } from '@/components/ui/actions-menu'
import { registry } from '@/contrib/registry'
import type { TileDock } from '@/store/session-states'

export interface PaneMirror<T> {
  /** Reactive source list. */
  source: ReadableAtom<T[]>
  /** Extra atoms whose changes should re-sync (e.g. titles living elsewhere). */
  also?: ReadableAtom<unknown>[]
  /** Stable key + pane-id seed for a tile. */
  key: (tile: T) => string
  /** Pane-id namespace — the id is `${prefix}:${key}`. */
  prefix: string
  /** Dock on adoption (default right; `center` = stack into anchor's zone). */
  dir?: (tile: T) => TileDock | undefined
  /** Pane to dock against (default `workspace`) — a drop's target zone. */
  anchor?: (tile: T) => string | undefined
  /** Center docks: the strip slot (stack before this pane id). */
  before?: (tile: T) => null | string | undefined
  minWidth: string
  title: (key: string) => string
  /** Custom lead NODE for the tile's tab (rendered before the label). A live,
   *  self-subscribing component (e.g. a session's status dot) so the strip needn't
   *  re-sync on status/color change — only `title` drives re-registration. */
  tabLead?: (key: string) => ReactNode
  /** Custom label NODE for the tile's tab, self-subscribing for the same reason
   *  as `tabLead` — a name that moves faster than re-registration (see
   *  PaneChrome.tabTitle). Falls back to `title`. */
  tabTitle?: (key: string) => ReactNode
  /** Mint another tile of this kind — the strip's "+" (see PaneChrome.newTab).
   *  Per tile so a mirror can offer it for some of its tabs and not others. */
  newTab?: (key: string) => (() => void) | undefined
  render: (key: string) => ReactNode
  /** Extra rows at the top of the zone tab menu (see PaneChrome.tabMenuPrefix). */
  tabMenuPrefix?: (key: string) => ((kit: MenuKit) => ReactNode) | undefined
  /** Wrap the tile's TAB (domain context menu — session verbs). */
  tabWrap?: (key: string, tab: ReactElement) => ReactNode
  /** Override the tile's TAB drag (session drop language: stack/split/link).
   *  Returns whether it took the drag (see PaneChrome.tabDrag). */
  tabDrag?: (key: string, event: ReactPointerEvent<HTMLElement>, onTap: () => void) => boolean
  /** Wired as the pane's closer (tab Close). */
  close: (key: string) => void
}

/** Build a `watch*` fn: syncs once, then re-syncs on every source/also change.
 *  Module-level state lives in the returned closure, so call it once per app. */
export function paneMirror<T>(cfg: PaneMirror<T>): () => void {
  const registered = new Map<string, { dispose: () => void; title: string }>()

  const paneId = (key: string) => `${cfg.prefix}:${key}`

  const sync = () => {
    const tiles = cfg.source.get()
    const wanted = new Set(tiles.map(cfg.key))

    for (const tile of tiles) {
      const key = cfg.key(tile)
      const title = cfg.title(key)
      const current = registered.get(key)

      // register() replaces same-id in place — safe for live title refreshes.
      if (current && current.title === title) {
        continue
      }

      const dispose = registry.register({
        id: paneId(key),
        area: 'panes',
        title,
        data: {
          tabLead: cfg.tabLead ? () => cfg.tabLead!(key) : undefined,
          tabTitle: cfg.tabTitle ? () => cfg.tabTitle!(key) : undefined,
          dock: {
            before: cfg.before?.(tile),
            pane: cfg.anchor?.(tile) ?? 'workspace',
            pos: cfg.dir?.(tile) ?? 'right'
          },
          minWidth: cfg.minWidth,
          newTab: cfg.newTab?.(key),
          // Every mirrored tile is a full workspace surface docked beside main —
          // and closeable, which is what keeps its tab when it lands in a zone of
          // its own (see strip-visibility.ts).
          placement: 'main',
          tabDrag: cfg.tabDrag
            ? (event: ReactPointerEvent<HTMLElement>, onTap: () => void) => cfg.tabDrag!(key, event, onTap)
            : undefined, // returns boolean (handled) — see PaneChrome.tabDrag
          tabMenuPrefix: cfg.tabMenuPrefix?.(key),
          tabWrap: cfg.tabWrap ? (tab: ReactElement) => cfg.tabWrap!(key, tab) : undefined
        },
        render: () => cfg.render(key)
      })

      registered.set(key, { dispose, title })

      if (!current) {
        registerPaneCloser(paneId(key), () => cfg.close(key))
      }
    }

    for (const [key, entry] of registered) {
      if (!wanted.has(key)) {
        entry.dispose()
        registered.delete(key)
        removeTreePane(paneId(key))
      }
    }

    // Prune tree panes the SHARED tree persisted for a tile we never registered
    // this session and that isn't wanted now — a profile switch reloads with the
    // other profile's tile panes still stacked in. (`registered` is empty after a
    // reload, so the loop above can't catch these.)
    for (const id of treePanesWithPrefix(`${cfg.prefix}:`)) {
      if (!wanted.has(id.slice(cfg.prefix.length + 1))) {
        removeTreePane(id)
      }
    }
  }

  return () => {
    sync()
    cfg.source.listen(sync)
    cfg.also?.forEach(atom => atom.listen(sync))
  }
}
