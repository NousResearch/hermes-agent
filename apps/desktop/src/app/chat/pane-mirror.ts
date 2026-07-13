/**
 * Mirror a reactive list of "tiles" into layout-tree pane contributions:
 * register a pane per tile, refresh its title in place, and dispose panes whose
 * tile is gone. This is the shared bookkeeping — a keyed registry, a wanted-set
 * diff, a one-time pane closer — behind BOTH session tiles and route (page)
 * tiles; each supplies only what differs (key, title, render, close, edge).
 */

import type { ReadableAtom } from 'nanostores'
import type { ReactElement, ReactNode } from 'react'

import { registerPaneCloser, removeTreePane, treePanesWithPrefix } from '@/components/pane-shell/tree/store'
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
  render: (key: string) => ReactNode
  /** Wrap the tile's TAB (domain context menu — session verbs). */
  tabWrap?: (key: string, tab: ReactElement) => ReactNode
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
          dock: { before: cfg.before?.(tile), pane: cfg.anchor?.(tile) ?? 'workspace', pos: cfg.dir?.(tile) ?? 'right' },
          minWidth: cfg.minWidth,
          placement: 'main',
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
