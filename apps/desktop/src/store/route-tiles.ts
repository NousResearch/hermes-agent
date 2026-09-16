import { atom } from 'nanostores'

import { routePathname } from '@/app/routes'
import { revealTreePane } from '@/components/pane-shell/tree/store'
import { readJson, writeJson } from '@/lib/storage'

import type { TileDock } from './session-states'

/**
 * Route (page) tiles — a full-page view (Capabilities / Messaging / Artifacts,
 * or any plugin route) rendered as a layout-tree pane BESIDE the main thread,
 * the page analog of session tiles. Persisted by path so they re-open on boot.
 */
export interface RouteTile {
  /** The route path this tile renders, e.g. `/skills`. */
  path: string
  /** Dock on adoption (default right; `center` = a tab in the workspace strip). */
  dir?: TileDock
}

const TILES_KEY = 'hermes.desktop.routeTiles.v1'

function loadTiles(): RouteTile[] {
  const parsed = readJson<unknown>(TILES_KEY)

  return Array.isArray(parsed)
    ? parsed.reduce<RouteTile[]>((tiles, t) => {
        if (!t || typeof (t as RouteTile).path !== 'string') {
          return tiles
        }

        const path = routePathname((t as RouteTile).path)

        return tiles.some(tile => tile.path === path) ? tiles : [...tiles, { dir: (t as RouteTile).dir, path }]
      }, [])
    : []
}

export const $routeTiles = atom<RouteTile[]>(loadTiles())

function saveTiles(tiles: RouteTile[]) {
  $routeTiles.set(tiles)
  writeJson(TILES_KEY, tiles.length === 0 ? null : tiles)
}

/** Open (or front) a page tile for a route, docked on `dir` (default right).
 *  Idempotent — an already-open tile keeps its original edge. */
export function openRouteTile(path: string, dir: TileDock = 'right') {
  const canonicalPath = routePathname(path)
  const tiles = $routeTiles.get()

  if (!tiles.some(t => t.path === canonicalPath)) {
    saveTiles([...tiles, { dir, path: canonicalPath }])
  }

  revealTreePane(`route-tile:${canonicalPath}`)
}

export function closeRouteTile(path: string) {
  const canonicalPath = routePathname(path)

  saveTiles($routeTiles.get().filter(t => t.path !== canonicalPath))
}
