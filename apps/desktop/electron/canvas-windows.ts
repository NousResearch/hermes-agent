// Popped-out canvas windows: one per canvas provider (pen today), hosting the
// same provider pane the docked tile renders, in its own OS window. Same
// query-before-hash contract as browser-windows: `?win=canvas` MUST sit in the
// search string before the '#', or HashRouter swallows it as part of the route.

import { pathToFileURL } from 'node:url'

export const CANVAS_WINDOW_WIDTH = 1200
export const CANVAS_WINDOW_HEIGHT = 800
export const CANVAS_WINDOW_MIN_WIDTH = 640
export const CANVAS_WINDOW_MIN_HEIGHT = 480

/** The docked tab, verbatim — the window is just another host for it. */
export interface CanvasWindowTab {
  provider: string
  docId: string
  title: string
  url: string
}

export function isCanvasWindowTab(value: unknown): value is CanvasWindowTab {
  if (!value || typeof value !== 'object') {
    return false
  }

  const tab = value as Record<string, unknown>

  return (
    typeof tab.provider === 'string' &&
    tab.provider.trim() !== '' &&
    typeof tab.docId === 'string' &&
    typeof tab.title === 'string' &&
    typeof tab.url === 'string'
  )
}

/**
 * Renderer URL for a popped-out canvas. The tab rides the query so the new
 * renderer can seat it without asking anyone — it has no layout tree and no
 * `$canvasTabs` of its own until it reads this.
 */
export function buildCanvasWindowUrl(
  tab: CanvasWindowTab,
  { devServer, rendererIndexPath }: { devServer?: null | string; rendererIndexPath?: string } = {}
): string {
  const params = new URLSearchParams({
    win: 'canvas',
    provider: tab.provider,
    doc: tab.docId,
    title: tab.title,
    url: tab.url
  })

  const query = `?${params.toString()}`

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}#/`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}${query}#/`
}
