// Popped-out in-app Browser windows. Same query-before-hash contract as
// session-windows / hud-url: `?win=browser` MUST sit in the search string
// before the '#', or HashRouter swallows it as part of the route.

import { pathToFileURL } from 'node:url'

export const BROWSER_WINDOW_WIDTH = 960
export const BROWSER_WINDOW_HEIGHT = 720
export const BROWSER_WINDOW_MIN_WIDTH = 480
export const BROWSER_WINDOW_MIN_HEIGHT = 400

/**
 * Renderer URL for a popped-out Browser. `tab` is the `$previewTabs` id the
 * window should show — the tab stays in storage so closing the window can
 * dock it again. Absent/blank tab is still a valid Browser window (blank page).
 */
export function buildBrowserWindowUrl(
  tabId: null | string | undefined,
  { devServer, rendererIndexPath }: { devServer?: null | string; rendererIndexPath?: string } = {}
): string {
  const tab = typeof tabId === 'string' ? tabId.trim() : ''
  const query = `?win=browser${tab ? `&tab=${encodeURIComponent(tab)}` : ''}`

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}#/`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}${query}#/`
}
