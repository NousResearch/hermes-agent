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
  {
    devServer,
    rendererIndexPath,
    scope
  }: { devServer?: null | string; rendererIndexPath?: string; scope?: 'ide' } = {}
): string {
  const tab = typeof tabId === 'string' ? tabId.trim() : ''
  // `scope=ide` keeps the pop-out reading the same tab store as the window it
  // came from — the IDE keeps its browser tabs under its own key, so a pop-out
  // without the scope finds no tab and renders blank.
  const scopeParam = scope ? `&scope=${encodeURIComponent(scope)}` : ''
  const query = `?win=browser${scopeParam}${tab ? `&tab=${encodeURIComponent(tab)}` : ''}`

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}#/`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}${query}#/`
}
