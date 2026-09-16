// Popped-out Skills Hub window. Same query-before-hash contract as
// browser-windows / session-windows: `?win=skills-hub` MUST sit in the search
// string before the '#', or HashRouter swallows it as part of the route.

import { pathToFileURL } from 'node:url'

// The hub window opens big — the whole point is that the embedded picker is
// capped at 75% of the app window and rendered scaled down (unreadable at a
// glance). This window shows the catalog at 100% with room for the card grid.
export const HUB_WINDOW_WIDTH = 1280
export const HUB_WINDOW_HEIGHT = 860
export const HUB_WINDOW_MIN_WIDTH = 640
export const HUB_WINDOW_MIN_HEIGHT = 480

/**
 * Renderer URL for the popped-out Skills Hub. `win=skills-hub` selects the
 * dedicated full-window shell (the same hub iframe the Capabilities pane
 * embeds, minus the 0.75 scale and the height cap). The window is a singleton:
 * the main process focuses an existing window instead of spawning a second one.
 */
export function buildHubWindowUrl({
  devServer,
  rendererIndexPath
}: { devServer?: null | string; rendererIndexPath?: string } = {}): string {
  const query = '?win=skills-hub'

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}#/`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}${query}#/`
}
