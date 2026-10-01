// Kirsin Agent Window's renderer URL. The pure, Electron-free piece lives here
// so it can be unit-tested (same split as hud-url.ts, and for the same reason:
// the contract below is invisible until it breaks at runtime).
//
// The Kirsin window is "a HUD pinned to the `kirsin` profile, with a
// Kirsin-branded shell": a full app renderer that adopts the kirsin backend at
// boot, rendered in a persistent always-on-top floating panel instead of the
// HUD's auto-hiding band.

import { pathToFileURL } from 'node:url'

/**
 * Build the renderer URL for the Kirsin window.
 *
 * Same query-before-hash contract as `buildHudWindowUrl`: `?win=kirsin` and the
 * `profile=` MUST sit in the search string before the '#', or HashRouter
 * swallows them as part of the route.
 *
 * `profile=kirsin` is what the window adopts at boot (see `windowProfileOverride`
 * + the gateway boot). Without it the renderer would adopt the PRIMARY backend
 * and show the wrong agent's conversations. Absent/blank means no override.
 */
export function buildKirsinWindowUrl(
  sessionId: null | string | undefined,
  {
    devServer,
    profile,
    rendererIndexPath
  }: { devServer?: null | string; profile?: null | string; rendererIndexPath?: string } = {}
): string {
  const profileKey = typeof profile === 'string' ? profile.trim() : ''
  const query = `?win=kirsin${profileKey ? `&profile=${encodeURIComponent(profileKey)}` : ''}`
  const route = sessionId ? `#/${encodeURIComponent(sessionId)}` : '#/'

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}${route}`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}${query}${route}`
}
