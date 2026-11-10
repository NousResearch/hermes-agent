// HUD mode's renderer URL. The pure, Electron-free piece lives here so it can
// be unit-tested (same split as session-windows.ts, and for the same reason:
// the contract below is invisible until it breaks at runtime).

import { pathToFileURL } from 'node:url'

/**
 * Build the renderer URL for the HUD window.
 *
 * Same query-before-hash contract as `buildSessionWindowUrl`: `?win=hud` and
 * the optional `profile=` MUST sit in the search string before the '#', or
 * HashRouter swallows them as part of the route.
 *
 * The profile is what the HUD renderer adopts at boot. Session ids are scoped
 * per profile, so without it a HUD opened on a non-primary conversation
 * resolves the id against the primary backend, misses, and falls back to that
 * backend's last session (#82285). Absent/blank means no override — ordinary
 * single-profile boots are unchanged.
 */
export function buildHudWindowUrl(
  sessionId: null | string | undefined,
  {
    devServer,
    profile,
    rendererIndexPath
  }: { devServer?: null | string; profile?: null | string; rendererIndexPath?: string } = {}
): string {
  const profileKey = typeof profile === 'string' ? profile.trim() : ''
  const query = `?win=hud${profileKey ? `&profile=${encodeURIComponent(profileKey)}` : ''}`
  const route = sessionId ? `#/${encodeURIComponent(sessionId)}` : '#/'

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}${route}`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}${query}${route}`
}
