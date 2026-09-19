// The Hermes IDE window's renderer URL. The pure, Electron-free piece lives
// here so it can be unit-tested (same split as hud-url.ts / session-windows.ts:
// the query-before-hash contract is invisible until it breaks at runtime).
//
// Same contract as every window URL: `?win=ide` and the optional profile /
// connectionId / cwd carries MUST sit in the search string before the '#', or
// HashRouter swallows them as part of the route.
//
// `cwd` seeds the IDE's workspace root from the opening window — a brand-new
// IDE has no session yet, and the workspace it should show is the one the user
// was already working in. Absent means no seed: the IDE opens on its empty
// state and follows whichever IDE session becomes active.

import { pathToFileURL } from 'node:url'

export const IDE_WINDOW_TITLE = 'Hermes IDE'

export interface IdeWindowUrlOptions {
  connectionId?: null | string
  cwd?: null | string
  devServer?: null | string
  profile?: null | string
  rendererIndexPath?: string
}

export function buildIdeWindowUrl({
  connectionId,
  cwd,
  devServer,
  profile,
  rendererIndexPath
}: IdeWindowUrlOptions = {}): string {
  const profileKey = typeof profile === 'string' ? profile.trim() : ''
  const parts = ['win=ide']

  if (profileKey) {
    // An empty connectionId is the registry-local route — same shape as the
    // peer window URL (buildInstanceWindowUrl).
    parts.push(`profile=${encodeURIComponent(profileKey)}`)
    parts.push(`connectionId=${encodeURIComponent(connectionId ?? '')}`)
  }

  const cwdKey = typeof cwd === 'string' ? cwd.trim() : ''

  if (cwdKey) {
    parts.push(`cwd=${encodeURIComponent(cwdKey)}`)
  }

  const query = `?${parts.join('&')}`
  const route = '#/'

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}${route}`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}${query}${route}`
}
