import { CAPABILITIES_ROUTE } from '../routes'

// Settings tabs that now live in Capabilities → the Capabilities tab that
// answers for them and the row-selector param each carries. A server on this
// Mac is one card on the Connectors tab now, so `/settings?tab=mcp&server=x`
// lands on `?tab=connectors&server=x` and opens that card. Old bookmarks and
// palette links keep resolving to the same row on the new page.
const MOVED_TO_CAPABILITIES: Record<string, { param: string; tab: string }> = {
  mcp: { param: 'server', tab: 'connectors' },
  plugins: { param: 'plugin', tab: 'plugins' }
}

/** The Capabilities URL an old `/settings?tab=<moved>` query should land on,
 *  or null when the tab still belongs to Settings. */
export function movedSettingsTabRedirect(search: string): null | string {
  const params = new URLSearchParams(search)
  const moved = MOVED_TO_CAPABILITIES[params.get('tab') ?? '']

  if (moved === undefined) {
    return null
  }

  const row = params.get(moved.param)
  const suffix = row ? `&${moved.param}=${encodeURIComponent(row)}` : ''

  return `${CAPABILITIES_ROUTE}?tab=${moved.tab}${suffix}`
}
