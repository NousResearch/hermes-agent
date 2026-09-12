import { SKILLS_ROUTE } from '../routes'

// Settings tabs that now live in Capabilities → the row-selector param each
// carries (`?server=` for MCP, `?plugin=` for Plugins). Old bookmarks and
// palette links keep resolving to the same row on the new page.
const MOVED_TO_CAPABILITIES: Record<string, string> = { mcp: 'server', plugins: 'plugin' }

/** The Capabilities URL an old `/settings?tab=<moved>` query should land on,
 *  or null when the tab still belongs to Settings. */
export function movedSettingsTabRedirect(search: string): null | string {
  const params = new URLSearchParams(search)
  const tab = params.get('tab')
  const rowParam = tab ? MOVED_TO_CAPABILITIES[tab] : undefined

  if (!tab || rowParam === undefined) {
    return null
  }

  const row = params.get(rowParam)
  const suffix = row ? `&${rowParam}=${encodeURIComponent(row)}` : ''

  return `${SKILLS_ROUTE}?tab=${tab}${suffix}`
}
