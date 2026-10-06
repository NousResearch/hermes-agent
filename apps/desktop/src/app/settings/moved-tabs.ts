import { CAPABILITIES_ROUTE } from '../routes'

// `plugins` used to redirect here too; it is a live Settings view again
// (Settings ▸ Plugins = each plugin's own settings pages).
const MOVED_TO_CAPABILITIES: Record<string, { param: string; tab: string }> = {
  mcp: { param: 'server', tab: 'connectors' }
}

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
