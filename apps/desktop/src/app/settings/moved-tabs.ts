import { CAPABILITIES_ROUTE } from '../routes'

const MOVED_TO_CAPABILITIES: Record<string, { param: string; tab: string }> = {
  mcp: { param: 'server', tab: 'connectors' },
  plugins: { param: 'plugin', tab: 'plugins' }
}

export function movedSettingsTabRedirect(search: string): null | string {
  const params = new URLSearchParams(search)

  if (
    params.get('tab') === 'config:advanced' &&
    (params.get('page') === 'delegation' || params.get('field')?.startsWith('delegation.'))
  ) {
    params.set('tab', 'config:model')
    params.set('page', 'delegation')

    return `/settings?${params}`
  }

  const moved = MOVED_TO_CAPABILITIES[params.get('tab') ?? '']

  if (moved === undefined) {
    return null
  }

  const row = params.get(moved.param)
  const suffix = row ? `&${moved.param}=${encodeURIComponent(row)}` : ''

  return `${CAPABILITIES_ROUTE}?tab=${moved.tab}${suffix}`
}
