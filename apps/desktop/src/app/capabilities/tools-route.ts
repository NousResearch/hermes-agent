import { CAPABILITIES_ROUTE, SETTINGS_ROUTE } from '@/app/routes'

/** Where "Tools" is: classic has a Tools tab on Capabilities; the redesign has none, because an
 *  agent's tools are switched in that agent's settings. */
export function toolsRoute(portal: boolean, profile: string): string {
  return portal
    ? `${SETTINGS_ROUTE}?${new URLSearchParams({ agentProfile: profile, panel: 'capabilities', tab: 'toolsets' })}`
    : `${CAPABILITIES_ROUTE}?tab=toolsets`
}
