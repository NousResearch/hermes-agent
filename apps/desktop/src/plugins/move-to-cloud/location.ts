/**
 * Where Backup & Move lives: Settings ▸ Plugins ▸ Backup & Move. The old `/move-to-cloud` route,
 * which the portal's `hermes://open/move-to-cloud?agent=<name>` link opens, forwards here.
 */

import { pluginSettingsHref } from '@hermes/plugin-sdk'

export const MOVE_PLUGIN_ID = 'move-to-cloud'

/** The route the portal link and older builds open; it only forwards to Settings now. */
export const MOVE_ROUTE = '/move-to-cloud'

// Settings reads `agent` as an agent plugin's schema form, so the portal's agent name travels
// under its own param once it reaches Settings.
const AGENT_PARAM = 'cloudAgent'

const queryOf = (hash: string) => new URLSearchParams(hash.split('?')[1] ?? '')

/** The Settings location of this page, optionally naming the cloud agent to preselect. */
export function moveSettingsHref(agent = ''): string {
  const href = pluginSettingsHref(MOVE_PLUGIN_ID)

  return agent ? `${href}&${new URLSearchParams({ [AGENT_PARAM]: agent })}` : href
}

/** `#/move-to-cloud?agent=<name>` (the portal link, or a saved route) → its Settings location. */
export function settingsHrefForRoute(hash: string): string {
  return moveSettingsHref((queryOf(hash).get('agent') ?? '').trim())
}

/** The cloud agent the portal link named. A hint only: an unknown or missing name leaves the
 *  choice to the user. */
export function agentHint(hash: string): string {
  return (queryOf(hash).get(AGENT_PARAM) ?? '').trim()
}
