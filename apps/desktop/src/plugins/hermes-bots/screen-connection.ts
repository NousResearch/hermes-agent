/**
 * Bot Screen — connection plumbing for a bot's Bot Desktop (the headless Xfce
 * screen its computer_use drives on the gateway host).
 *
 * Two legs share one authenticated origin: JSON-RPC (`display.*`) rides the
 * bot's pooled gateway socket through `host.requestProfile`; the RFB stream
 * rides a SIBLING WebSocket to `/api/display/ws`, minted per attach by
 * `display.observe` (single-use, 30 s). noVNC's Websock takes ownership of the
 * socket it is handed, so it can never share the JSON-RPC one — same reason
 * voice playback opens `/api/audio/speak-stream` beside `/api/ws`.
 */

import { host, resolveSiblingWsUrl } from '@hermes/plugin-sdk'
import type { PluginProfileRoute } from '@hermes/plugin-sdk'

import { botConnectionRoute } from './routing'
import type { RosterRow } from './types'

export interface DisplayLease {
  holder: 'agent' | 'human'
  viewer_id: null | string
  since: number
  reason: string
  pending_handoff: null | string
}

export interface DisplayStatus {
  profile: string
  profile_key: string
  supported: boolean
  installed: boolean
  missing: string[]
  running: boolean
  pid: null | number
  display: null | string
  socket: null | string
  geometry: string
  install_command: null | string
  lease: DisplayLease
}

export interface DisplayObserveResult extends DisplayStatus {
  ticket: string
  path: string
  viewer_id: string
}

/** Stable per-window viewer identity: the lease names who holds control, and a
 *  reload must NOT silently inherit a stale holder's authority. */
export const VIEWER_ID = `desktop-${Math.random().toString(36).slice(2, 10)}`

/** Bare-profile fallback so a v1 local bot (no registry route) still resolves. */
export function botScreenRoute(bot: RosterRow): PluginProfileRoute | string {
  return botConnectionRoute(bot) ?? bot.name
}

export function displayRequest<T>(bot: RosterRow, method: string, params: Record<string, unknown> = {}): Promise<T> {
  return host.requestProfile<T>(botScreenRoute(bot), method, params)
}

/**
 * Resolve the RFB WebSocket URL for `bot`: the bot's gateway `/api/ws` origin
 * (fresh credential for OAuth remotes) with the path swapped for the display
 * bridge and the single-use display ticket attached.
 */
export async function resolveScreenWsUrl(bot: RosterRow, ticket: string): Promise<string> {
  const route = botConnectionRoute(bot)

  // The /api/ws credential authenticated the RPC that minted the display ticket;
  // the bridge authenticates on the ticket alone, so the gateway credential is
  // dropped rather than spending a second one-shot ticket.
  const url = new URL(
    await resolveSiblingWsUrl({ connectionId: route?.connectionId ?? null, profile: route?.profile ?? bot.name }, '/api/display/ws', {
      stripGatewayCredential: true
    })
  )

  url.searchParams.set('display_ticket', ticket)

  return url.toString()
}
