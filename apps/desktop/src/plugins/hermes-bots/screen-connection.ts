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
import type { PluginProfileRoute, RpcEvent } from '@hermes/plugin-sdk'

import { botConnectionRoute } from './routing'
import type { RosterRow } from './types'

export interface DisplayLease {
  holder: 'agent' | 'human'
  /** Raw holder id — only older backends still broadcast it; newer ones send `viewer_hash`. */
  viewer_id: null | string
  /** First 12 hex of sha256(viewer_id): names the holder without leaking a usable id. */
  viewer_hash?: null | string
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

export interface DisplayThumbnail {
  data_url: string | null
  /** Set while a human holds the screen: the frame is withheld, not missing. */
  suppressed?: 'human_has_control' | null
}

export interface DisplayObserveResult extends DisplayStatus {
  ticket: string
  path: string
  /** Server-minted per attach: the only id the lease will ever be compared against. */
  viewer_id: string
}

/** This window's identity for one attach: the minted id plus its lease-payload hash. */
export interface ScreenViewer {
  id: string
  hash: string
}

const VIEWER_HASH_HEX = 12

/** `viewer_hash` as the lease broadcasts it: first 12 hex of sha256(viewer_id). */
export async function viewerHash(viewerId: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(viewerId))

  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0'))
    .join('')
    .slice(0, VIEWER_HASH_HEX)
}

/** Does `viewer` (this window's attach) hold `lease`? Newer backends name the
 *  holder by hash only; a payload still carrying the raw id is compared raw. */
export function leaseHeldBy(lease: DisplayLease | null | undefined, viewer: ScreenViewer | null | undefined): boolean {
  if (!lease || !viewer || lease.holder !== 'human') {
    return false
  }

  return lease.viewer_id != null ? lease.viewer_id === viewer.id : lease.viewer_hash === viewer.hash
}

/** JSON-RPC method-not-found: the bot's Hermes predates the `display.*` surface. */
export function isDisplayUnavailable(error: unknown): boolean {
  const record = typeof error === 'object' && error !== null ? (error as { code?: unknown; message?: unknown }) : null

  if (record?.code === -32601) {
    return true
  }

  const message = typeof record?.message === 'string' ? record.message.toLowerCase() : ''

  return message.includes('method not found') || message.includes('method-not-found')
}

/** Bare-profile fallback so a v1 local bot (no registry route) still resolves. */
export function botScreenRoute(bot: RosterRow): PluginProfileRoute | string {
  return botConnectionRoute(bot) ?? bot.name
}

/**
 * Does a `display.*` event belong to `bot`'s screen? Two hosts can share the same
 * `~/.hermes` path, so the profile key alone is ambiguous: the event must also have
 * arrived on the bot's registry connection (local/legacy events carry no tag).
 */
export function isEventForBotScreen(bot: RosterRow, event: RpcEvent, profileKey: null | string | undefined): boolean {
  const payload = event.payload as { profile_key?: string } | undefined

  if (!profileKey || payload?.profile_key !== profileKey) {
    return false
  }

  const expected = botConnectionRoute(bot)?.connectionId ?? null
  const actual = event.connectionId ?? null

  return expected === actual || (expected === 'local' && actual === null)
}

export function displayRequest<T>(bot: RosterRow, method: string, params: Record<string, unknown> = {}): Promise<T> {
  return host.requestProfile<T>(botScreenRoute(bot), method, params)
}

/**
 * Hold the bot's pooled gateway socket open across a `display.*` sequence. The
 * SDK disposes an inactive registry-routed socket once its request count hits
 * zero, so without this the `display.install.*` / `display.lease` events that
 * follow the request never arrive. Feature-detected: older hosts (and local
 * routes, which never close) get a no-op release.
 */
export async function retainBotScreen(bot: RosterRow): Promise<() => void> {
  const noop = () => undefined

  if (typeof host.retainProfile !== 'function') {
    return noop
  }

  try {
    const release = await host.retainProfile(botScreenRoute(bot))

    return typeof release === 'function' ? release : noop
  } catch {
    return noop
  }
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
