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
import type { DisplayLease, PluginProfileRoute, RpcEvent } from '@hermes/plugin-sdk'

import { resolveBotConnectionRoute } from './routing'
import type { RosterRow } from './types'

// Wire shapes come from the generated contract (Python is the source: `tui_gateway/contracts/display.py`).
export type {
  DisplayLease,
  DisplayObserveResult,
  DisplayStatus,
  DisplayThumbnailResult as DisplayThumbnail
} from '@hermes/plugin-sdk'

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

/** Does `viewer` (this window's attach) hold `lease`? The gateway names the holder by hash only
 *  (the raw id is a capability and never travels). */
export function leaseHeldBy(lease: DisplayLease | null | undefined, viewer: ScreenViewer | null | undefined): boolean {
  if (!lease || !viewer || lease.holder !== 'human') {
    return false
  }

  return lease.viewer_hash === viewer.hash
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

/** The bot's backend is a Portal-managed runtime (Hermes Cloud): its Hermes is updated by the
 *  platform, never by the user, so "update the bot's Hermes" is not an instruction the user
 *  can follow. A `display.*` method-not-found from a managed release simply means Screen has
 *  not reached that release yet (#120852). */
export function isManagedBackend(bot: RosterRow): boolean {
  return bot.connectionKind === 'cloud'
}

/**
 * Where `display.*` for `bot` goes. Passive resolution (the typed resolver, never
 * the throwing dispatch wrapper): a source-scoped row whose connection was deleted
 * is a settled "no screen" (null), and a v1 local bot with no registry route falls
 * back to its bare profile name.
 */
export function botScreenRoute(bot: RosterRow): PluginProfileRoute | string | null {
  const resolved = resolveBotConnectionRoute(bot)

  if (resolved.status === 'resolved') {
    return resolved.route
  }

  return resolved.status === 'not_scoped' ? bot.name : null
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

  // Only now resolve the route: this runs for every display.* event, and an orphaned row (its
  // connection deleted) owns no screen rather than being an error to throw from a listener.
  const resolved = resolveBotConnectionRoute(bot)

  if (resolved.status === 'owner_removed') {
    return false
  }

  const expected = resolved.route?.connectionId ?? null
  const actual = event.connectionId ?? null

  return expected === actual || (expected === 'local' && actual === null)
}

export function displayRequest<T>(bot: RosterRow, method: string, params: Record<string, unknown> = {}): Promise<T> {
  const route = botScreenRoute(bot)

  if (route === null) {
    return Promise.reject(new Error(`Bot ${bot.name} has no connection owner`))
  }

  return host.requestProfile<T>(route, method, params)
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

  const route = botScreenRoute(bot)

  if (typeof host.retainProfile !== 'function' || route === null) {
    return noop
  }

  try {
    const release = await host.retainProfile(route)

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
  const resolved = resolveBotConnectionRoute(bot)

  if (resolved.status === 'owner_removed') {
    throw new Error(`Bot ${bot.name} has no connection owner`)
  }

  const route = resolved.route

  // The /api/ws credential authenticated the RPC that minted the display ticket;
  // the bridge authenticates on the ticket alone, so the gateway credential is
  // dropped rather than spending a second one-shot ticket.
  const url = new URL(
    await resolveSiblingWsUrl(
      { connectionId: route?.connectionId ?? null, profile: route?.profile ?? bot.name },
      '/api/display/ws',
      {
        stripGatewayCredential: true
      }
    )
  )

  url.searchParams.set('display_ticket', ticket)

  return url.toString()
}
