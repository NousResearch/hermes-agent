/**
 * Cross-connection routing: resolving a roster row's owning connection, the
 * RPC door that rides that owner, and the alias identity that keeps a
 * configured Desktop alias attached to the backend row it activates into.
 *
 * The bottom of the Bot Mode module graph — it imports no sibling module, so
 * everything else can dispatch through it without a cycle.
 */

import { host } from '@hermes/plugin-sdk'

import type { BotMeta, ProfileRoute, RosterRow } from './types'

export function botRouteKey(route: ProfileRoute): string {
  return `${route.connectionId}::${route.profile}`
}

// ── cross-connection routing ─────────────────────────────────────────────────
// A bot from another registered connection (remoteSource rows) is reached
// through host.requestProfile with a route descriptor; local bots keep the
// active-gateway door. Feature-detected: older desktops without
// requestProfile simply have no remote routes (callers fall back / disable).

/** Outcome of resolving a row's owning connection. `profile` is only carried
 *  on the `owner_removed` branch, where the strict wrapper needs it for the
 *  error message. */
interface BotRouteResolution {
  profile?: string
  route: ProfileRoute | null
  status: 'not_scoped' | 'owner_removed' | 'resolved'
}

/** Non-throwing resolver behind botConnectionRoute(). Returns a typed status
 *  instead of throwing, so passive callers (display/meta lookups) can branch
 *  on `resolved | owner_removed | not_scoped` rather than catching whatever
 *  exception the strict wrapper below happens to throw. */
export function resolveBotConnectionRoute(bot: Partial<RosterRow> | null | undefined): BotRouteResolution {
  if (!bot?.sourceScoped && !bot?.remoteSource) {
    return {
      status: 'not_scoped',
      route: null
    }
  }
  const candidate = bot.route || {
    connectionId: bot.connectionId,
    mode: bot.connectionKind === 'local' ? 'local' : 'remote',
    profile: bot.name,
    targetProfile: bot.targetProfile || bot.name
  }
  const connectionId = String(candidate?.connectionId || '').trim()
  const profile = String(candidate?.profile || bot?.name || '').trim() || 'default'
  const targetProfile = String(candidate?.targetProfile || profile).trim() || profile
  if (!connectionId) {
    return {
      status: 'owner_removed',
      route: null,
      profile
    }
  }
  return {
    status: 'resolved',
    route: Object.freeze({
      connectionId,
      mode: candidate.mode === 'local' || connectionId === 'local' ? 'local' : 'remote',
      profile,
      targetProfile
    })
  }
}

/** Immutable owner descriptor for every source-scoped row. The active
 *  gateway is presentation state and is never consulted here. Strict: throws
 *  when the owning connection is gone -- correct for real dispatch
 *  (requestForBot, session creation, etc.), covered by
 *  remote-routing-races.test.mjs. Passive lookups (rendering, meta) must call
 *  resolveBotConnectionRoute() directly instead of catching this throw. */
export function botConnectionRoute(bot: Partial<RosterRow> | null | undefined): null | ProfileRoute {
  const resolved = resolveBotConnectionRoute(bot)
  if (resolved.status === 'owner_removed') {
    throw new Error(`Bot ${resolved.profile} has no connection owner`)
  }
  return resolved.route
}
export function botWorkspaceOwnerKey(bot: RosterRow) {
  // Render-reachable (sidebar sync, context menus): an orphaned row must
  // yield a stable degraded key, never the dispatch throw.
  const route = resolveBotConnectionRoute(bot).route
  return `bot:${route ? botRouteKey(route) : String(bot?.name || 'default')}`
}
export function setBotsWorkspaceOwner(
  ownerKey: string,
  bot: null | RosterRow = null,
  blockedMessage = 'Select a Bot or group first.'
) {
  // Render-reachable (sidebar listener fires on visibility flips). An
  // orphaned row degrades to the blocked target instead of throwing.
  const route = bot ? resolveBotConnectionRoute(bot).route : null
  const target: { kind: 'blocked'; message: string } | { kind: 'route'; route: ProfileRoute } = route
    ? {
        kind: 'route',
        route
      }
    : {
        kind: 'blocked',
        message: blockedMessage
      }
  host.setWorkspaceScope?.('bots', ownerKey, target)
}
export function backendTargetProfile(route: null | ProfileRoute | undefined, fallbackProfile = 'default') {
  if (!route) {
    return fallbackProfile
  }
  return route.targetProfile || route.profile
}
function rewriteCliProfileOperands(argv: string[], logical: string, target: string) {
  const next = [...argv]
  for (let index = 0; index < next.length; index += 1) {
    if (next[index] === '--profile' && next[index + 1] === logical) {
      next[index + 1] = target
      index += 1
    } else if (next[index] === `--profile=${logical}`) {
      next[index] = `--profile=${target}`
    }
  }
  const profileCommand = next.indexOf('profile')
  const operand = profileCommand >= 0 ? profileCommand + 2 : -1
  if (operand < next.length && next[operand] === logical) {
    next[operand] = target
  }
  return next
}
function scopedBotParams(route: ProfileRoute, method: string, params: Record<string, unknown>) {
  const logical = route.profile
  const target = backendTargetProfile(route, logical)
  let next = params
  if (Object.prototype.hasOwnProperty.call(params, 'profile')) {
    next = {
      ...next,
      profile: target
    }
  }
  if (method.startsWith('profiles.') && method !== 'profiles.create' && params.name === logical) {
    next = {
      ...next,
      name: target
    }
  }
  if (params.clone_from === logical) {
    next = {
      ...next,
      clone_from: target
    }
  }
  if (method === 'cli.exec' && Array.isArray(params.argv)) {
    next = {
      ...next,
      argv: rewriteCliProfileOperands(params.argv, logical, target)
    }
  }
  return next
}
export function botBackendProfileScope(route: null | ProfileRoute | undefined, fallbackProfile = 'default') {
  if (!route) {
    return fallbackProfile
  }
  return {
    connectionId: route.connectionId,
    profile: backendTargetProfile(route, fallbackProfile)
  }
}

/** Gateway RPC on the bot's OWN source. Source-scoped rows always use the
 * explicit descriptor, including a registered local source. */
export async function requestForBot<T = unknown>(
  bot: Partial<RosterRow> | null | undefined,
  method: string,
  params: Record<string, unknown> = {}
): Promise<T> {
  const route = botConnectionRoute(bot)
  if (route) {
    if (typeof host.requestProfile !== 'function') {
      throw new Error(`Cannot route ${method} for ${route.connectionId}::${route.profile}`)
    }
    try {
      return await host.requestProfile(route, method, scopedBotParams(route, method, params))
    } catch (error) {
      // React 19 formats query errors with `(error.name || '').trim()`. IPC /
      // JSON-RPC rejections are often plain objects whose `name` is a number,
      // which crashes the Routines pane and hides the original failure (#94471).
      throw asRpcError(error, `Gateway request ${method} failed`)
    }
  }
  try {
    return await host.request(method, params)
  } catch (error) {
    throw asRpcError(error, `Gateway request ${method} failed`)
  }
}

/** A rejection duck-typed across realms: an Error-like whose fields are only
 *  conventionally typed, so every read stays `unknown` until it is checked. */
export interface RpcErrorLike {
  message?: unknown
  name?: unknown
  stack?: unknown
}

/** Coerce an IPC/JSON-RPC rejection into an Error with a string `name`.
 *
 *  React Query stores whatever the queryFn throws. React 19 then formats it
 *  with `(e.name || '').trim()`, which throws TypeError when `name` is a
 *  number (JSON-RPC codes) or another non-string — the Routines pane crash
 *  in #94471. Real Error instances are returned as-is when already safe.
 */
function asRpcError(value: unknown, fallback: string): unknown {
  // Duck-type across realms (plugin tests run the source in `vm`, and IPC
  // can deliver Error-like objects whose prototype is not this realm's
  // Error). React 19 only needs a string `name`. Never mutate the rejection:
  // frozen/sealed objects make `name = 'Error'` a silent no-op in sloppy
  // mode, so a non-string name always becomes a fresh Error with cause.
  const isObject = value != null && typeof value === 'object'
  const name = isObject ? (value as RpcErrorLike).name : undefined
  const message = isObject ? (value as RpcErrorLike).message : undefined
  const hasStringName = typeof name === 'string'
  const hasStringMessage = typeof message === 'string'
  const hasStack = isObject && typeof (value as RpcErrorLike).stack === 'string'
  if (isObject && hasStringName && (hasStack || hasStringMessage)) {
    return value
  }
  if (isObject) {
    const text = hasStringMessage && String(message).trim() ? String(message) : fallback
    const error = new Error(text)
    error.cause = value
    return error
  }
  return new Error(value == null || value === '' ? fallback : String(value))
}

// ── alias identity for connection rows (#89131) ─────────────────────────────
// A Desktop per-profile alias (profile `moxie` with a Cloud/URL/SSH override)
// routes to a remote backend's root profile: its route reads
// { connectionId: C, profile: 'moxie', targetProfile: 'default' }. Once that
// backend answers the roster itself, the row's identity is (C, 'default') —
// a DIFFERENT key than the alias meta (C::moxie / 'moxie') — so the friendly
// name fell off after source/session activation: the row regressed to the
// raw Cloud hostname, or to generic 'Hermes' in Cloud-only mode.
//
// aliasRouteIndex bridges the backend row identity back to its configured
// alias. It is keyed by (connectionId, targetProfile), so two same-named
// `default` rows on different connections can never share a title, and it
// fails closed when two aliases claim the same backend row (mirroring the
// fail-closed route resolution). This is the one sanctioned exception to
// "remote rows never borrow local meta": the alias IS the local identity of
// exactly this connection row, proven by the configured route — never by a
// bare name match.
interface AliasIdentity {
  /** Both the source-qualified v2 key and the bare v1 name key. */
  metaKeys: string[]
  name: string
}

/** Keyed `${connectionId}::${targetProfile}`; null marks an ambiguous claim. */
let aliasRouteIndex = new Map<string, AliasIdentity | null>()

/** Rebuild the alias index from the credential-free route inventory. Only
 *  genuine aliases (route.profile !== route.targetProfile) participate. */
export function indexAliasRoutes(routes: ProfileRoute[]) {
  const next = new Map<string, AliasIdentity | null>()
  for (const route of Array.isArray(routes) ? routes : []) {
    const connectionId = String(route?.connectionId || '').trim()
    const profile = String(route?.profile || '').trim()
    const target = String(route?.targetProfile || '').trim()
    if (!connectionId || !profile || !target || profile === target) {
      continue
    }
    const key = `${connectionId}::${target}`

    // Two aliases pointing at the same backend row are ambiguous — neither
    // may claim the identity.
    next.set(
      key,
      next.has(key)
        ? null
        : {
            name: profile,
            // Alias meta can live under the source-qualified v2 key or the bare
            // v1 name key (aliases predate the v2 migration on mixed setups).
            metaKeys: [`${connectionId}::${profile}`, profile]
          }
    )
  }
  aliasRouteIndex = next
}

/** The configured alias identity claiming this roster row, or null. Matches
 *  strictly by (connectionId, backend target profile); the alias row itself
 *  keeps resolving its own meta directly. */
export function aliasIdentityFor(bot: Partial<RosterRow> | null | undefined): AliasIdentity | null {
  if (!aliasRouteIndex.size) {
    return null
  }
  const connectionId = String(
    bot?.connectionId ||
      bot?.route?.connectionId ||
      // Unannotated rich rows (no host.agents on this build) still belong to
      // the ACTIVE gateway — Cloud-only mode must resolve the alias too.
      (!bot?.remoteSource && !bot?.sourceScoped ? host.state.connectionId?.get?.() || '' : '')
  ).trim()
  if (!connectionId) {
    return null
  }
  const target = String(bot?.targetProfile || bot?.route?.targetProfile || bot?.name || '').trim() || 'default'
  const entry = aliasRouteIndex.get(`${connectionId}::${target}`) || null
  return entry && entry.name !== String(bot?.name || '').trim() ? entry : null
}

// Bot metadata is scoped to the active gateway until the server exposes a
// union of rich profile rows. Never paint that metadata onto a thin row from
// another source: two `default` agents must not borrow each other's title,
// pin, avatar, group, unread state, or canonical-chat pointer. The ONE
// exception is a configured alias route claiming the row — see
// aliasRouteIndex above — which is connection-exact, never name-based.
export function botRosterMeta(bot: RosterRow, metaByName: Record<string, BotMeta>) {
  if (bot?.sourceScoped || bot?.remoteSource) {
    // Passive meta lookup: branch on the typed status instead of catching
    // botConnectionRoute's throw, so an owner_removed row (e.g. a stale
    // persisted group roster after its connection was deleted) reads as "no
    // route" without masking an unrelated failure under the same catch.
    const resolved = resolveBotConnectionRoute(bot)
    const route = resolved.status === 'resolved' ? resolved.route : null
    const direct = route ? metaByName?.[botRouteKey(route)] : null
    if (direct) {
      return direct
    }
    const alias = aliasIdentityFor(bot)
    if (alias) {
      for (const key of alias.metaKeys) {
        if (metaByName?.[key]) {
          return metaByName[key]
        }
      }
    }
    return direct
  }
  const own = metaByName?.[bot?.name]
  if (own) {
    return own
  }
  const alias = aliasIdentityFor(bot)
  if (alias) {
    for (const key of alias.metaKeys) {
      if (metaByName?.[key]) {
        return metaByName[key]
      }
    }
  }
  return own
}
