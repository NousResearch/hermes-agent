/**
 * Bot Mode's domain layer: the roster query, bot metadata storage, the
 * identity keys everything else is filed under, and needs-attention state.
 *
 * Framework-light — useRoster is the one React binding — and it depends only
 * on routing.ts, for the owner descriptors it dispatches through.
 */

import { atom, host, queryClient, useQuery, useValue } from '@hermes/plugin-sdk'

import { displayName } from './labels'
import {
  aliasIdentityFor,
  beginAliasRouteIndex,
  botConnectionRoute,
  botRosterMeta,
  botRouteKey,
  botWorkspaceOwnerKey,
  indexAliasRoutes,
  requestForBot,
  setBotsWorkspaceOwner
} from './routing'
import { getPluginCtx, ID } from './shared'
import type {
  AttentionClass,
  BotMeta,
  CanonicalSession,
  GatewaySource,
  GroupMember,
  ProfileRoute,
  RosterRow,
  SessionPreview
} from './types'

export const ROSTER_KEY = [ID, 'roster']
// Bounded retries. `retry: true` keeps React Query in isLoading until the
// first success, so a stalled profiles.list (live state.db write lock, SSH
// flap) leaves the Bots sidebar on a spinner with no error card. The 5s
// refetchInterval and the gateway-open effect already recover drops.
const ROSTER_QUERY_RETRY = 2

export const BOT_META_V1_KEY = 'bot-meta'
const BOT_META_V2_KEY = 'bot-meta-v2'
const BOT_META_MIGRATION_KEY = 'bot-meta-v2-migrated'
export let botMetaV2Active = false
let botMetaV2Commit = Promise.resolve()
const migratedLocalRoutes = new Map<string, ProfileRoute>()

/** Live roster snapshot for imperative handlers (context menus). */
export const $lastRoster = atom<RosterRow[]>([])

// ── needs-attention badge (#93091 item 3) ───────────────────────────────────
// Attention-worthy failure classes — matches the #93091 item-1 reason-code
// enum (shipped separately). Until reason codes flow end-to-end,
// attentionReasonFromError ALSO classifies raw error text as a fallback so
// the badge works against current gateway error strings.
const BOT_ATTENTION_CLASSES: ReadonlySet<string> = new Set<AttentionClass>([
  'agent_blocked',
  'provider_auth_or_access',
  'provider_quota_limit',
  'missing_config'
])

/** One-line user hint per attention class (roster badge tooltip). */
export const BOT_ATTENTION_HINTS: Record<string, string> = {
  provider_auth_or_access: 'Sign in again for this profile',
  provider_quota_limit: 'Quota or balance exhausted',
  missing_config: 'Provider not configured — run hermes model',
  agent_blocked: 'Bot is blocked — see its last message'
}

/** Map an error (a #93091 reason code or raw error text) to an attention
 *  class, or null when the failure is transient (rate limit, server error,
 *  timeout) — transient classes must NEVER badge. Pure; tested directly. */
function attentionReasonFromError(errorTextOrReason: unknown) {
  const raw = String(errorTextOrReason || '').trim()

  if (!raw) {
    return null
  }

  if (BOT_ATTENTION_CLASSES.has(raw)) {
    return raw
  }

  const text = raw.toLowerCase()

  // Transient failures first, so a retryable error never sticks a badge.
  if (
    /rate.?limit|too many requests|\b429\b|\b5\d\d\b|server error|overloaded|timed?.?out|timeout|temporar/.test(text)
  ) {
    return null
  }

  if (/no llm provider|no access token|not configured|no api key|missing api key/.test(text)) {
    return 'missing_config'
  }

  if (
    /\b401\b|\b403\b|unauthorized|forbidden|authentication|invalid.?api.?key|credentials? (are )?(invalid|expired)/.test(
      text
    )
  ) {
    return 'provider_auth_or_access'
  }

  if (/quota|out of funds|insufficient (credits?|funds|balance)|payment required|\b402\b|billing/.test(text)) {
    return 'provider_quota_limit'
  }

  if (/\bblocked\b/.test(text)) {
    return 'agent_blocked'
  }

  return null
}

/** Per-bot needs-attention state: roster key -> {reason, at, message}.
 *  Display-only presentation state (never persisted, never alters delivery).
 *  Latest failure wins; the bot's next good turn clears it. Hidden bots keep
 *  their entry — hiding is a roster-DISPLAY concern only. */
export const $botAttention = atom<Record<string, { at: number; message: string; reason: string }>>({})

/** Record attention for a bot after a failed turn/delivery. Transient errors
 *  classify to null and set nothing. Latest failure wins. */
export function noteBotAttention(key: string, errorTextOrReason: unknown) {
  const reason = attentionReasonFromError(errorTextOrReason)

  if (!key || !reason) {
    return
  }

  $botAttention.set({
    ...$botAttention.get(),
    [key]: {
      reason,
      at: Date.now(),
      message: String(errorTextOrReason || '')
        .trim()
        .slice(0, 200)
    }
  })
}

/** A good turn clears the badge. */
export function clearBotAttention(key: string) {
  if (!key || !$botAttention.get()[key]) {
    return
  }

  const next = {
    ...$botAttention.get()
  }

  delete next[key]
  $botAttention.set(next)
}

/** The persisted record adds two keys the shared domain model deliberately
 *  omits: `chat`, the dead canonical-chat pointer that mergeServerMeta strips
 *  on sight, and `pet`, the extracted pet icon that stays local and is never
 *  sent to the server. */
interface StoredBotMeta extends BotMeta {
  chat?: unknown
  pet?: unknown
}

/** Appearance records keyed by meta key — `connectionId::profile`, or the bare
 *  bot name on legacy single-source installs. */
export type BotMetaSnapshot = Record<string, StoredBotMeta>

/** `ctx.storage` as this file consumes it. `get` is optional because the write
 *  path feature-detects it mid-body, and it is called without the SDK's
 *  required `fallback`, so it resolves whatever JSON was persisted. */
interface BotMetaStorage {
  get?: (key: string, fallback?: unknown) => any
  remove: (key: string) => unknown
  set: (key: string, value: unknown) => unknown
}

/** Per-bot appearance + display meta, persisted via ctx.storage:
 *  { [botName]: { shape, color, title } } */
export const $botMeta = atom<BotMetaSnapshot>({})

export function commitBotMetaV2(storage: BotMetaStorage | undefined, snapshot: BotMetaSnapshot) {
  const commit = botMetaV2Commit.then(async () => {
    if (typeof storage?.remove !== 'function' || typeof storage?.set !== 'function') {
      throw new Error('bot metadata v2 storage is unavailable')
    }

    const [previousSnapshot, previousMarker] =
      typeof storage.get === 'function'
        ? await Promise.all([storage.get(BOT_META_V2_KEY), storage.get(BOT_META_MIGRATION_KEY)])
        : [null, null]

    const hasCommittedPrevious =
      previousMarker === true &&
      previousSnapshot &&
      typeof previousSnapshot === 'object' &&
      !Array.isArray(previousSnapshot)

    try {
      await storage.remove(BOT_META_MIGRATION_KEY)
      await storage.set(BOT_META_V2_KEY, snapshot)
      await storage.set(BOT_META_MIGRATION_KEY, true)
    } catch (error) {
      if (hasCommittedPrevious) {
        try {
          await storage.set(BOT_META_V2_KEY, previousSnapshot)
          await storage.set(BOT_META_MIGRATION_KEY, true)
        } catch {
          await Promise.allSettled([storage.remove(BOT_META_MIGRATION_KEY), storage.remove(BOT_META_V2_KEY)])
        }
      } else {
        await Promise.allSettled([BOT_META_MIGRATION_KEY, BOT_META_V2_KEY].map(key => storage.remove(key)))
      }

      throw error
    }
  })

  botMetaV2Commit = commit.catch(() => undefined)

  return commit
}

/** A save target resolved into the four things every write site needs. */
interface BotOwner {
  bot: RosterRow
  key: string
  name: string
  route: null | ProfileRoute
}

export function botOwner(owner: RosterRow | string): BotOwner {
  if (typeof owner === 'string') {
    const name = owner.trim()
    const route = migratedLocalRoutes.get(name)

    return {
      bot: route
        ? {
            name,
            sourceScoped: true,
            route
          }
        : {
            name
          },
      name,
      key: route ? botRouteKey(route) : name,
      route: route || null
    }
  }

  const name = String(owner?.name || '').trim()
  const route = botConnectionRoute(owner)

  return {
    bot: owner,
    name,
    key: route ? botRouteKey(route) : name,
    route
  }
}

/** Freshness fence for the server-meta overlay: a roster snapshot fetched
 * before the latest local/server metadata write must not overwrite it. */
export const botMetaWriteAt = new Map<string, number>()

/** How long a write can still outrank a roster snapshot. The overlay only
 *  skips a bot while some snapshot ISSUED before the write is still landing,
 *  and the roster refetches every 5s with retry backoff capped at 15s — well
 *  inside a minute. Past that the stamp can never fence anything again, so
 *  keeping it just grows the map for the life of the renderer. */
const BOT_META_WRITE_FENCE_MS = 60_000

export function noteBotMetaWrite(key: string) {
  const now = Date.now()

  for (const [written, at] of botMetaWriteAt) {
    if (now - at > BOT_META_WRITE_FENCE_MS) {
      botMetaWriteAt.delete(written)
    }
  }

  botMetaWriteAt.set(key, now)
}

/** Three-way outcome of the server half of a save — see the block comment on
 *  `serverOutcome` below for what separates 'unsupported' from 'failed'. */
type BotMetaServerOutcome = 'failed' | 'persisted' | 'unsupported'

interface BotMetaSaveResult {
  serverOutcome: BotMetaServerOutcome
  serverPersisted: boolean
}

/** `profiles.configure` reply. Older gateways answer without `applied` at all,
 *  which is what makes the field optional rather than the contract. */
interface ProfilesConfigureResult {
  applied?: { ui_meta?: boolean }
}

export async function saveBotMeta(owner: RosterRow | string, patch: StoredBotMeta): Promise<BotMetaSaveResult> {
  const { bot, key, name, route } = botOwner(owner)
  const prevMeta = $botMeta.get()[key] || {}

  const next = {
    ...$botMeta.get(),
    [key]: {
      ...prevMeta,
      ...patch
    }
  }

  noteBotMetaWrite(key)
  $botMeta.set(next)

  // Local plugin storage: instant, and the fallback for older gateways.
  let localPersistence = Promise.resolve()

  try {
    const persisted =
      route || botMetaV2Active
        ? commitBotMetaV2(getPluginCtx()?.storage, next)
        : Promise.resolve(getPluginCtx()?.storage?.set?.(BOT_META_V1_KEY, next))

    localPersistence = persisted.catch(() => undefined)
  } catch {
    /* storage unavailable — look persists for this window only */
  }

  // Server-side (source of truth when supported): profile.yaml ui_meta,
  // namespaced under this plugin's id — every client machine sees the same
  // roster. Return the outcome so user-initiated saves can distinguish a
  // cross-machine save from a local-only fallback instead of reporting a
  // false success. Data-URL fields are stripped from ui_meta (64KB cap,
  // rides every profiles.list); the avatar IMAGE goes to the profile asset
  // store instead (profiles.set_asset), which is server-side and uncapped by
  // the list call — so pfps follow the profile across machines too.
  let serverRequest: null | Promise<ProfilesConfigureResult> = null

  try {
    const { image, pet, ...rest } = next[key] || {}

    const request = route
      ? requestForBot(bot, 'profiles.configure', {
          name,
          ui_meta: {
            'hermes-bots': rest
          }
        })
      : host.request('profiles.configure', {
          name,
          ui_meta: {
            'hermes-bots': rest
          }
        })

    serverRequest = Promise.resolve(request) as Promise<ProfilesConfigureResult>
  } catch {
    /* older/unavailable gateway — the local fallback remains saved */
  }

  // Avatar image → profile asset store (feature-detected; local storage
  // remains the fallback rendering source on older gateways) — but only when
  // the image actually CHANGED. Every Edit Profile save sends the image key
  // (changed or not); a no-op `clear` from one machine can race another
  // machine's just-pushed avatar and wipe it server-side, and a no-op
  // `data` push re-uploads the full data URL for nothing.
  if ('image' in patch && patch.image !== (prevMeta.image ?? null)) {
    try {
      const req = patch.image
        ? route
          ? requestForBot(bot, 'profiles.set_asset', {
              name,
              asset: 'avatar',
              data: patch.image
            })
          : host.request('profiles.set_asset', {
              name,
              asset: 'avatar',
              data: patch.image
            })
        : route
          ? requestForBot(bot, 'profiles.set_asset', {
              name,
              asset: 'avatar',
              clear: true
            })
          : host.request('profiles.set_asset', {
              name,
              asset: 'avatar',
              clear: true
            })

      req.catch(() => undefined)
    } catch {
      /* older gateway */
    }
  }

  // Three-way outcome so callers can tell a REAL remote failure from the
  // documented legacy fallback ("older gateways reject the param shape;
  // that's fine, local wins"):
  //   'persisted'   — gateway confirmed applied.ui_meta === true
  //   'unsupported' — older gateway: request rejected, or response carries
  //                   no `applied` contract at all. Silent local fallback;
  //                   an error toast here would fire on EVERY save forever.
  //   'failed'      — gateway speaks the contract and explicitly reported
  //                   the ui_meta write did NOT apply.
  let serverOutcome: BotMetaServerOutcome = 'unsupported'

  if (serverRequest) {
    try {
      const result = await serverRequest

      if (result?.applied?.ui_meta === true) {
        serverOutcome = 'persisted'
      } else if (result && typeof result === 'object' && result.applied && typeof result.applied === 'object') {
        serverOutcome = 'failed'
      }
    } catch {
      /* older/unavailable gateway — the local fallback remains saved */
    }

    // Re-stamp now that the server write settled: a roster snapshot fetched
    // while profiles.configure was still in flight predates the new ui_meta
    // just as surely as one fetched before the local write.
    noteBotMetaWrite(key)
  }

  await localPersistence

  return {
    serverPersisted: serverOutcome === 'persisted',
    serverOutcome
  }
}

/** Migrate name-keyed appearance state only when the live registry proves
 * there is one local source. A v1 name cannot identify a machine in a
 * multi-source desktop, so the conservative result there is to retain v1 as
 * rollback data and leave remote rows unpainted. */
function hydrateBotMeta(snapshot: BotMetaSnapshot, remap: Map<string, string> | null = null): BotMetaSnapshot {
  const next = {
    ...snapshot
  }

  for (const [key, meta] of Object.entries($botMeta.get())) {
    const target = remap?.get(key) || key
    next[target] = {
      ...(next[target] || {}),
      ...meta
    }
  }

  $botMeta.set(next)

  return next
}

export async function migrateBotMeta(storage: BotMetaStorage | undefined = getPluginCtx()?.storage) {
  let v1: BotMetaSnapshot | null = null
  let v2: BotMetaSnapshot | null = null
  let v2Committed = false

  try {
    ;[v1, v2, v2Committed] = await Promise.all([
      storage?.get?.(BOT_META_V1_KEY),
      storage?.get?.(BOT_META_V2_KEY),
      storage?.get?.(BOT_META_MIGRATION_KEY)
    ])
  } catch {
    return false
  }

  if (v2Committed === true && v2 && typeof v2 === 'object' && !Array.isArray(v2)) {
    hydrateBotMeta(v2)
    botMetaV2Active = true

    return true
  }

  if (!v1 || typeof v1 !== 'object' || Array.isArray(v1) || typeof host.agents !== 'function') {
    if (v1 && typeof v1 === 'object' && !Array.isArray(v1)) {
      hydrateBotMeta(v1)
    }

    return false
  }

  let union
  let routes

  try {
    union = await host.agents()
    routes = typeof host.profileRoutes === 'function' ? await host.profileRoutes() : []
  } catch {
    hydrateBotMeta(v1)

    return false
  }

  const sources = Array.isArray(union?.sources) ? union.sources : []
  const localAgents = (union?.agents || []).filter(agent => agent?.connectionKind === 'local')

  const soleLocal =
    sources.length === 1
      ? sources[0]?.kind === 'local'
      : sources.length === 0 &&
        localAgents.length > 0 &&
        (union?.agents || []).every(agent => agent?.connectionKind === 'local')

  if (!soleLocal) {
    hydrateBotMeta(v1)

    return false
  }

  const migrated: BotMetaSnapshot = {}
  const pendingLocalRoutes = new Map<string, ProfileRoute>()

  for (const [name, meta] of Object.entries(v1)) {
    const route =
      (routes || []).find(candidate => candidate?.mode === 'local' && candidate?.profile === name) ||
      (() => {
        const agent = localAgents.find(candidate => candidate.profile === name)

        return agent
          ? {
              connectionId: agent.connectionId,
              mode: 'local',
              profile: name,
              targetProfile: agent.targetProfile || name
            }
          : null
      })()

    if (!route?.connectionId) {
      // A missing route makes the topology proof unusable for this key. Keep
      // the v1 record intact rather than guessing a local/remote projection.
      hydrateBotMeta(v1)

      return false
    }

    const captured: ProfileRoute = {
      connectionId: route.connectionId,
      mode: 'local',
      profile: name,
      targetProfile: route.targetProfile || name
    }

    migrated[botRouteKey(captured)] = meta
    pendingLocalRoutes.set(name, captured)
  }

  const remap = new Map([...pendingLocalRoutes].map(([name, route]) => [name, botRouteKey(route)]))

  const hydrated = {
    ...migrated
  }

  for (const [key, meta] of Object.entries($botMeta.get())) {
    const target = remap.get(key) || key
    hydrated[target] = {
      ...(hydrated[target] || {}),
      ...meta
    }
  }

  try {
    await commitBotMetaV2(storage, hydrated)
  } catch {
    botMetaV2Active = false
    hydrateBotMeta(v1)

    return false
  }

  migratedLocalRoutes.clear()

  for (const [name, route] of pendingLocalRoutes) {
    migratedLocalRoutes.set(name, route)
  }

  hydrateBotMeta(hydrated)
  botMetaV2Active = true

  return true
}

// ── data ─────────────────────────────────────────────────────────────────────

/** True once profiles.list reports the backend injects the bot-to-bot
 *  protocol into the system prompt itself (hermes-agent bot_mode_probe).
 *  Gates every SOUL.md protocol append below. */
export let serverInjectsProtocol = false
let foregroundRosterRequest: symbol | undefined

/** A `profiles.list` answer as Bot Mode consumes it, plus the fields the
 *  multi-source merge and the roster query stamp on afterwards. Not in
 *  types.ts: this is the query-cache envelope around RosterRow, not a domain
 *  object. */
interface RosterSnapshot {
  /** Newer backends inject the teammate protocol into the system prompt. */
  bot_mode_protocol?: boolean
  /** Time the request was ISSUED, the conservative bound mergeServerMeta wants. */
  fetchedAt?: number
  primaryConnectionId?: string
  profiles?: RosterRow[]
  sources?: GatewaySource[]
}

/** One row of the desktop-wide union agent roster (`host.agents`). */
interface UnionAgentRow {
  connectionId?: string
  connectionKind?: string
  connectionLabel?: string
  handle?: string
  profile?: string
  targetProfile?: string
}

/** The union roster payload. `primaryConnectionId` is served by Electron but
 *  absent from the SDK's `DesktopAgentRoster`, so the merge reads it here. */
interface UnionRoster {
  agents?: UnionAgentRow[]
  primaryConnectionId?: string
  sources?: GatewaySource[]
}

export function useRoster() {
  const activeConnectionId = useValue(host.state.connectionId)

  return useQuery({
    queryKey: [...ROSTER_KEY, activeConnectionId],
    queryFn: () => fetchRoster(activeConnectionId, { publishProtocol: true }),
    refetchInterval: 5000,
    staleTime: 5000,
    retry: ROSTER_QUERY_RETRY,
    retryDelay: attempt => Math.min(15000, 1000 * 2 ** attempt)
  })
}

async function fetchRoster(
  activeConnectionId?: null | string,
  { publishProtocol = true }: { publishProtocol?: boolean } = {}
): Promise<RosterSnapshot> {
  const connectionId = String(activeConnectionId || host.activeConnectionId?.() || 'local')
  const profile = String(host.state.profile?.get?.() || 'default').trim() || 'default'
  const requestOwner = publishProtocol ? Symbol(connectionId) : undefined

  if (requestOwner) {
    foregroundRosterRequest = requestOwner
  }

  const ownerIsCurrent = () =>
    connectionId === String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || 'local') &&
    profile === (String(host.state.profile?.get?.() || 'default').trim() || 'default')

  const requestMayPublish = () => requestOwner !== undefined && foregroundRosterRequest === requestOwner && ownerIsCurrent()

  // Stamp the ISSUE time on the snapshot: mergeServerMeta compares it
  // against each bot's last local meta write, and a fetch issued before
  // a write can only carry pre-write ui_meta. (Issue time is the
  // conservative bound — the server answered no earlier than this.)
  const issuedAt = Date.now()

  // Rich rows (last_session, canonical_session, ui_meta, has_avatar)
  // come from the ACTIVE gateway's profiles.list — the canonical Bot
  // Chat is resolved server-side by NAME (the "Bot Chat" registry row),
  // so the roster never sends session pointers.
  // Refresh the alias identity index alongside the roster: alias routes
  // (Desktop profile → remote backend root) are what let a backend row
  // keep its configured friendly identity after activation (#89131).
  // Best-effort and feature-detected — a failed read keeps the last
  // good index rather than dropping identities mid-session.
  let activeRoute: ProfileRoute | undefined

  if (typeof host.profileRoutes === 'function') {
    const epoch = publishProtocol ? beginAliasRouteIndex() : undefined

    try {
      const routes = await host.profileRoutes()

      activeRoute = routes.find(
        route => String(route.connectionId) === connectionId && String(route.profile) === profile
      )

      if (epoch !== undefined && requestMayPublish()) {
        indexAliasRoutes(routes, epoch)
      }
    } catch {
      /* keep the previous alias index */
    }
  }

  // Dispatch through the source captured before profileRoutes awaited. A
  // connection switch must not retarget this request through ambient state.
  const activeBot: RosterRow = {
    connectionId,
    name: profile,
    route: activeRoute || {
      connectionId,
      mode: connectionId === 'local' ? 'local' : 'remote',
      profile,
      targetProfile: profile
    },
    sourceScoped: true
  }

  // Older hosts only have the ambient request door. It is safe while the
  // captured owner is still active; after a switch, fail closed instead of
  // letting that ambient door silently retarget the request.
  if (typeof host.requestProfile !== 'function' && !ownerIsCurrent()) {
    throw new Error(`Cannot route profiles.list for ${connectionId}::${profile}`)
  }

  const local = await requestForBot<RosterSnapshot>(
    typeof host.requestProfile === 'function' ? activeBot : { name: profile },
    'profiles.list',
    {}
  )

  // Newer backends inject the teammate-messaging protocol into every
  // session's system prompt (agent.bot_mode_protocol) — SOUL.md must not
  // carry a second copy. Older gateways lack the flag: keep appending.
  if (requestMayPublish()) {
    serverInjectsProtocol = Boolean(local?.bot_mode_protocol)
  }

  // Multi-source desktops (hermes-agent #86875) also expose the union
  // agent roster across every registered connection. Merge agents from
  // OTHER sources in as additional rows. Feature-detected + best-effort:
  // an older Desktop build (no host.agents) or a roster hiccup leaves
  // the local list exactly as it was.
  if (typeof host.agents === 'function') {
    try {
      const union = await host.agents()
      const previous: RosterRow[] = $lastRoster.get().filter(row => !row?.ghost)
      const merged = mergeMultiSourceRoster(local, union, activeConnectionId, previous)
      const sources = Array.isArray(union?.sources) ? union.sources : []

      return {
        ...merged,
        profiles: (merged?.profiles || []).map(row => annotateBotSource(row, sources)),
        sources,
        fetchedAt: issuedAt
      }
    } catch {
      /* older build or roster failure — single-source list stands */
    }
  }

  return {
    ...(local && typeof local === 'object' ? local : {}),
    fetchedAt: issuedAt
  }
}

/** Synchronous union-roster read for the composer surfaces (autocomplete
 *  provider + mention middleware). useRoster caches under
 *  [...ROSTER_KEY, activeConnectionId] — a 3-element key — so a bare
 *  getQueryData(ROSTER_KEY) exact-match lookup returns undefined forever
 *  (issue #89303: remote handles absent from @ autocomplete, mentions
 *  unrouted). Read the live connection's entry first, then fall back to a
 *  prefix scan keeping the freshest snapshot. Never throws: cold cache or
 *  legacy queryClient returns null and callers fall back to their own path. */
export function cachedUnionRoster(): RosterSnapshot | null {
  if (typeof queryClient === 'undefined' || !queryClient || typeof queryClient.getQueryData !== 'function') {
    return null
  }

  try {
    const connectionId = String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || 'local')
    const exact = queryClient.getQueryData<RosterSnapshot>([...ROSTER_KEY, connectionId])

    if (Array.isArray(exact?.profiles)) {
      return exact
    }

    if (typeof queryClient.getQueriesData === 'function') {
      let best: RosterSnapshot | null = null

      // v5 takes a filters object; a legacy v3 queryClient treats the same
      // object as the key itself and simply matches nothing — harmless.
      for (const [, data] of queryClient.getQueriesData<RosterSnapshot>({
        queryKey: ROSTER_KEY
      })) {
        if (Array.isArray(data?.profiles) && (!best || Number(data.fetchedAt || 0) > Number(best.fetchedAt || 0))) {
          best = data
        }
      }

      return best
    }
  } catch {
    /* cache hiccup — caller falls back (middleware refetches) */
  }

  return null
}

/** Cold cache (no Bots pane mounted yet): fetch once and seed the query cache
 * so the completion popup answers on the next keystroke. */
const warmingRosterFor = new Map<string, symbol>()

export function warmUnionRoster() {
  if (
    typeof queryClient === 'undefined' ||
    !queryClient ||
    typeof queryClient.getQueryData !== 'function' ||
    typeof queryClient.setQueryData !== 'function'
  ) {
    return
  }

  const connectionId = String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || 'local')
  const cacheKey = [...ROSTER_KEY, connectionId]
  const cacheBefore = queryClient.getQueryData<RosterSnapshot>(cacheKey)

  if (warmingRosterFor.has(connectionId)) {
    return
  }

  const owner = Symbol(connectionId)

  warmingRosterFor.set(connectionId, owner)

  // A hung fetch must not block warms for this connection forever.
  const evict = setTimeout(() => {
    if (warmingRosterFor.get(connectionId) === owner) {
      warmingRosterFor.delete(connectionId)
    }
  }, 15000)

  void fetchRoster(connectionId, { publishProtocol: false })
    .then(roster => {
      if (
        warmingRosterFor.get(connectionId) === owner &&
        queryClient.getQueryData<RosterSnapshot>(cacheKey) === cacheBefore &&
        Array.isArray(roster?.profiles)
      ) {
        queryClient.setQueryData(cacheKey, roster)
      }
    })
    .catch(() => undefined)
    .finally(() => {
      clearTimeout(evict)

      if (warmingRosterFor.get(connectionId) === owner) {
        warmingRosterFor.delete(connectionId)
      }
    })
}

/** Merge the union agent roster (host.agents) over the active gateway's
 *  profiles.list. Active-source rows — matched by the LIVE connection id,
 *  falling back to the roster's primaryConnectionId, then the legacy
 *  kind==='local' rule on older desktops — are the agents profiles.list
 *  already returned: they only ANNOTATE the rich rows (handle, connection
 *  fields); rich fields stay authoritative and they are NOT duplicated.
 *  Rows from other sources become new roster entries tagged with their
 *  source label so BotRow can badge them, warm the captured agent, and route
 *  every open directly through that descriptor. Pure — exercised directly by
 *  the tests. */
function mergeMultiSourceRoster(
  local: RosterSnapshot | null | undefined,
  union: UnionRoster | null | undefined,
  activeConnectionId?: null | string,
  previous: RosterRow[] = []
): RosterSnapshot {
  const localProfiles = Array.isArray(local?.profiles) ? local.profiles : []
  const agents = Array.isArray(union?.agents) ? union.agents : []
  // A live id of null/'' means the window is on the unscoped local backend
  // (legacy hosts reported null for mode:'local'; the SDK now reports
  // 'local'). Do NOT fall back to registry primary when the third argument
  // was passed — primary can still say "spark" after the user clicked a
  // local bot, which skipped every Spark row as "active" and invented a
  // This-device shadow of default.
  const liveProvided = arguments.length >= 3
  const liveId = String(activeConnectionId || '').trim()
  let activeId = liveId || (liveProvided ? '' : String(union?.primaryConnectionId || '').trim())

  // Migrated remote-primary windows can still expose a legacy remote
  // descriptor without connectionId. That produces a null live id even
  // though profiles.list is answering from the registry primary. Infer the
  // primary only when its inventory matches the rich rows and the local
  // inventory does not. A genuinely local window has a matching local row,
  // so it keeps the null-is-local behavior used after clicking This device.
  if (!activeId && liveProvided) {
    const primaryId = String(union?.primaryConnectionId || '').trim()
    const richNames = new Set(localProfiles.map(profile => String(profile?.name || '').trim()).filter(Boolean))

    const localMatches = agents.some(
      agent => agent?.connectionKind === 'local' && richNames.has(String(agent?.profile || '').trim())
    )

    const primaryMatches = agents.some(
      agent =>
        String(agent?.connectionId || '').trim() === primaryId && richNames.has(String(agent?.profile || '').trim())
    )

    if (!localMatches && primaryId && primaryMatches) {
      activeId = primaryId
    }
  }

  const activeByName = new Map<string, RosterRow>()

  // Treat the rich list as one row per active-source profile. Clone every
  // row: some gateway clients reuse response objects, and annotating those in
  // place made each five-second refresh feed the previous union back into the
  // next merge, growing duplicate source rows indefinitely.
  for (const profile of localProfiles) {
    const name = String(profile?.name || '').trim()

    if (!name || profile?.remoteSource) {
      continue
    }

    if (profile?.sourceScoped && activeId && profile.connectionId !== activeId) {
      continue
    }

    if (!activeByName.has(name)) {
      activeByName.set(name, {
        ...profile,
        name
      })
    }
  }

  const profiles = [...activeByName.values()]

  // host.agents is an Electron/main-process capability. Defend the plugin
  // boundary too: older shells or reconnect races can still hand us repeated
  // identities even after the core roster deduplicates them.
  const seenSources = new Set<string>()

  for (const agent of agents) {
    const profile = String(agent?.profile || '').trim()
    const connectionId = String(agent?.connectionId || '').trim()
    const sourceKey = `${connectionId}::${profile || 'default'}`

    if (!profile || seenSources.has(sourceKey)) {
      continue
    }

    seenSources.add(sourceKey)

    // The union enumerates EVERY registered connection, including the active
    // gateway that already answered profiles.list. Without this the active
    // gateway's own agents (connectionKind 'remote' on a remote-primary
    // desktop) would be appended as phantom duplicates — every bot listed
    // twice. Older Electron builds predate the connection ids; fall back to
    // the legacy local-source rule so single-source behavior stays intact.
    const isActiveSource = activeId ? connectionId === activeId : agent.connectionKind === 'local'
    const row = isActiveSource ? activeByName.get(profile) : null

    if (row) {
      // Annotate in place: the @name-device handle only differs from the
      // bare name when the profile exists on several sources.
      row.handle = agent.handle
      row.connectionId = agent.connectionId
      row.connectionKind = agent.connectionKind
      row.connectionLabel = agent.connectionLabel
      row.targetProfile = agent.targetProfile || profile
      row.route = {
        connectionId,
        mode: agent.connectionKind === 'local' ? 'local' : 'remote',
        profile,
        targetProfile: agent.targetProfile || profile
      }
      row.sourceScoped = true

      continue
    }

    if (isActiveSource) {
      // Union saw an active-source profile profiles.list didn't return (older
      // backend mid-refresh) — skip rather than invent a thin row.
      continue
    }

    profiles.push({
      name: profile,
      handle: agent.handle,
      connectionId,
      connectionKind: agent.connectionKind,
      connectionLabel: agent.connectionLabel,
      targetProfile: agent.targetProfile || profile,
      route: {
        connectionId,
        mode: agent.connectionKind === 'local' ? 'local' : 'remote',
        profile,
        targetProfile: agent.targetProfile || profile
      },
      remoteSource: true,
      sourceScoped: true
    })
  }

  // SSH sources drop to connect-on-demand the moment their tunnel is not
  // the live gateway. Keep previously painted remote rows so clicking the
  // local agent does not empty Bot Mode.
  if (Array.isArray(previous) && previous.length > 0) {
    const present = new Set(profiles.map(row => `${row.connectionId || ''}::${row.name}`))
    const unionSourceIds = new Set(agents.map(agent => String(agent?.connectionId || '').trim()).filter(Boolean))

    const omitted = new Set(
      (Array.isArray(union?.sources) ? union.sources : [])
        .filter(source => source?.error === 'connect-on-demand' || source?.reachable === false)
        .map(source => String(source.connectionId || '').trim())
        .filter(Boolean)
    )

    const registered = new Set(
      (Array.isArray(union?.sources) ? union.sources : [])
        .map(source => String(source?.connectionId || '').trim())
        .filter(Boolean)
    )

    for (const row of previous) {
      const connectionId = String(row?.connectionId || '').trim()
      const name = String(row?.name || '').trim()
      const key = `${connectionId}::${name || 'default'}`

      if (!row?.remoteSource || !connectionId || !name || present.has(key)) {
        continue
      }

      if (registered.size > 0 && !registered.has(connectionId)) {
        continue
      }

      if (omitted.has(connectionId) || !unionSourceIds.has(connectionId)) {
        profiles.push({
          ...row,
          remoteSource: true,
          sourceScoped: true
        })
        present.add(key)
      }
    }
  }

  return {
    ...local,
    profiles
  }
}

/** The @handle users tag a bot with. Multi-source rosters precompute the
 *  handle (bare name, or name-device when the profile exists on several
 *  registered sources) — prefer it when present. The primary profile's
 *  callable alias is 'hermes' — the mention middleware resolves it back to
 *  'default' — so the word 'default' never surfaces in the UI. */
export function botHandle(name: string, bot?: Partial<RosterRow> | null): string {
  if (bot?.handle && bot.handle !== name) {
    return bot.handle
  }

  return (name || '').trim().toLowerCase() === 'default' ? 'hermes' : name
}

/** Taggable @-forms derived from a bot's friendly names — the core profile
 *  display name (`hermes profile rename`) and the Bot Mode title. Free text
 *  reduces to the mention charset two ways: slugified ("Research Buddy" →
 *  research-buddy, the form autocomplete inserts) and collapsed
 *  (researchbuddy). Reserved tokens are dropped so a bot renamed "Hermes"
 *  can never hijack the primary profile's @hermes alias. */
export function mentionNameForms(value: null | string | undefined): string[] {
  const name = String(value || '')
    .trim()
    .toLowerCase()

  if (!name) {
    return []
  }

  const slug = name.replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '')
  const collapsed = name.replace(/[^a-z0-9_-]+/g, '')

  return [...new Set([slug, collapsed])].filter(
    form => /^[a-z0-9][a-z0-9_-]*$/.test(form) && !['all', 'everyone', 'user', 'default', 'hermes'].includes(form)
  )
}

/** Every friendly (renameable) name a roster row carries: the Bot Mode title
 *  (server-synced via ui_meta, locally stored, or persisted on a durable
 *  group descriptor) and the core profile display_name — in displayName's
 *  precedence order. Remote rows never borrow local meta (two `default`s
 *  must not share a title) — EXCEPT the connection-exact alias identity
 *  (#89131): a backend row claimed by a configured alias route carries the
 *  alias's friendly names, so @moxie keeps resolving after handoff. */
export function botFriendlyNames(bot: Partial<RosterRow> | null | undefined): Array<null | string | undefined> {
  const metaByName: Record<string, BotMeta | undefined> | null = typeof $botMeta !== 'undefined' ? $botMeta.get() : null
  const localTitle = !bot?.remoteSource ? metaByName?.[bot?.name!]?.title : null
  const alias = aliasIdentityFor(bot)

  const aliasTitle = alias
    ? alias.metaKeys.map(key => metaByName?.[key]?.title).find(title => typeof title === 'string' && title.trim()) ||
      alias.name
    : null

  return [bot?.ui_meta?.['hermes-bots']?.title, localTitle, aliasTitle, bot?.title, bot?.display_name]
}

/** The tag autocomplete inserts for a bot: the renamed (friendly) slug when
 *  the user gave the bot a real name, otherwise the profile @handle. The
 *  resolvers accept both, so older muscle memory keeps working. */
export function botMentionTag(bot: GroupMember | RosterRow): string {
  for (const friendly of botFriendlyNames(bot)) {
    const forms = mentionNameForms(friendly)

    if (forms.length) {
      return forms[0]
    }
  }

  return botHandle(bot?.name, bot)
}

/** Who a roster row is being compared against: a profile name plus the
 *  registry connection it lives on. Covers the focused chat's owner, the live
 *  gateway pair, and the bare `{ name }` the mention provider builds. */
interface RosterOwnerRef {
  connectionId?: null | string
  name?: null | string
}

export function isActiveRosterBot(
  bot: Partial<RosterRow> | null | undefined,
  active: RosterOwnerRef | null | undefined
) {
  if (!active) {
    return false
  }

  const activeName = String(active.name || 'default').trim() || 'default'
  const activeId = String(active?.connectionId || '').trim()
  const botId = String(bot?.connectionId || '').trim()
  const botName = String(bot?.name || '').trim() || 'default'

  if (bot?.remoteSource) {
    return Boolean(activeId) && activeId === botId && botName === activeName
  }

  if (activeId && activeId !== 'local' && botId && activeId !== botId) {
    return false
  }

  return botName === activeName
}

/** The key a bot is selected/badged/watermarked under. `undefined` is
 *  load-bearing on the nullable overload — callers use it to CLEAR a
 *  selection, so it must never be coerced to a 'default' fallback the way
 *  botRosterKey does. A real row always yields a key (`name` is required). */
/* eslint-disable no-redeclare -- overload signatures; the rule predates TS */
export function botSelectionKey(bot: RosterRow): string
export function botSelectionKey(bot: Partial<RosterRow> | null | undefined): string | undefined

export function botSelectionKey(bot: Partial<RosterRow> | null | undefined): string | undefined {
  return bot?.sourceScoped || bot?.remoteSource ? botRosterKey(bot) : bot?.name
}
/* eslint-enable no-redeclare */

export function isDefaultBot(bot: Partial<RosterRow> | null | undefined): boolean {
  const route = botConnectionRoute(bot)

  return (
    String(route?.profile || bot?.name || '')
      .trim()
      .toLowerCase() === 'default'
  )
}

export function newBotChat(bot: RosterRow) {
  if (typeof host.newChat !== 'function') {
    host.notify?.({
      kind: 'error',
      message:
        getPluginCtx()?.i18n?.t('bot.openAnotherChatUnsupported') ?? 'Update Hermes Desktop to open another Bot chat.'
    })

    return
  }

  const route = botConnectionRoute(bot)

  if (!route) {
    host.notify?.({
      kind: 'error',
      message:
        getPluginCtx()?.i18n?.t('bot.openAnotherChatUnsupported') ?? 'Update Hermes Desktop to open another Bot chat.'
    })

    return
  }

  const ownerKey = botWorkspaceOwnerKey(bot)
  setBotsWorkspaceOwner(ownerKey, bot)
  host.newChat(route, {
    workspaceMode: 'bots',
    workspaceOwnerKey: ownerKey
  })
}

/** Resolve @handles in prose against the Bot Mode roster (local + Connections).
 *  Skips the bot already speaking in this chat. Unique bare names match;
 *  duplicate names require the @name-device handle. */
export function resolveRosterMentions(
  text: null | string | undefined,
  roster: RosterRow[] | null | undefined,
  active: RosterOwnerRef = {}
): RosterRow[] {
  const members = Array.isArray(roster) ? roster : []

  const prose = String(text || '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`\n]*`/g, ' ')

  const byForm = new Map<string, RosterRow | null>()

  for (const bot of members) {
    if (!bot?.name || isActiveRosterBot(bot, active)) {
      continue
    }

    const handle = String(botHandle(bot.name, bot) || '').toLowerCase()
    const name = String(bot.name || '').toLowerCase()
    const forms = new Set([handle, name])

    if (bot.handle) {
      forms.add(String(bot.handle).toLowerCase())
    }

    // Renamed bots are taggable by their friendly names too — the core
    // profile display_name and the Bot Mode title (issue: renaming a bot
    // didn't change what you @-tag it with).
    for (const friendly of botFriendlyNames(bot)) {
      for (const form of mentionNameForms(friendly)) {
        forms.add(form)
      }
    }

    for (const form of forms) {
      if (!form) {
        continue
      }

      const existing = byForm.get(form)

      if (existing && existing !== bot) {
        byForm.set(form, null)

        continue
      }

      if (!existing) {
        byForm.set(form, bot)
      }
    }
  }

  const mentioned: RosterRow[] = []
  const seen = new Set<string>()

  for (const match of prose.matchAll(/(^|\s)@([a-z0-9][a-z0-9_-]*)/gi)) {
    let token = match[2].toLowerCase()

    if (token === 'hermes') {
      token = byForm.has('hermes') ? 'hermes' : token
    }

    const bot = byForm.get(token)

    if (!bot) {
      continue
    }

    const key = botRosterKey(bot)

    if (seen.has(key)) {
      continue
    }

    seen.add(key)
    mentioned.push(bot)
  }

  return mentioned
}

/** Source-qualified identity for a roster row — the React list key AND the
 *  cross-surface roster identity. Names alone are NOT unique in a
 *  multi-source roster (two connections can both expose 'default');
 *  duplicate keys make React reconciliation repeat whole blocks of the list
 *  on every poll repaint (the Aug 2026 dupe-bots smear). */
export function botRosterKey(bot: Partial<RosterRow> | null | undefined): string {
  return `${bot?.connectionId || 'legacy'}::${bot?.name || 'default'}`
}

/** The key a bot's persisted appearance/meta lives under. Same nullable
 *  contract as botSelectionKey: a real row always resolves to a key, a
 *  partial/absent one may not. */
/* eslint-disable no-redeclare -- overload signatures; the rule predates TS */
export function botMetaKey(bot: RosterRow): string
export function botMetaKey(bot: Partial<RosterRow> | null | undefined): string | undefined

export function botMetaKey(bot: Partial<RosterRow> | null | undefined): string | undefined {
  const route = botConnectionRoute(bot)

  return route ? botRouteKey(route) : bot?.name
}
/* eslint-enable no-redeclare */

export function persistBotMetaSnapshot(value: Record<string, BotMeta>, scoped = false): Promise<void> {
  try {
    const persisted = scoped
      ? commitBotMetaV2(getPluginCtx()?.storage, value)
      : Promise.resolve(getPluginCtx()?.storage?.set?.(BOT_META_V1_KEY, value))

    return persisted.catch(() => undefined)
  } catch {
    return Promise.resolve()
  }
}

export function sourceByConnection(sources: GatewaySource[] | null | undefined): Map<string, GatewaySource> {
  return new Map(
    (Array.isArray(sources) ? sources : [])
      .filter(source => source?.connectionId)
      .map(source => [String(source.connectionId), source])
  )
}

/** Copy current source health onto a row without changing its owner. */
export function annotateBotSource(bot: RosterRow, sources: GatewaySource[] | null | undefined): RosterRow {
  const id = String(bot?.connectionId || '').trim()

  if (!id) {
    return bot
  }

  const list = Array.isArray(sources) ? sources : []
  const source = sourceByConnection(list).get(id)

  if (!source) {
    return list.length && bot?.sourceScoped
      ? {
          ...bot,
          sourceMissing: true,
          sourceReachable: false
        }
      : bot
  }

  return {
    ...bot,
    connectionKind: bot.connectionKind || source.kind,
    connectionLabel: bot.connectionLabel || source.label,
    sourceError: source.error || null,
    sourceMissing: false,
    sourceReachable: source.reachable
  }
}

/** The source-health fields botSourceStatus reads. Both a full RosterRow and
 *  the reduced GroupMember satisfy it, and so do the ad-hoc literals the
 *  group/connection headers build. */
interface BotSourceFields {
  ghost?: boolean
  name?: string
  sourceError?: null | string
  sourceMissing?: boolean
  sourceReachable?: boolean | null
}

/** How a row's owning gateway is doing, as the roster paints it. */
interface BotSourceStatus {
  available: boolean
  key: 'missing' | 'on-demand' | 'ready' | 'unavailable' | 'unknown'
  label: string
  tone: 'bad' | 'good' | 'muted' | 'warn'
}

/** Called from plain functions rather than components, so there is no `useBots`
 *  to lean on and the ctx may not be installed yet — the English text is the
 *  floor, not the intended reading. Guard `i18n` as well as the ctx: this runs
 *  on every roster row, and a throw here paints an empty rail. */
function sourceLabel(key: string, fallback: string): string {
  return getPluginCtx()?.i18n?.t(`roster.${key}`) ?? fallback
}

export function botSourceStatus(bot: BotSourceFields | null | undefined): BotSourceStatus {
  const error = String(bot?.sourceError || '').trim()

  if (bot?.sourceMissing) {
    return {
      available: false,
      key: 'missing',
      label: sourceLabel('gatewayRemoved', 'Gateway removed'),
      tone: 'bad'
    }
  }

  if (error === 'connect-on-demand') {
    return {
      available: true,
      key: 'on-demand',
      label: sourceLabel('onDemand', 'On demand'),
      tone: 'muted'
    }
  }

  if (error || bot?.sourceReachable === false) {
    return {
      available: false,
      key: 'unavailable',
      label: sourceLabel('unavailable', 'Unavailable'),
      tone: 'warn'
    }
  }

  if (bot?.sourceReachable === true) {
    return {
      available: true,
      key: 'ready',
      label: sourceLabel('ready', 'Ready'),
      tone: 'good'
    }
  }

  return {
    available: true,
    key: 'unknown',
    label: sourceLabel('statusUnknown', 'Status unknown'),
    tone: 'muted'
  }
}

/** Drop unreachable same-name copies from the top-level agent list.
 *
 *  Group rooms persist source-qualified members. After Desktop moves to the
 *  built-in This-device source, the old loopback row (127.0.0.1:port, not
 *  listening) still sits next to the live profile and looks like a second
 *  agent (#92286). Routing identities stay intact: this only filters sidebar
 *  tiles. $lastRoster, group members, and @-mentions still see every
 *  (connectionId, profile) row.
 *
 *  Ghosts stay: a selected-but-offline owner must remain visible rather than
 *  being replaced by a same-named twin on another gateway. A name with no
 *  reachable copy is kept, so a genuinely down source still has a row. */
export function preferReachableSameNameRows(bots: RosterRow[] | null | undefined): RosterRow[] {
  const rows = Array.isArray(bots) ? bots : []
  const reachableNames = new Set<string>()

  for (const bot of rows) {
    if (botSourceStatus(bot).available) {
      reachableNames.add(bot?.name)
    }
  }

  return rows.filter(bot => botSourceStatus(bot).available || bot?.ghost || !reachableNames.has(bot?.name))
}

/** Filter by the two stable identities rendered in every roster row: the
 * customizable display name and the profile's @handle. Keep the current
 * activity order — search narrows the roster, it never re-ranks it. */
export function filterBots(roster: RosterRow[], metaByName: Record<string, BotMeta>, query: string) {
  const needle = query.trim().toLowerCase().replace(/^@/, '')

  if (!needle) {
    return roster
  }

  return roster.filter(bot => {
    const meta = botRosterMeta(bot, metaByName)
    const display = displayName(bot, meta).toLowerCase()
    const profile = (bot.name || '').toLowerCase()
    const handle = botHandle(bot.name, bot).toLowerCase()
    // Multi-source rows also match on their device name ("homelab" finds
    // every bot living on the Homelab connection).
    const sourceLabel = (bot.connectionLabel || '').toLowerCase()
    const role = `${meta?.description || ''} ${bot.description || ''}`.toLowerCase()
    const preview = String(botActivitySession(bot)?.preview || '').toLowerCase()

    return (
      display.includes(needle) ||
      profile.includes(needle) ||
      handle.includes(needle) ||
      sourceLabel.includes(needle) ||
      role.includes(needle) ||
      preview.includes(needle)
    )
  })
}

/** The session whose activity best represents this bot — the FRESHER of the
 *  canonical Bot Chat (canonical_session, the profile's "Bot Chat" registry
 *  row resolved server-side by name) and the profile's newest visible
 *  conversation (last_session).
 *
 *  Canonical Bot Chats are hidden from the session list by design, so
 *  last_session alone never sees them: a bot you talk to all day through its
 *  Bot Chat reads "6d ago" because its newest VISIBLE session is a week old.
 *  Every activity signal (age label, pulse dot, unread watermark, recency
 *  sort) keys off this helper. Older gateways without the canonical_session
 *  field degrade to last_session unchanged. */
export function botActivitySession(bot: null | RosterRow | undefined): CanonicalSession | SessionPreview | null {
  const preferred = bot?.canonical_session
  const last = bot?.last_session

  if (!preferred || !last) {
    return preferred || last || null
  }

  return (preferred.last_active || 0) >= (last.last_active || 0) ? preferred : last
}
