/**
 * The always-hidden reconciliation sweep: everything that walks Bot Mode's
 * own sessions on load and on each reconnect and pushes them back to hidden.
 *
 * Durable-visibility plumbing, not a surface. It reads the room store and the
 * roster cache and renders nothing, so the plugin's lifecycle can start the
 * scheduler without pulling a view in.
 */

import { host } from '@hermes/plugin-sdk'

import { PROFILE_SESSION_LIST_LIMIT } from './canonical-chat'
import { $lastRoster } from './data'
import { $groupChats } from './group-chat'
import { groupMemberKey } from './group-membership'
import { backendTargetProfile, botConnectionRoute, requestForBot } from './routing'
import type { GroupMember, RosterRow } from './types'

/** The slice of the plugin context the scheduler needs to park its timer. */
interface HideSweepContext {
  onDispose?: (fn: () => void) => void
}

/** One-time reconciliation: Bot Mode sessions are always hidden, but rooms
 *  and Bot Chats created before this policy (or while the old pref was off)
 *  left visible rows behind. On every plugin load, sweep the session ids we
 *  own by id (each group room's member sessions) through the core
 *  session.set_hidden RPC, then run the TITLE-based ownership sweep for
 *  everything else — canonical Bot Chats are identified by name (the
 *  registry row titled "Bot Chat"), so the title sweep is what hides them;
 *  no stored-id pointer is consulted. Idempotent (the DB setter is a no-op
 *  on already-hidden rows) and feature-detected: older Desktop hosts defer
 *  reconciliation rather than activating an absent profile backend. */
export function startHideSweepScheduler(ctx: HideSweepContext) {
  let timer: ReturnType<typeof setTimeout> | null = null
  let inflight: Promise<unknown> | null = null
  let pending = false
  let disposed = false
  const run = () => {
    timer = null
    if (disposed) {
      return
    }
    if (inflight) {
      pending = true
      return
    }
    inflight = Promise.resolve()
      .then(() => hideOwnedBotSessions())
      .catch(() => undefined)
      .finally(() => {
        inflight = null
        if (pending && !disposed) {
          pending = false
          schedule()
        }
      })
  }
  const schedule = () => {
    if (disposed) {
      return
    }
    try {
      if (timer !== null) {
        clearTimeout(timer)
      }
      timer = setTimeout(run, 0)
    } catch {
      run()
    }
  }
  const stopGatewayListener = host.state.gateway.listen(state => {
    if (state === 'open') {
      schedule()
    }
  })
  const teardown = () => {
    disposed = true
    stopGatewayListener()
    if (timer !== null) {
      clearTimeout(timer)
      timer = null
    }
  }
  if (typeof ctx.onDispose === 'function') {
    ctx.onDispose(teardown)
  }
  schedule()
}
/** One (owner, session id) pair the sweep will hide, plus the key it dedupes
 *  on when the same member session is seated in several rooms. */
interface RoomSessionEntry {
  dedupe: string
  id: string
  owner: RosterRow
}
function hideOwnedBotSessions() {
  // `.filter(Boolean)` doesn't narrow away the nulls the map returns, so the
  // element type is restated here rather than at every read below.
  const roomEntries = Object.values($groupChats.get()).flatMap(room =>
    Object.entries(room?.sessions || {})
      .map(([key, id]) => {
        if (!id || id === true) {
          return null
        }
        const persisted = room?.sessionOwners?.[key]
        const derived = (room?.members || []).find((member: GroupMember) => groupMemberKey(member) === key)
        // Bare keys are legacy local rooms. A source-qualified key without its
        // immutable owner is unsafe: never let it fall through ambient routing.
        const owner =
          persisted ||
          derived ||
          (!key.includes('::')
            ? {
                name: key
              }
            : null)
        if (key.includes('::')) {
          const route = owner?.route
          const sourceMarked = owner?.sourceScoped || owner?.remoteSource
          const routeKey = route?.connectionId && route?.profile ? `${route.connectionId}::${route.profile}` : ''
          if (!sourceMarked || !route?.targetProfile || routeKey !== key) {
            return null
          }
        }
        return owner
          ? {
              owner,
              id,
              dedupe: `${key}\u0000${id}`
            }
          : null
      })
      .filter(Boolean)
  ) as RoomSessionEntry[]

  // The same member session can appear in several rooms (and legacy rooms can
  // share ids) — hide each (owner, id) pair exactly once.
  const rooms = [...new Map(roomEntries.map(entry => [entry.dedupe, entry])).values()]
  const known = Promise.all(rooms.map(({ owner, id }) => hidePersistedBotSession(owner, id).catch(() => undefined)))
  return Promise.all([known, sweepBotProfileSessions().catch(() => undefined)])
}

/** Reconcile durable visibility through the source's primary REST backend.
 *  Never fall back to requestForBot: that compatibility path activates an
 *  absent profile backend, which is worse than deferring this best-effort sweep. */
function hidePersistedBotSession(bot: RosterRow, sessionId: string, profileOverride = '') {
  if (typeof host.setPersistedSessionHidden !== 'function') {
    return Promise.resolve()
  }
  const route = botConnectionRoute(bot)
  const fallback = String(bot?.name || '').trim() || 'default'
  const profile = profileOverride || backendTargetProfile(route, fallback)
  return Promise.resolve(
    host.setPersistedSessionHidden(route, {
      sessionId,
      profile,
      hidden: true
    })
  )
}

// Titles Bot Mode itself mints for its plumbing sessions. Bot-to-bot CLI
// handoffs (`hermes -p <bot> chat --in ~ -c "Bot Chat" --create-if-missing`)
// create sessions with EXACTLY these titles; the "Group: " prefix is the
// member-session title ensureGroupChatSession has
// used since group chats shipped. Exact/prefix matching is deliberate — a
// user's real conversation inside a bot profile keeps whatever title the
// user gave it and is never touched.
const BOT_MODE_SWEEP_TITLES = new Set(['Bot Chat', 'Agent Inbox'])
const BOT_MODE_SWEEP_MIN_AGE_SECONDS = 5 * 60
function isBotModeSweepTitle(title: null | string | undefined) {
  const t = String(title || '').trim()
  return BOT_MODE_SWEEP_TITLES.has(t) || t.startsWith('Group: ')
}

/** A persisted session row as the sweep reads it. Structural so both the REST
 *  list rows and a raw `session.list` payload satisfy it. */
interface SweepSessionRow {
  id: string
  /** Epoch seconds. */
  started_at?: number
  title?: null | string
}
function isBotModeSweepCandidate(row: SweepSessionRow | null | undefined, nowSeconds = Date.now() / 1000) {
  const startedAt = Number(row?.started_at)
  return (
    row &&
    row.id &&
    isBotModeSweepTitle(row.title) &&
    Number.isFinite(startedAt) &&
    startedAt > 0 &&
    nowSeconds - startedAt >= BOT_MODE_SWEEP_MIN_AGE_SECONDS
  )
}

/** `profiles.list` as Bot Mode reads it. */
interface ProfilesListResult {
  profiles?: RosterRow[]
}

/** Ownership-based sweep: the id-based sweep above only covers sessions the
 *  plugin recorded ($botMeta canonical chats, $groupChats member sids), but
 *  Bot Mode sessions are ALSO minted outside the plugin — bot-to-bot CLI
 *  handoffs ("Agent Inbox" / extra "Bot Chat" rows born visible in a bot's
 *  profile) — and those ids the plugin never learns. So: enumerate each
 *  roster bot's OWN profile sessions (only bot profiles — a non-bot profile
 *  is never listed, so its sessions are never touched) and hide any VISIBLE
 *  row whose title is Bot Mode plumbing and whose creation grace period has
 *  elapsed. The grace period protects a new desktop draft while its first-turn
 *  title is pending; after five minutes an unchanged plumbing title is treated
 *  as Bot Mode-owned. session.list supplies epoch seconds; missing, malformed,
 *  millisecond, or future timestamps fail closed and stay visible. session.list
 *  without include_hidden returns only visible rows, which keeps the sweep
 *  naturally idempotent.
 *  Reads and writes go through the owning source's primary REST backend, which
 *  opens persisted state directly and never starts an inactive profile backend.
 *  Feature-detected + fire-and-forget: older Desktop hosts defer the sweep. */
async function sweepBotProfileSessions(nowSeconds = Date.now() / 1000) {
  if (typeof host.listPersistedSessions !== 'function' || typeof host.setPersistedSessionHidden !== 'function') {
    return
  }
  const cached = $lastRoster.get()
  let roster: RosterRow[] | null = Array.isArray(cached) && cached.length ? cached : null
  if (!roster) {
    // Plugin load can run before the Bots pane hydrates $lastRoster — fall
    // back to the active gateway's own profile list (local bots; remote
    // sources get covered by the next sweep once the roster cache exists).
    try {
      const activeBot = {
        name: String(host.state.profile?.get?.() || 'default').trim() || 'default'
      }
      const res = (await requestForBot(activeBot, 'profiles.list', {})) as ProfilesListResult
      roster = Array.isArray(res?.profiles) ? res.profiles : []
    } catch {
      return
    }
  }
  await Promise.all(
    roster.map(async (bot: RosterRow) => {
      const name = String(bot?.name || '').trim()
      if (!name) {
        return
      }
      try {
        const route = botConnectionRoute(bot)
        const profile = backendTargetProfile(route, name)
        const res = await host.listPersistedSessions(route, {
          profile,
          limit: PROFILE_SESSION_LIST_LIMIT
        })
        const rows = Array.isArray(res?.sessions) ? res.sessions : []
        await Promise.all(
          rows
            .filter(row => isBotModeSweepCandidate(row, nowSeconds))
            .map(row => Promise.resolve(hidePersistedBotSession(bot, row.id, profile)).catch(() => undefined))
        )
      } catch {
        /* older gateway / unreachable source — leave this profile alone */
      }
    })
  )
}
