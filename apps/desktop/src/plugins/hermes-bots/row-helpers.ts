/**
 * The row-level reads a roster row is assembled from: who sent the last
 * message, whether a bot counts as live, and whether its row owns the
 * highlight.
 *
 * Pure below the surfaces. Every one of these takes a roster row and returns
 * a value, so the bot row, the roster pane and the open path can share one
 * answer instead of each deriving its own.
 */

import { botActivitySession, botHandle, botRosterKey, isActiveRosterBot } from './data'
import type { RosterActivityFilter, RosterRow } from './types'

// ── human-readable row helpers ───────────────────────────────────────────────

/** Bot-to-bot delivery prefix (see messagingProtocolSection): either the
 *  current "Message from 🤖 name (@handle):" form or the older
 *  "[Message from agent 'name']" shape. Captures the sender's handle. */
const A2A_RE = /^Message from (?:agent '([^']+)'|🤖\s*([^\s(@]+))/i

/** Strip the delivery prefix so a DM preview reads like a DM, not a log line. */
export const A2A_PREFIX_RE = /^Message from (?:agent '[^']+'|🤖[^:]+):\s*/i

/** Classify a roster preview: `{ fromBot: handle|null }`. A preview that
 *  starts with the delivery prefix is a bot-to-bot message — the receiving
 *  bot's row should show WHO sent it, not present it as the human's chat. */
export function previewKind(preview: null | string | undefined): { fromBot: null | string } {
  const text = (preview || '').trim()

  if (!text) {
    return {
      fromBot: null
    }
  }

  const match = text.match(A2A_RE)

  if (match) {
    // The captured name is whatever the delivery prefix carried — a raw
    // profile name. Map it the way every other surface does so the primary
    // profile reads @hermes, never @default (#89484).
    const sender = (match[1] || match[2] || '').trim().toLowerCase()

    return {
      fromBot: sender ? botHandle(sender) : null
    }
  }

  return {
    fromBot: null
  }
}

/** Roster liveness window: a bot whose last message landed within this many
 *  seconds is treated as "active now" (pulsing dot in its row). */
export const ACTIVE_WINDOW_S = 90
const RECENT_ACTIVITY_WINDOW_S = 7 * 24 * 60 * 60
export const BOT_ROSTER_SEARCH_THRESHOLD = 8

/** The stored session id this bot's canonical Bot Chat answers to — the
 *  compression-lineage tip the live-state atoms are keyed by, falling back to
 *  the durable registry id. THE id for anything core-keyed: the row's status
 *  dot reads it and the unread writes below key by it, so the two can never
 *  describe different conversations. Deliberately NOT botActivitySession —
 *  that may answer with `last_session`, a bare preview with no id belonging to
 *  another conversation entirely. */
export function botCanonicalSessionId(bot: null | RosterRow | undefined): null | string {
  return bot?.canonical_session?.resolved_id ?? bot?.canonical_session?.id ?? null
}

/** Worker liveness window: kanban/tool workers heartbeat last_activity_at
 *  at least every 60s while running (agent/session_activity.py), so a
 *  worker whose stamp is older than this is finished or stalled. Wider
 *  than ACTIVE_WINDOW_S to bridge one missed heartbeat. */
const WORKER_ACTIVE_WINDOW_S = 150

/** True while this bot's freshest kanban/tool worker looks alive. Workers
 *  never surface in conversation lists, so without this a profile grinding
 *  through a 30-minute kanban task reads idle ("3 hr ago") the whole time
 *  (hermes-agent#90268). Older gateways omit worker_session — always false. */
export function workerActiveAt(bot: null | RosterRow | undefined, now = Date.now()): boolean {
  const ts = bot?.worker_session?.last_active || 0

  return Boolean(ts && now / 1000 - ts < WORKER_ACTIVE_WINDOW_S)
}

/** Bots that are working right now: the profile the gateway is running a
 *  turn for (busy), any bot whose last message landed inside the liveness
 *  window, any live kanban/tool worker, plus members on a group turn (keys
 *  from `activeGroupMemberKeys`). Pure — output follows the input roster's
 *  order, so presence never reorders or hides the normal list. */
export function activeBots(
  roster: null | RosterRow[] | undefined,
  activeProfile: string,
  gatewayState: string,
  now = Date.now(),
  groupKeys: string[] = []
): RosterRow[] {
  const groupTurnKeys = new Set(groupKeys)

  return (roster || []).filter(bot => {
    const busyTurn = !bot.remoteSource && bot.name === activeProfile && gatewayState === 'busy'
    const last = botActivitySession(bot)?.last_active || 0
    const inWindow = Boolean(last && now / 1000 - last < ACTIVE_WINDOW_S)
    const groupTurn = groupTurnKeys.has(botRosterKey(bot))

    return busyTurn || inWindow || workerActiveAt(bot, now) || groupTurn
  })
}

/** What the roster list wraps around a bot or group before filtering:
 *  `activity` is a millisecond stamp, `active` the live pulse. */
interface RosterActivityRow {
  active?: boolean
  activity?: number
}

export function rosterActivityMatches(
  row: null | RosterActivityRow | undefined,
  filter: null | RosterActivityFilter | undefined,
  now = Date.now()
): boolean {
  if (!filter || filter === 'all') {
    return true
  }

  if (filter === 'active') {
    return Boolean(row?.active)
  }

  const activity = Number(row?.activity || 0)
  const recent = Boolean(activity && now - activity <= RECENT_ACTIVITY_WINDOW_S * 1000)

  return filter === 'recent' ? recent : !recent
}

export function botRowOwnsWorkspace(
  bot: RosterRow,
  activeGroup: null | string,
  botChatFocused: boolean,
  focusedOwner: null | { authoritative: boolean; connectionId: string; name: string },
  selectedRosterKey: string
): boolean {
  if (activeGroup) {
    return false
  }

  if (!botChatFocused) {
    return selectedRosterKey === botRosterKey(bot)
  }

  return isActiveRosterBot(bot, focusedOwner)
}
