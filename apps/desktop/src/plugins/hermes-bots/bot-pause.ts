/**
 * Pause / Resume a bot and its per-bot notification level.
 *
 * Both live on the bot's own meta (`profile.yaml` `ui_meta['hermes-bots']`),
 * so they follow the profile to every desktop. `paused` is ALSO read by the
 * bot's backend: its scheduled jobs stop firing (each missed slot is skipped,
 * so a resume never bursts) and a teammate's `message_agent` gets a typed
 * `recipient_paused` refusal. The user can still open and chat with a paused
 * bot — pause stops the work it does on its own, not the user's conversation.
 * Hide stays display-only.
 *
 * `notifications` filters this Desktop's native notifications for the bot's
 * sessions: 'all' (the default, today's behaviour), 'needs-you' (approval,
 * clarify, a failed turn) or 'off'. Core applies it through the
 * `host.sessions.registerNotificationLevel` door; this module answers per bot.
 */

import { host } from '@hermes/plugin-sdk'

import { ensureBotMetadata } from './canonical-chat'
import {
  $botMeta,
  $lastRoster,
  type BotMetaSnapshot,
  botMetaV2Active,
  botOwner,
  cachedUnionRoster,
  persistBotMetaSnapshot,
  saveBotMeta
} from './data'
import { botsText } from './i18n'
import { displayName } from './labels'
import { botRosterMeta } from './routing'
import type { BotNotificationLevel, RosterRow } from './types'

export const BOT_NOTIFICATION_LEVELS: readonly BotNotificationLevel[] = ['all', 'needs-you', 'off']

/** Each level's menu label key under `b.pause`. */
export const BOT_NOTIFICATION_LEVEL_LABELS = {
  all: 'levelAll',
  'needs-you': 'levelNeedsYou',
  off: 'levelOff'
} as const satisfies Record<BotNotificationLevel, string>

export function isBotPaused(bot: RosterRow, metaByName: BotMetaSnapshot): boolean {
  return botRosterMeta(bot, metaByName)?.paused === true
}

export function botNotificationLevel(bot: RosterRow, metaByName: BotMetaSnapshot): BotNotificationLevel {
  const level = botRosterMeta(bot, metaByName)?.notifications

  return level && BOT_NOTIFICATION_LEVELS.includes(level) ? level : 'all'
}

/** Undo a failed pause save in this window's cache and its local storage
 *  (the save stored the attempt there, and a restart would load it back). A
 *  second server write would be a fresh save that could overwrite another
 *  window's newer pause, and a value someone changed since (`wrote` no longer
 *  cached) stays. */
export function revertCachedPause(bot: RosterRow | string, wrote: boolean, previous: boolean): Promise<void> {
  const { key, route } = botOwner(bot)
  const snapshot = $botMeta.get()
  const entry = snapshot[key]

  if (entry?.paused !== wrote) {
    return Promise.resolve()
  }

  const next = { ...snapshot, [key]: { ...entry, paused: previous } }

  $botMeta.set(next)

  return persistBotMetaSnapshot(next, Boolean(route) || botMetaV2Active)
}

/** Flip the bot's pause. The backend enforces it, so a save that never
 *  reached the bot's gateway is reported and rolled back, never shown as done. */
export async function toggleBotPaused(bot: RosterRow): Promise<void> {
  const text = botsText().pause

  try {
    const current = await ensureBotMetadata(bot)
    const name = displayName(bot, current)
    const paused = current.paused === true
    const saved = await saveBotMeta(bot, { paused: !paused })

    if (!saved.serverPersisted) {
      await revertCachedPause(bot, !paused, paused)
      host.notify({ kind: 'error', message: text.saveFailed(name) })

      return
    }

    if (!paused && !saved.pauseEnforced) {
      // An older gateway stored the flag but runs the bot's routines and
      // teammate messages anyway: take the inert pause back off rather than
      // show a paused bot that is still working.
      await saveBotMeta(bot, { paused: false })
      host.notify({ kind: 'error', message: text.updateRequired(name) })

      return
    }

    host.notify({ kind: 'info', message: paused ? text.resumedToast(name) : text.pausedToast(name) })
  } catch (error) {
    host.notifyError?.(error, botsText().bot.metadataLoadFailed)
  }
}

export async function setBotNotificationLevel(bot: RosterRow, level: BotNotificationLevel): Promise<void> {
  try {
    await ensureBotMetadata(bot)
    await saveBotMeta(bot, { notifications: level })
  } catch (error) {
    host.notifyError?.(error, botsText().bot.metadataLoadFailed)
  }
}

interface NotificationLevelQuery {
  owner: null | { connectionId: null | string; profile: string; targetProfile?: string }
  storedSessionId: string
}

const isLocalRow = (bot: RosterRow, activeConnectionId: string) =>
  !bot.connectionId || bot.connectionId === 'local' || bot.connectionId === activeConnectionId

/** The roster row a session belongs to: its owner `(connection, profile)`,
 *  else the bot whose canonical Bot Chat it is. Two same-named bots on
 *  different connections never answer for each other. An alias owner (a
 *  Desktop name routed to a differently named backend profile) also matches
 *  that backend's row, so its level survives the roster swapping rows. */
export function botForSession(roster: RosterRow[], query: NotificationLevelQuery, activeConnectionId: string) {
  const { owner, storedSessionId } = query

  if (owner) {
    const byName = roster.filter(bot => (bot.name || 'default') === owner.profile)
    const target = owner.targetProfile

    const named =
      byName.length || !target || target === owner.profile
        ? byName
        : roster.filter(bot => (bot.targetProfile || bot.name || 'default') === target)

    return owner.connectionId
      ? named.find(bot => bot.connectionId === owner.connectionId) ||
          (owner.connectionId === activeConnectionId ? named.find(bot => !bot.connectionId) : undefined)
      : named.find(bot => isLocalRow(bot, activeConnectionId))
  }

  return roster.find(bot =>
    [bot.canonical_session?.id, bot.canonical_session?.resolved_id].includes(storedSessionId || undefined)
  )
}

/** The pane publishes `$lastRoster` only once it mounts; before that (a launch
 *  restored onto Sessions) the roster cache primed at gateway open answers. */
function notificationRoster(): RosterRow[] {
  const published = $lastRoster.get()

  if (published.length) {
    return published
  }

  const cached = cachedUnionRoster()?.profiles

  return Array.isArray(cached) ? cached.filter(row => !row?.ghost) : []
}

export function resolveBotNotificationLevel(query: NotificationLevelQuery): BotNotificationLevel | null {
  const activeConnectionId = String(host.state.connectionId?.get?.() || 'local').trim()
  const bot = botForSession(notificationRoster(), query, activeConnectionId)

  return bot ? botNotificationLevel(bot, $botMeta.get()) : null
}

/** Wire the per-bot level into core's native notifications; returns the
 *  disposer. Feature-detected: an older shell keeps its per-kind toggles. */
export function startBotNotificationLevels(): () => void {
  const register = host.sessions?.registerNotificationLevel

  return typeof register === 'function' ? register(resolveBotNotificationLevel) : () => undefined
}
