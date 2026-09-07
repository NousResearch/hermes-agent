/** Pure presentation helpers for the Sessions bot pinboard. */

import { botActivitySession, botRosterKey } from './data'
import type { BotMetaSnapshot } from './data'
import { botRosterMeta } from './routing'
import type { BotMeta, RosterRow } from './types'

/** Read server metadata immediately, before the roster hydration effect has
 * copied it into the local snapshot. The fallback keeps a cold-start pinboard
 * useful without weakening source scoping in `botRosterMeta`. */
export function pinboardMeta(bot: RosterRow, metaByName: BotMetaSnapshot): BotMeta | null {
  return botRosterMeta(bot, metaByName) || bot.ui_meta?.['hermes-bots'] || null
}

function activityAt(bot: RosterRow, metaByName: BotMetaSnapshot): number {
  const created = pinboardMeta(bot, metaByName)?.created || bot.ui_meta?.['hermes-bots']?.created || 0
  const lastMessage = (botActivitySession(bot)?.last_active || 0) * 1000

  return Math.max(created, lastMessage)
}

/** Return displayable bots in pinboard order. Pinned bots lead, then the most
 * recently active bots fill the rail. The source-qualified roster key is the
 * final tie-breaker so two `default` profiles never collapse or jump around. */
export function pinnedBotRows(
  roster: readonly RosterRow[],
  metaByName: BotMetaSnapshot,
  limit = Number.POSITIVE_INFINITY
): RosterRow[] {
  return roster
    .filter(bot => {
      const meta = pinboardMeta(bot, metaByName)

      return !bot?.ghost && !meta?.hidden
    })
    .slice()
    .sort((a, b) => {
      const pinnedOrder =
        Number(Boolean(pinboardMeta(b, metaByName)?.pinned)) - Number(Boolean(pinboardMeta(a, metaByName)?.pinned))

      if (pinnedOrder) {
        return pinnedOrder
      }

      const activityOrder = activityAt(b, metaByName) - activityAt(a, metaByName)

      return activityOrder || botRosterKey(a).localeCompare(botRosterKey(b))
    })
    .slice(0, Number.isFinite(limit) ? Math.max(0, limit) : undefined)
}
