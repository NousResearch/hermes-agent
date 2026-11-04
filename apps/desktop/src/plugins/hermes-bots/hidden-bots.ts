/**
 * Roster-display hiding: the session-only reveal toggle, the two per-row
 * predicates every surface reads, and the selection re-home a hide performs.
 *
 * A leaf by design. Hiding is presentation, so it knows nothing about the
 * rows that render it — the bot row, the roster pane and selection
 * reconciliation all read these and none of them owns the state.
 */

import { atom } from '@hermes/plugin-sdk'

import { $selectedBot } from './bot-state'
import { $botMeta, $lastRoster, botSelectionKey, isDefaultBot } from './data'
import type { BotMetaSnapshot } from './data'
import { botRosterMeta } from './routing'
import type { RosterRow } from './types'

// ── hidden bots (right-click → Hide Bot) ────────────────────────────────────
// Hiding is a ROSTER-DISPLAY concern only: a hidden bot keeps working,
// remains mentionable, keeps group membership, and any open chat stays open.

/** Session-only view toggle: reveal hidden bots (dimmed) in the roster. */
export const $showHiddenBots = atom(false)

export function isBotHidden(bot: RosterRow, metaByName: BotMetaSnapshot) {
  return Boolean(botRosterMeta(bot, metaByName)?.hidden)
}

export function isBotPinned(bot: RosterRow, metaByName: BotMetaSnapshot) {
  return Boolean(botRosterMeta(bot, metaByName)?.pinned)
}

/** Hiding the selected bot re-homes the selection to the next visible owner. */
export function fallbackSelectionAfterHide(name: string) {
  if ($selectedBot.get() !== name) {
    return
  }

  const meta = $botMeta.get()
  const visible = $lastRoster.get().filter(bot => botSelectionKey(bot) !== name && !botRosterMeta(bot, meta)?.hidden)

  if (visible.length) {
    $selectedBot.set(botSelectionKey(visible[0]))

    return
  }

  const defaultBot = $lastRoster.get().find(bot => isDefaultBot(bot) && !botRosterMeta(bot, meta)?.hidden)

  if (defaultBot && botSelectionKey(defaultBot) !== name) {
    $selectedBot.set(botSelectionKey(defaultBot))
  } else if (!$lastRoster.get().some(isDefaultBot)) {
    // Legacy sole-local rosters can transiently omit the default row. Keep the
    // historic fallback rather than leaving Routines attached to a hidden bot.
    $selectedBot.set('default')
  }
}
