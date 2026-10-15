/**
 * A bot's role: the short job label ("Deep Research") shown beside its name and
 * listed next to it in teammates' rosters. `title` stays the NAME; the profile
 * `description` stays the long prose. A leaf by design: the roster row, the
 * edit/create dialogs and the save patch all read the same normalisation.
 */

import type { BotMeta, RosterRow } from './types'

/** Longest stored role. Mirrored by `_ROLE_MAX` in `tools/bot_mode_probe.py`. */
export const BOT_ROLE_MAX = 40

/** A role as stored: one line, trimmed, capped to `BOT_ROLE_MAX` characters. */
export function normalizeBotRole(value: unknown): string {
  if (typeof value !== 'string') {
    return ''
  }

  return Array.from(value.replace(/\s+/g, ' ').trim()).slice(0, BOT_ROLE_MAX).join('').trim()
}

/** The role a row shows: this desktop's meta first, else what the row's own
 *  backend reports in `ui_meta` (rows from another connection). '' when unset. */
export function botRole(bot: Partial<RosterRow> | null | undefined, meta?: BotMeta | null): string {
  return normalizeBotRole(meta?.role) || normalizeBotRole(bot?.ui_meta?.['hermes-bots']?.role)
}
