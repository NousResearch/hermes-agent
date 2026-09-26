/**
 * "Open recent session" (hermes-agent#93054): a bot row's left click always
 * lands on the canonical Bot Chat — that is the bot's stable identity and
 * stays so. Multi-task users also want the freshest ORDINARY conversation
 * (a cron run, a delegated job, a `+` side thread) one gesture away, so the
 * row's context menu opens the bot's most recently active listed session.
 *
 * No new backend surface: `profiles.list` already ships `last_session` — the
 * newest row that `session.list` would show (hidden rows such as the Bot Chat
 * and group member sessions never win, tool/kanban workers are denied
 * server-side). Opened through the same `host.openSession` contract the
 * canonical open uses, in the bot's own workspace and connection route.
 */

import * as sdk from '@hermes/plugin-sdk'
import { haptic, host } from '@hermes/plugin-sdk'

import { saveSelectedRosterBot } from './bot-state'
import { prepareBotSource } from './canonical-chat'
import { openRosterBot } from './roster-actions'
import { botConnectionRoute, botWorkspaceOwnerKey, setBotsWorkspaceOwner } from './routing'
import type { RosterRow } from './types'

/** The listed session the menu item would open, or null when the profile has
 *  none yet (a brand-new bot whose only conversation is its hidden Bot Chat). */
export function botRecentSession(bot: null | RosterRow | undefined): null | { id: string; title: string } {
  const last = bot?.last_session
  const id = String(last?.id || '').trim()

  return id ? { id, title: String(last?.title || '').trim() } : null
}

/** Open the bot's most recent listed session as a tab in its workspace.
 *  Falls back to the row click (canonical chat) when nothing is listable. */
export async function openBotRecentSession(bot: RosterRow): Promise<boolean> {
  const recent = botRecentSession(bot)

  if (!recent || typeof host.openSession !== 'function') {
    return openRosterBot(bot)
  }

  haptic('tap')
  saveSelectedRosterBot(bot)
  const ownerKey = botWorkspaceOwnerKey(bot)
  setBotsWorkspaceOwner(ownerKey, bot)

  try {
    await prepareBotSource(bot)
    const route = botConnectionRoute(bot)

    await host.openSession(recent.id, {
      ...(route ? { route } : {}),
      profile: bot.name,
      // A tab beside the Bot Chat, never in its place: the canonical chat
      // stays where the row click expects it (same shape as `+` side threads).
      intent: 'tab',
      awaitHydration: true,
      expectHistory: (bot.last_session?.message_count ?? 1) > 0,
      forceResume: true,
      hydrationTimeoutMs: Number.isFinite(sdk.BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS)
        ? sdk.BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS
        : 60_000,
      keepAllProfilesScope: true,
      workspaceMode: 'bots',
      workspaceOwnerKey: ownerKey,
      // No tabTitle: the stored row supplies its own title. Passing the same
      // string as tabTitle would make the tab caption read as the bot's name
      // (workspaceOwnerTitle treats title === workspaceTabTitle as the
      // canonical chat), hiding which session this tab is.
      retryHydrationTimeoutOnce: true
    })

    return true
  } catch (error) {
    host.notifyError?.(error, 'Could not open the recent session')

    return false
  }
}
