/**
 * The two roster actions that reach outside a component: the unread/toast
 * poll every roster refresh feeds, and the click path that fronts one exact
 * bot's canonical chat.
 *
 * They sit beside the surfaces rather than inside them — the bot row, the
 * roster pane and the plugin's own lifecycle all invoke them, and none of
 * them can own an action the others call without importing a sibling surface.
 */

import { ackStoredSessionId, atom, haptic, host, markSessionUnreadFinished } from '@hermes/plugin-sdk'

import { $openBotChat, $selectedBot, rosterWatermarks, saveSelectedRosterBot } from './bot-state'
import { notifyBotOpenFailure, openBotCanonicalChat, prepareBotSource } from './canonical-chat'
import { $botMeta, botActivitySession, botRosterKey, botSelectionKey, newBotChat } from './data'
import { $groupChats, $groupChatWorkspace } from './group-chat'
import { openGroupChat } from './group-chat-view'
import { liveGroupChatNames } from './group-membership'
import { closeGroupChatMainTab } from './group-panes'
import { displayName } from './labels'
import { botRosterMeta, botWorkspaceOwnerKey, setBotsWorkspaceOwner } from './routing'
import { botCanonicalSessionId } from './row-helpers'
import { bumpBotOpenGeneration, getBotOpenGeneration, getPluginCtx } from './shared'
import type { RosterRow } from './types'

// last_active watermark per source-qualified bot, seeded on first poll so a
// fresh mount doesn't mark ancient history unread.
let watermarksSeeded = false

/** User pref: toast on every new bot activity. Default OFF — a busy roster
 *  (cron runs, bot-to-bot chatter) turns the toasts into a firehose, and the
 *  unread badge already carries the signal. Persisted via ctx.storage. */
export const $activityToasts = atom(false)

/** Flip the activity-toast pref and persist it. */
export function setActivityToasts(enabled: boolean) {
  $activityToasts.set(enabled)

  try {
    Promise.resolve(getPluginCtx()?.storage?.set?.('activity-toasts', enabled)).catch(() => undefined)
  } catch {
    /* storage unavailable — pref holds for this window only */
  }
}

/** Detect new inbound activity from a fresh roster: last_active moved past
 *  the watermark for a bot whose chat isn't on screen -> unread + toast.
 *  Watermarks follow botActivitySession (canonical Bot Chat included) —
 *  last_session alone never sees the hidden Bot Chat, so DMs delivered
 *  there would neither badge nor toast.
 *
 *  This poll is the ONLY unread signal a canonical Bot Chat can have: it is
 *  unconditionally hidden, so it never reaches the session list the backend's
 *  own unread watermark iterates, and deliveries from the CLI, cron, another
 *  bot, or another machine never touch this window's live turn edge either. */
export function trackInboundActivity(roster: RosterRow[]) {
  const seeding = !watermarksSeeded
  watermarksSeeded = true

  for (const bot of roster) {
    const key = botSelectionKey(bot)
    const activity = botActivitySession(bot)
    const ts = activity?.last_active || 0
    const prev = rosterWatermarks.get(key) || 0
    rosterWatermarks.set(key, Math.max(prev, ts))

    if (seeding || ts <= prev) {
      continue
    }

    // Activity in the exact bot owner the user is currently looking at is
    // already visible — never badge the open chat or its same-named twin.
    if ($selectedBot.get() === key) {
      continue
    }

    // Straight into core's unread store, keyed by the same canonical id the
    // row's SessionStatusDot reads — a parallel map here would be a second
    // badge that drifts from the dot.
    const canonicalSessionId = botCanonicalSessionId(bot)

    if (canonicalSessionId) {
      markSessionUnreadFinished(canonicalSessionId, bot.name)
    }

    // Roster-hidden bots stay quiet: the mark above accumulates silently
    // (unhiding reveals the dot) but a hidden bot never toasts.
    if (botRosterMeta(bot, $botMeta.get())?.hidden) {
      continue
    }

    // Toasts are opt-in: the unread mark is recorded above regardless, but the
    // per-message notification fires only when the user enabled it.
    if ($activityToasts.get()) {
      const meta = botRosterMeta(bot, $botMeta.get())
      const label = displayName(bot, meta)
      const preview = (activity?.preview || '').trim()
      const inbound = /^Message from/i.test(preview)
      host.notify({
        kind: 'info',
        title: inbound ? `\uD83E\uDD16 New message for ${label}` : `${label} has new activity`,
        message: preview.slice(0, 140) || 'Open the chat to see it.'
      })
    }
  }
}

/** Select one exact roster owner, then open its named canonical chat only when
 *  the current Desktop can route that owner without guessing. The workspace
 *  remembers only this transient opened-view observation; it never stores or
 *  resolves a canonical-chat id. */
export async function openRosterBot(bot: RosterRow): Promise<boolean> {
  const generation = bumpBotOpenGeneration()
  const key = botRosterKey(bot)
  const meta = botRosterMeta(bot, $botMeta.get())
  // Keep the currently visible group as a fallback until this explicit action
  // has actually fronted a new owner; a failed open must not steal the center
  // from a group the user was reading.
  const previousGroup = $groupChatWorkspace.get()

  const previousGroupRef = previousGroup
    ? {
        group: previousGroup,
        roomId: String($groupChats.get()[previousGroup]?.roomId || '')
      }
    : null

  haptic('tap')
  saveSelectedRosterBot(bot)
  setBotsWorkspaceOwner(botWorkspaceOwnerKey(bot), bot)
  const dismissedGroup = bot.remoteSource ? null : dismissGroupChatForLocalBotOpen()

  if (!dismissedGroup) {
    $groupChatWorkspace.set(null)
  }

  const restorePreviousGroup = () => {
    if (!previousGroupRef || $groupChatWorkspace.get()) {
      return
    }

    const restoreRef = dismissedGroup || previousGroupRef
    const rooms = $groupChats.get()

    const currentGroup = restoreRef.roomId
      ? Object.keys(rooms).find(
          name => !rooms[name]?.tombstone && String(rooms[name]?.roomId || '') === restoreRef.roomId
        )
      : liveGroupChatNames().includes(restoreRef.group)
        ? restoreRef.group
        : null

    if (!currentGroup) {
      return
    }

    openGroupChat(currentGroup)
  }

  // The persisted half of clear-on-open. The transient dot is retired by
  // core's own selection path once the chat lands; this retires the marker,
  // which the selection listener alone would file against the wrong profile —
  // a bot open deliberately leaves the gateway on the launch profile.
  ackStoredSessionId(botCanonicalSessionId(bot), bot.name)

  try {
    // Activation selects this row's source only. Canonical identity is resolved
    // after that by the owner profile's "Bot Chat" title registry.
    await prepareBotSource(bot)
  } catch (error) {
    if (generation === getBotOpenGeneration()) {
      $openBotChat.set(null)
      restorePreviousGroup()
      notifyBotOpenFailure(error, bot, `Could not reach ${bot.connectionLabel || 'the gateway'}`)
    }

    return false
  }

  if (generation !== getBotOpenGeneration()) {
    return false
  }

  try {
    const opened = await openBotCanonicalChat(bot, () => generation === getBotOpenGeneration())

    if (generation !== getBotOpenGeneration()) {
      return false
    }

    if (opened) {
      // This is not an identity preference: opening already completed through
      // the name registry. Keep only enough ephemeral state to release the
      // claim if another tab later claims the center. Track BOTH identities —
      // session focus reports the compression-lineage tip (openedId), not the
      // durable registry row, and matching focus against the registry id
      // alone released this claim on the first click of every compressed
      // Bot Chat.
      $openBotChat.set({
        key,
        openedRegistryId: opened.registryId,
        openedSessionId: opened.openedId
      })

      return true
    }
  } catch (error) {
    if (generation === getBotOpenGeneration()) {
      $openBotChat.set(null)
      restorePreviousGroup()
      notifyBotOpenFailure(error, bot, `Could not open ${displayName(bot, meta)}'s chat — try again`)
    }

    return false
  }

  // An older Desktop without the profile-scoped draft API has no safe fallback:
  // do not navigate the current workspace or create a draft on the wrong owner.
  if (typeof host.newChat !== 'function') {
    $openBotChat.set(null)
    restorePreviousGroup()

    return false
  }

  $openBotChat.set({
    key,
    openedRegistryId: ''
  })
  newBotChat(bot)

  return true
}

/** Bot-open handoff: capture the selected group and retire its registered
 * main tab (or clear the in-panel selection) before async source prep /
 * canonical open. */
function dismissGroupChatForLocalBotOpen(): null | { group: string; roomId: string } {
  const group = $groupChatWorkspace.get()

  if (!group) {
    return null
  }

  const roomId = String($groupChats.get()[group]?.roomId || '')
  closeGroupChatMainTab(group)

  return {
    group,
    roomId
  }
}
