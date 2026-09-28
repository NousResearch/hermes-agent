/**
 * The two roster actions that reach outside a component: the unread/toast
 * poll every roster refresh feeds, and the click path that fronts one exact
 * bot's canonical chat.
 *
 * They sit beside the surfaces rather than inside them — the bot row, the
 * roster pane and the plugin's own lifecycle all invoke them, and none of
 * them can own an action the others call without importing a sibling surface.
 */

import { ackStoredSessionId, atom, haptic, host, markSessionRead, markSessionUnreadFinished } from '@hermes/plugin-sdk'

import {
  $openBotChat,
  $pendingBotOpen,
  $selectedBot,
  lastToastedPreview,
  rosterWatermarks,
  saveSelectedRosterBot
} from './bot-state'
import { isStaleBotChatTile, notifyBotOpenFailure, openBotCanonicalChat, prepareBotSource } from './canonical-chat'
import { $botMeta, $lastRoster, botActivitySession, botRosterKey, botSelectionKey, newBotChat } from './data'
import { $groupChats, $groupChatWorkspace } from './group-chat'
import { openGroupChat } from './group-chat-view'
import { liveGroupChatNames } from './group-membership'
import { closeGroupChatMainTab } from './group-panes'
import { displayName } from './labels'
import { botRosterMeta, botWorkspaceOwnerKey, setBotsWorkspaceOwner } from './routing'
import { botCanonicalSessionId } from './row-helpers'
import { bumpBotOpenGeneration, getBotOpenGeneration, getPluginCtx } from './shared'
import type { RosterRow } from './types'

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
  for (const bot of roster) {
    const key = botSelectionKey(bot)
    const activity = botActivitySession(bot)
    const ts = activity?.last_active || 0
    const seeded = rosterWatermarks.has(key)
    const prev = rosterWatermarks.get(key) || 0
    rosterWatermarks.set(key, Math.max(prev, ts))

    if (!seeded || ts <= prev) {
      // Seed (or refresh) the last-toasted preview so a fresh mount, or a row
      // whose activity hasn't advanced, treats current content as already-seen
      // rather than replaying it — or a busy bridge's unchanged preview — as a
      // duplicate toast.
      lastToastedPreview.set(key, (activity?.preview || '').trim())

      continue
    }

    // Roster selection survives a group switch and a retained Bot Chat tab.
    // Only the visible chat consumes this activity; the group hides it.
    if ($selectedBot.get() === key && !$groupChatWorkspace.get()) {
      refreshOpenBotChat(bot)

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
    const preview = (activity?.preview || '').trim()

    // Content-level dedup, tracked independently of the toast pref so the
    // memory stays accurate whether or not toasts are on: skip re-surfacing an
    // identical preview a busy bridge keeps re-pinging (last_active advances
    // but the visible content is unchanged). Unread marking above is unaffected.
    if (lastToastedPreview.get(key) === preview) {
      continue
    }

    lastToastedPreview.set(key, preview)

    if ($activityToasts.get()) {
      const meta = botRosterMeta(bot, $botMeta.get())
      const label = displayName(bot, meta)
      const inbound = /^Message from/i.test(preview)
      host.notify({
        kind: 'info',
        title: inbound ? `\uD83E\uDD16 New message for ${label}` : `${label} has new activity`,
        message: preview.slice(0, 140) || 'Open the chat to see it.'
      })
    }
  }
}

// Focus epochs distinguish leaving and returning to the same cached tab. A
// roster click and its focus edge share one flight rather than racing two reads.
let botFocusEpoch = 0
let focusedRefresh: { key: string; run: ReturnType<typeof openBotCanonicalChat> } | null = null

/** Tab-strip/keyboard focus bypasses openRosterBot. Resolve the focused owner,
 * not the selected roster row, after the tree's derived stores have settled. */
export function refreshBotChatOnFocus(focusedId: null | string | undefined): void {
  const epoch = ++botFocusEpoch

  if (!focusedId) {
    return
  }

  queueMicrotask(() => {
    if (epoch !== botFocusEpoch || host.state.focusedStoredSessionId?.get?.() !== focusedId) {
      return
    }

    const owner = host.state.focusedSessionOwner?.get?.()

    const bot = owner && $lastRoster.get().find(row =>
      botRosterKey(row) === `${owner.connectionId}::${owner.profile}` &&
      [row.canonical_session?.id, row.canonical_session?.resolved_id].includes(focusedId)
    )

    if (bot) {
      void refreshOpenBotChat(bot, { allowWhileBusy: true })
    }
  })
}

/** Reconcile off-window deliveries (#99393) on roster activity, roster click,
 * or direct tab focus. All paths share hydration-before-ack and identity fences.
 * Background refreshInPlace never navigates (#121874); activity polls also skip
 * busy chats, whose current turn is already streaming. */
function refreshOpenBotChat(bot: RosterRow, { allowWhileBusy = false }: { allowWhileBusy?: boolean } = {}) {
  const canonicalIds = [bot.canonical_session?.id, bot.canonical_session?.resolved_id].filter(Boolean).map(String)
  const focused = String(host.state.focusedStoredSessionId?.get?.() || '')
  const owner = host.state.focusedSessionOwner?.get?.()
  const key = botRosterKey(bot)

  if (
    // A cold roster open owns its own awaited hydration. Its focus edge can
    // arrive before that open completes; a second background SDK open would
    // advance openSessionGeneration and reject the foreground wait as superseded.
    $pendingBotOpen.get()?.key === key ||
    !focused || !canonicalIds.includes(focused) || (!allowWhileBusy && host.state.busy.get()) ||
    (owner && key !== `${owner.connectionId}::${owner.profile}`)
  ) {
    return
  }

  const generation = getBotOpenGeneration()
  const epoch = botFocusEpoch
  const activity = rosterWatermarks.get(botSelectionKey(bot))
  const flightKey = JSON.stringify([key, focused, generation, epoch, activity])

  if (focusedRefresh?.key === flightKey) {
    return focusedRefresh.run
  }

  const stillCurrent = () => {
    const currentOwner = host.state.focusedSessionOwner?.get?.()

    return generation === getBotOpenGeneration() && epoch === botFocusEpoch &&
      activity === rosterWatermarks.get(botSelectionKey(bot)) &&
      host.state.focusedStoredSessionId?.get?.() === focused &&
      (!owner || (currentOwner?.connectionId === owner.connectionId && currentOwner?.profile === owner.profile))
  }

  const run = openBotCanonicalChat(bot, { background: true, openingStillCurrent: stillCurrent })
    .then(opened => {
      if (opened && stillCurrent() && opened.registryId === String(bot.canonical_session?.id) && opened.openedId === focused) {
        markSessionRead(focused)
        ackStoredSessionId(focused, bot.name)
      }

      return opened
    })
    .catch(() => null) // A later user focus/click can retry; failure keeps unread.
    .finally(() => {
      if (focusedRefresh?.run === run) {
        focusedRefresh = null
      }
    })

  focusedRefresh = { key: flightKey, run }

  return run
}

/** Release the pending-open mark, but only for the flight that set it: a
 *  superseded flight settling late must not clear its successor's mark. */
function settlePendingBotOpen(generation: number) {
  if ($pendingBotOpen.get()?.generation === generation) {
    $pendingBotOpen.set(null)
  }
}

/** Front the bot's canonical Bot Chat when it is ALREADY open as a tab —
 *  presentation only, no registry round-trip. Returns the fronted stored id,
 *  or null when the chat is not on screen (or this shell cannot tell) and the
 *  caller must resolve the registry.
 *
 *  Only the canonical chat qualifies: a tile whose stored id is the roster's
 *  server-resolved `canonical_session` (registry row or its lineage tip). An
 *  earlier version fronted whatever bots-workspace tab the user last had
 *  active — a `+` side thread persisted in Local Storage across restarts and
 *  won every click forever while the row kept previewing the Bot Chat, so
 *  sidebar and center described two different conversations ("[Bots] -
 *  Sessions is not in sync again"). Side tabs stay open; they never answer a
 *  click aimed at the bot. Canonical-titled tiles at a foreign id are stale
 *  (hermes-agent#90102) and are discarded. Without `canonical_session` (older
 *  gateway) nothing can be verified, so nothing is fronted. */
function focusExistingBotTab(bot: RosterRow): null | { registryId: string; storedSessionId: string } {
  if (typeof host.focusOpenWorkspaceSession !== 'function') {
    return null
  }

  const canonical = bot?.canonical_session
  const canonicalIds = [canonical?.id, canonical?.resolved_id].filter(Boolean).map(String)

  if (canonicalIds.length === 0) {
    return null
  }

  try {
    const focused = host.focusOpenWorkspaceSession(
      botWorkspaceOwnerKey(bot),
      isStaleBotChatTile(canonicalIds),
      canonicalIds
    )

    return typeof focused === 'string' && focused
      ? { registryId: String(canonical!.id), storedSessionId: focused }
      : null
  } catch {
    return null
  }
}

/** Select one exact roster owner and open its canonical Bot Chat — the same
 *  session the row previews. Resolution always goes through the owner
 *  profile's "Bot Chat" title registry: an already-open canonical tab is
 *  fronted (focusExistingBotTab), otherwise openBotCanonicalChat resolves and
 *  opens it in place; side tabs the user opened with `+` stay open beside it. A click never fronts a side tab: an
 *  earlier "return to the last open tab" shortcut left the center on a `+`
 *  thread (persisted in Local Storage across restarts) while the row kept
 *  previewing the Bot Chat — sidebar and center described two different
 *  conversations ("[Bots] - Sessions is not in sync again"). The workspace
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
  const dismissedGroup = dismissGroupChatForBotOpen()

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

  const fronted = focusExistingBotTab(bot)

  if (fronted) {
    // The canonical chat is on screen: no source activation, no registry
    // round-trip. Both identities are recorded so the reclaim listener and
    // the roster-activity refresh treat it exactly like a registry open.
    $openBotChat.set({ key, openedRegistryId: fronted.registryId, openedSessionId: fronted.storedSessionId })
    // Front immediately, but do not acknowledge the cached transcript. A
    // background in-place re-resume actually waits for the tile's REST merge;
    // failure (or a newer click/activity/focus) keeps its unread marker.
    void refreshOpenBotChat(bot, { allowWhileBusy: true })

    return true
  }

  // The click missed an already-open tab. Publish the target before the cold
  // backend start so the row can acknowledge it in this same turn (#120277).
  // Highlight, routing, drafts, and running turns are unchanged.
  $pendingBotOpen.set({ generation, key })

  try {
    // Activation selects this row's source only. Canonical identity is resolved
    // after that by the owner profile's "Bot Chat" title registry.
    await prepareBotSource(bot)
  } catch (error) {
    if (generation === getBotOpenGeneration()) {
      $openBotChat.set(null)
      restorePreviousGroup()
      notifyBotOpenFailure(error, bot, 'reach')
    }

    settlePendingBotOpen(generation)

    return false
  }

  if (generation !== getBotOpenGeneration()) {
    settlePendingBotOpen(generation)

    return false
  }

  try {
    const opened = await openBotCanonicalChat(bot, { openingStillCurrent: () => generation === getBotOpenGeneration() })

    if (generation !== getBotOpenGeneration()) {
      settlePendingBotOpen(generation)

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
      // Reading acknowledgement follows a successful foreground transition.
      // A failed source lookup/open leaves the persisted marker intact so
      // Retry still tells the truth. The owner hint is required because a bot
      // open deliberately leaves the gateway on the launch profile.
      ackStoredSessionId(botCanonicalSessionId(bot), bot.name)
      settlePendingBotOpen(generation)

      return true
    }
  } catch (error) {
    if (generation === getBotOpenGeneration()) {
      $openBotChat.set(null)
      restorePreviousGroup()
      notifyBotOpenFailure(error, bot, 'open', displayName(bot, meta))
    }

    settlePendingBotOpen(generation)

    return false
  }

  // An older Desktop without the profile-scoped draft API has no safe fallback:
  // do not navigate the current workspace or create a draft on the wrong owner.
  if (typeof host.newChat !== 'function') {
    $openBotChat.set(null)
    restorePreviousGroup()
    settlePendingBotOpen(generation)

    return false
  }

  $openBotChat.set({
    key,
    openedRegistryId: ''
  })
  newBotChat(bot)
  settlePendingBotOpen(generation)

  return true
}

/** Bot-open handoff: capture the selected group and retire its registered
 * main tab (or clear the in-panel selection) before async source prep /
 * canonical open. */
function dismissGroupChatForBotOpen(): null | { group: string; roomId: string } {
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
