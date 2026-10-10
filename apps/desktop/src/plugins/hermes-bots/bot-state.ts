/**
 * The window-local state Bot Mode's surfaces share: which roster row is
 * selected, which owner's chat is on screen, and the per-bot activity
 * watermarks the unread poll compares against.
 *
 * A leaf by design. The roster, the routines tile, the create dialog and the
 * delete path all read and write this, and it reads none of them — so no
 * surface has to import a sibling surface to know what is selected.
 */

import { atom, host } from '@hermes/plugin-sdk'

import { botRosterKey, botSelectionKey } from './data'
import { getPluginCtx } from './shared'
import type { RosterRow } from './types'

// last_active watermark per source-qualified bot, seeded on first poll so a
// fresh mount doesn't mark ancient history unread.
export const rosterWatermarks = new Map<string, number>()

// Last preview text we actually surfaced (toasted) per bot, used to suppress
// duplicate activity toasts when last_active keeps advancing (a busy bridge
// re-pinging the same message) but the visible content is unchanged. Seeded on
// the first poll alongside rosterWatermarks.
export const lastToastedPreview = new Map<string, string>()

// Bot Mode sessions are ALWAYS hidden from the global Sessions sidebar:
// canonical Bot Chats are plugin-owned forever-chats and group-chat member
// sessions are room plumbing — neither is a scratch conversation, and a
// 6-member room would otherwise dump six identical "Group: ..." rows into
// recents. Backed by the core generic `hidden` session flag (session.create
// hidden:true / REST PATCH /api/sessions/{id}). Older gateways ignore the flag and the
// sessions simply stay visible there.

/** Bot the Routines tile is scoped to. Follows the live gateway profile
 *  (the bot you're actually chatting with) and roster clicks. */
export const $selectedBot = atom('default')

/** Owner of the chat the user is LOOKING AT. Newer desktops expose a
 *  connection-qualified owner. Older builds synthesize the previous
 *  profile/gateway fallback and listen to both atoms when available. */
/** Source-qualified Bot Mode selection. Restoring it is presentation-only:
 *  it never activates a gateway or creates a session. */
export const $selectedRosterKey = atom('')
export const $selectedRosterHydrated = atom(false)
export const $rosterHydrated = atom(false)
/** Mirrors host.paneVisibility('hermes-bots:pane') — wired in register(). */
export const $botsPaneVisible = atom(false)
/** An explicit open landed: {key, openedRegistryId, openedSessionId}. The
 *  registry id is empty for the legacy newChat draft fallback and for a click
 *  that came back to the bot's already-open tabs (only openedSessionId set — no
 *  canonical chat was resolved). This transient view observation is never an
 *  identity preference. */
export const $openBotChat = atom<{ key: string; openedRegistryId: string; openedSessionId?: string } | null>(null)
export { $pendingBotOpen } from './shared'
/** A session owns the main workspace. The roster highlight and the Cronjobs
 *  lifecycle both key off this rather than reading host.state conditionally
 *  from render. */
export const $botChatFocused = atom(false)

/** Set when the persisted selection was proven retired and dropped. The roster
 *  then stays deliberately UNselected until the user picks a successor: seating
 *  the next surviving bot would be the silent redirect this design forbids (a
 *  retired bot-builder must not come back as whoever happens to sort first).
 *  Persisted, because a deferral that a reload forgets is not a deferral —
 *  reconciliation re-runs on the next render either way. */
export const $rosterSelectionDeferred = atom(false)

const SELECTION_DEFERRED_KEY = 'roster-selection-deferred-v1'

/** When the standing selection was established in this window (ms). The tile
 *  path fences on when a TAB was opened; the selection path needs the same
 *  clock, because a roster answer ISSUED before the user picked a bot never saw
 *  the pick and must not clear it. Stamped by every path that establishes a
 *  selection — hydration included, since a choice restored from storage is the
 *  user's standing choice. Not reactive: nothing renders it. */
let selectionEstablishedAt = 0

export function rosterSelectionEstablishedAt(): number {
  return selectionEstablishedAt
}

function persistSelectedRosterKey(key: string) {
  if (key) {
    selectionEstablishedAt = Date.now()
  }

  $selectedRosterKey.set(key)

  try {
    Promise.resolve(getPluginCtx()?.storage?.set?.('selected-roster-bot-v1', key)).catch(() => undefined)
  } catch {
    /* storage unavailable — selection lasts for this window */
  }
}

/** Hydrate the selection a previous window persisted. Same clock as a live
 *  pick: the user chose this bot, and it is still their standing choice. */
export function restoreSelectedRosterKey(key: string) {
  selectionEstablishedAt = Date.now()
  $selectedRosterKey.set(key)
}

function persistSelectionDeferred(deferred: boolean) {
  $rosterSelectionDeferred.set(deferred)

  try {
    Promise.resolve(getPluginCtx()?.storage?.set?.(SELECTION_DEFERRED_KEY, deferred)).catch(() => undefined)
  } catch {
    /* storage unavailable — the deferral lasts for this window */
  }
}

/** The USER chose this bot (roster click, chat open, recent row). An explicit
 *  choice is exactly what a deferral is waiting for, so it ends here. */
export function saveSelectedRosterBot(bot: RosterRow) {
  $selectedBot.set(botSelectionKey(bot))
  resumeRosterSelection()
  persistSelectedRosterKey(botRosterKey(bot))
}

/** The ROSTER seated a bot because nothing was selected (first run). Not a user
 *  choice, so it must never clear a deferral a retired selection left behind. */
export function seatRosterSelection(bot: RosterRow) {
  $selectedBot.set(botSelectionKey(bot))
  persistSelectedRosterKey(botRosterKey(bot))
}

/** Retire the persisted selection WITHOUT seating a replacement. */
export function deferRosterSelection(key: string) {
  clearSelectedRosterKey(key)
  persistSelectionDeferred(true)
}

/** The user picked a bot, so the roster may auto-seat again. */
export function resumeRosterSelection() {
  if ($rosterSelectionDeferred.get()) {
    persistSelectionDeferred(false)
  }
}

export function clearSelectedRosterBot(bot: RosterRow) {
  clearSelectedRosterKey(botRosterKey(bot))
}

/** Drop the persisted selection when it is exactly this key — the caller has
 *  proven the owner is gone, not merely unreachable. An unreachable source
 *  KEEPS its key so the selection reconciles when the gateway returns. */
export function clearSelectedRosterKey(key: string) {
  if ($selectedRosterKey.get() !== key) {
    return
  }

  persistSelectedRosterKey('')
}

/** Split a roster key back into its owner parts. Profile names cannot contain
 *  ':' (NAME_RE), so the first '::' is unambiguous. */
export function parseRosterKey(key: null | string | undefined) {
  const raw = String(key || '')
  const at = raw.indexOf('::')

  if (at < 0) {
    return {
      connectionId: '',
      name: ''
    }
  }

  return {
    connectionId: raw.slice(0, at),
    name: raw.slice(at + 2)
  }
}

const $focusedBotProfile = host.state.focusedSessionProfile || host.state.profile

/** Profile that owns the chat currently on screen. Bot Mode opens another
 *  profile's session without moving the gateway socket, so mention filtering
 *  and sender identity must follow focus rather than host.state.profile. */
export function focusedMentionProfile() {
  return String($focusedBotProfile.get?.() || '').trim() || 'default'
}

function fallbackFocusedBotOwner(profile: string = $focusedBotProfile.get?.()) {
  const focusedProfile = String(profile || 'default').trim() || 'default'
  const activeProfile = String(host.state.profile?.get?.() || 'default').trim() || 'default'

  // focusedSessionProfile without focusedSessionOwner is a legacy half-shape:
  // it carries no source identity. Only reuse the active connection when the
  // focused profile is also the active profile; otherwise fail closed rather
  // than manufacturing a cross-source owner from unrelated atoms.
  if (host.state.focusedSessionProfile && focusedProfile !== activeProfile) {
    return null
  }

  const connectionId = String(
    host.state.connectionId?.get?.() ||
      (typeof host.activeConnectionId === 'function' ? host.activeConnectionId() : '') ||
      ''
  ).trim()

  return {
    authoritative: false,
    connectionId,
    profile: focusedProfile
  }
}

export const $focusedBotOwner = host.state.focusedSessionOwner || {
  get: () => fallbackFocusedBotOwner(),
  listen: (listener: (value: ReturnType<typeof fallbackFocusedBotOwner>) => void) => {
    const emit = (profile: string) => listener(fallbackFocusedBotOwner(profile))
    const unbindProfile = $focusedBotProfile.listen(emit)
    const unbindConnection = host.state.connectionId?.listen?.(() => emit($focusedBotProfile.get?.()))

    return () => {
      unbindProfile?.()
      unbindConnection?.()
    }
  }
}

export function focusedRosterOwner(
  owner: {
    authoritative?: boolean
    connectionId?: string
    name?: string
    profile?: string
  } | null
) {
  // TODO(bot-mode-types): `owner.name` cannot exist. Every caller passes
  // $focusedBotOwner, whose two shapes (host.state.focusedSessionOwner and
  // fallbackFocusedBotOwner) both key the profile as `profile`, so the
  // `owner?.name` arm is unreachable and a name-only owner would be dropped.
  const name = String(owner?.profile || owner?.name || '').trim()

  if (!owner || !name) {
    return null
  }

  return {
    authoritative: owner.authoritative !== false,
    connectionId: String(owner.connectionId || '').trim(),
    name
  }
}
