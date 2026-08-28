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

// Bot Mode sessions are ALWAYS hidden from the global Sessions sidebar:
// canonical Bot Chats are plugin-owned forever-chats and group-chat member
// sessions are room plumbing — neither is a scratch conversation, and a
// 6-member room would otherwise dump six identical "Group: ..." rows into
// recents. Backed by the core generic `hidden` session flag (session.create
// hidden:true / session.set_hidden). Older gateways ignore the flag and the
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
/** An explicit open landed: {key, openedRegistryId}. This transient view
 *  observation is empty only for the legacy newChat draft fallback. */
export const $openBotChat = atom<{ key: string; openedRegistryId: string; openedSessionId?: string } | null>(null)
/** A session owns the main workspace. The roster highlight and the Cronjobs
 *  lifecycle both key off this rather than reading host.state conditionally
 *  from render. */
export const $botChatFocused = atom(false)

export function saveSelectedRosterBot(bot: RosterRow) {
  const key = botRosterKey(bot)
  $selectedBot.set(botSelectionKey(bot))
  $selectedRosterKey.set(key)

  try {
    Promise.resolve(getPluginCtx()?.storage?.set?.('selected-roster-bot-v1', key)).catch(() => undefined)
  } catch {
    /* storage unavailable — selection lasts for this window */
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

  $selectedRosterKey.set('')

  try {
    Promise.resolve(getPluginCtx()?.storage?.set?.('selected-roster-bot-v1', '')).catch(() => undefined)
  } catch {
    /* storage unavailable — selection is cleared for this window */
  }
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
