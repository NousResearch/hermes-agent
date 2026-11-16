import { atom, computed } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'

import { $showsAdvancedChrome } from './interface-mode'
import { $focusedSessionIsTile, $focusedStoredSessionId } from './session-focus'
import { $portalShell } from './shell-variant'

// The redesign has no statusbar strip. The readouts that describe the chat being worked in (its
// workspace, context, timers, approvals) sit on one line under that chat's composer instead; the
// rest stay in the rail's System popover. Its own preference, so hiding the line never touches the
// classic bar's visibility for a later rollback.
const $portalChatStatusPref = persistentAtom('hermes.desktop.portalChatStatus.v1', true, Codecs.bool)

/** The line's own on/off, as the System popover's switch shows it. */
export const $portalChatStatusEnabled = $portalChatStatusPref

export function setPortalChatStatusEnabled(enabled: boolean) {
  $portalChatStatusPref.set(enabled)
}

/** The line is on screen: the redesigned shell with developer chrome, and the preference on. */
export const $portalChatStatusShown = computed(
  [$portalShell, $showsAdvancedChrome, $portalChatStatusPref],
  (portal, advanced, enabled) => portal && advanced && enabled
)

/** The composer the line belongs under: the focused chat's (`ComposerScope.target`). */
export const $portalChatStatusTarget = computed([$focusedSessionIsTile, $focusedStoredSessionId], (tile, id) =>
  tile && id ? `tile:${id}` : 'main'
)

/** The focused composer's mount point for the line; null while no such composer is on screen. */
export const $portalChatStatusSlot = atom<HTMLElement | null>(null)
