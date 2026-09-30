/**
 * Sticky user messages — pin the user's latest message at the top of a
 * conversation while scrolling through long threads. On by default; the
 * Settings → Appearance toggle turns the pinning off so bubbles scroll in
 * normal flow (see StickyHumanMessageContainer in thread/user-message.tsx).
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.stickyUserMessages.v1'

/** Desktop-local appearance preference, shared by all threads in this window. */
export const $stickyUserMessagesEnabled = atom(
  typeof window === 'undefined' ? true : storedBoolean(KEY, true)
)

export function setStickyUserMessagesEnabled(enabled: boolean): void {
  $stickyUserMessagesEnabled.set(enabled)
}

if (typeof window !== 'undefined') {
  $stickyUserMessagesEnabled.subscribe(enabled => persistBoolean(KEY, enabled))
}
