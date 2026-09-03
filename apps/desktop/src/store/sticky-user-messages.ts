/**
 * Sticky user messages — pin the user's latest message at the top of a
 * conversation while scrolling through long threads.
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.stickyUserMessages.v1'

export const $stickyUserMessagesEnabled = atom<boolean>(
  typeof window === 'undefined' ? true : storedBoolean(KEY, true),
)

export function setStickyUserMessagesEnabled(enabled: boolean): void {
  $stickyUserMessagesEnabled.set(enabled)
}

if (typeof window !== 'undefined') {
  $stickyUserMessagesEnabled.listen(enabled => persistBoolean(KEY, enabled))
}
