/**
 * Show full user messages (#42992).
 *
 * Off by default: sticky user bubbles clamp long prompts with a fade, and a
 * click expands them. On renders every prompt at its natural height instead.
 * Settings → Appearance owns the lever, next to the message bubble options.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

import { recordFeatureToggle } from './desktop-metrics'

const KEY = 'hermes.desktop.showFullUserMessages.v1'

export const $showFullUserMessages = atom<boolean>(typeof window === 'undefined' ? false : storedString(KEY) === 'on')

export function setShowFullUserMessages(enabled: boolean): void {
  recordFeatureToggle('show_full_user_messages', $showFullUserMessages.get(), enabled)
  $showFullUserMessages.set(enabled)
}

if (typeof window !== 'undefined') {
  $showFullUserMessages.listen(enabled => {
    persistString(KEY, enabled ? 'on' : 'off')
  })
}
