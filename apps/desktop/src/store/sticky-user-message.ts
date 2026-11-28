/**
 * Sticky user messages — opt-out (#38372, #80146).
 *
 * Every user bubble carries `position: sticky` so the prompt that started the
 * turn stays on screen while its reply streams. That is a preference, not a
 * law: some users want the transcript to scroll like a plain feed. Default ON
 * preserves the shipped behavior; the consumer gate lives in
 * `StickyHumanMessageContainer`, so anything that becomes pinnable later
 * (#145785: clarify answers) respects the same switch instead of growing its own.
 *
 * Presentation-scoped, so the renderer owns it (desktop AGENTS.md: state lives
 * with its authority), but it mirrors to `display.sticky_user_message` on the
 * live gateway like the other Appearance switches — the backend persists the
 * key today and consumes no behavior from it, which is exactly what the
 * two-beat mirror's "only re-send what the user touched" semantics make safe.
 */

import { atom } from 'nanostores'

import { persistStore } from '@/lib/persist-store'
import { persistBoolean, storedBoolean } from '@/lib/storage'
import { mirrorDisplayToggle } from '@/store/display-toggles'

import { recordFeatureToggle } from './desktop-metrics'

const STICKY_USER_MESSAGE_STORAGE_KEY = 'hermes.desktop.stickyUserMessage'

/** Desktop-local appearance preference, shared by all threads in this window. */
export const $stickyUserMessage = atom(storedBoolean(STICKY_USER_MESSAGE_STORAGE_KEY, true))

persistStore(
  $stickyUserMessage,
  (value, options) => persistBoolean(STICKY_USER_MESSAGE_STORAGE_KEY, value, options),
  true
)

export function setStickyUserMessage(value: boolean) {
  recordFeatureToggle('sticky_user_message', $stickyUserMessage.get(), value)
  $stickyUserMessage.set(value)
}

mirrorDisplayToggle('display.sticky_user_message', STICKY_USER_MESSAGE_STORAGE_KEY, $stickyUserMessage)
