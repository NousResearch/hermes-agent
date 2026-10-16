import { computed } from 'nanostores'

import { $sessionStates, isLiveTurnAwaitingEvents } from '@/store/session-states'
import { $temporaryChats } from '@/store/temporary-chat'

/** Is a turn still running (or parked on the user) in this temporary chat's runtime? A running chat
 *  never starts its idle clock, so a long reply is never cut off by the TTL. */
export function temporaryChatBusy(runtimeId: null | string): boolean {
  if (!runtimeId) {
    return false
  }

  const state = $sessionStates.get()[runtimeId]

  return Boolean(state && (isLiveTurnAwaitingEvents(state) || state.needsInput))
}

/** Stored ids of temporary chats with a turn running, as one stable key — `computed` only notifies
 *  when it changes, so streaming tokens never repaint the private-mode bar. */
export const $runningTemporaryChats = computed([$sessionStates, $temporaryChats], (_states, chats) =>
  Object.values(chats)
    .filter(chat => temporaryChatBusy(chat.runtimeId))
    .map(chat => chat.storedId)
    .sort()
    .join('|')
)
