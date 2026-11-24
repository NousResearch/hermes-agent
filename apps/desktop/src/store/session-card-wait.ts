import { atom } from 'nanostores'

import type { ClientSessionState } from '@/app/types'

// Sessions with a folder or site approval card parked, by runtime id. The prompt store keeps the
// set current. Such a card outlives its turn, and the turn's `needsInput` flag ends with the turn.
export const $sessionsWaitingOnCard = atom<ReadonlySet<string>>(new Set())

export function setSessionsWaitingOnCard(runtimeIds: readonly string[]): void {
  const current = $sessionsWaitingOnCard.get()

  if (runtimeIds.length !== current.size || runtimeIds.some(id => !current.has(id))) {
    $sessionsWaitingOnCard.set(new Set(runtimeIds))
  }
}

/** Whether a session waits on the user: its turn is blocked on a prompt, or a folder or site card
 *  is parked for it. What tells a person that a chat is waiting asks this. */
export const sessionNeedsInput = (state: ClientSessionState | undefined, runtimeId: string): boolean =>
  Boolean(state?.needsInput) || $sessionsWaitingOnCard.get().has(runtimeId)
