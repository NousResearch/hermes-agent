import { atom } from 'nanostores'

interface SessionListAuthority {
  profile: string
  scope: string
  epoch: number
  status: 'loaded' | 'failed'
}

// $sessions merges rows across profiles; its length and the skeleton-loading
// flag cannot prove that the current source/profile has answered a list read.
export const $sessionListAuthority = atom<SessionListAuthority | null>(null)

export function setSessionListAuthority(next: SessionListAuthority): void {
  const previous = $sessionListAuthority.get()

  if (
    previous?.profile === next.profile &&
    previous.scope === next.scope &&
    previous.epoch === next.epoch &&
    previous.status === next.status
  ) {
    return
  }

  $sessionListAuthority.set(next)
}
