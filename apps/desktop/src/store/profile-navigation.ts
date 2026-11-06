import { atom } from 'nanostores'

// Restore is optional: allow warm activation + a list read, but never navigate
// after the user has spent 15 seconds on the fallback draft (cold dials may take longer).
export const PROFILE_NAVIGATION_RESTORE_TIMEOUT_MS = 15_000

export interface ProfileNavigationRequest {
  profile: string
  sequence: number
  activation: 'pending' | 'ready' | 'failed'
  // Published after activation, when connection-scoped history is safe to read.
  scope?: string
}

export const $profileNavigationRequest = atom<ProfileNavigationRequest | null>(null)
let profileNavigationSequence = 0

export const currentProfileNavigationSequence = (): number => profileNavigationSequence
let profileNavigationCleanup: { sequence: number; dispose: () => void } | undefined

$profileNavigationRequest.listen(() => {
  if (profileNavigationCleanup?.sequence !== $profileNavigationRequest.get()?.sequence) {
    profileNavigationCleanup?.dispose()
  }
})

export function dropProfileNavigationRestore(sequence: number): void {
  if ($profileNavigationRequest.get()?.sequence === sequence) {
    cancelProfileNavigationRestore()
  }
}

export function requestProfileNavigationRestore(profile: string): ProfileNavigationRequest {
  const request: ProfileNavigationRequest = {
    profile,
    sequence: ++profileNavigationSequence,
    activation: 'pending'
  }

  cancelProfileNavigationRestore()
  // Arm synchronously at the pick, not in a React effect: input followed by a
  // send can empty the composer before activation or the next render. Once
  // touched, this request can never become eligible again. Synthetic events
  // (including editor bookkeeping) are not user intent.
  const events = ['keydown', 'pointerdown', 'paste', 'drop', 'input', 'compositionstart'] as const

  const onInteraction = (event: Event) => {
    if (!event.isTrusted || (event.target instanceof Element && event.target.closest('[data-slot="profile-rail"]'))) {
      return
    }

    if (event.type === 'keydown') {
      const key = event as KeyboardEvent

      if (
        key.repeat ||
        /^(Shift|Control|Alt|Meta|CapsLock|AltGraph|NumLock|ScrollLock|Fn|FnLock|Symbol|SymbolLock)$/.test(key.key)
      ) {
        return
      }
    }

    dropProfileNavigationRestore(request.sequence)
  }

  const target = window
  events.forEach(type => target.addEventListener(type, onInteraction, true))

  const timer = setTimeout(() => dropProfileNavigationRestore(request.sequence), PROFILE_NAVIGATION_RESTORE_TIMEOUT_MS)
  let disposed = false
  profileNavigationCleanup = {
    sequence: request.sequence,
    dispose: () => {
      if (disposed) {
        return
      }

      disposed = true
      clearTimeout(timer)
      events.forEach(type => target.removeEventListener(type, onInteraction, true))
    }
  }
  $profileNavigationRequest.set(request)

  return request
}

export function cancelProfileNavigationRestore(): void {
  // Dispose directly: nanostores batch() can coalesce the intermediate null.
  profileNavigationCleanup?.dispose()
  profileNavigationCleanup = undefined
  $profileNavigationRequest.set(null)
}
