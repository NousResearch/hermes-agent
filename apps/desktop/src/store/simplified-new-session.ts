import { $portalShell } from './shell-variant'

let claimant: (() => void) | null = null

/** The mounted redesigned sidebar's picker claims the New session command; returns the release. */
export function claimSimplifiedNewSession(handler: () => void): () => void {
  claimant = handler

  return () => {
    if (claimant === handler) {
      claimant = null
    }
  }
}

/** Runs the claimed picker when the portal shell has one mounted; false keeps the existing path. */
export function requestSimplifiedNewSession(): boolean {
  if (!$portalShell.get() || !claimant) {
    return false
  }

  claimant()

  return true
}
