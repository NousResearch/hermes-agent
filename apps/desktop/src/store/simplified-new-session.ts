import { $portalShell } from './shell-variant'

/** Opens a chat with this window's own fresh draft, or with the agent the redesigned New-chat row
 *  targets; answers whether it opened one. */
type NewSessionCommand = () => boolean

let claimant: NewSessionCommand | null = null

/** The mounted redesigned sidebar's New-chat row claims the New session command while it is on
 *  screen; returns the release. The command answers whether it opened a chat, so a row that has
 *  nothing to start hands the chord back instead of eating it. */
export function claimSimplifiedNewSession(command: NewSessionCommand): () => void {
  claimant = command

  return () => {
    if (claimant === command) {
      claimant = null
    }
  }
}

/** Runs the claimed command when the portal shell has one mounted; a `false` answer — no portal
 *  shell, no sidebar row mounted, or a row that could not open a chat — keeps the caller's own
 *  fresh-draft path, so the shortcut never dies on a sidebar it cannot reach. */
export function requestSimplifiedNewSession(): boolean {
  if (!$portalShell.get() || !claimant) {
    return false
  }

  return claimant()
}
