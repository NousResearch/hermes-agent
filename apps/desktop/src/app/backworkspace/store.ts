import { atom } from 'nanostores'

import { prefersReducedMotion } from '@/hooks/use-media-query'
import { $remoteDisplayReason } from '@/store/remote-display'
import { isBrowserWindow, isHudWindow } from '@/store/windows'

import { flipSurface } from './flip'
import { settleLiveEditors, TURNING_ATTRIBUTE } from './live-editor'
import { flushBackworkspacePage } from './page'

/** Which side of this window faces the user. Window presentation only — never persisted. */
export const $backworkspaceOpen = atom(false)

let flipping = false
let returnFocus: HTMLElement | null = null

/** Turn the window over, front ⇄ back. A toggle while a flip is running is dropped. */
export async function toggleBackworkspace(): Promise<void> {
  const root = document.getElementById('root')

  // HUD and popped-out browser windows have no back page to turn to.
  if (flipping || !root || isHudWindow() || isBrowserWindow()) {
    return
  }

  flipping = true

  const opening = !$backworkspaceOpen.get()

  if (opening) {
    returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
  } else {
    void flushBackworkspacePage()
  }

  document.documentElement.setAttribute(TURNING_ATTRIBUTE, '')

  try {
    await flipSurface(root, () => $backworkspaceOpen.set(opening), {
      direction: opening ? 1 : -1,
      reducedMotion: prefersReducedMotion(),
      softwareComposited: $remoteDisplayReason.get() !== null
    })
  } finally {
    flipping = false
    // The page's editor was built mid-turn, under a transform it cannot see
    // end. Asked first, shown second: the measurement runs before the next
    // paint, so the caret's first frame on screen is already the right one.
    settleLiveEditors()
    document.documentElement.removeAttribute(TURNING_ATTRIBUTE)
  }

  // Back on the front: hand the caret back to wherever it was before the flip.
  if (!opening && returnFocus?.isConnected) {
    returnFocus.focus({ preventScroll: true })
    returnFocus = null
  }
}
