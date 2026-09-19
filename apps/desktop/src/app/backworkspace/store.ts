import { atom } from 'nanostores'

import { prefersReducedMotion } from '@/hooks/use-media-query'
import { isBrowserWindow, isHudWindow } from '@/store/windows'

import { flipSurface } from './flip'
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

  try {
    await flipSurface(root, () => $backworkspaceOpen.set(opening), {
      direction: opening ? 1 : -1,
      reducedMotion: prefersReducedMotion()
    })
  } finally {
    flipping = false
  }

  // Back on the front: hand the caret back to wherever it was before the flip.
  if (!opening && returnFocus?.isConnected) {
    returnFocus.focus({ preventScroll: true })
    returnFocus = null
  }
}
