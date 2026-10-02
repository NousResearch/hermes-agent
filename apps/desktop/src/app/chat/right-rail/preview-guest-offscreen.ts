/**
 * A Browser guest that is mounted but OFF SCREEN — an inactive tab, a
 * collapsed side, or a hidden session's kept-alive page — must not act like
 * the page the user is on.
 *
 * Chromium keeps tracking a hidden guest as the focused webContents (nothing
 * moves a guest's focus when its `<webview>` is hidden), so main's
 * focused-guest gestures (mouse back/forward, swipe, ⌘R) would navigate it.
 * Only this renderer knows what is on screen, so it tells main.
 */

import { type RefObject, useEffect } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'

export interface OffscreenGuest {
  getWebContentsId?: () => number
}

/** The guest's webContents id, or null before attach / after removal (Electron
 *  throws rather than returning nothing). */
function guestId(webview: null | OffscreenGuest): null | number {
  try {
    return webview?.getWebContentsId?.() ?? null
  } catch {
    return null
  }
}

export function usePreviewGuestOffscreen(webviewRef: RefObject<null | OffscreenGuest>): void {
  const hidden = !usePaneVisible()

  useEffect(() => {
    const id = guestId(webviewRef.current)

    if (id !== null) {
      window.hermesDesktop?.setPreviewGuestHidden?.(id, hidden)
    }
  }, [hidden, webviewRef])
}
