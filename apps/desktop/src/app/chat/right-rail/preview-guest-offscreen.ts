/**
 * A Browser guest that is mounted but OFF SCREEN — an inactive tab, a
 * collapsed side, or a hidden session's kept-alive page — must not act like
 * the page the user is on.
 *
 * Chromium keeps tracking a hidden guest as the focused webContents (nothing
 * moves a guest's focus when its `<webview>` is hidden), so main's
 * focused-guest gestures (mouse back/forward, swipe, ⌘R) would navigate it.
 * Only this renderer knows what is on screen, so it tells main.
 *
 * A hidden SESSION's page (parked, see parked-panes.ts) keeps running — that
 * is the point of keeping it — but must not be heard from a chat the user has
 * left. It is muted while parked and handed back as it was. An inactive tab in
 * the session on screen keeps playing, like a background tab in any browser.
 */

import { useStore } from '@nanostores/react'
import { type RefObject, useCallback, useEffect, useState } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { $parkedTreePanes } from '@/components/pane-shell/tree/parked-panes'
import { PREVIEW_TILE_PREFIX } from '@/store/preview-explicit'

export interface OffscreenGuest {
  getWebContentsId?: () => number
  isConnected?: boolean
  isAudioMuted?: () => boolean
  setAudioMuted?: (muted: boolean) => void
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

/** Mute `webview` unless it already is; returns whether this call muted it, so
 *  the caller never unmutes a page that was muted before. */
function muteGuest(webview: OffscreenGuest): boolean {
  try {
    if (webview.isAudioMuted?.()) {
      return false
    }

    webview.setAudioMuted?.(true)

    return true
  } catch {
    // Not attached yet / already gone: nothing is playing.
    return false
  }
}

/** Reports and mutes the guest as it goes off screen. Returns a `dom-ready`
 *  listener the pane must attach to its `<webview>`: a guest has no id until it
 *  attaches, and one that attaches while its session is already hidden would
 *  otherwise never be reported or muted (the effects only re-run on change). */
export function usePreviewGuestOffscreen(webviewRef: RefObject<null | OffscreenGuest>, tabId?: string): () => void {
  const hidden = !usePaneVisible()
  const parkedPanes = useStore($parkedTreePanes)
  const parked = Boolean(tabId) && parkedPanes.has(`${PREVIEW_TILE_PREFIX}:${tabId}`)
  const [attachedId, setAttachedId] = useState<null | number>(null)
  const noteGuestReady = useCallback(() => setAttachedId(guestId(webviewRef.current)), [webviewRef])

  useEffect(() => {
    const id = guestId(webviewRef.current)

    if (id !== null) {
      window.hermesDesktop?.setPreviewGuestHidden?.(id, hidden)
    }
  }, [attachedId, hidden, webviewRef])

  useEffect(() => {
    const webview = webviewRef.current

    if (!parked || !webview || !muteGuest(webview)) {
      return
    }

    return () => {
      // Unmounted while parked (closed, or let go past the retention cap):
      // the guest is being destroyed, and unmuting it would only blip.
      if (!webview.isConnected) {
        return
      }

      try {
        webview.setAudioMuted?.(false)
      } catch {
        // The guest went away while parked; nothing to restore.
      }
    }
  }, [attachedId, parked, webviewRef])

  return noteGuestReady
}
