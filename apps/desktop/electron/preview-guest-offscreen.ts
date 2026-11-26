/**
 * Preview guests that are mounted but OFF SCREEN.
 *
 * A hidden session's Browser stays mounted (hidden, inert) so returning to
 * that session shows the same page. Chromium still tracks its guest as the
 * focused webContents after the switch — nothing on the host side moves a
 * guest's webContents focus when its `<webview>` is hidden — so a gesture
 * that acts on "the focused guest" (mouse back/forward, swipe, ⌘R) would
 * navigate a page the user cannot see, in a session they have left.
 *
 * Only the host renderer knows which `<webview>` is on screen (its layout
 * tree, not Chromium, decides), so it reports each guest's visibility here
 * and main skips a hidden one; the gesture then falls through to the
 * renderer, which routes it to the visible pane.
 */

import type { WebContents } from 'electron'

type GuestCommand = 'back' | 'forward' | 'reload'

const hiddenGuests = new Set<number>()
const watchedGuests = new WeakSet<WebContents>()

/** Record what the host renderer `sender` reports for one of ITS guests. A
 *  renderer may only speak for the guests it embeds. */
export function notePreviewGuestHidden(sender: WebContents, guest: null | undefined | WebContents, hidden: boolean) {
  if (!guest || guest.isDestroyed() || guest.getType() !== 'webview' || guest.hostWebContents !== sender) {
    return
  }

  if (!hidden) {
    hiddenGuests.delete(guest.id)

    return
  }

  hiddenGuests.add(guest.id)

  if (!watchedGuests.has(guest)) {
    watchedGuests.add(guest)
    guest.once('destroyed', () => hiddenGuests.delete(guest.id))
  }
}

export function isPreviewGuestHidden(guest: WebContents): boolean {
  return hiddenGuests.has(guest.id)
}

/**
 * Run a browser gesture on the guest page the user is actually in, if any.
 *
 * A `<webview>` guest is its own out-of-process webContents: pointer and focus
 * events inside the page never reach the host document, so NOTHING in the
 * renderer — not `document.activeElement`, not the layout tree's hover/focus
 * ladder — can see that the user is in there. Main can: Electron tracks the
 * focused webContents across processes. A guest its host reported hidden is
 * not "the page the user is in", whatever Chromium's focus says.
 *
 * Returns false when focus is in the app's own chrome (or on a hidden guest),
 * where the renderer is the one that knows which pane is active.
 */
export function commandFocusedGuest(command: GuestCommand, focused: null | undefined | WebContents): boolean {
  if (!focused || focused.isDestroyed() || focused.getType() !== 'webview' || isPreviewGuestHidden(focused)) {
    return false
  }

  const history = focused.navigationHistory

  if (command === 'reload') {
    focused.reload()
  } else if (command === 'back') {
    if (history.canGoBack()) {
      history.goBack()
    }
  } else if (history.canGoForward()) {
    history.goForward()
  }

  return true
}
