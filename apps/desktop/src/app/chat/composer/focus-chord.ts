/**
 * ⌘/Ctrl+L moves focus to the composer from anywhere. This matches the
 * address-bar muscle memory of a browser. The other handlers of the chord
 * keep priority:
 *
 *   - A terminal or preview selection sends itself to the composer through
 *     its own capture-phase listener. That listener stops propagation, so
 *     this handler never sees the press. This handler registers on the
 *     BUBBLE phase for that reason.
 *   - A user-rebound action on the same chord dispatches in the capture
 *     listener of use-keybinds and marks the event handled. This handler
 *     yields on `defaultPrevented`.
 *   - A focused terminal with no selection keeps the chord as clear-screen.
 *     `composerFocusBlockedBySurface()` owns that gate. The same gate covers
 *     type-to-focus and paste-to-focus, dialogs, full workspace pages, and
 *     the session switcher.
 */

import { isComposerChord } from '@/lib/keybinds/chords'
import { composerFocusBlockedBySurface } from '@/lib/keybinds/composer-focus-keys'

import { requestComposerFocus } from './focus'

/** The window-level keydown fallback. use-keybinds registers it beside the
 *  paste listener. When this handler claims the press, it prevents the
 *  default action. */
export function handleComposerFocusChord(event: KeyboardEvent): void {
  if (event.defaultPrevented || !isComposerChord(event)) {
    return
  }

  if (composerFocusBlockedBySurface()) {
    return
  }

  event.preventDefault()
  requestComposerFocus('active')
}
