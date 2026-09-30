/**
 * Refocus the composer when the WINDOW regains focus (#38993).
 *
 * The opt-in companion of ⌘/Ctrl+L: returning to Hermes should put the caret
 * back in the chat input instead of eating the first keystroke into whatever
 * chrome last held focus. Rides the same focus-request bus the chord uses, so
 * the composer heals its own caret across React commit + browser focus
 * restore.
 *
 * Keyboard ownership follows focus (desktop AGENTS.md): the request goes out
 * only when no other surface holds the caret — an editable (a dialog input,
 * a terminal's textarea) keeps it, and open overlays / full pages / the
 * session switcher keep their keys via `composerFocusBlockedBySurface()`.
 * Raw keystroke capture (typing into an unfocused composer) is deliberately
 * not attempted here: IME/composition edge cases make it unreliable, and once
 * the input is focused every normal keystroke lands in it anyway.
 */

import { isEditableTarget } from '@/lib/keybinds/combo'
import { composerFocusBlockedBySurface } from '@/lib/keybinds/composer-focus-keys'
import { $autoFocusComposer } from '@/store/auto-focus-composer'

import { requestComposerFocus } from './focus'

/** window `focus` listener, registered beside the chord in use-keybinds. */
export function handleWindowActivate(): void {
  if (!$autoFocusComposer.get() || composerFocusBlockedBySurface() || isEditableTarget(document.activeElement)) {
    return
  }

  requestComposerFocus('active')
}
