/**
 * The find bar owns the keyboard while it is open.
 *
 * `composerFocusBlockedBySurface()` stands type-to-focus down for dialogs,
 * menus, the terminal and full pages — but the find bar is none of those. It is
 * a plain overlay input, so an unbound printable key still reached
 * `requestComposerFocus('active', { typeChar })`, and the composer's focus
 * helper retries focus three times (sync, rAF, setTimeout 0). The find input,
 * focused once inside a single rAF, always lost the race: the user pressed
 * Ctrl+F, typed, and nothing appeared in the find field.
 *
 * Keyed off the STORE rather than `document.activeElement` on purpose. The
 * store flips `active` a commit before the input mounts and focuses, and it is
 * exactly that window in which the first keystroke used to escape to the
 * composer.
 */
export function findBarOwnsTyping(findBarActive: boolean): boolean {
  return findBarActive
}
