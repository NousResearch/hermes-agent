// Which action a left-click on a sidebar session row triggers, given the
// modifier keys held. Kept as a pure resolver (separate from the row
// component) so the precedence — the part that's easy to get subtly wrong —
// is unit-testable without rendering the whole sidebar.

export type SessionRowClickAction = 'archive' | 'newTab' | 'newWindow' | 'pin' | 'resume'

export interface SessionRowClickModifiers {
  altKey: boolean
  ctrlKey: boolean
  metaKey: boolean
  shiftKey: boolean
}

/**
 * Resolve the click action from its modifiers.
 *
 * Precedence matters: the multi-modifier gestures (⌥+⇧ archive, ⌘/⌃+⇧ new
 * window) MUST be checked before the single-modifier pin (⇧) and new-tab
 * (⌘/⌃) gestures, because they set those flags too — testing `shiftKey`
 * first would swallow both into "pin".
 *
 * Archive is independent of window support (it works in the web embed too);
 * only the new-window gesture needs standalone windows, and without them
 * ⌘/⌃+⇧ falls through to the plain ⌘/⌃ new-tab behaviour.
 */
export function resolveSessionRowClick(
  { altKey, ctrlKey, metaKey, shiftKey }: SessionRowClickModifiers,
  opts: { canOpenWindow: boolean }
): SessionRowClickAction {
  const primaryModifier = metaKey || ctrlKey

  if (altKey && shiftKey) {
    return 'archive'
  }

  if (primaryModifier && shiftKey && opts.canOpenWindow) {
    return 'newWindow'
  }

  if (primaryModifier) {
    return 'newTab'
  }

  if (shiftKey) {
    return 'pin'
  }

  return 'resume'
}
