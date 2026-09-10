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
 * Precedence matters: exact ⌃+⇧ and ⌥+⇧ archive, while ⌘+⇧ and
 * primary-modifier supersets open a new window. These MUST be checked before
 * the single-modifier pin (⇧) and new-tab (⌘/⌃) gestures, because they set
 * those flags too — testing `shiftKey` first would swallow both into "pin".
 *
 * Archive is independent of window support (it works in the web embed too);
 * only the new-window gesture needs standalone windows, and without them
 * a window gesture falls through to the plain ⌘/⌃ new-tab behaviour.
 */
export function resolveSessionRowClick(
  { altKey, ctrlKey, metaKey, shiftKey }: SessionRowClickModifiers,
  opts: { canOpenWindow: boolean }
): SessionRowClickAction {
  const primaryModifier = metaKey || ctrlKey

  // Exact physical Ctrl+Shift is the cross-platform archive gesture. Keep it
  // exact so Cmd+Shift and modifier supersets retain the window route.
  if (ctrlKey && shiftKey && !metaKey && !altKey) {
    return 'archive'
  }

  if (primaryModifier && shiftKey && opts.canOpenWindow) {
    return 'newWindow'
  }

  if (altKey && shiftKey && !primaryModifier) {
    return 'archive'
  }

  if (primaryModifier) {
    return 'newTab'
  }

  if (shiftKey) {
    return 'pin'
  }

  return 'resume'
}
