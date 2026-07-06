// Which action a left-click on a sidebar session row triggers, given the
// modifier keys held. Kept as a pure resolver (separate from the row
// component) so the precedence — the part that's easy to get subtly wrong —
// is unit-testable without rendering the whole sidebar.

export type SessionRowClickAction = 'archive' | 'newWindow' | 'pin' | 'resume'

export interface SessionRowClickModifiers {
  ctrlKey: boolean
  metaKey: boolean
  shiftKey: boolean
}

/**
 * Resolve the click action from its modifiers.
 *
 * Precedence matters: the combined ⌘/⌃+⇧ archive gesture MUST be checked
 * before the single-modifier pin (⇧) and new-window (⌘/⌃) gestures, because
 * it sets all of those flags at once — testing `shiftKey` first would swallow
 * it into "pin".
 *
 * Archive is independent of window support (it works in the web embed too);
 * only the new-window gesture falls back to a plain resume when standalone
 * windows aren't available.
 */
export function resolveSessionRowClick(
  { ctrlKey, metaKey, shiftKey }: SessionRowClickModifiers,
  opts: { canOpenWindow: boolean }
): SessionRowClickAction {
  const primaryModifier = metaKey || ctrlKey

  if (primaryModifier && shiftKey) {
    return 'archive'
  }

  if (shiftKey) {
    return 'pin'
  }

  if (primaryModifier && opts.canOpenWindow) {
    return 'newWindow'
  }

  return 'resume'
}
