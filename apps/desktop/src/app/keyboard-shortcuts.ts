/**
 * Keyboard shortcut cascade logic for Hermes Desktop.
 *
 * `resolveCloseAction` implements the Cmd/Ctrl+W cascade:
 *   overlay → right sidebar → preview rail → left sidebar → window close/minimize
 *
 * The other shortcuts (Cmd/Ctrl + , . K) are handled as renderer-side keydown
 * listeners, following the same pattern as Cmd/Ctrl+B and Cmd/Ctrl+N.
 */

export type CloseAction =
  | { type: 'close-overlay' }
  | { type: 'close-right-sidebar' }
  | { type: 'close-preview-rail' }
  | { type: 'close-left-sidebar' }
  | { type: 'close-window' }

export interface CloseState {
  overlayOpen: boolean
  rightSidebarOpen: boolean
  previewRailOpen: boolean
  leftSidebarOpen: boolean
}

/**
 * Determines the first UI layer to close in the Cmd/Ctrl+W cascade.
 *
 * Cascade order: overlay → right sidebar → preview rail → left sidebar → window.
 * Returns the first open layer, or 'close-window' if nothing is open.
 */
export function resolveCloseAction(state: CloseState): CloseAction {
  if (state.overlayOpen) return { type: 'close-overlay' }
  if (state.rightSidebarOpen) return { type: 'close-right-sidebar' }
  if (state.previewRailOpen) return { type: 'close-preview-rail' }
  if (state.leftSidebarOpen) return { type: 'close-left-sidebar' }
  return { type: 'close-window' }
}

/** Check if Cmd/Ctrl modifier is pressed (platform-aware). */
export function isModifierPressed(event: KeyboardEvent): boolean {
  return event.metaKey || event.ctrlKey
}

/** Check if the event is a clean modifier+key combo (no alt/shift). */
export function isCleanShortcut(event: KeyboardEvent, key: string): boolean {
  return isModifierPressed(event) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === key
}

/**
 * Sets up the Cmd/Ctrl+W cascade keydown listener.
 *
 * @param getState - returns the current UI layer state (overlay, sidebars, preview)
 * @param dispatch - called with the resolved close action
 * @returns cleanup function to remove the listener
 */
export function setupCloseCascadeListener(
  getState: () => CloseState,
  dispatch: (action: CloseAction) => void
): () => void {
  const onKeyDown = (event: KeyboardEvent) => {
    if (!isCleanShortcut(event, 'w')) return

    event.preventDefault()
    event.stopPropagation()
    dispatch(resolveCloseAction(getState()))
  }

  window.addEventListener('keydown', onKeyDown, { capture: true })
  return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
}

/**
 * Sets up a generic Cmd/Ctrl+<key> shortcut listener.
 *
 * @param key - the key to match (e.g. ',', '.', 'k')
 * @param callback - called when the shortcut is pressed
 * @returns cleanup function to remove the listener
 */
export function setupShortcutListener(key: string, callback: () => void): () => void {
  const onKeyDown = (event: KeyboardEvent) => {
    if (!isCleanShortcut(event, key)) return

    event.preventDefault()
    callback()
  }

  window.addEventListener('keydown', onKeyDown)
  return () => window.removeEventListener('keydown', onKeyDown)
}
