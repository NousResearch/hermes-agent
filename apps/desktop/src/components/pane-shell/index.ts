/**
 * Cross-surface event: toggle-reveal a collapsed pane. Dispatched by the
 * keybinds (⌘B / ⌘G / titlebar toggles on narrow viewports) with the pane id
 * in `detail`; the layout tree's narrow overlays (tree/renderer.tsx) listen
 * and slide the pane over the grid.
 */
export const PANE_TOGGLE_REVEAL_EVENT = 'hermes:pane-toggle-reveal'

/** Ask the narrow overlays to open, close or (with no mode) flip a collapsed pane. */
export function togglePaneReveal(id: string, mode?: 'close' | 'open' | 'toggle'): void {
  window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: mode ? { id, mode } : { id } }))
}
