/**
 * `display.hover_reveal_file_browser` — one config key for the file browser
 * pane's pointer-hover edge reveal.
 *
 * On (default, matching the pre-config behavior): hovering the window edge
 * near a collapsed file browser slides it over the layout as an edge overlay.
 * Off: only manual reveal (keyboard / PANE_TOGGLE_REVEAL_EVENT) opens it —
 * pointer hover leaves the pane alone. Mirrors the CLI default
 * `hover_reveal_file_browser: True` in hermes_cli/config_defaults.py.
 */

import { atom } from 'nanostores'

export const DEFAULT_FILE_BROWSER_HOVER_REVEAL = true

export const $fileBrowserHoverReveal = atom<boolean>(DEFAULT_FILE_BROWSER_HOVER_REVEAL)

/** config.yaml hands back whatever the user wrote — only an explicit `false`
 *  disables pointer-hover reveal; every other value keeps the enabled default. */
export function isFileBrowserHoverRevealEnabled(config: { display?: { hover_reveal_file_browser?: boolean } }): boolean {
  return config.display?.hover_reveal_file_browser !== false
}

export function setFileBrowserHoverRevealFromConfig(config: { display?: { hover_reveal_file_browser?: boolean } }): void {
  $fileBrowserHoverReveal.set(isFileBrowserHoverRevealEnabled(config))
}
