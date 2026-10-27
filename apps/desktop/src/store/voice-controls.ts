/**
 * `voice.show_controls` — whether the composer shows its voice/dictation
 * controls (the mic hub and, when the row is narrow, the folded voice menu).
 *
 * Some users never dictate and keep hitting the mic instead of Send, so the
 * control row can be trimmed without disabling voice: `false` removes the
 * buttons only. Dictation stays reachable through the record shortcut
 * (`voice.record_key`), and every voice setting keeps working.
 *
 * Defaults to true, matching the CLI default in
 * hermes_cli/config_defaults.py.
 */

import { atom } from 'nanostores'

export const $showVoiceControls = atom<boolean>(true)

export function setShowVoiceControlsFromConfig(value: unknown): void {
  $showVoiceControls.set(value !== false && value !== 'false' && value !== 0)
}
