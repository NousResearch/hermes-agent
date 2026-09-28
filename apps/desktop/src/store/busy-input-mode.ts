/**
 * `display.busy_input_mode` — how a plain-text submit behaves mid-turn.
 *
 * `interrupt` (default) redirects the live turn (the composer's historical
 * stop-and-correct); `queue` parks the text as the next turn and lets the
 * current one run to completion — the same routing the classic CLI applies
 * (`hermes_cli/cli_tui_mixin.py`). The desktop composer ignored the key
 * entirely and always steered (#125963).
 *
 * Config-fed and display-only like `display-timestamps`: reading it never
 * mutates model context, so it stays prompt-cache safe. Unknown values fall
 * back to the config default in `hermes_cli/config_defaults.py`.
 */

import { atom } from 'nanostores'

export type BusyInputMode = 'interrupt' | 'queue' | 'steer'

export const $busyInputMode = atom<BusyInputMode>('interrupt')

export function setBusyInputModeFromConfig(value: unknown): void {
  const mode = typeof value === 'string' ? value.trim().toLowerCase() : ''
  $busyInputMode.set(mode === 'queue' || mode === 'steer' ? mode : 'interrupt')
}
