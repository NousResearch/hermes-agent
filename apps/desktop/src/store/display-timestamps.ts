/**
 * `display.timestamps` — one config key for message timestamps everywhere.
 *
 * The same config.yaml key that puts [HH:MM] stamps on classic-CLI labels
 * gates the desktop transcript's per-message / per-tool-run timeline
 * timestamps (#41531, #65272, #68052). Off by default, matching the CLI
 * default in hermes_cli/config_defaults.py.
 *
 * Display-only: reading or toggling it never mutates model context, so it is
 * prompt-cache safe. Hover tooltips with the exact time (#70450) are NOT
 * gated — they add no visual noise until the user asks for them.
 */

import { atom } from 'nanostores'

export const $displayTimestamps = atom<boolean>(false)

export function setDisplayTimestampsFromConfig(value: unknown): void {
  $displayTimestamps.set(value === true || value === 'true' || value === 1)
}
