/**
 * `display.double_enter_continue` — Desktop-only convenience gesture.
 *
 * When enabled, an empty Enter fired shortly after the composer last emptied
 * (the classic "double Enter") submits a filler "Continue" message instead of
 * doing nothing — the fastest way to nudge a paused turn along without typing.
 * Off by default, matching the CLI default in hermes_cli/config_defaults.py;
 * the deliberate empty-Enter no-op stays the shipped behavior.
 *
 * Display/interaction-only: the filler text goes through the normal composer
 * submit path (steer while busy, send when idle), so it never mutates model
 * context behind the conversation — it is just another user message.
 */

import { atom } from 'nanostores'

export const $doubleEnterContinue = atom<boolean>(false)

export function setDoubleEnterContinueFromConfig(value: unknown): void {
  $doubleEnterContinue.set(value === true || value === 'true' || value === 1)
}
