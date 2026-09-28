import { atom } from 'nanostores'

// Mirrors `display.tool_progress`, independent of `display.show_reasoning`:
// hiding thinking must not hide the work. Answer-only is both keys off.
// Parsing matches the gateway's `_load_tool_progress_mode` so the renderer and
// the event stream agree: only false / "off" silence tool rows; every other
// mode (all, new, verbose, unknown, missing) keeps them on.
export const $showToolActivity = atom(true)

export function toolProgressVisible(value: unknown): boolean {
  if (typeof value === 'boolean') {
    return value
  }

  return !(typeof value === 'string' && value.trim().toLowerCase() === 'off')
}

export function setShowToolActivityFromConfig(value: unknown): void {
  $showToolActivity.set(toolProgressVisible(value))
}
