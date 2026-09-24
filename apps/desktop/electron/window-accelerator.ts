export type WindowAcceleratorAction = 'close-tab' | 'reload' | 'zoom-in' | 'zoom-out' | 'zoom-reset' | 'ignore'

export interface WindowAcceleratorInput {
  alt?: boolean
  control?: boolean
  key?: string
  meta?: boolean
  shift?: boolean
  type?: string
}

/**
 * Classify a main-process before-input-event for Close Tab, Reload, and Zoom.
 *
 * `before-input-event` fires for keydown and keyup. A keyup that arrives after
 * Windows transfers focus (Ctrl+W started in another app) is not a chord this
 * window owns, so only `keyDown` is an accelerator.
 */
export function windowAcceleratorAction(input: WindowAcceleratorInput, isMac: boolean): WindowAcceleratorAction {
  if (input.type !== 'keyDown') {
    return 'ignore'
  }

  const key = String(input.key || '')
  const folded = key.toLowerCase()
  const accel = Boolean((isMac ? input.meta : input.control) && !input.alt)

  if (!accel) {
    return 'ignore'
  }

  if (folded === 'w' && !input.shift) {
    return 'close-tab'
  }

  if (folded === 'r' && !input.shift) {
    return 'reload'
  }

  if (key === '0') {
    // Ctrl/Cmd+Shift+0 is not a zoom chord.
    return input.shift ? 'ignore' : 'zoom-reset'
  }

  if (key === '=' || key === '+') {
    // Zoom-in accepts Shift: on US layouts Plus is physically Shift+=, so
    // Cmd+Plus arrives as Cmd+Shift+'+' or '=' (#43517).
    return 'zoom-in'
  }

  if (key === '-' && !input.shift) {
    // Shift+'-' is '_' on most layouts, not zoom-out.
    return 'zoom-out'
  }

  return 'ignore'
}
