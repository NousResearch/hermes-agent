interface WindowStatePayload {
  isMinimized?: boolean
  isVisible?: boolean
}

export const RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE = 'data-renderer-animations-paused'

// Fail-open: with no busy signal the hidden/minimized pause still applies.
// Live controllers subscribe so a busy flip re-syncs CSS and StatusPulse.
let rendererBusyOverride = false
const busyOverrideListeners = new Set<() => void>()

/**
 * Keep renderer activity indicators animating while Hermes is mid-turn,
 * even if the window is occluded or minimized. Idle (false) restores pause.
 */
export function setRendererBusyOverride(busy: boolean): void {
  if (rendererBusyOverride === busy) {
    return
  }

  rendererBusyOverride = busy

  for (const listener of busyOverrideListeners) {
    listener()
  }
}

export function createRendererLoopPauseController(onChange: () => void, { pauseWhenUnfocused = false } = {}) {
  let windowPaused = false
  let windowFocused = document.hasFocus()

  const onVisibilityChange = () => onChange()

  const onBlur = () => {
    if (windowFocused) {
      windowFocused = false
      onChange()
    }
  }

  const onFocus = () => {
    if (!windowFocused) {
      windowFocused = true
      onChange()
    }
  }

  const offWindowState = window.hermesDesktop?.onWindowStateChanged?.((payload: WindowStatePayload) => {
    const next = payload?.isMinimized === true || payload?.isVisible === false

    if (windowPaused === next) {
      return
    }

    windowPaused = next
    onChange()
  })

  document.addEventListener('visibilitychange', onVisibilityChange)

  if (pauseWhenUnfocused) {
    window.addEventListener('blur', onBlur)
    window.addEventListener('focus', onFocus)
  }

  busyOverrideListeners.add(onChange)

  return {
    dispose: () => {
      busyOverrideListeners.delete(onChange)
      document.removeEventListener('visibilitychange', onVisibilityChange)
      window.removeEventListener('blur', onBlur)
      window.removeEventListener('focus', onFocus)
      offWindowState?.()
    },
    isPaused: () =>
      !rendererBusyOverride &&
      (document.visibilityState === 'hidden' || (pauseWhenUnfocused && !windowFocused) || windowPaused)
  }
}

/**
 * Mirrors the main window's observability onto :root so continuous decorative
 * CSS animations can sleep with the JS renderer loops. The caller owns the
 * returned cleanup; overlay windows intentionally do not install this state.
 */
export function installRendererAnimationPauseState(): () => void {
  const root = document.documentElement
  let controller: ReturnType<typeof createRendererLoopPauseController>

  const sync = () => root.toggleAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE, controller.isPaused())

  controller = createRendererLoopPauseController(sync)
  sync()

  return () => {
    controller.dispose()
    root.removeAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)
  }
}
