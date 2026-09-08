interface WindowStatePayload {
  isMinimized?: boolean
  isVisible?: boolean
}

export const RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE = 'data-renderer-animations-paused'

// `pauseWhenUnfocused` defaults to false. Blur is not "not being looked at":
// on a multi-monitor desktop an unfocused window is usually still fully
// visible, and pausing its animations there reads as the app having frozen.
// Real invisibility (document hidden, or the main process reporting
// minimized/not-visible) still pauses, which is where the renderer-wake saving
// actually mattered. Callers that want blur to pause opt in explicitly.
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
  window.addEventListener('blur', onBlur)
  window.addEventListener('focus', onFocus)

  return {
    dispose: () => {
      document.removeEventListener('visibilitychange', onVisibilityChange)
      window.removeEventListener('blur', onBlur)
      window.removeEventListener('focus', onFocus)
      offWindowState?.()
    },
    isPaused: () => document.visibilityState === 'hidden' || (pauseWhenUnfocused && !windowFocused) || windowPaused
  }
}

/**
 * Mirrors the main window's observability onto :root so continuous decorative
 * CSS animations can sleep with the JS renderer loops. Sleeping is keyed to the
 * window being genuinely invisible (hidden/minimized), never to mere blur.
 * The caller owns the returned cleanup; overlay windows intentionally do not
 * install this state.
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
