interface WindowStatePayload {
  isMinimized?: boolean
  isVisible?: boolean
}

export const RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE = 'data-renderer-animations-paused'

// Minimized/hidden is tracked once per window, not per controller. The bridge
// only reports changes, so a controller that kept its own flag started as
// "shown" whenever it was created while the window was already hidden (a
// StatusPulse remounting in a hidden window played at once) and stayed wrong
// until the next hide or show. One subscription serves every live controller;
// in the main window installRendererAnimationPauseState holds one for the
// window's lifetime, so the state survives any other controller coming and
// going.
type WindowStateSubscribe = (callback: (payload: WindowStatePayload) => void) => (() => void) | undefined

let windowHidden = false
let subscribedBridge: WindowStateSubscribe | undefined
let offBridge: (() => void) | undefined
const windowStateListeners = new Set<() => void>()

function subscribeWindowHidden(listener: () => void): () => void {
  const bridge = window.hermesDesktop?.onWindowStateChanged as WindowStateSubscribe | undefined

  // A different bridge (a test's stub) starts over from "shown".
  if (bridge !== subscribedBridge) {
    offBridge?.()
    subscribedBridge = bridge
    windowHidden = false
    offBridge = bridge?.(payload => {
      const next = payload?.isMinimized === true || payload?.isVisible === false

      if (windowHidden === next) {
        return
      }

      windowHidden = next

      for (const notify of [...windowStateListeners]) {
        notify()
      }
    })
  }

  windowStateListeners.add(listener)

  return () => {
    windowStateListeners.delete(listener)

    if (windowStateListeners.size === 0) {
      offBridge?.()
      offBridge = undefined
      subscribedBridge = undefined
      windowHidden = false
    }
  }
}

export function createRendererLoopPauseController(onChange: () => void, { pauseWhenUnfocused = false } = {}) {
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

  const offWindowState = subscribeWindowHidden(onChange)

  document.addEventListener('visibilitychange', onVisibilityChange)

  if (pauseWhenUnfocused) {
    window.addEventListener('blur', onBlur)
    window.addEventListener('focus', onFocus)
  }

  return {
    dispose: () => {
      document.removeEventListener('visibilitychange', onVisibilityChange)
      window.removeEventListener('blur', onBlur)
      window.removeEventListener('focus', onFocus)
      offWindowState()
    },
    isPaused: () => document.visibilityState === 'hidden' || (pauseWhenUnfocused && !windowFocused) || windowHidden
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
