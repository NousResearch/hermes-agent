interface WindowStatePayload {
  isMinimized?: boolean
  isVisible?: boolean
}

/** Hard pause: the window is hidden or minimized, nothing should animate. */
export const RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE = 'data-renderer-animations-paused'

/** Soft pause: the window is visible but has no keyboard focus. Only
 *  decoration sleeps — status cues (spinners, the streaming cursor, progress
 *  and timeline bars) keep moving, because a frozen spinner on a window the
 *  user is actually watching reads as "hung", not "parked". */
export const RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE = 'data-renderer-animations-unfocused'

export function createRendererLoopPauseController(onChange: () => void, { pauseWhenUnfocused = false } = {}) {
  let windowPaused = false
  let windowFocused = document.hasFocus()

  const onVisibilityChange = () => onChange()

  const onBlur = () => {
    windowFocused = false
    onChange()
  }

  const onFocus = () => {
    windowFocused = true
    onChange()
  }

  // A missed `focus` event (system dialog, IME handoff, a window manager that
  // does focus-follows-mouse) must not strand the pause. Any input event while
  // the document really is focused is proof the latch is stale, so heal it.
  const healIfFocused = () => {
    if (!windowFocused && document.hasFocus()) {
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
    // Capture phase so a child that stops propagation cannot hide the heal.
    window.addEventListener('pointerdown', healIfFocused, true)
    window.addEventListener('keydown', healIfFocused, true)
    document.addEventListener('visibilitychange', healIfFocused)
  }

  const unfocused = () =>
    pauseWhenUnfocused &&
    !windowPaused &&
    document.visibilityState === 'visible' &&
    // Latch AND live check: the latch catches every blur, the live check means
    // a stuck latch cannot outlive the next moment the document really has
    // focus. Requiring both means neither alone can stall a visible UI.
    !windowFocused &&
    !document.hasFocus()

  return {
    dispose: () => {
      document.removeEventListener('visibilitychange', onVisibilityChange)
      window.removeEventListener('blur', onBlur)
      window.removeEventListener('focus', onFocus)
      window.removeEventListener('pointerdown', healIfFocused, true)
      window.removeEventListener('keydown', healIfFocused, true)
      document.removeEventListener('visibilitychange', healIfFocused)
      offWindowState?.()
    },
    /** Hidden or minimized: everything stops. */
    isPaused: () => document.visibilityState === 'hidden' || windowPaused,
    /** Visible but unfocused: decoration stops, status keeps moving. */
    isUnfocused: unfocused
  }
}

/**
 * Mirrors the main window's observability onto :root so continuous decorative
 * CSS animations can sleep with the JS renderer loops. The caller owns the
 * returned cleanup; overlay windows intentionally do not install this state.
 */
export interface RendererAnimationPauseOptions {
  /** Also stop decorative animations while the window is visible but has no
   *  keyboard focus. Chromium only throttles an occluded/minimized page, so a
   *  visible-but-backgrounded main window otherwise paints every infinite
   *  animation (session arcs, shimmer, glows) at display cadence — measured at
   *  30-50% of a core with the window behind another app. Only the primary
   *  window should opt in: a popped-out session window is often the one the
   *  user is watching from another display, where "unfocused" does not mean
   *  "unwatched". Defaults to false so overlays keep the old always-animate
   *  look. */
  pauseWhenUnfocused?: boolean
}

export function installRendererAnimationPauseState({
  pauseWhenUnfocused = false
}: RendererAnimationPauseOptions = {}): () => void {
  const root = document.documentElement
  let controller: ReturnType<typeof createRendererLoopPauseController>

  const sync = () => {
    const paused = controller.isPaused()
    root.toggleAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE, paused)
    root.toggleAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE, !paused && controller.isUnfocused())
  }

  controller = createRendererLoopPauseController(sync, { pauseWhenUnfocused })
  sync()

  return () => {
    controller.dispose()
    root.removeAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)
    root.removeAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)
  }
}
