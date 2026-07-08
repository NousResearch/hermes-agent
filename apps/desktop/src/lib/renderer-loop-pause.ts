interface WindowStatePayload {
  isMinimized?: boolean
  isVisible?: boolean
}

export function createRendererLoopPauseController(onChange: () => void) {
  let windowPaused = false

  const onVisibilityChange = () => onChange()
  const offWindowState = window.hermesDesktop?.onWindowStateChanged?.((payload: WindowStatePayload) => {
    const next = payload?.isMinimized === true || payload?.isVisible === false

    if (windowPaused === next) {
      return
    }

    windowPaused = next
    onChange()
  })

  document.addEventListener('visibilitychange', onVisibilityChange)

  return {
    dispose: () => {
      document.removeEventListener('visibilitychange', onVisibilityChange)
      offWindowState?.()
    },
    isPaused: () => document.visibilityState === 'hidden' || windowPaused
  }
}
