import { useEffect, useRef } from 'react'

/** Run a UI-only clock while this document is actually being viewed.
 *
 * macOS can leave an occluded BrowserWindow `visible`, and active streaming
 * deliberately disables Chromium's background timer throttling. Pairing focus
 * with visibility avoids waking React for elapsed labels nobody can see while
 * a leading tick on return catches the UI up immediately.
 */
export function useViewedInterval(callback: () => void, intervalMs: number, enabled = true): void {
  const callbackRef = useRef(callback)

  // eslint-disable-next-line no-restricted-syntax -- latest-callback ref avoids restarting the interval each render
  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    if (!enabled) {
      return
    }

    let intervalId: null | number = null
    // ponytail: shared timer guard — coalesce overlapping ticks + throttle to 80% interval, one fix covers all callers
    let lastInvokeMs = 0
    let inFlight = false

    const stop = () => {
      if (intervalId !== null) {
        window.clearInterval(intervalId)
        intervalId = null
      }
    }

    const guardedCallback = () => {
      const now = Date.now()
      if (inFlight) {
        return
      }
      if (now - lastInvokeMs < Math.max(200, intervalMs * 0.8)) {
        return
      }
      inFlight = true
      lastInvokeMs = now
      try {
        callbackRef.current()
      } finally {
        inFlight = false
      }
    }

    const sync = () => {
      const viewed = document.visibilityState === 'visible' && document.hasFocus()

      if (!viewed) {
        stop()

        return
      }

      if (intervalId === null) {
        guardedCallback()
        intervalId = window.setInterval(guardedCallback, intervalMs)
      }
    }

    window.addEventListener('focus', sync)
    window.addEventListener('blur', sync)
    document.addEventListener('visibilitychange', sync)
    sync()

    return () => {
      stop()
      window.removeEventListener('focus', sync)
      window.removeEventListener('blur', sync)
      document.removeEventListener('visibilitychange', sync)
    }
  }, [enabled, intervalMs])
}
