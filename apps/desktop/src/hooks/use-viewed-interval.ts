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

    const stop = () => {
      if (intervalId !== null) {
        window.clearInterval(intervalId)
        intervalId = null
      }
    }

    const sync = () => {
      const viewed = document.visibilityState === 'visible' && document.hasFocus()

      if (!viewed) {
        stop()

        return
      }

      if (intervalId === null) {
        callbackRef.current()
        intervalId = window.setInterval(() => callbackRef.current(), intervalMs)
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
