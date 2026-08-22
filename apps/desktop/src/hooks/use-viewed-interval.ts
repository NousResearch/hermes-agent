import { useEffect, useRef } from 'react'

/** Run a UI-only clock while this document is actually being viewed.
 *
 * "Viewed" is visibility ONLY, deliberately not focus. An unfocused window is
 * usually still fully on screen -- on a multi-monitor desktop it is normal to
 * watch one window while typing in another -- and stopping an elapsed clock
 * there makes a working turn look hung, which is the exact bug this used to
 * cause. Focus was standing in for occlusion (macOS can leave an occluded
 * window `visible`), but a 1s interval is far too cheap to be worth showing a
 * frozen timer to everyone whose window is merely unfocused. A hidden document
 * still stops the clock, and the leading tick on return catches the UI up.
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
      const viewed = document.visibilityState === 'visible'

      if (!viewed) {
        stop()

        return
      }

      if (intervalId === null) {
        callbackRef.current()
        intervalId = window.setInterval(() => callbackRef.current(), intervalMs)
      }
    }

    document.addEventListener('visibilitychange', sync)
    sync()

    return () => {
      stop()
      document.removeEventListener('visibilitychange', sync)
    }
  }, [enabled, intervalMs])
}
