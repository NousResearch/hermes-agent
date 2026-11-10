import { noteVisibleTime } from '@/store/desktop-metrics'

/**
 * Visible window time, the long-frame rate's denominator: credited on visibility changes and on every
 * tick, only for spans the document was visible. A span longer than `maxSpanMs` (the tick period plus
 * slack) means the machine slept with the window up — no tick ran, so none of it is credited.
 */
export function visibleClock(maxSpanMs: number): { credit: () => void } {
  const visibleNow = () => document.visibilityState === 'visible'
  let since: number | null = visibleNow() ? Date.now() : null

  return {
    credit: () => {
      const now = Date.now()

      if (since !== null && now - since <= maxSpanMs) {
        noteVisibleTime(now - since)
      }

      since = visibleNow() ? now : null
    }
  }
}
