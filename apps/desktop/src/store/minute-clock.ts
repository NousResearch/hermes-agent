import { atom, onMount } from 'nanostores'

// Wall clock for coarse age labels (the sidebar's "now", "12m", "3h", "18d").
//
// A row's age is derived from Date.now() at render, but memoized rows only
// re-render when their data changes, and the session list is event-driven and
// signature-gated: between list changes an idle row never re-renders, so its
// age froze at whatever it read last. Rows subscribe to this atom through a selector that
// returns the label string, so a tick re-renders only the rows whose label
// actually moved.
//
// Ticks only while something listens (nanostores onMount), and catches up at
// once when the window comes back into view, since background timers throttle.
export const MINUTE_CLOCK_INTERVAL_MS = 30_000

export const $minuteClock = atom(Date.now())

function tick() {
  $minuteClock.set(Date.now())
}

function tickWhenVisible() {
  if (document.visibilityState === 'visible') {
    tick()
  }
}

onMount($minuteClock, () => {
  tick()
  const intervalId = setInterval(tick, MINUTE_CLOCK_INTERVAL_MS)
  const hasDom = typeof document !== 'undefined' && typeof window !== 'undefined'

  if (hasDom) {
    document.addEventListener('visibilitychange', tickWhenVisible)
    window.addEventListener('focus', tick)
  }

  return () => {
    clearInterval(intervalId)

    if (hasDom) {
      document.removeEventListener('visibilitychange', tickWhenVisible)
      window.removeEventListener('focus', tick)
    }
  }
})
