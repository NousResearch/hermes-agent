// Trailing coalescer for gateway-event-driven refreshes. At swarm scale every
// session heartbeat / turn completion wants to re-fetch some REST surface;
// firing them 1:1 turns N busy sessions into an O(N) request storm against the
// (possibly remote, Tailscale'd) backend. schedule() arms a single trailing
// timer — repeat calls inside the window coalesce into one run — and the run
// itself is single-flight: calls arriving while a run is in flight queue at
// most ONE follow-up run after it settles, so results can't interleave.

export interface TrailingCoalescer {
  /** Ask for a run. Coalesces with any already-pending request. */
  schedule: () => void
  /** Drop the pending timer + queued follow-up (unmount cleanup). */
  cancel: () => void
}

export function createTrailingCoalescer(run: () => Promise<unknown> | unknown, delayMs: number): TrailingCoalescer {
  let timer: ReturnType<typeof setTimeout> | null = null
  let inFlight = false
  let rerunQueued = false
  let cancelled = false

  const execute = (): void => {
    if (cancelled) {
      return
    }

    if (inFlight) {
      rerunQueued = true

      return
    }

    inFlight = true
    void Promise.resolve()
      .then(run)
      .catch(() => undefined)
      .then(() => {
        inFlight = false

        if (rerunQueued && !cancelled) {
          rerunQueued = false
          execute()
        }
      })
  }

  return {
    schedule() {
      if (cancelled || timer !== null) {
        return
      }

      timer = setTimeout(() => {
        timer = null
        execute()
      }, delayMs)
    },
    cancel() {
      cancelled = true
      rerunQueued = false

      if (timer !== null) {
        clearTimeout(timer)
        timer = null
      }
    }
  }
}
