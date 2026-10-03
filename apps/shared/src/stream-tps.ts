/**
 * Streaming throughput (TPS) counter — the shared, surface-agnostic tracker
 * for live "tokens per second" during a streaming response.
 *
 * This is the canonical form of the rolling-window logic historically inlined
 * in the TUI's `turnController.recordMessageDelta` (ui-tui/src/app/turnController.ts).
 * It is a PURE helper: no React/Ink imports, no stores, zero dependencies. Both
 * the TUI and the desktop surface consume it so the live TPS computation stays
 * single-sourced instead of drifting between surfaces.
 *
 * Mechanics (kept 1:1 with the TUI inline version so existing assertions stay
 * green without change):
 *   - `accumulate(now)` pushes the timestamp, then drops any older than ~2 s
 *     (rolling window: timestamps `< now - 2000` are removed).
 *   - It throttles: a rate is (re)computed only when `now - lastEmit >= 1000`.
 *   - With `count = window length`, it returns `Math.round((count/2)*10)/10`
 *     when `count >= 3`, otherwise `undefined` (not enough samples yet).
 *   - `reset()` clears the window and the throttle so a new turn / session
 *     starts from scratch.
 */

/** Closed counter returned by {@link createStreamTpsCounter}. */
export interface StreamTpsCounter {
  /**
   * Record a streaming delta at wall-clock `now` (ms) and return the live
   * TPS rate, or `undefined` when there are not enough samples yet or the
   * ~1 s throttle window since the last emit has not elapsed.
   */
  accumulate(now: number): number | undefined
  /** Drop all accumulated state (start of a TUI turn / end of a desktop session). */
  reset(): void
}

const ROLLING_WINDOW_MS = 2000
const THROTTLE_MS = 1000
const MIN_SAMPLES = 3

/** Create a {@link StreamTpsCounter} returning a closed object. */
export function createStreamTpsCounter(): StreamTpsCounter {
  let _deltaTimestamps: number[] = []
  let _lastTpsEmit = 0

  return {
    accumulate(now: number): number | undefined {
      _deltaTimestamps.push(now)

      // Drop timestamps older than ~2 s (rolling window).
      while (_deltaTimestamps[0]! < now - ROLLING_WINDOW_MS) {
        _deltaTimestamps.shift()
      }

      const age = now - (_lastTpsEmit || 0)

      // Emit only every ~1 s so the status bar doesn't flicker.
      if (age >= THROTTLE_MS) {
        _lastTpsEmit = now
        const count = _deltaTimestamps.length

        // Minimum samples for a meaningful rate.
        return count >= MIN_SAMPLES ? Math.round((count / 2) * 10) / 10 : undefined
      }

      return undefined
    },

    reset(): void {
      _deltaTimestamps = []
      _lastTpsEmit = 0
    }
  }
}
