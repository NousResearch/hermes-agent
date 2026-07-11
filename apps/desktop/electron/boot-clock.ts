/**
 * Boot milestone clock (Phase 0 of the desktop startup-latency work).
 *
 * The desktop.log had NO timestamps, so cold-launch latency could only be
 * measured by external screenshot polling — nobody should ever have to do that
 * again. This module stamps a small set of boot milestones (window-created,
 * cache-paint, handshake-done, list-loaded, transcript-loaded, ready) with a
 * millisecond offset from a single T0 captured as early as possible in the main
 * process, and formats them as `[boot:t+<ms>ms] <milestone>[ <detail>]` lines
 * that go through the existing desktop.log writer.
 *
 * It also formats the Phase-0 cache-observability counters
 * (`[boot] cache-hit|miss` and `[boot] cache-divergence rows=N`) so the reconcile
 * path has a single honest place to emit "was the cached paint ever wrong?".
 *
 * Pure + unit-testable: the clock takes an injectable `now()` (defaults to a
 * monotonic clock) so tests assert exact offsets without wall-clock flake. The
 * actual desktop.log write is done by the caller (main.ts `rememberLog`), so
 * this module never touches the filesystem.
 */

export type BootMilestone =
  | 'process-start'
  | 'app-ready'
  | 'window-created'
  | 'cache-paint'
  | 'handshake-done'
  | 'list-loaded'
  | 'transcript-loaded'
  | 'ready'

/** Monotonic millisecond clock; falls back to Date.now if performance is absent. */
function defaultNow(): number {
  // performance.now() is monotonic and immune to wall-clock steps; Date.now is
  // the fallback for any exotic runtime without a global performance.
  const perf = (globalThis as any).performance
  if (perf && typeof perf.now === 'function') {
    return perf.now()
  }
  return Date.now()
}

export interface BootClock {
  /** Format a milestone line, e.g. `[boot:t+1412ms] window-created`. */
  mark(milestone: BootMilestone, detail?: string): string
  /** Milliseconds elapsed since T0 (rounded, never negative). */
  elapsedMs(): number
  /** The raw T0 value (for tests / continuity). */
  readonly t0: number
}

/**
 * Create a boot clock anchored at `t0` (default: now). Every `mark()` returns a
 * ready-to-log line; it never throws and never writes anywhere.
 */
export function createBootClock(opts: { now?: () => number; t0?: number } = {}): BootClock {
  const now = opts.now ?? defaultNow
  const t0 = opts.t0 ?? now()

  const elapsedMs = (): number => {
    const delta = now() - t0
    // A monotonic source can't go backwards, but clamp defensively so a swapped
    // clock never emits a negative offset that reads as corrupt.
    return Math.max(0, Math.round(delta))
  }

  const mark = (milestone: BootMilestone, detail?: string): string => {
    const base = `[boot:t+${elapsedMs()}ms] ${milestone}`
    const extra = detail == null ? '' : String(detail).trim()
    return extra ? `${base} ${extra}` : base
  }

  return { mark, elapsedMs, t0 }
}

/**
 * Format the cache hit/miss counter line (Opus pass-2 RC4). `rows` is how many
 * cached rows were painted on a hit (omit/undefined on a miss).
 */
export function formatCacheHit(hit: boolean, rows?: number): string {
  if (!hit) {
    return '[boot] cache-miss'
  }
  const n = Number.isFinite(rows as number) ? Math.max(0, Math.round(rows as number)) : 0
  return `[boot] cache-hit rows=${n}`
}

/**
 * Format the reconcile-divergence counter line (Opus pass-2 RC4): how many rows
 * differed between the cached paint and the first live snapshot. `rows=0` means
 * the cache matched live exactly — the signal that "reconciles within one cycle"
 * actually held this launch.
 */
export function formatCacheDivergence(rows: number): string {
  const n = Number.isFinite(rows) ? Math.max(0, Math.round(rows)) : 0
  return `[boot] cache-divergence rows=${n}`
}
