// Long Animation Frame observation + attribution, shared by the dev-only
// profiler (`src/debug/perf-live.ts`) and the production capture module.
//
// This is the half of a hitch the render counter cannot see: a frame can cost
// 900ms with almost no React in it, and only LoAF says whether that was
// style/layout, a ResizeObserver callback loop, or some timer. The attribution
// pitfalls (entry-type support probing, `buffered` so a frame already in flight
// is still caught, the styleAndLayoutStart → frame-end tail) are solved once
// here rather than duplicated per consumer.
//
// Unlike `debug/`, this module IS production-buildable: it registers nothing
// until a consumer calls `start()`.

/** One Long Animation Frame, attributed. `styleMs` is the engine's style+layout
 *  time inside the frame; `scripts` names who ran JS and for how long. */
export interface LongFrameSample {
  ms: number
  styleMs: number
  blockingMs: number
  scripts: LongFrameScript[]
}

export interface LongFrameScript {
  invoker: string
  ms: number
  src: string
}

export interface LongFrameObserver {
  start: () => void
  stop: () => void
}

// Scripts cheaper than this are noise in every real frame.
const MIN_SCRIPT_MS = 5

// Attribution strings come from the engine, so they can carry element ids,
// handler names and script URLs. Capture records only counts/durations/IDs
// (R2), so every attribution token is reduced at record time to a bounded
// identifier-shaped string and every source URL to its bare file name — no
// path, no query, no origin.
const UNSAFE_TOKEN_CHARS = /[^A-Za-z0-9_.:#$-]+/g
const MAX_TOKEN_LENGTH = 64

const sanitizeToken = (value: string) => value.replace(UNSAFE_TOKEN_CHARS, '_').slice(0, MAX_TOKEN_LENGTH)

const sanitizeSource = (sourceURL: string) => sanitizeToken((sourceURL.split(/[?#]/)[0].split('/').pop() ?? '').trim())

interface LongAnimationFrameEntry extends PerformanceEntry {
  blockingDuration?: number
  styleAndLayoutStart?: number
  renderStart?: number
  scripts?: Array<{
    duration: number
    invoker?: string
    invokerType?: string
    sourceURL?: string
    sourceFunctionName?: string
  }>
}

/** True when this runtime reports `long-animation-frame` entries at all. */
export const supportsLongAnimationFrames = () =>
  typeof PerformanceObserver !== 'undefined' &&
  Boolean(PerformanceObserver.supportedEntryTypes?.includes('long-animation-frame'))

/** Shape one raw LoAF entry into a sanitized sample. Exported for tests and for
 *  consumers that receive entries from somewhere other than the observer. */
export function toLongFrameSample(entry: PerformanceEntry): LongFrameSample {
  const e = entry as LongAnimationFrameEntry

  return {
    blockingMs: Math.round(e.blockingDuration ?? 0),
    ms: Math.round(e.duration),
    scripts: (e.scripts ?? [])
      .filter(s => s.duration >= MIN_SCRIPT_MS)
      .map(s => ({
        invoker: sanitizeToken(`${s.invokerType ?? ''}:${s.invoker ?? s.sourceFunctionName ?? '?'}`),
        ms: Math.round(s.duration),
        src: sanitizeSource(s.sourceURL ?? '')
      })),
    // styleAndLayoutStart -> frame end is the engine's style+layout tail.
    styleMs: e.styleAndLayoutStart ? Math.round(e.startTime + e.duration - e.styleAndLayoutStart) : 0
  }
}

/** Create a LoAF observer that hands each attributed frame to `onSample`.
 *  Returns null when the runtime has no LoAF support — callers keep working
 *  with empty attribution. NOTHING is registered until `start()` is called. */
export function createLongFrameObserver(onSample: (sample: LongFrameSample) => void): LongFrameObserver | null {
  if (!supportsLongAnimationFrames()) {
    return null
  }

  const observer = new PerformanceObserver(list => {
    for (const entry of list.getEntries()) {
      onSample(toLongFrameSample(entry))
    }
  })

  return {
    start: () => {
      try {
        // Buffered so a long frame already in flight when observation starts is
        // still attributed to it.
        observer.observe({ buffered: true, type: 'long-animation-frame' })
      } catch {
        // Older runtime without LoAF — the caller still works, attribution is
        // simply empty.
      }
    },
    stop: () => observer.disconnect()
  }
}
