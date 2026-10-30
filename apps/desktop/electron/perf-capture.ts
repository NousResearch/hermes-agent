/**
 * Env-gated performance capture for the desktop main window.
 *
 * The packaged app deliberately has no remote debugging port (see dev-cdp.ts),
 * so a developer asking "where is the renderer/GPU time actually going" on an
 * installed build has no way in. This module is the way in: with an env var set
 * the main process records a Chromium trace of the window and writes it to
 * disk, and (separately) can ablate every CSS animation to measure what the
 * decorative layer costs.
 *
 * Both hooks are inert unless their env var is set, so a normal launch pays
 * nothing: no tracer, no inserted stylesheet, no timers.
 *
 *   HERMES_DESKTOP_TRACE_MS=8000        record a trace for 8s after load
 *   HERMES_DESKTOP_TRACE_OUT=/path      where to write it (default: userData)
 *   HERMES_DESKTOP_ABLATE_ANIMATIONS=1  insert `animation: none` for every node
 *
 * Categories are the ones that answer the questions we keep asking: how many
 * frames the compositor actually produced, whether paint/raster runs without a
 * commit (non-compositor animation), and where the main thread spent its time.
 */

export const TRACE_CATEGORIES = [
  'toplevel',
  'v8',
  'v8.execute',
  'blink',
  'blink.user_timing',
  'cc',
  'gpu',
  'viz',
  'disabled-by-default-devtools.timeline',
  'disabled-by-default-devtools.timeline.frame',
  'disabled-by-default-v8.cpu_profiler',
  'disabled-by-default-blink.invalidation'
] as const

/** Chromium refuses an empty category list; also keep one runaway list from
 *  producing a gigabyte trace by capping the requested window. */
export const MIN_TRACE_MS = 500
export const MAX_TRACE_MS = 30_000

export type TracerLike = {
  startRecording(options: { recordMode?: string; categories: string[]; traceBufferSizeInKb?: number }): Promise<void> | void
  stopRecording(): Promise<string> | string
}

export interface CaptureTraceOptions {
  tracer: TracerLike
  durationMs: number
  outFile: string
  write: (file: string, data: string) => Promise<void>
  /** Electron's `stopRecording` resolves to the PATH of the trace file, while
   *  other implementations resolve to the JSON itself. Supply `read` so a path
   *  result can be pulled in and written to `outFile` under our own name. */
  read?: (file: string) => Promise<string>
  sleep?: (ms: number) => Promise<void>
  categories?: readonly string[]
}

/** Clamp a requested capture window into something a trace can hold. */
export function traceDurationMs(raw: string | undefined): number | null {
  const parsed = Number(raw)

  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null
  }

  return Math.min(MAX_TRACE_MS, Math.max(MIN_TRACE_MS, Math.round(parsed)))
}

/**
 * Record one trace of `durationMs` and write it to `outFile`. Returns the path,
 * or null when the tracer refused (already tracing) — a dev hook must never
 * take the app down, so every failure is swallowed after the tracer is stopped.
 */
export async function captureTraceOnce({
  tracer,
  durationMs,
  outFile,
  write,
  read,
  sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
  categories = TRACE_CATEGORIES
}: CaptureTraceOptions): Promise<string | null> {
  let started = false

  try {
    await tracer.startRecording({
      categories: [...categories],
      recordMode: 'recordAsMuchAsPossible',
      traceBufferSizeInKb: 100_000
    })
    started = true
    await sleep(durationMs)
    // Stop is claimed before it is awaited: a stop that throws means the tracer
    // is already done with us, and retrying it in `finally` only stacks another
    // rejected promise on top of the real failure.
    started = false
    const result = await tracer.stopRecording()

    if (!result) {
      return null
    }

    // Electron hands back the trace FILE path; some tracers hand back the JSON
    // itself. Normalise both into one payload written under our own name.
    const json = result.trimStart().startsWith('{') ? result : read ? await read(result) : null

    if (!json) {
      return null
    }

    await write(outFile, json)

    return outFile
  } catch {
    return null
  } finally {
    if (started) {
      try {
        await tracer.stopRecording()
      } catch {
        // A tracer that cannot stop is already gone with the process.
      }
    }
  }
}
