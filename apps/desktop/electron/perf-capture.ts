/**
 * Env-gated performance capture for the desktop main window.
 *
 * The packaged app deliberately has no remote debugging port (see dev-cdp.ts),
 * so a developer asking "where is the renderer/GPU time actually going" on an
 * installed build has no way in. This module is the way in: with an env var set
 * the main process records a Chromium trace of the window and writes it to
 * disk. A second hook can ablate every CSS animation, which is how the
 * decorative layer's cost was attributed.
 *
 * All hooks are inert unless their env var is set, so a normal launch pays
 * nothing: no tracer, no inserted stylesheet, no timers.
 *
 *   HERMES_DESKTOP_TRACE_MS=8000        record a trace for 8s after load
 *   HERMES_DESKTOP_TRACE_OUT=/path      where to write it (REQUIRED)
 *   HERMES_DESKTOP_ABLATE_ANIMATIONS=1  insert `animation: none` for every node
 *
 * PRIVACY: a Chromium trace is not a neutral artefact. `devtools.timeline`
 * records every resource URL the renderer requested (query strings included,
 * and those can carry tokens), `v8.cpu_profiler` records script URLs and sampled
 * function names, and `blink.user_timing` records app marks. It does not
 * generally contain DOM text or keystrokes, but it is still diagnostic data
 * about one person's session: write it somewhere private, never attach it to a
 * public issue, and delete it once the question is answered. There is
 * deliberately no default destination — a capture needs an explicit path, so a
 * stray env var cannot litter a user's profile with 100MB files.
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

/** `stopRecording` flushes the whole trace before it resolves. If it never
 *  settles, the trace buffer stays pinned for the life of the process, so the
 *  wait is bounded and the failure is reported rather than swallowed. */
export const STOP_TIMEOUT_MS = 30_000

export type TraceFailure = 'start-refused' | 'stop-failed' | 'stop-timeout' | 'empty' | 'read-failed' | 'write-failed'

export type TraceResult = { ok: true; path: string } | { ok: false; reason: TraceFailure }

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
   *  result can be pulled in, and `remove` so the tracer's copy is deleted
   *  afterwards — two 100MB copies per capture is not a diagnostic aid. */
  read?: (file: string) => Promise<string>
  remove?: (file: string) => Promise<void>
  sleep?: (ms: number) => Promise<void>
  stopTimeoutMs?: number
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
 * Record one trace of `durationMs` and write it to `outFile`. Never throws: a
 * diagnostic hook must not take the app down. It also never claims a capture
 * that did not happen — every failure names itself, because the one line this
 * tool prints is the only evidence the developer has.
 */
export async function captureTraceOnce({
  tracer,
  durationMs,
  outFile,
  write,
  read,
  remove,
  sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
  stopTimeoutMs = STOP_TIMEOUT_MS,
  categories = TRACE_CATEGORIES
}: CaptureTraceOptions): Promise<TraceResult> {
  let started = false

  try {
    await tracer.startRecording({
      categories: [...categories],
      recordMode: 'recordAsMuchAsPossible',
      traceBufferSizeInKb: 100_000
    })
    started = true
  } catch {
    return { ok: false, reason: 'start-refused' }
  }

  try {
    await sleep(durationMs)

    // Claim the stop before awaiting it: a stop that rejects means the tracer is
    // already done with us, and retrying in `finally` only stacks another
    // rejected promise on top of the real failure.
    started = false

    let timer: ReturnType<typeof setTimeout> | undefined
    let result: string

    try {
      result = await Promise.race([
        Promise.resolve(tracer.stopRecording()),
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Error('stopRecording timed out')), stopTimeoutMs)
        })
      ])
    } catch (error) {
      const timedOut = error instanceof Error && error.message === 'stopRecording timed out'

      return { ok: false, reason: timedOut ? 'stop-timeout' : 'stop-failed' }
    } finally {
      if (timer !== undefined) {
        clearTimeout(timer)
      }
    }

    if (!result) {
      return { ok: false, reason: 'empty' }
    }

    // Electron hands back the trace FILE path; some tracers hand back the JSON
    // itself. Normalise both into one payload written under our own name, then
    // drop the tracer's copy so a capture leaves exactly one file behind.
    let json: string | null = null

    if (result.trimStart().startsWith('{')) {
      json = result
    } else if (read) {
      try {
        json = await read(result)
      } catch {
        return { ok: false, reason: 'read-failed' }
      } finally {
        try {
          await remove?.(result)
        } catch {
          // A leftover tracer copy is untidy, not fatal.
        }
      }
    }

    if (!json) {
      return { ok: false, reason: 'empty' }
    }

    try {
      await write(outFile, json)
    } catch {
      return { ok: false, reason: 'write-failed' }
    }

    return { ok: true, path: outFile }
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
