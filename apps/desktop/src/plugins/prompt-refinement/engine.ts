/**
 * One refinement attempt: build the call, run it with a deadline and a cancel
 * signal, and turn every outcome — including the unexpected — into a typed
 * `RefineResult`. Never throws: a caller can only get text back through
 * `{ ok: true }`, and only the review store can turn that into a send.
 */

import { buildRefinementCall, MAX_REFINE_CHARS, type OneShotParams, parseRefinement } from './prompt'
import type { RefinementFailure, RefineRequest, RefineResult } from './types'

/** Just under the backend's own 60 s ceiling on `llm.oneshot`, so our clear
 *  timeout message wins over a generic transport error. */
export const DEFAULT_TIMEOUT_MS = 58_000

/** Performs the model call. `signal` aborts it; `timeoutMs` is the transport's
 *  own ceiling (set above ours so our deadline is the one that fires). */
export type OneShotCaller = (params: OneShotParams, opts: { signal: AbortSignal; timeoutMs: number }) => Promise<string>

interface ErrorShape {
  code?: unknown
  data?: unknown
  message?: unknown
  name?: unknown
}

/** The backend refuses a `task` no loaded plugin registered: the shared gateway answers
 *  `invalid_params` with `data.field === 'task'`, the sidecar puts the reason in the message. */
const UNKNOWN_TASK_RE = /unknown auxiliary task/i

function isUnknownTask(shape: ErrorShape, message: string): boolean {
  const data = (shape.data && typeof shape.data === 'object' ? shape.data : {}) as { detail?: unknown; field?: unknown }

  return (
    data.field === 'task' ||
    UNKNOWN_TASK_RE.test(message) ||
    (typeof data.detail === 'string' && UNKNOWN_TASK_RE.test(data.detail))
  )
}

const UNAVAILABLE_RE =
  /no (?:auxiliary|provider|model|api key|credentials)|not configured|unavailable|auxiliaryclientunavailable|no access token|invalid_grant|\b40[13]\b|unauthori[sz]ed|quota|rate.?limit|insufficient/i

const OFFLINE_RE = /not connected|gateway unavailable|socket|connection (?:closed|lost|refused)|disconnected|network/i

/** Map a transport / RPC failure to the copy the user sees. */
export function classifyFailure(error: unknown): { failure: RefinementFailure; detail?: string } {
  const shape = (error && typeof error === 'object' ? error : {}) as ErrorShape
  const message = typeof shape.message === 'string' ? shape.message : String(error ?? '')
  const detail = message.slice(0, 300) || undefined

  // Checked first: its message ("invalid_params") says nothing the patterns below would read.
  if (isUnknownTask(shape, message)) {
    return { failure: 'slot-missing', detail }
  }

  if (shape.code === -32601 || /method not found|unknown method/i.test(message)) {
    return { failure: 'unsupported', detail }
  }

  if (OFFLINE_RE.test(message)) {
    return { failure: 'offline', detail }
  }

  if (UNAVAILABLE_RE.test(message)) {
    return { failure: 'unavailable', detail }
  }

  if (/timed? ?out|deadline/i.test(message)) {
    return { failure: 'timeout', detail }
  }

  return { failure: 'failed', detail }
}

export async function runRefinement(
  request: RefineRequest,
  call: OneShotCaller,
  options: { signal: AbortSignal; timeoutMs?: number }
): Promise<RefineResult> {
  const text = request.text.trim()

  if (text.length > MAX_REFINE_CHARS) {
    return { ok: false, failure: 'too-long' }
  }

  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const controller = new AbortController()
  let timedOut = false

  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  const forwardAbort = () => controller.abort()

  if (options.signal.aborted) {
    controller.abort()
  } else {
    options.signal.addEventListener('abort', forwardAbort, { once: true })
  }

  try {
    if (controller.signal.aborted) {
      return { ok: false, failure: 'canceled' }
    }

    // Race the call against our own abort: a transport that ignores the signal
    // must still not hold the review open past the deadline or a Cancel.
    const aborted = new Promise<never>((_, reject) =>
      controller.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), {
        once: true
      })
    )

    const raw = await Promise.race([
      call(buildRefinementCall({ ...request, text }), { signal: controller.signal, timeoutMs: timeoutMs + 5_000 }),
      aborted
    ])

    const answer = typeof raw === 'string' ? raw.trim() : ''

    if (!answer) {
      return { ok: false, failure: 'empty' }
    }

    const parsed = parseRefinement(answer)

    if (!parsed) {
      return { ok: false, failure: 'malformed' }
    }

    return {
      ok: true,
      text: parsed.text,
      unchanged: parsed.text === text,
      analysis: parsed.analysis,
      ...(parsed.chosenDepth ? { chosenDepth: parsed.chosenDepth } : {})
    }
  } catch (error) {
    if (options.signal.aborted) {
      return { ok: false, failure: 'canceled' }
    }

    if (timedOut) {
      return { ok: false, failure: 'timeout' }
    }

    return { ok: false, ...classifyFailure(error) }
  } finally {
    clearTimeout(timer)
    options.signal.removeEventListener('abort', forwardAbort)
  }
}
