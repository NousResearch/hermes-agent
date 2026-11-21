import { atom, computed } from 'nanostores'

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $providerWaitSessions = atom<Record<string, string>>({})

export function sessionProviderWait(sessionId: null | string) {
  return computed($providerWaitSessions, sessions => sessions[keyFor(sessionId)] ?? '')
}

export function setSessionProviderWait(sessionId: string | null | undefined, text: string): void {
  const key = keyFor(sessionId)
  const sessions = $providerWaitSessions.get()
  const nextText = text.trim()

  if (!nextText) {
    if (!(key in sessions)) {
      return
    }

    const next = { ...sessions }
    delete next[key]
    $providerWaitSessions.set(next)

    return
  }

  if (sessions[key] === nextText) {
    return
  }

  $providerWaitSessions.set({ ...sessions, [key]: nextText })
}

export function clearSessionProviderWait(sessionId: string | null | undefined): void {
  setSessionProviderWait(sessionId, '')
}

export function clearAllProviderWaits(): void {
  $providerWaitSessions.set({})
}

/** Only the core's explained wait/reconnect frames belong in Desktop's status
 * row. Generic kawaii spinner rewrites remain presentation noise. */
export function providerWaitText(text: string): string {
  const value = text.trim()

  return /^(?:⏳|⚠|↻|⚙)\s*(?:(?:still\s+)?waiting on|waiting for a free slot|loading|processing prompt|no (?:output|response)|model returned|rate limited|provider (?:overloaded|temporarily unavailable))/i.test(
    value
  )
    ? value
    : ''
}

/** Parse a managed-local progress frame into bar-renderable parts, or null
 * for every other wait frame. Three shapes, all minted by the backend
 * (the load frame by `agent/chat_completion_local._managed_local_load_notice`,
 * the prefill frame by that function and by the stream monitor's
 * `_show_prompt_progress`, the queue frame by `_managed_local_load_notice`):
 *   "⏳ loading <model> into memory — 43% …"                        -> kind: 'load'
 *   "⚙ processing prompt — 46,500 / 80,000 tokens (58%)"           -> kind: 'prefill'
 *   "⚙ processing prompt — 12,288 / ~39,551 tokens (31%)"          -> kind: 'prefill'
 *       (the "~" total is only an estimate, from the /slots fallback)
 *   "⚙ processing prompt — 12,288 tokens read"                     -> kind: 'prefill'
 *       (no honest denominator: count alone, never a fake percent)
 *   "⏳ waiting for a free slot — …"                                -> kind: 'queued'
 * A percentless prefill frame ("⚙ processing prompt") parses with
 * percent/processed/total all null and renders as label-only, no fake bar.
 * The percents are real — per-tensor load callback / live prefill counter
 * (stream progress chunks) / /slots counter — so a determinate bar is honest. */
export type ModelLoadWait =
  | { kind: 'load'; model: string; percent: number }
  | {
      kind: 'prefill'
      model: ''
      percent: null | number
      /** Tokens the server has read for this call, or null when unreported. */
      processed: null | number
      /** Tokens to read: the server's own total, or null when unknown. */
      total: null | number
      /** True when the total is only the request-token estimate. */
      totalEstimated: boolean
    }
  | { kind: 'queued'; model: '' }

const parseCount = (text: string): number => Number(text.replace(/,/g, ''))
const clampPercent = (percent: number): number => Math.max(0, Math.min(100, percent))

export function parseModelLoadWait(text: string): ModelLoadWait | null {
  const value = text.trim()
  const load = /^⏳\s*loading\s+(.+?)\s+into memory\s+—\s+(\d{1,3})%/i.exec(value)

  if (load) {
    return {
      kind: 'load',
      model: load[1],
      percent: clampPercent(Number(load[2]))
    }
  }

  const queued = /^⏳\s*waiting for a free slot\b/i.exec(value)

  if (queued) {
    return { kind: 'queued', model: '' }
  }

  const prefill = /^⚙\s*processing prompt(?:\s+—\s+(.+))?$/i.exec(value)

  if (prefill) {
    const rest = prefill[1]?.trim()
    const none = {
      kind: 'prefill',
      model: '',
      percent: null,
      processed: null,
      total: null,
      totalEstimated: false
    } as const

    if (rest === undefined || rest === '') {
      return { ...none }
    }

    const counts = /^([\d,]+)\s*\/\s*(~?)\s*([\d,]+)\s*tokens\s*\((\d{1,3})%\)$/.exec(rest)

    if (counts) {
      return {
        kind: 'prefill',
        model: '',
        percent: clampPercent(Number(counts[4])),
        processed: parseCount(counts[1]),
        total: parseCount(counts[3]),
        totalEstimated: counts[2] === '~'
      }
    }

    const read = /^([\d,]+)\s+tokens read$/.exec(rest)

    if (read) {
      return {
        kind: 'prefill',
        model: '',
        percent: null,
        processed: parseCount(read[1]),
        total: null,
        totalEstimated: false
      }
    }

    const percent = /^(\d{1,3})%$/.exec(rest)

    if (percent) {
      return {
        kind: 'prefill',
        model: '',
        percent: clampPercent(Number(percent[1])),
        processed: null,
        total: null,
        totalEstimated: false
      }
    }

    return { ...none }
  }

  return null
}

/** A generic provider wait — the backend's neutral first-chunk / first-event
 * notice (`agent/chat_completion_wait_notice.wait_notice_text`) or the
 * reconnect diagnostic — parsed so the row can name the stage honestly
 * instead of repeating the frame's frozen "Ns" as if it were live. The
 * seconds inside the frame freeze at emission (the backend re-emits only on
 * a phase/watchdog change); the live timer beside the label carries the
 * duration. Returns null for any frame whose shape is not recognised, which
 * renders verbatim as before (older backends, retry/backoff frames). */
export type GenericWait = {
  kind: 'waiting'
  model: string
  phase: 'first-chunk' | 'first-event' | 'post-chunk' | 'post-event' | 'pre-progress' | 'reconnect'
  /** Which watchdog will reconnect, without its frozen countdown. */
  watchdog: null | string
}

export function parseGenericWait(text: string): GenericWait | null {
  const value = text.trim()

  const reconnect = /^⚠\s*no output from provider for \d+s\s+—\s+reconnecting/i.exec(value)

  if (reconnect) {
    return { kind: 'waiting', model: '', phase: 'reconnect', watchdog: null }
  }

  const waiting = /^⏳\s*(?:still\s+)?waiting on\s+(.+?)\s+—\s+(?:\d+s\s+)?(.+)$/i.exec(value)

  if (!waiting) {
    return null
  }

  const model = waiting[1].trim()
  const rest = waiting[2].trim()
  const watchdog = /\(auto-reconnect:\s*(.+?)\s+watchdog in \d+s\)\s*$/i.exec(rest)?.[1]

  if (/waiting for the first stream chunk/i.test(rest)) {
    return { kind: 'waiting', model, phase: 'first-chunk', watchdog: watchdog ? `${watchdog} watchdog` : null }
  }

  if (/waiting for the first provider event/i.test(rest)) {
    return { kind: 'waiting', model, phase: 'first-event', watchdog: watchdog ? `${watchdog} watchdog` : null }
  }

  if (/without substantive model progress/i.test(rest)) {
    return { kind: 'waiting', model, phase: 'pre-progress', watchdog: watchdog ? `${watchdog} watchdog` : null }
  }

  if (/without stream events/i.test(rest)) {
    return { kind: 'waiting', model, phase: 'post-event', watchdog: watchdog ? `${watchdog} watchdog` : null }
  }

  if (/without stream output/i.test(rest)) {
    return { kind: 'waiting', model, phase: 'post-chunk', watchdog: watchdog ? `${watchdog} watchdog` : null }
  }

  return null
}
