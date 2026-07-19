// Canonical time/date formatting. Shared `Intl` instances (created once, not
// per-render) + relative-time helpers. Every surface that shows a timestamp or
// an age pulls from here so the rendered strings stay consistent app-wide.

export const SECOND = 1000
export const MINUTE = 60_000
export const HOUR = 3_600_000
export const DAY = 86_400_000

// ── Absolute date/time formatters ──────────────────────────────────────────
const DAY_TIME_OPTIONS: Intl.DateTimeFormatOptions = {
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
  month: 'short'
}

const DATE_TIME_OPTIONS: Intl.DateTimeFormatOptions = { dateStyle: 'medium', timeStyle: 'short' }
const DATE_OPTIONS: Intl.DateTimeFormatOptions = { day: 'numeric', month: 'short', year: 'numeric' }

// `hh:mm` clock (thread today/yesterday lines).
export const fmtClock = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })

// Compact "day + clock", no year/seconds (artifacts, thread fallback, cron runs).
export const fmtDayTime = new Intl.DateTimeFormat(undefined, DAY_TIME_OPTIONS)

// Medium date + short time (command center session detail).
export const fmtDateTime = new Intl.DateTimeFormat(undefined, DATE_TIME_OPTIONS)

// Date only, "5 Jun 2026" (starmap tooltip).
export const fmtDate = new Intl.DateTimeFormat(undefined, DATE_OPTIONS)

// ── UI-locale-aware date formatters ────────────────────────────────────────
// Dates keep following the OS/browser locale, exactly as they always have: an
// en-GB user must still read `15 Jun 2025` whichever UI language is selected,
// so every non-Arabic locale resolves to `undefined` and reuses the shared
// instance above. Arabic is the sole opt-in — its digits and month names have
// no Latin fallback. Instances are built once per locale and cached, never
// rebuilt inside a render loop.

/** `undefined` (= OS/browser locale) unless the UI locale needs its own script. */
export function dateLocaleTag(locale: string): string | undefined {
  return locale === 'ar' ? 'ar-EG' : undefined
}

function byLocale(shared: Intl.DateTimeFormat, options: Intl.DateTimeFormatOptions) {
  const cache = new Map<string, Intl.DateTimeFormat>()

  return (locale: string): Intl.DateTimeFormat => {
    const tag = dateLocaleTag(locale)

    if (tag === undefined) {
      return shared
    }

    const cached = cache.get(tag)

    if (cached) {
      return cached
    }

    const formatter = new Intl.DateTimeFormat(tag, options)
    cache.set(tag, formatter)

    return formatter
  }
}

export const dayTimeFor = byLocale(fmtDayTime, DAY_TIME_OPTIONS)
export const dateTimeFor = byLocale(fmtDateTime, DATE_TIME_OPTIONS)
export const dateFor = byLocale(fmtDate, DATE_OPTIONS)

// ── Relative time ──────────────────────────────────────────────────────────
const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto', style: 'short' })

// Localized bidirectional "in 5 min" / "2 hr ago" — coarsest sensible unit so a
// daily job reads "in 14 hr", not "in 840 min".
export function relativeTime(targetMs: number, nowMs = Date.now()): string {
  const diff = targetMs - nowMs
  const abs = Math.abs(diff)
  const sign = diff < 0 ? -1 : 1

  if (abs < MINUTE) {
    return rtf.format(sign * Math.round(abs / SECOND), 'second')
  }

  if (abs < HOUR) {
    return rtf.format(sign * Math.round(abs / MINUTE), 'minute')
  }

  if (abs < DAY) {
    return rtf.format(sign * Math.round(abs / HOUR), 'hour')
  }

  return rtf.format(sign * Math.round(abs / DAY), 'day')
}

export type ElapsedUnit = 'day' | 'hour' | 'minute' | 'second'

// Coarsest elapsed bucket for a (clamped-nonnegative) duration, floored. The
// caller owns rendering — compact "5m", "5m ago", etc. — so no format is baked
// in here.
export function coarseElapsed(deltaMs: number): { unit: ElapsedUnit; value: number } {
  const ms = Math.max(0, deltaMs)

  if (ms >= DAY) {
    return { unit: 'day', value: Math.floor(ms / DAY) }
  }

  if (ms >= HOUR) {
    return { unit: 'hour', value: Math.floor(ms / HOUR) }
  }

  if (ms >= MINUTE) {
    return { unit: 'minute', value: Math.floor(ms / MINUTE) }
  }

  return { unit: 'second', value: Math.floor(ms / SECOND) }
}
