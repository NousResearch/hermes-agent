// Canonical time/date formatting. Shared `Intl` instances (created once, not
// per-render) + relative-time helpers. Every surface that shows a timestamp or
// an age pulls from here so the rendered strings stay consistent app-wide.

export const SECOND = 1000
export const MINUTE = 60_000
export const HOUR = 3_600_000
export const DAY = 86_400_000

// ── Absolute date/time formatters ──────────────────────────────────────────
// `hh:mm` clock (thread today/yesterday lines).
export const fmtClock = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })

// Compact "day + clock", no year/seconds (artifacts, thread fallback, cron runs).
export const fmtDayTime = new Intl.DateTimeFormat(undefined, {
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
  month: 'short'
})

// Medium date + short time (command center session detail).
export const fmtDateTime = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' })

// Date only, "5 Jun 2026" (starmap tooltip).
export const fmtDate = new Intl.DateTimeFormat(undefined, { day: 'numeric', month: 'short', year: 'numeric' })

// Month name alone / with year — session-list date-bucket dividers ("September",
// "September 2025").
export const fmtMonth = new Intl.DateTimeFormat(undefined, { month: 'long' })
export const fmtMonthYear = new Intl.DateTimeFormat(undefined, { month: 'long', year: 'numeric' })

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

// Calendar groups used by the chronological sessions sidebar. Keys are
// technical local-calendar identities: they never contain translated copy.
export type SessionBucketKind = 'day' | 'lastWeek' | 'month' | 'monthYear' | 'today' | 'week' | 'yesterday'

export interface SessionBucket {
  at: number
  key: string
  kind: SessionBucketKind
  rangeEnd?: number
}

export interface SessionBucketLabels {
  lastWeek: string
  thisMonth: string
  thisWeek: string
  today: string
  yesterday: string
}

export const CALENDAR_WEEKS_BEFORE_MONTHS = 4

export const startOfLocalDay = (ms: number): number => {
  const d = new Date(ms)

  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

// The human day doesn't end at midnight — it ends when you sleep. Sessions
// from the small hours belong to the previous evening's run, so the day
// boundary sits at 4 AM local (same trick activity/sleep trackers use).
// A 12:30 AM session groups with 11:50 PM instead of splitting off.
export const DAY_ROLLOVER_HOUR = 4

// Start of the *nominal* local day a timestamp belongs to, honoring the 4 AM
// rollover: Saturday 1 AM → start of Friday.
export const nominalDayStart = (ms: number): number => startOfLocalDay(ms - DAY_ROLLOVER_HOUR * HOUR)

// Locale-aware first day of week in JS getDay() convention (0=Sun … 6=Sat).
// Intl.Locale weekInfo reports 1=Mon … 7=Sun; unsupported → Monday.
export function localeWeekStartDay(): number {
  try {
    const locale = new Intl.Locale(new Intl.DateTimeFormat().resolvedOptions().locale)
    const withWeekInfo = locale as { getWeekInfo?: () => { firstDay?: number }; weekInfo?: { firstDay?: number } }
    const firstDay = (withWeekInfo.getWeekInfo?.() ?? withWeekInfo.weekInfo)?.firstDay

    return typeof firstDay === 'number' ? firstDay % 7 : 1
  } catch {
    return 1
  }
}

// Start of the local calendar week containing `ms` (DST-safe Date field math).
export function startOfLocalWeek(ms: number, weekStartsOn: number): number {
  const d = new Date(startOfLocalDay(ms))
  const back = (d.getDay() - weekStartsOn + 7) % 7

  return new Date(d.getFullYear(), d.getMonth(), d.getDate() - back).getTime()
}

const endOfLocalWeek = (weekStart: number): number => {
  const d = new Date(weekStart)

  return new Date(d.getFullYear(), d.getMonth(), d.getDate() + 6).getTime()
}

const localCalendarOrdinal = (ms: number): number => {
  const d = new Date(ms)

  return Math.floor(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) / DAY)
}

const localDateKey = (ms: number): string => {
  const d = new Date(ms)
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')

  return `${d.getFullYear()}-${month}-${day}`
}

// Today, yesterday, each remaining day of the current locale week, four full
// calendar weeks (the first is "last week"), then exact calendar months.
export function calendarBucket(
  seconds: number,
  nowMs = Date.now(),
  weekStartsOn = localeWeekStartDay()
): SessionBucket {
  const activityDay = startOfLocalDay(seconds * SECOND)
  const today = startOfLocalDay(nowMs)
  const dayDiff = localCalendarOrdinal(today) - localCalendarOrdinal(activityDay)

  if (dayDiff <= 0) {
    return { at: today, key: `day:${localDateKey(today)}`, kind: 'today' }
  }

  if (dayDiff === 1) {
    return { at: activityDay, key: `day:${localDateKey(activityDay)}`, kind: 'yesterday' }
  }

  const currentWeekStart = startOfLocalWeek(today, weekStartsOn)

  if (activityDay >= currentWeekStart) {
    return { at: activityDay, key: `day:${localDateKey(activityDay)}`, kind: 'day' }
  }

  const activityWeekStart = startOfLocalWeek(activityDay, weekStartsOn)
  const weeksBack = (localCalendarOrdinal(currentWeekStart) - localCalendarOrdinal(activityWeekStart)) / 7

  if (weeksBack >= 1 && weeksBack <= CALENDAR_WEEKS_BEFORE_MONTHS) {
    return {
      at: activityWeekStart,
      key: `week:${localDateKey(activityWeekStart)}`,
      kind: weeksBack === 1 ? 'lastWeek' : 'week',
      rangeEnd: endOfLocalWeek(activityWeekStart)
    }
  }

  const d = new Date(activityDay)
  const monthStart = new Date(d.getFullYear(), d.getMonth(), 1).getTime()
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const sameYear = d.getFullYear() === new Date(today).getFullYear()

  return {
    at: monthStart,
    key: `month:${d.getFullYear()}-${month}`,
    kind: sameYear ? 'month' : 'monthYear'
  }
}

const formatCalendarRange = (start: number, end: number, locale: string, nowMs: number): string => {
  const startDate = new Date(start)
  const endDate = new Date(end)
  const currentYear = new Date(nowMs).getFullYear()
  const includeYear = startDate.getFullYear() !== endDate.getFullYear() || endDate.getFullYear() !== currentYear

  const formatter = new Intl.DateTimeFormat(locale, {
    day: 'numeric',
    month: 'long',
    ...(includeYear ? { year: 'numeric' as const } : {})
  })

  return formatter.formatRange(startDate, endDate)
}

export function sessionBucketLabel(
  bucket: SessionBucket,
  labels: SessionBucketLabels,
  locale = new Intl.DateTimeFormat().resolvedOptions().locale,
  nowMs = Date.now()
): string {
  switch (bucket.kind) {
    case 'today':
      return labels.today

    case 'yesterday':
      return labels.yesterday

    case 'day':
      return new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'long', weekday: 'long' }).format(bucket.at)

    case 'lastWeek':
      return labels.lastWeek

    case 'week':
      return formatCalendarRange(bucket.at, bucket.rangeEnd ?? bucket.at, locale, nowMs)

    case 'month':
      return fmtMonth.format(bucket.at)

    case 'monthYear':
      return fmtMonthYear.format(bucket.at)
  }
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

// Localized strings for `formatAgo`; shaped to accept `t.agents` directly.
export interface AgoLabels {
  ageNow: string
  ageSeconds: (seconds: number) => string
  ageMinutes: (minutes: number) => string
  ageHours: (hours: number) => string
  ageDays: (days: number) => string
}

// Compact localized "2h ago" / "3m ago" / "now" for a past timestamp, bucketed
// via `coarseElapsed` so every age label reads consistently.
export function formatAgo(fromMs: number, labels: AgoLabels, nowMs = Date.now()): string {
  const { unit, value } = coarseElapsed(nowMs - fromMs)

  if (unit === 'second') {
    return value < 2 ? labels.ageNow : labels.ageSeconds(value)
  }

  const by = { day: labels.ageDays, hour: labels.ageHours, minute: labels.ageMinutes }

  return by[unit](value)
}
