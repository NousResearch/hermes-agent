import { fmtClock, fmtDayTime } from '@/lib/time'

const fmtMessageClock = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  hour12: true,
  minute: '2-digit'
})

const fmtTimelineClock = new Intl.DateTimeFormat(undefined, {
  fractionalSecondDigits: 3,
  hour: 'numeric',
  minute: '2-digit',
  second: '2-digit'
})

const timelineDate = (seconds: number | undefined): Date | null => {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds <= 0) {
    return null
  }

  const date = new Date(seconds * 1000)

  return Number.isNaN(date.getTime()) ? null : date
}

/** Millisecond-precise local clock for transcript activity boundaries. */
export function formatTimelineTimestamp(seconds: number | undefined): string {
  const date = timelineDate(seconds)

  return date ? fmtTimelineClock.format(date) : ''
}

/**
 * Minute-precision wall clock (e.g. `4:30 PM`) for message bubbles. Chat rows
 * answer "when was this sent / when did it land", not "how long did it run",
 * so they deliberately drop the seconds and milliseconds that tool activity
 * boundaries still need.
 */
export function formatClockTimestamp(seconds: number | undefined): string {
  const date = timelineDate(seconds)

  return date ? fmtMessageClock.format(date) : ''
}

export function formatTimelineRange(start: number | undefined, end: number | undefined): string {
  const from = formatTimelineTimestamp(start)

  if (!from) {
    return ''
  }

  const to = formatTimelineTimestamp(end)

  return to ? `${from} → ${to}` : from
}

function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

export function formatMessageTimestamp(
  value: Date | string | number | undefined,
  labels: { today: (time: string) => string; yesterday: (time: string) => string }
): string {
  if (!value) {
    return ''
  }

  const date = value instanceof Date ? value : new Date(value)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  const dayDelta = Math.round((startOfDay(new Date()) - startOfDay(date)) / 86_400_000)

  if (dayDelta === 0) {
    return labels.today(fmtClock.format(date))
  }

  if (dayDelta === 1) {
    return labels.yesterday(fmtClock.format(date))
  }

  return fmtDayTime.format(date)
}
