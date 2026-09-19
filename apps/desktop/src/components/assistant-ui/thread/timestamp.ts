import { fmtClock, fmtDayTime } from '@/lib/time'

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

/** Precise `start → end` range. Tooltip / expanded-detail only — the default
 *  label is a duration (`formatTimelineDuration`). */
export function formatTimelineRange(start: number | undefined, end: number | undefined): string {
  const from = formatTimelineTimestamp(start)

  if (!from) {
    return ''
  }

  const to = formatTimelineTimestamp(end)

  return to ? `${from} → ${to}` : from
}

/** Friendly local clock ("1:02 PM") — no seconds, no milliseconds. The
 *  transcript's default label for an event that has no measured duration. */
export function formatTimelineClock(seconds: number | undefined): string {
  const date = timelineDate(seconds)

  return date ? fmtClock.format(date) : ''
}

/** How long a settled event took, in human units — "21s", "2m 5s", and "<1s"
 *  for anything under a second. A start→end range said the same thing twice at
 *  millisecond precision, which is why the transcript stopped rendering it. */
export function formatTimelineDuration(start: number | undefined, end: number | undefined): string {
  const from = timelineDate(start)
  const to = timelineDate(end)

  if (!from || !to || to.getTime() < from.getTime()) {
    return ''
  }

  const seconds = Math.floor((to.getTime() - from.getTime()) / 1000)

  if (seconds < 1) {
    return '<1s'
  }

  const minutes = Math.floor(seconds / 60)

  return minutes < 1 ? `${seconds}s` : `${minutes}m ${seconds % 60}s`
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
