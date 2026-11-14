import {
  type ClockHourCycle,
  formatClock,
  formatDayTime,
  formatTimelineClock
} from '@/lib/time'

const timelineDate = (seconds: number | undefined): Date | null => {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds <= 0) {
    return null
  }

  const date = new Date(seconds * 1000)

  return Number.isNaN(date.getTime()) ? null : date
}

/** Millisecond-precise local clock for transcript activity boundaries. */
export function formatTimelineTimestamp(seconds: number | undefined, hourCycle?: ClockHourCycle): string {
  const date = timelineDate(seconds)

  return date ? formatTimelineClock(date, hourCycle) : ''
}

export function formatTimelineRange(
  start: number | undefined,
  end: number | undefined,
  hourCycle?: ClockHourCycle
): string {
  const from = formatTimelineTimestamp(start, hourCycle)

  if (!from) {
    return ''
  }

  const to = formatTimelineTimestamp(end, hourCycle)

  return to ? `${from} → ${to}` : from
}

function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

export function formatMessageTimestamp(
  value: Date | string | number | undefined,
  labels: { today: (time: string) => string; yesterday: (time: string) => string },
  hourCycle?: ClockHourCycle
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
    return labels.today(formatClock(date, hourCycle))
  }

  if (dayDelta === 1) {
    return labels.yesterday(formatClock(date, hourCycle))
  }

  return formatDayTime(date, hourCycle)
}
