import { useMemo, useState } from 'react'

import { useViewedInterval } from '@/hooks/use-viewed-interval'

const MINUTE_MS = 60_000

export interface StatusbarDateTimeParts {
  date: string
  time: string
}

export function formatStatusbarDateTime(
  value: Date,
  locales?: Intl.LocalesArgument,
  timeZone?: string
): StatusbarDateTimeParts {
  return {
    date: new Intl.DateTimeFormat(locales, {
      day: 'numeric',
      month: 'short',
      timeZone,
      weekday: 'short'
    }).format(value),
    time: new Intl.DateTimeFormat(locales, {
      hour: '2-digit',
      minute: '2-digit',
      timeZone
    }).format(value)
  }
}

function currentMinute(): number {
  const now = Date.now()

  return now - (now % MINUTE_MS)
}

export function StatusbarDateTime() {
  const [minute, setMinute] = useState(currentMinute)

  // The one-second probe catches the minute boundary promptly, but React only
  // receives a new scalar once per minute. The shared viewed-interval hook
  // parks the probe while the window is hidden or unfocused.
  useViewedInterval(() => setMinute(currentMinute()), 1_000)

  const value = useMemo(() => new Date(minute), [minute])
  const parts = useMemo(() => formatStatusbarDateTime(value), [value])

  return (
    <time
      className="inline-flex items-center gap-1.5 whitespace-nowrap font-medium tracking-[-0.01em] tabular-nums"
      dateTime={value.toISOString()}
    >
      <span className="text-(--ui-text-secondary)">{parts.date}</span>
      <span aria-hidden="true" className="size-0.5 rounded-full bg-(--ui-text-quaternary)" />
      <span className="text-foreground/90">{parts.time}</span>
    </time>
  )
}
