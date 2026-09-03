import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import type { FC } from 'react'

import { cn } from '@/lib/utils'
import { $displayTimestamps } from '@/store/display-timestamps'

import { formatClockTimestamp, formatTimelineRange } from './timestamp'

const preciseDateTime = new Intl.DateTimeFormat(undefined, {
  day: 'numeric',
  fractionalSecondDigits: 3,
  hour: 'numeric',
  minute: '2-digit',
  month: 'short',
  second: '2-digit',
  year: 'numeric'
})

const validUnixSeconds = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

const unixDate = (value: unknown): Date | null => {
  if (!validUnixSeconds(value)) {
    return null
  }

  const date = new Date(value * 1000)

  return Number.isNaN(date.getTime()) ? null : date
}

export const TimelineTimestamp: FC<{
  className?: string
  completedAt?: number
  /**
   * `exact` keeps the millisecond range used by tool/activity boundaries.
   * `clock` collapses the row to a single minute-precision wall clock — the
   * moment a chat bubble was sent or landed (#41531 follow-up).
   */
  precision?: 'clock' | 'exact'
  timestamp?: number
}> = ({ className, completedAt, precision = 'exact', timestamp }) => {
  // One config key everywhere (#41531): `display.timestamps` in config.yaml
  // gates transcript timestamps here exactly as it gates the classic CLI's
  // [HH:MM] labels. Display-only, so toggling never touches model context.
  const enabled = useStore($displayTimestamps)
  const started = unixDate(timestamp)

  if (!enabled || !started || !validUnixSeconds(timestamp)) {
    return null
  }

  const completed = validUnixSeconds(completedAt) && completedAt > timestamp ? unixDate(completedAt) : null

  const validCompletedAt = completed && validUnixSeconds(completedAt) ? completedAt : undefined

  const title = completed
    ? `${preciseDateTime.format(started)} → ${preciseDateTime.format(completed)}`
    : preciseDateTime.format(started)

  if (precision === 'clock') {
    // A chat bubble answers "when did this arrive", so it shows the settled
    // moment as a single `4:30 PM`. Full precision stays on hover.
    //
    // Two deliberate fallbacks to the start time: a turn that never completed
    // (still streaming, errored, or cancelled) has no completion to show, and
    // an instant turn whose completion equals its start has nothing to add.
    // While streaming, the row therefore shows the send minute and settles to
    // the landing minute — invisible for sub-minute turns, a single quiet
    // change for long ones.
    const landedSeconds = validCompletedAt ?? timestamp

    return (
      <span
        className={cn('text-[0.625rem] leading-4 tabular-nums text-muted-foreground/55', className)}
        data-slot="timeline-timestamp"
        title={title}
      >
        <time dateTime={(completed ?? started).toISOString()}>{formatClockTimestamp(landedSeconds)}</time>
      </span>
    )
  }

  const startLabel = formatTimelineRange(timestamp, undefined)
  const completedLabel = validCompletedAt === undefined ? '' : formatTimelineRange(validCompletedAt, undefined)

  return (
    <span
      className={cn('text-[0.625rem] leading-4 tabular-nums text-muted-foreground/55', className)}
      data-slot="timeline-timestamp"
      title={title}
    >
      <time dateTime={started.toISOString()}>{startLabel}</time>
      {completed && validCompletedAt !== undefined && (
        <>
          {' → '}
          <time dateTime={completed.toISOString()}>{completedLabel}</time>
        </>
      )}
    </span>
  )
}

/** Timestamp for the current assistant-ui message lifecycle. */
export const MessageTimelineTimestamp: FC<{
  className?: string
}> = ({ className }) => {
  const timestamp = useAuiState(s => {
    const value = (s.message.metadata?.custom as { timelineTimestamp?: unknown } | undefined)?.timelineTimestamp

    return validUnixSeconds(value) ? value : undefined
  })

  const completedAt = useAuiState(s => {
    const value = (s.message.metadata?.custom as { timelineCompletedAt?: unknown } | undefined)?.timelineCompletedAt

    return validUnixSeconds(value) ? value : undefined
  })

  return <TimelineTimestamp className={className} completedAt={completedAt} precision="clock" timestamp={timestamp} />
}
