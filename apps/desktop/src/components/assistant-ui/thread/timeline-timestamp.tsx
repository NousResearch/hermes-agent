import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import type { FC } from 'react'

import { cn } from '@/lib/utils'
import { $displayTimestamps } from '@/store/display-timestamps'

import { formatTimelineRange } from './timestamp'

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
  timestamp?: number
}> = ({ className, completedAt, timestamp }) => {
  // One config key everywhere (#41531): `display.timestamps` in config.yaml
  // gates transcript timestamps here exactly as it gates the classic CLI's
  // [HH:MM] labels. Display-only, so toggling never touches model context.
  const enabled = useStore($displayTimestamps)
  const started = unixDate(timestamp)

  if (!enabled || !started || !validUnixSeconds(timestamp)) {
    return null
  }

  const completed =
    validUnixSeconds(completedAt) && completedAt > timestamp ? unixDate(completedAt) : null

  const validCompletedAt = completed && validUnixSeconds(completedAt) ? completedAt : undefined
  const startLabel = formatTimelineRange(timestamp, undefined)
  const completedLabel = validCompletedAt === undefined ? '' : formatTimelineRange(validCompletedAt, undefined)

  const title = completed
    ? `${preciseDateTime.format(started)} → ${preciseDateTime.format(completed)}`
    : preciseDateTime.format(started)

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
  suppressIfDuplicatePart?: boolean
}> = ({ className, suppressIfDuplicatePart = false }) => {
  const timestamp = useAuiState(s => {
    const value = (s.message.metadata?.custom as { timelineTimestamp?: unknown } | undefined)?.timelineTimestamp

    return validUnixSeconds(value) ? value : undefined
  })

  const completedAt = useAuiState(s => {
    const value = (s.message.metadata?.custom as { timelineCompletedAt?: unknown } | undefined)?.timelineCompletedAt

    return validUnixSeconds(value) ? value : undefined
  })

  const duplicatePart = useAuiState(s => {
    const custom = (s.message.metadata?.custom ?? {}) as {
      timelineCompletedAt?: unknown
      timelineTimestamp?: unknown
    }

    const solePart = s.message.parts.length === 1 ? (s.message.parts[0] as { completedAt?: unknown; timestamp?: unknown }) : null

    return (
      Boolean(solePart) &&
      solePart?.timestamp === custom.timelineTimestamp &&
      (solePart?.completedAt === custom.timelineCompletedAt ||
        (!validUnixSeconds(solePart?.completedAt) && !validUnixSeconds(custom.timelineCompletedAt)))
    )
  })

  if (suppressIfDuplicatePart && duplicatePart) {
    return null
  }

  return <TimelineTimestamp className={className} completedAt={completedAt} timestamp={timestamp} />
}
