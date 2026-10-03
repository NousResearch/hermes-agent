import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import type { FC } from 'react'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $displayTimestamps } from '@/store/display-timestamps'

import { formatMessageTimestamp, formatTimelineClock, formatTimelineDuration } from './timestamp'

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
  /** Message rows read "Today, 1:02 PM"; event rows read as a duration. */
  friendly?: boolean
  timestamp?: number
}> = ({ className, completedAt, friendly = false, timestamp }) => {
  // One config key everywhere (#41531): `display.timestamps` in config.yaml
  // gates transcript timestamps here exactly as it gates the classic CLI's
  // [HH:MM] labels. Display-only, so toggling never touches model context.
  const enabled = useStore($displayTimestamps)
  const { t } = useI18n()
  const started = unixDate(timestamp)

  if (!enabled || !started || !validUnixSeconds(timestamp)) {
    return null
  }

  const completed = validUnixSeconds(completedAt) && completedAt > timestamp ? unixDate(completedAt) : null

  const validCompletedAt = completed && validUnixSeconds(completedAt) ? completedAt : undefined

  // The default label, not the debug one (#103608): a friendly day+clock for a
  // message, the duration for a settled event — `11:12:34.809 AM → 11:12:55.905
  // AM` was a range printed at reply weight for what is one number — and a
  // plain clock otherwise. The exact boundaries stay in the tooltip below.
  const label = friendly
    ? formatMessageTimestamp(started, t.assistant.thread)
    : validCompletedAt === undefined
      ? formatTimelineClock(timestamp)
      : formatTimelineDuration(timestamp, validCompletedAt)

  const title = completed
    ? `${preciseDateTime.format(started)} → ${preciseDateTime.format(completed)}`
    : preciseDateTime.format(started)

  return (
    <span
      // The same meta ink as the tool rows' stamps (SCAFFOLD_META_CLASS).
      // `text-muted-foreground/55` doubled an alpha: muted-foreground is
      // already a 54% mix of the base ink, so the stamp landed near 30% —
      // faint over a dark chrome and fainter over a light one.
      className={cn('text-[0.625rem] leading-4 tabular-nums text-(--conversation-scaffold-meta)', className)}
      data-slot="timeline-timestamp"
      title={title}
    >
      <time dateTime={started.toISOString()}>{label}</time>
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

    const solePart =
      s.message.parts.length === 1 ? (s.message.parts[0] as { completedAt?: unknown; timestamp?: unknown }) : null

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

  return <TimelineTimestamp className={className} completedAt={completedAt} friendly timestamp={timestamp} />
}
