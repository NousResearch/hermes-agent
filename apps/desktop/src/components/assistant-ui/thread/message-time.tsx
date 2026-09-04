import { useAuiState } from '@assistant-ui/react'
import { type FC } from 'react'

import { useI18n } from '@/i18n'
import { fmtClock, fmtDayTime } from '@/lib/time'
import { cn } from '@/lib/utils'

function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

/**
 * Friendly timestamp ("Today at 14:32" / "Yesterday at 21:05" / full date
 * for anything older). Inlined here (rather than importing a shared
 * `timestamp.ts`) so this component cannot break when upstream replaces
 * files at that path on a future sync.
 */
function friendlyTimestamp(
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

/**
 * Always-visible wall-clock time for a message bubble ("14:32"). The full
 * friendly form ("Today at 14:32" / "Yesterday at 21:05") rides the hover
 * title — the same labels MessageAge uses in assistant-message.tsx.
 *
 * Renders nothing only when createdAt is missing or unparseable; every real
 * message carries one (toRuntimeMessage → messageCreatedAt), including fresh
 * optimistic rows (they fall back to *now*).
 */
export const MessageTimeLabel: FC<{ className?: string }> = ({ className }) => {
  const { t } = useI18n()
  const createdAt = useAuiState(s => s.message.createdAt)
  const date = createdAt ? new Date(createdAt) : null

  if (!date || Number.isNaN(date.getTime())) {
    return null
  }

  return (
    <span
      className={cn(
        'inline-flex items-center text-[0.6875rem] leading-none tabular-nums text-muted-foreground/60',
        className
      )}
      title={friendlyTimestamp(date, t.assistant.thread) || undefined}
    >
      {fmtClock.format(date)}
    </span>
  )
}