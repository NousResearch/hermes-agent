import { useAuiState } from '@assistant-ui/react'
import { memo } from 'react'

import { fmtDate, startOfLocalDay } from '@/lib/time'

interface DaySeparatorProps {
  index: number
  previousIndex?: number
}

const timestamp = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined

export const DaySeparator = memo(function DaySeparator({ index, previousIndex }: DaySeparatorProps) {
  // Read only the two group openers, not streamed content or createdAt (which
  // may be a synthetic "now"). Dates must not enter the list's structural key.
  const current = useAuiState(s => timestamp(s.thread.messages[index]?.metadata.custom?.timelineTimestamp))

  const previous = useAuiState(s =>
    previousIndex === undefined ? undefined : timestamp(s.thread.messages[previousIndex]?.metadata.custom?.timelineTimestamp)
  )

  const date = new Date((current ?? NaN) * 1000)

  if (Number.isNaN(date.getTime()) || startOfLocalDay(date.getTime()) === startOfLocalDay((previous ?? NaN) * 1000)) {
    return null
  }

  const label = fmtDate.format(date)

  return (
    <div
      aria-label={label}
      className="flex min-w-0 items-center gap-3 text-xs text-(--ui-text-tertiary)"
      data-slot="aui_day-separator"
      role="separator"
    >
      <span aria-hidden="true" className="h-px min-w-0 flex-1 bg-(--ui-stroke-tertiary)" />
      <time className="min-w-0 text-center wrap-anywhere" dateTime={date.toISOString()}>
        {label}
      </time>
      <span aria-hidden="true" className="h-px min-w-0 flex-1 bg-(--ui-stroke-tertiary)" />
    </div>
  )
})
