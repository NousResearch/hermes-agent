import type { FC } from 'react'

import { formatMessageTimestamp } from '@/components/assistant-ui/thread/timestamp'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { formatAgo } from '@/lib/time'

interface MessageAgeProps {
  createdAt: Date | string | number | undefined
}

export const MessageAge: FC<MessageAgeProps> = ({ createdAt }) => {
  const { t } = useI18n()
  const date = createdAt instanceof Date ? createdAt : createdAt ? new Date(createdAt) : null

  if (!date || Number.isNaN(date.getTime())) {
    return null
  }

  const age = formatAgo(date.getTime(), t.agents)
  const absoluteAge = formatMessageTimestamp(date, t.assistant.thread)

  return (
    <Tip label={absoluteAge} side="top">
      <time
        aria-label={`${age}, ${absoluteAge}`}
        className="px-0.5 text-[0.6875rem] tabular-nums text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        dateTime={date.toISOString()}
        tabIndex={0}
      >
        {age}
      </time>
    </Tip>
  )
}
