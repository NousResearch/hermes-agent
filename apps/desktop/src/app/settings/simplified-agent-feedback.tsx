import { useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { recordDislike } from '@/store/desktop-metrics'

import { ListRow } from './primitives'

/** Optional inline feedback after switching back; never delays restoring the original sidebar. */
export function SimplifiedAgentFeedback() {
  const { t } = useI18n()
  const copy = t.simplifiedAgents
  const [submitted, setSubmitted] = useState(false)

  if (submitted) {
    return (
      <p className="px-3 py-2 text-xs text-(--ui-text-tertiary)" role="status">
        {copy.feedbackThanks}
      </p>
    )
  }

  return (
    <ListRow
      action={
        <Select
          onValueChange={reason => {
            recordDislike('sidebar_disabled_reason', reason)
            setSubmitted(true)
          }}
        >
          <SelectTrigger aria-label={copy.feedbackTitle}>
            <SelectValue placeholder={copy.feedbackPlaceholder} />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(copy.feedbackReasons).map(([reason, label]) => (
              <SelectItem key={reason} value={reason}>
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
      title={copy.feedbackTitle}
    />
  )
}
