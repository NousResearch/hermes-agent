'use client'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'

/** Heads a card whose request never reached this window: nothing on it can
 *  answer, so say so and point at the way out instead of a silent wait. */
export function UndeliveredNotice() {
  const { t } = useI18n()

  return (
    <Alert className="gap-x-2 px-3 py-2 text-xs" role="status" variant="warning">
      <AlertTriangle />
      <AlertDescription>{t.assistant.clarify.notDelivered}</AlertDescription>
    </Alert>
  )
}
