import { memo, useCallback, useState } from 'react'

import type { SubmitTextOptions } from '@/app/session/hooks/use-prompt-actions/utils'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import type { SessionControlEntry } from '@/store/session-control'

import { SessionControlGoalSection } from './session-control-goal'
import { SessionControlHeartbeatSection } from './session-control-heartbeat'
import { SessionControlLoopSection } from './session-control-loop'

export interface SessionControlSectionsProps {
  entry: SessionControlEntry
  onSubmit?: (value: string, options?: SubmitTextOptions) => Promise<boolean> | boolean
  sessionId: string
}

export const SessionControlSections = memo(function SessionControlSections({
  entry,
  onSubmit,
  sessionId
}: SessionControlSectionsProps) {
  const { t } = useI18n()
  const ctrl = t.statusStack.control

  const [feedbackError, setFeedbackError] = useState<string | null>(null)
  const [feedbackSuccess, setFeedbackSuccess] = useState<string | null>(null)
  const [dismissedStoreError, setDismissedStoreError] = useState<string | null>(null)

  const handleFeedback = useCallback((error: string | null, success: string | null) => {
    setFeedbackError(error)
    setFeedbackSuccess(success)
  }, [])

  // Action failures arrive through onFeedback (which wins below); a store
  // error with no feedback is a read/hydration failure, not a failed action.
  const storeError = entry.error ? ctrl.controlUnavailable(entry.error) : null
  const displayError = feedbackError ?? (entry.error === dismissedStoreError ? null : storeError)

  const handleDismissError = useCallback(() => {
    if (feedbackError) {
      setFeedbackError(null)
    } else {
      setDismissedStoreError(entry.error)
    }
  }, [entry.error, feedbackError])

  const snapshot = entry.snapshot

  if (!snapshot && !displayError) {
    return null
  }

  return (
    <>
      {displayError && (
        <div
          className="flex items-center justify-between gap-2 border-b border-destructive/20 bg-destructive/10 px-3 py-1.5 text-xs text-destructive"
          role="alert"
        >
          <div className="flex items-center gap-1.5 truncate">
            <Codicon className="shrink-0" name="error" size="0.85rem" />
            <span className="truncate">{displayError}</span>
          </div>
          <Button
            aria-label={ctrl.dismissError}
            className="size-6 shrink-0 p-0 text-destructive hover:bg-destructive/20"
            onClick={handleDismissError}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="close" size="0.75rem" />
          </Button>
        </div>
      )}
      {feedbackSuccess && (
        <div aria-live="polite" className="sr-only">
          {feedbackSuccess}
        </div>
      )}
      {snapshot?.goal && (
        <SessionControlGoalSection
          goal={snapshot.goal}
          onFeedback={handleFeedback}
          onSubmit={onSubmit}
          pendingAction={entry.pendingAction}
          sessionId={sessionId}
        />
      )}
      {snapshot?.loop && (
        <SessionControlLoopSection
          loop={snapshot.loop}
          onFeedback={handleFeedback}
          pendingAction={entry.pendingAction}
          sessionId={sessionId}
        />
      )}
      {snapshot?.heartbeat && (
        <SessionControlHeartbeatSection
          heartbeat={snapshot.heartbeat}
          onFeedback={handleFeedback}
          pendingAction={entry.pendingAction}
          sessionId={sessionId}
        />
      )}
    </>
  )
})
