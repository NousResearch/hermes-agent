import { useStore } from '@nanostores/react'
import { memo, useCallback, useState } from 'react'

import type { SubmitTextOptions } from '@/app/session/hooks/use-prompt-actions/utils'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $gatewayUpdateDrains, sessionUpdateDrain, updateDrainTurns } from '@/store/gateway-update-drain'
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
  // The cause read from THIS session's own gateway: another gateway's update is not this refusal.
  const updateDrain = sessionUpdateDrain(useStore($gatewayUpdateDrains), sessionId)
  // A refusal because the owner is finishing its turns for an update is that update, not a fault.
  const updating = entry.error === 'runtime_draining' && updateDrain

  const storeError = updating
    ? t.gatewayUpdate.updating(updateDrain.target ?? null, updateDrainTurns(updateDrain))
    : entry.error
      ? ctrl.controlUnavailable(entry.error)
      : null

  const displayError = feedbackError ?? (entry.error === dismissedStoreError ? null : storeError)
  const calm = Boolean(updating) && !feedbackError

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
          className={
            calm
              ? 'flex items-center justify-between gap-2 border-b border-border px-3 py-1.5 text-xs text-muted-foreground'
              : 'flex items-center justify-between gap-2 border-b border-destructive/20 bg-destructive/10 px-3 py-1.5 text-xs text-destructive'
          }
          role={calm ? 'status' : 'alert'}
        >
          <div className="flex items-center gap-1.5 truncate">
            <Codicon className="shrink-0" name={calm ? 'sync' : 'error'} size="0.85rem" />
            <span className="truncate">{displayError}</span>
          </div>
          <Button
            aria-label={ctrl.dismissError}
            className={cn('size-6 shrink-0 p-0', calm ? 'text-muted-foreground' : 'text-destructive hover:bg-destructive/20')}
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
