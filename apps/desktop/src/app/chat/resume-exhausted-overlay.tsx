// Full-window fallback for a routed session whose resume exhausted every
// retry (#106217 remainder). The overlay covers the thread AND the composer
// is hidden underneath it, so a Retry-only action row is a wall with no way
// out — the same dead-end class as the live-owner refusal. Offer Start new
// session (the existing fresh-draft path, same copy as the turn-error card)
// next to Retry. No lease/takeover touch: this only navigates away from the
// stranded route; the failed session and its owner are left intact.
import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { useI18n } from '@/i18n'
import { requestFreshSession } from '@/store/profile'

export function ResumeExhaustedOverlay({
  onRetryResume,
  sessionId
}: {
  onRetryResume: (sessionId: string) => void
  sessionId: string
}) {
  const { t } = useI18n()

  return (
    <div className="absolute inset-0 z-10 grid place-items-center bg-(--ui-chat-surface-background) px-8 py-10">
      <ErrorState
        className="max-w-sm"
        description={t.desktop.resumeStrandedBody}
        title={t.desktop.resumeStrandedTitle}
      >
        <div className="grid justify-items-center gap-1.5">
          <Button onClick={() => requestFreshSession()} size="sm" variant="outline">
            {t.assistant.thread.errorStartNewSession}
          </Button>
          <Button onClick={() => onRetryResume(sessionId)} size="sm" variant="outline">
            {t.desktop.resumeRetry}
          </Button>
        </div>
      </ErrorState>
    </div>
  )
}
