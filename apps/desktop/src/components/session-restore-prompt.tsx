import { useStore } from '@nanostores/react'
import { useCallback } from 'react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Clock } from '@/lib/icons'
import {
  $sessionRestorePromptVisible,
  $sessionRestoreSnapshot
} from '@/store/session-restore'

interface SessionRestorePromptProps {
  onRestore: () => Promise<void> | void
  onDiscard: () => Promise<void> | void
}

function formatRelativeTime(createdAt: number): string {
  const seconds = Math.floor((Date.now() - createdAt) / 1000)

  if (seconds < 60) {
    return 'just now'
  }

  const minutes = Math.floor(seconds / 60)

  if (minutes < 60) {
    return minutes === 1 ? '1 minute ago' : `${minutes} minutes ago`
  }

  const hours = Math.floor(minutes / 60)

  if (hours < 24) {
    return hours === 1 ? '1 hour ago' : `${hours} hours ago`
  }

  const days = Math.floor(hours / 24)

  return days === 1 ? '1 day ago' : `${days} days ago`
}

export function SessionRestorePrompt({ onRestore, onDiscard }: SessionRestorePromptProps) {
  const visible = useStore($sessionRestorePromptVisible)
  const snapshot = useStore($sessionRestoreSnapshot)
  const { t } = useI18n()

  const handleRestore = useCallback(async () => {
    await onRestore()
  }, [onRestore])

  const handleDiscard = useCallback(async () => {
    await onDiscard()
  }, [onDiscard])

  if (!visible || !snapshot || snapshot.entries.length === 0) {
    return null
  }

  const copy = t.sessionRestore
  const count = snapshot.entries.length
  const timeLabel = copy.timestamp(formatRelativeTime(snapshot.createdAt))

  return (
    <div className="pointer-events-none fixed left-1/2 top-[calc(var(--titlebar-height,34px)+0.75rem)] z-[200] w-[min(32rem,calc(100%-2rem))] -translate-x-1/2">
      <Alert className="pointer-events-auto grid-cols-[auto_minmax(0,1fr)] border-(--stroke-nous) bg-popover/95 pb-2.5 pt-2.5 shadow-nous backdrop-blur-md">
        <Clock className="mt-0.5 text-muted-foreground" />
        <div className="col-start-2 grid gap-1.5">
          <div>
            <AlertTitle className="mb-0.5 text-[0.9375rem] font-semibold">{copy.title}</AlertTitle>
            <AlertDescription className="text-[0.8125rem] leading-5 text-(--ui-text-tertiary)">
              <p className="m-0">{copy.body(count)}</p>
              <p className="m-0 mt-0.5 text-[0.75rem] text-(--ui-text-quaternary)">{timeLabel}</p>
            </AlertDescription>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              onClick={() => void handleRestore()}
              size="sm"
            >
              {copy.restore}
            </Button>
            <Button
              onClick={() => void handleDiscard()}
              size="sm"
              variant="secondary"
            >
              {copy.discard}
            </Button>
          </div>
        </div>
      </Alert>
    </div>
  )
}
