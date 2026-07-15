import { useEffect, useState } from 'react'

import type { Translations } from '@/i18n'
import type {
  PetActionCenterLiveStatus,
  PetActionCenterLiveTurnStatus
} from '@/store/pet-action-center'

type ActionCenterStrings = Translations['pet']['actionCenter']

const STATUS_LABEL = {
  done: 'statusDone',
  failed: 'statusFailed',
  idle: 'statusIdle',
  reviewing: 'statusReviewing',
  waiting: 'statusWaiting',
  working: 'statusWorking'
} as const satisfies Record<PetActionCenterLiveTurnStatus, keyof ActionCenterStrings>

export interface LiveStatusStripProps {
  ac: ActionCenterStrings
  profileLabel: string
  sessionTitle: string | null
  status: PetActionCenterLiveStatus
  storedSessionId: string | null
}

function elapsedSeconds(startedAt: number, now: number): number {
  if (!Number.isFinite(startedAt) || !Number.isFinite(now)) {
    return 0
  }

  return Math.max(0, Math.floor((now - startedAt) / 1000))
}

function formatElapsed(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainingSeconds = seconds % 60

  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
    : `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
}

function connectionLabel(ac: ActionCenterStrings, connectionState: PetActionCenterLiveStatus['connectionState']) {
  switch (connectionState) {
    case 'connecting':

    case 'idle':
      return ac.connectionConnecting

    case 'closed':
      return ac.connectionOffline

    case 'error':
      return ac.connectionError

    case 'open':
      return null
  }
}

export function LiveStatusStrip({
  ac,
  profileLabel,
  sessionTitle,
  status,
  storedSessionId
}: LiveStatusStripProps) {
  const [now, setNow] = useState(() => Date.now())
  const hasStartedAt = status.turnStartedAt !== null

  useEffect(() => {
    if (!hasStartedAt) {
      return
    }

    setNow(Date.now())
    const timerId = globalThis.setInterval(() => setNow(Date.now()), 1000)

    return () => globalThis.clearInterval(timerId)
  }, [hasStartedAt, status.turnStartedAt])

  const statusLabel = ac[STATUS_LABEL[status.status]] as string
  const safeProfileLabel = profileLabel.trim() || ac.unknownProfile
  const safeSessionTitle = sessionTitle?.trim() || (storedSessionId ? ac.untitledSession : ac.newSession)
  const connection = connectionLabel(ac, status.connectionState)
  const activityName = status.activityName?.trim() || null

  return (
    <div
      className="flex flex-col gap-1 rounded-md border-l-2 border-(--ui-stroke-primary) bg-(--ui-bg-quaternary) px-2 py-1.5"
      data-testid="live-status-strip"
    >
      <div className="flex min-w-0 items-baseline gap-1.5">
        <span className="text-[0.6875rem] font-semibold text-(--ui-text-primary)">{safeProfileLabel}</span>
        <span className="min-w-0 truncate text-[0.6875rem] text-(--ui-text-secondary)">{safeSessionTitle}</span>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span
          aria-live="polite"
          className="text-[0.6875rem] font-semibold text-(--ui-text-primary)"
          data-testid="live-status-announcement"
        >
          {statusLabel}
        </span>
        {hasStartedAt && (
          <span aria-hidden="true" className="text-[0.625rem] text-(--ui-text-tertiary)">
            {formatElapsed(elapsedSeconds(status.turnStartedAt ?? 0, now))}
          </span>
        )}
        {status.queuedCount > 0 && (
          <span className="rounded-sm bg-(--ui-bg-tertiary) px-1 py-px text-[0.625rem] text-(--ui-text-secondary)">
            {ac.queuedCount(status.queuedCount)}
          </span>
        )}
        {connection && (
          <span className="rounded-sm border border-(--ui-stroke-secondary) px-1 py-px text-[0.625rem] text-(--ui-text-secondary)">
            {connection}
          </span>
        )}
      </div>

      {(status.activityKind === 'reasoning' || activityName) && (
        <div className="flex flex-wrap gap-1 text-[0.625rem] text-(--ui-text-tertiary)">
          {status.activityKind === 'reasoning' && <span>{ac.reasoning}</span>}
          {activityName && <span>{activityName}</span>}
        </div>
      )}
    </div>
  )
}
