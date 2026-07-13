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
      data-testid="live-status-strip"
      style={{
        background: 'var(--ui-bg-quaternary)',
        borderLeft: '2px solid var(--ui-stroke-primary)',
        borderRadius: 3,
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        padding: '6px 8px'
      }}
    >
      <div style={{ alignItems: 'baseline', display: 'flex', gap: 6, minWidth: 0 }}>
        <span style={{ color: 'var(--ui-text-primary)', fontSize: 11, fontWeight: 600 }}>{safeProfileLabel}</span>
        <span
          style={{
            color: 'var(--ui-text-secondary)',
            fontSize: 11,
            minWidth: 0,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap'
          }}
        >
          {safeSessionTitle}
        </span>
      </div>

      <div style={{ alignItems: 'center', display: 'flex', flexWrap: 'wrap', gap: 5 }}>
        <span
          aria-live="polite"
          data-testid="live-status-announcement"
          style={{ color: 'var(--ui-text-primary)', fontSize: 11, fontWeight: 600 }}
        >
          {statusLabel}
        </span>
        {hasStartedAt && (
          <span aria-hidden="true" style={{ color: 'var(--ui-text-tertiary)', fontSize: 10 }}>
            {formatElapsed(elapsedSeconds(status.turnStartedAt ?? 0, now))}
          </span>
        )}
        {status.queuedCount > 0 && (
          <span
            style={{
              background: 'var(--ui-bg-tertiary)',
              borderRadius: 3,
              color: 'var(--ui-text-secondary)',
              fontSize: 10,
              padding: '1px 4px'
            }}
          >
            {ac.queuedCount(status.queuedCount)}
          </span>
        )}
        {connection && (
          <span
            style={{
              border: '1px solid var(--ui-stroke-secondary)',
              borderRadius: 3,
              color: 'var(--ui-text-secondary)',
              fontSize: 10,
              padding: '1px 4px'
            }}
          >
            {connection}
          </span>
        )}
      </div>

      {(status.activityKind === 'reasoning' || activityName) && (
        <div style={{ color: 'var(--ui-text-tertiary)', display: 'flex', flexWrap: 'wrap', fontSize: 10, gap: 4 }}>
          {status.activityKind === 'reasoning' && <span>{ac.reasoning}</span>}
          {activityName && <span>{activityName}</span>}
        </div>
      )}
    </div>
  )
}
