import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { type CloudChannelParticipant, loadCloudChannelParticipants, loadSessionCloudStatus } from '@/lib/cloud-share'
import { cn } from '@/lib/utils'
import { $localDeviceName, $sessionParticipants, type SessionParticipant } from '@/store/session'

interface ParticipantChipsProps {
  sessionId: string
}

const CLOUD_ROSTER_REFRESH_MS = 5000

const mergeParticipants = (local: SessionParticipant[], cloud: CloudChannelParticipant[]): SessionParticipant[] => {
  const merged = new Map<string, SessionParticipant>()

  for (const participant of [...local, ...cloud]) {
    const device = participant.device?.trim()

    if (!device) {
      continue
    }

    const count = Math.max(1, Number(participant.count || 1))
    const existing = merged.get(device)
    merged.set(device, {
      device,
      count: existing ? Math.max(existing.count, count) : count
    })
  }

  return [...merged.values()]
}

// Co-viewer presence for a channel (channels Phase 3): small chips naming the
// OTHER devices currently viewing this session. Local meshes feed the
// session.participants store; cloud-shared sessions also poll the cloud roster
// so NAT-crossing viewers show in the same header chips.
export function ParticipantChips({ sessionId }: ParticipantChipsProps) {
  const rosters = useStore($sessionParticipants)
  const localDevice = useStore($localDeviceName)
  const [cloudParticipants, setCloudParticipants] = useState<CloudChannelParticipant[]>([])

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setInterval> | null = null
    const clearCloudParticipants = () => setCloudParticipants(current => (current.length === 0 ? current : []))

    clearCloudParticipants()

    if (!sessionId) {
      return () => undefined
    }

    const refreshCloudRoster = async () => {
      const status = await loadSessionCloudStatus(sessionId, { quiet: true })

      if (cancelled) {
        return
      }

      if (!status?.configured || !status.shared || !status.channel_id) {
        clearCloudParticipants()

        return
      }

      const channelId = status.channel_id
      const result = await loadCloudChannelParticipants(channelId, { quiet: true })

      if (!cancelled) {
        setCloudParticipants(result?.participants ?? [])
      }
    }

    void refreshCloudRoster()
    timer = setInterval(() => void refreshCloudRoster(), CLOUD_ROSTER_REFRESH_MS)

    return () => {
      cancelled = true

      if (timer) {
        clearInterval(timer)
      }
    }
  }, [sessionId])

  const participants = useMemo(
    () => mergeParticipants(rosters[sessionId] ?? [], cloudParticipants),
    [cloudParticipants, rosters, sessionId]
  )
  const others = participants.filter(
    participant => participant.device && participant.device !== localDevice
  )

  if (others.length === 0) {
    return null
  }

  return (
    <div className="flex min-w-0 items-center gap-1 [-webkit-app-region:no-drag]" data-testid="session-participants">
      {others.map(participant => (
        <span
          className={cn(
            'inline-flex min-w-0 items-center gap-1 rounded-full border border-(--ui-stroke-tertiary)',
            'bg-(--ui-control-hover-background) px-2 py-0.5 text-[0.6875rem] leading-none text-(--ui-text-secondary)'
          )}
          key={participant.device}
          title={participant.device}
        >
          <span aria-hidden className="size-1.5 shrink-0 rounded-full bg-emerald-500" />
          <span className="max-w-28 truncate">{participant.device}</span>
          {participant.count > 1 && <span className="shrink-0 text-(--ui-text-tertiary)">×{participant.count}</span>}
        </span>
      ))}
    </div>
  )
}
