import { useStore } from '@nanostores/react'

import { cn } from '@/lib/utils'
import { $localDeviceName, $sessionParticipants } from '@/store/session'

interface ParticipantChipsProps {
  sessionId: string
}

// Co-viewer presence for a channel (channels Phase 3): small chips naming the
// OTHER devices currently viewing this session. THIS device is filtered out, so
// a chip only appears once someone else is here — a solo local session (no
// MeshBoard/Syncthing, no Tailscale) renders nothing. The visible content is
// the device name itself, so there's no localized copy to translate.
export function ParticipantChips({ sessionId }: ParticipantChipsProps) {
  const rosters = useStore($sessionParticipants)
  const localDevice = useStore($localDeviceName)

  const others = (rosters[sessionId] ?? []).filter(
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
