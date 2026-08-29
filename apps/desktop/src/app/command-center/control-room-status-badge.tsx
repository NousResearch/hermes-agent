import { useEffect, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { cn } from '@/lib/utils'

interface CrCounts {
  needs_you?: number
  agents_active?: number
  messages_unread?: number
}

interface CrSnapshotBadge {
  counts?: CrCounts
}

const POLL_MS = 5000

/**
 * CR-405: compact, non-focus-stealing Control Room attention badge for the
 * status bar. Renders `● N need you` (or the count summary) and opens Control
 * Room on click. Bounded polling (5s) rides the gateway's 2s snapshot cache —
 * never a high-frequency loop, never auto-opens.
 */
export function ControlRoomStatusBadge({ onOpen }: { onOpen: () => void }) {
  const { requestGateway } = useGatewayRequest()
  const [counts, setCounts] = useState<CrCounts | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null

    const fetchCounts = () => {
      requestGateway<CrSnapshotBadge>('control.room.snapshot', { profile: 'default' })
        .then(snap => {
          if (snap && !cancelled) {setCounts(snap.counts ?? null)}
        })
        .catch(() => {
          // Gateway unavailable — stay silent (never render a fake zero).
        })
    }

    fetchCounts()
    timer = setInterval(fetchCounts, POLL_MS)

    return () => {
      cancelled = true

      if (timer) {clearInterval(timer)}
    }
  }, [requestGateway])

  if (!counts) {
    return null
  }

  const needsYou = counts.needs_you ?? 0
  const active = counts.agents_active ?? 0

  return (
    <button
      aria-label="Open Control Room"
      className={cn(
        'inline-flex items-center gap-1 rounded px-1.5 text-[0.6875rem] leading-none text-(--ui-text-secondary)',
        'hover:bg-(--chrome-action-hover) hover:text-foreground'
      )}
      onClick={onOpen}
      title="Control Room (⌘P)"
      type="button"
    >
      <span className={cn('size-1.5 rounded-full', needsYou > 0 ? 'bg-(--ui-red)' : 'bg-(--ui-text-tertiary)')} />
      <span>{needsYou > 0 ? `${needsYou} need you` : active > 0 ? `${active} agents` : 'idle'}</span>
    </button>
  )
}
