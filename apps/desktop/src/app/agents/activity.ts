import { useEffect, useState } from 'react'

import type { AgentRow } from '@/store/agent-overview'

export const RECENT_ACTIVITY_MS = 15 * 60_000

export function isRecentActivity(row: AgentRow, now: number): boolean {
  return (
    row.attention === 'working' ||
    row.attention === 'needs-you' ||
    (row.stale && (row.lastKnownAttention === 'working' || row.lastKnownAttention === 'needs-you')) ||
    (row.lastActive > 0 && now - row.lastActive * 1000 <= RECENT_ACTIVITY_MS)
  )
}

// A presentation deadline, independent of inventory polling/reference changes.
export function useActivityNow(rows: readonly AgentRow[], visible: boolean): number {
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    if (!visible) {
      return
    }

    let next = Infinity

    for (const row of rows) {
      const expires = row.lastActive * 1000 + RECENT_ACTIVITY_MS + 1

      if (row.lastActive > 0 && expires > now && row.attention !== 'working' && row.attention !== 'needs-you') {
        next = Math.min(next, expires)
      }
    }

    if (!Number.isFinite(next)) {
      return
    }

    const timer = setTimeout(() => setNow(Date.now()), Math.max(0, next - Date.now()))

    return () => clearTimeout(timer)
  }, [rows, visible, now])

  return now
}
