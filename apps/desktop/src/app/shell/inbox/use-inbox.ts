import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { $gateway } from '@/store/gateway'
import { $inbox, clearInbox, type InboxEntry, refreshInbox } from '@/store/inbox'
import { $activeGatewayProfile } from '@/store/profile'

export const INBOX_POLL_INTERVAL_MS = 15_000

/**
 * Single source of truth for inbox state. The statusbar chip owns this hook and
 * passes `inbox` down to the panel — both surfaces must not each create their
 * own poll timer. A gateway/profile switch is handled at the seams:
 * `gateway-switch.ts` calls `clearInbox()`, and the store's scope token
 * (`connection\u0000profile`) rejects any late response from a superseded
 * backend (A-B-A). The store's single-flight guard prevents overlapping
 * requests. Opening/dismissing the panel never resolves anything.
 */
export function useInbox(): { inbox: InboxEntry; open: boolean; setOpen: (open: boolean) => void } {
  const inbox = useStore($inbox)
  const profile = useStore($activeGatewayProfile) ?? ''
  const gateway = useStore($gateway)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    // Never retain actionable data from a previous connection or profile.
    clearInbox()

    if (!gateway) {return}

    void refreshInbox(profile)
    const timer = window.setInterval(() => void refreshInbox(profile), INBOX_POLL_INTERVAL_MS)

    return () => window.clearInterval(timer)
  }, [gateway, profile])

  return { inbox, open, setOpen }
}
