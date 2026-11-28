import { useEffect } from 'react'
import { useLocation } from 'react-router'

import { isBrowserHostedDesktop } from '@/lib/platform'
import { applyNewVersionOnNewChat, watchForNewVersion } from '@/store/new-version'
import { $freshSessionRequest } from '@/store/profile'

import { NEW_CHAT_ROUTE } from '../routes'

/**
 * A Webapp tab learns a newer build is deployed, without reloading under the person: the chat says so
 * (update-ready-notice.tsx, E4) and the tray keeps it. Opening a new chat loads it when nothing is
 * running (store/new-version.ts); "Refresh now" loads it at once; the next visit gets it anyway.
 * Renders nothing itself. The desktop app has its own update flow and never watches.
 */
export function NewVersionNotice() {
  const { pathname } = useLocation()

  useEffect(() => (isBrowserHostedDesktop() ? watchForNewVersion() : undefined), [])

  // A new chat: arriving at its route, or asking for one while already there (⌘N on a new chat).
  useEffect(() => {
    if (pathname === NEW_CHAT_ROUTE) {
      applyNewVersionOnNewChat()
    }
  }, [pathname])
  useEffect(() => $freshSessionRequest.listen(() => applyNewVersionOnNewChat()), [])

  return null
}
