import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'
import { useLocation } from 'react-router'

import { Button } from '@/components/ui/button'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { type ButterbarItem, useButterbar } from '@/store/butterbar'
import { $newVersionReady, applyNewVersionOnNewChat, watchForNewVersion } from '@/store/new-version'
import { $freshSessionRequest } from '@/store/profile'

import { NEW_CHAT_ROUTE } from '../routes'

import { useDiscoveryCopy } from './discovery-copy'

function NewVersionMessage() {
  const copy = useDiscoveryCopy().newVersion

  return (
    <>
      {copy.notice}{' '}
      <Button onClick={() => window.location.reload()} size="inline" variant="link">
        {copy.refresh}
      </Button>
    </>
  )
}

/**
 * W1 (Figma 61:2037): a Webapp tab learns a newer build is deployed and says so in the notice strip,
 * without reloading under the person. Opening a new chat loads it when nothing is running
 * (store/new-version.ts); "Refresh now" loads it at once; the next visit gets it anyway. Renders
 * nothing itself. The desktop app has its own update flow and never watches.
 */
export function NewVersionNotice() {
  const ready = useStore($newVersionReady)
  const { pathname } = useLocation()

  useEffect(() => (isBrowserHostedDesktop() ? watchForNewVersion() : undefined), [])

  // A new chat: arriving at its route, or asking for one while already there (⌘N on a new chat).
  useEffect(() => {
    if (pathname === NEW_CHAT_ROUTE) {
      applyNewVersionOnNewChat()
    }
  }, [pathname])
  useEffect(() => $freshSessionRequest.listen(() => applyNewVersionOnNewChat()), [])

  const item = useMemo<ButterbarItem | null>(
    () => (ready ? { id: 'new-version', node: <NewVersionMessage />, priority: 10, tone: 'accent' } : null),
    [ready]
  )

  useButterbar(item)

  return null
}
