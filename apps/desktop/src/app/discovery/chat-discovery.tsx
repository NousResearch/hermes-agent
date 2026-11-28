import { useStore } from '@nanostores/react'

import { $newVersionNoticeShown } from '@/store/new-version'

import { DeveloperViewOffer } from './developer-view-offer'
import { HeldExchange } from './held-exchange'
import { UpdateReadyNotice } from './update-ready-notice'

interface ChatDiscoveryProps {
  /** The window's own chat, not a tile. */
  primary: boolean
  /** The stored chat the route shows, or null for a new chat. */
  routedSessionId: null | string
}

/** What the redesign's discovery lays over the window's own chat: a message held until sign up, and
 *  over a stored chat's composer either a newer build's notice or the developer-view offer (artifacts
 *  are filed under the stored chat the route names, artifact-card.tsx); the notice comes first. A new
 *  chat loads the newer build itself when nothing is running, so it never says it there. None shows
 *  anywhere else. */
export function ChatDiscovery({ primary, routedSessionId }: ChatDiscoveryProps) {
  const updateReady = useStore($newVersionNoticeShown)

  if (!primary) {
    return null
  }

  return (
    <>
      <HeldExchange primary routed={Boolean(routedSessionId)} />
      {updateReady && routedSessionId ? <UpdateReadyNotice /> : <DeveloperViewOffer sessionId={routedSessionId} />}
    </>
  )
}
