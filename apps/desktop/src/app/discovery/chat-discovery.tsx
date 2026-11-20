import { DeveloperViewOffer } from './developer-view-offer'
import { HeldExchange } from './held-exchange'

interface ChatDiscoveryProps {
  /** The window's own chat, not a tile. */
  primary: boolean
  /** The stored chat the route shows, or null for a new chat. */
  routedSessionId: null | string
}

/** What the redesign's discovery lays over the window's own chat: a message held until sign up, and
 *  the developer-view offer (artifacts are filed under the stored chat the route names,
 *  artifact-card.tsx). Neither shows anywhere else. */
export function ChatDiscovery({ primary, routedSessionId }: ChatDiscoveryProps) {
  if (!primary) {
    return null
  }

  return (
    <>
      <HeldExchange primary routed={Boolean(routedSessionId)} />
      <DeveloperViewOffer sessionId={routedSessionId} />
    </>
  )
}
