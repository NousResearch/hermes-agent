import { runFreeTierChallenge } from '@/store/free-tier-challenge'
import { activeGatewayConnectionId, requestGatewayForAgent } from '@/store/gateway'

import type { GatewayEventContext } from './types'

/**
 * free_tier.challenge: the account service wants a browser challenge cleared
 * before it mints the free-tier token. The backend polls for the result on its
 * own; this only gets the page loaded (hidden). Any source's challenge is worth
 * running: it is the install's identity, not the focused session's.
 */
export function handleFreeTierEvent({ deps, event, payload }: GatewayEventContext): boolean {
  if (event.type !== 'free_tier.challenge') {
    return false
  }

  const connectionId = event.connectionId ?? activeGatewayConnectionId()
  const profile = event.profile ?? deps.activeGatewayProfile

  void runFreeTierChallenge(payload, (method, params) => requestGatewayForAgent(connectionId, profile, method, params))

  return true
}
