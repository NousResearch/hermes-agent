import { useStore } from '@nanostores/react'
import { useCallback } from 'react'

import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $sessionListSpansProfiles } from '@/store/shell-policy'

import type { GatewayRequester } from './types'

/** The requester the status bar polls with. Classic asks the app's gateway, as main does. When the
 *  list spans profiles the focused chat can belong to another owner, and status describes the
 *  selected gateway and profile, so it asks that backend by name. */
export function useStatusRequester(requestGateway: GatewayRequester): GatewayRequester {
  const spansProfiles = useStore($sessionListSpansProfiles)
  const connectionId = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)

  const requestSelected = useCallback<GatewayRequester>(
    (method, params, timeoutMs, signal) =>
      requestGatewayForAgent(connectionId, profile, method, params, timeoutMs, signal),
    [connectionId, profile]
  )

  return spansProfiles ? requestSelected : requestGateway
}
