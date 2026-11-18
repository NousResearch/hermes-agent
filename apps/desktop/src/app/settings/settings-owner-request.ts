import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { $apiRequestScope } from '@/api/client'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

import { useSettingsOwner } from './profile-scope'

/** Classic settings follows its gateway; embedded agent controls never do. */
export function useSettingsGatewayRequest() {
  const { requestGateway } = useGatewayRequest()
  const owner = useSettingsOwner()

  return useMemo<typeof requestGateway & { ownerKey?: string }>(
    () =>
      owner
        ? Object.assign(
            <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number): Promise<T> =>
              requestGatewayForAgent<T>(owner.connectionId ?? null, owner.profile, method, params, timeoutMs),
            { ownerKey: JSON.stringify([owner.connectionId ?? '', owner.profile]) }
          )
        : requestGateway,
    [owner, requestGateway]
  )
}

/** Whether a settings page edits the window's ACTIVE agent, whose model and workspace the live
 *  composer and repo discovery mirror. An embedded owner does only while it is the active
 *  (connection, profile); without one, only a page that requested no explicit profile does. */
export function useSettingsEditsActiveAgent(requestedProfile: null | string | undefined): boolean {
  const owner = useSettingsOwner()
  const apiScope = useStore($apiRequestScope)
  const activeConnection = useStore($activeConnectionId)
  const activeProfile = useStore($activeGatewayProfile)

  if (!owner) {
    return requestedProfile == null
  }

  return (
    normalizeProfileKey(owner.profile) === normalizeProfileKey(activeProfile) &&
    (owner.connectionId ?? 'local') === (apiScope.connectionId ?? activeConnection ?? 'local')
  )
}
