import { useMemo } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { requestGatewayForAgent } from '@/store/gateway'

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
