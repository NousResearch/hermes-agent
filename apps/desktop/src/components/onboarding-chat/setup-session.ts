import { atom } from 'nanostores'

import { activeGatewayConnectionId } from '@/store/gateway'
import { getSessionOwnerHint } from '@/store/session'

export interface SetupSession {
  connectionId: null | string
  profile: string
  runtimeId: string
  storedId: null | string
}

export const $setupSession = atom<null | SetupSession>(null)

export function guideSourceConnectionId(guideStoredId: null | string | undefined): null | string {
  return (guideStoredId && getSessionOwnerHint(guideStoredId)?.connectionId) || activeGatewayConnectionId() || null
}
