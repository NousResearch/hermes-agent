import { atom } from 'nanostores'
import type { RpcMethods } from '@hermes/shared'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeConnectionId } from '@/store/connections'
import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'

export type MemoryPending = RpcMethods['memory.pending']['result']
export const $memoryReview = atom<null | { request: GatewayRequest; sessionId: string }>(null)

export function openMemoryReview(request: GatewayRequest, sessionId: string) {
  $memoryReview.set({ request, sessionId })
}

let scopeGeneration = 0
export const memoryReviewScopeGeneration = () => scopeGeneration
const invalidate = () => {
  scopeGeneration += 1
  $memoryReview.set(null)
}
$activeGatewayProfile.listen(invalidate)
$activeConnectionId.listen(invalidate)
