import { requestGatewayForAgent } from '@/store/gateway'
import { requestForOwnedSession } from '@/store/session-states'
import type { RpcEvent } from '@/types/hermes'

type GatewayRequest = (method: string, params?: Record<string, unknown>) => Promise<unknown>

export async function respondOnOwnedGateway(
  event: RpcEvent,
  method: string,
  params: Record<string, unknown>,
  ambientRequest: GatewayRequest | null,
): Promise<void> {
  if (ambientRequest) {
    try {
      await requestForOwnedSession(event.session_id, ambientRequest, method, params)
      return
    } catch {
      // Unknown owner (fresh Bot Chat runtime): fall through to the event's profile tag.
    }
  }

  const profile = typeof event.profile === 'string' ? event.profile.trim() : ''

  if (profile) {
    await requestGatewayForAgent(event.connectionId ?? null, profile, method, params)
    return
  }

  if (ambientRequest) {
    await ambientRequest(method, params)
  }
}
