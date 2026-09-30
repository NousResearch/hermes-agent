import { isOutOfSyncRpcParams } from '@/lib/gateway-rpc'
import { requestGatewayForAgent } from '@/store/gateway'
import type { AgentProfileRoute } from '@/store/profile'
import type { SessionCreateResponse } from '@/types/hermes'

type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

/** A backend predating #122899 (contract without `cwd_explicit`) rejects the
 *  whole create at admission (`tui_gateway/contracts/registry.py::validate_params`,
 *  code 4000, handler never runs) — e.g. a Hermes Cloud backend behind a
 *  Desktop that updates from main (#128971). Those backends always honoured the
 *  client `cwd`, so resending without the flag reproduces their behaviour.
 *  Delete once no supported backend predates #122899. */
function rejectsCwdExplicit(params: Record<string, unknown>, error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return 'cwd_explicit' in params && isOutOfSyncRpcParams(message) && /:\s*cwd_explicit:/.test(message)
}

/** `session.create` on the captured owner route (or the window's gateway). */
export async function createGatewaySession(
  route: AgentProfileRoute | null,
  params: Record<string, unknown>,
  requestGateway: RequestGateway
): Promise<SessionCreateResponse> {
  const send = (requestParams: Record<string, unknown>) =>
    route
      ? requestGatewayForAgent<SessionCreateResponse>(
          route.connectionId,
          route.profile,
          'session.create',
          requestParams,
          undefined,
          undefined,
          { spawnPriority: 'foreground' }
        )
      : requestGateway<SessionCreateResponse>('session.create', requestParams)

  try {
    return await send(params)
  } catch (error) {
    if (!rejectsCwdExplicit(params, error)) {
      throw error
    }

    const { cwd_explicit: _cwdExplicit, ...compatible } = params

    return send(compatible)
  }
}
