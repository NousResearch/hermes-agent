import { requestGatewayForAgent } from '@/store/gateway'
import type { AgentProfileRoute } from '@/store/profile'
import type { SessionCreateResponse } from '@/types/hermes'

type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

/** A backend predating a `session.create` field rejects the whole create at
 *  admission (`tui_gateway/contracts/registry.py::validate_params`, code 4000,
 *  handler never runs) — e.g. a Hermes Cloud backend behind a Desktop that
 *  updates from main (#128971). Each field below is safe to drop for them:
 *  - `cwd_explicit` (#122899): those backends always honoured the client `cwd`.
 *  - `service_tier` (Ultrafast): `fast` still rides, so they get Priority.
 *  Matched on the stable prefix, not `isOutOfSyncRpcParams`: v0.21.3 already
 *  rejects but predates the "out of sync" suffix.
 *  Delete a field once no supported backend predates it. */
const DROPPABLE_CREATE_FIELDS = ['cwd_explicit', 'service_tier'] as const

function rejectedField(params: Record<string, unknown>, error: unknown): string | undefined {
  const message = error instanceof Error ? error.message : String(error)

  return DROPPABLE_CREATE_FIELDS.find(
    field => field in params && message.includes(`invalid params for session.create: ${field}:`)
  )
}

/** `session.create` on the captured owner route (or the window's gateway).
 *  A null `connectionId` is a legacy named-profile owner (coding workspaces). */
export async function createGatewaySession(
  route: (Omit<AgentProfileRoute, 'connectionId'> & { connectionId: null | string }) | null,
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

  // One resend per dropped field: a backend predating both rejects them one at a time.
  const create = async (requestParams: Record<string, unknown>): Promise<SessionCreateResponse> => {
    try {
      return await send(requestParams)
    } catch (error) {
      const field = rejectedField(requestParams, error)

      if (!field) {
        throw error
      }

      const { [field]: _dropped, ...compatible } = requestParams

      return create(compatible)
    }
  }

  return create(params)
}
