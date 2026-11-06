import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import { handlePlayerAgentServerRequest, isPlayerAgentScopeLinked } from '@/lib/player-agent-bridge'
import type { PlayerAgentScope } from '@/lib/player-contract'
import type { ScopedServerRequest } from '@/store/gateway'

import type { ServerRequestContext } from './server-requests'

/** Transport tags and the server-named runtime are authority; model fields are not. */
export function playerAgentScopeFromServerRequest(
  request: Pick<ScopedServerRequest, 'connectionId' | 'profile'>,
  sessionId: string
): PlayerAgentScope {
  return {
    connectionId: request.connectionId ?? LOCAL_CONNECTION_ID,
    profileId: request.profile.trim() || 'default',
    sessionId
  }
}

export function playerAgentRequest({ request, sessionId }: ServerRequestContext): void {
  const scope = playerAgentScopeFromServerRequest(request, sessionId)

  // Two windows may show the same conversation. Only the explicitly linked
  // window can answer; an unlinked window must not win the fanout response race.
  if (!sessionId || !isPlayerAgentScopeLinked(scope)) {
    request.decline?.('This window does not own the linked player.')

    return
  }

  const operation = request.params.operation
  void handlePlayerAgentServerRequest(scope, operation).then(
    response => request.respond({ value: JSON.stringify(response) }),
    () =>
      request.respond({
        value: JSON.stringify({
          operation: typeof operation === 'object' && operation !== null && 'type' in operation ? operation.type : '',
          error: { code: 'player-request-unavailable', message: 'The linked player is unavailable for this session.' }
        })
      })
  )
}
