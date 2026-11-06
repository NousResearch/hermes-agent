import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'
import { useCallback, useEffect, useState } from 'react'

import { composerHost } from '@/sdk/composer'
import { activeGatewayConnectionId, activeGatewayProfileKey } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $connection, $sessions } from '@/store/session'
import { isSessionOwnerRoute } from '@/store/session-request-router'
import { $focusedRuntimeId, $focusedSessionState, $sessionTiles, knownOwnerForSession } from '@/store/session-states'

import type { PlayerAgentOperation } from './player-context-contract'
import type {
  PlayerAgentResponse,
  PlayerAgentScope,
  PlayerAwarenessLinkState,
  PlayerMediaJobCapabilities,
  PlayerSnapshot
} from './player-contract'
import { decodePlayerAgentOperation } from './player-operation-decoder'

type ScopeListener = (scope: PlayerAgentScope | null) => void
let linkedScope: PlayerAgentScope | null = null
const scopeListeners = new Set<ScopeListener>()

const sameScope = (a: PlayerAgentScope | null, b: PlayerAgentScope | null): boolean =>
  Boolean(a && b && a.connectionId === b.connectionId && a.profileId === b.profileId && a.sessionId === b.sessionId)

function publishLinkedScope(scope: PlayerAgentScope | null): void {
  linkedScope = scope

  for (const listener of scopeListeners) {listener(scope)}
}

// Keep the derivation in the store graph. A render-time .get() hidden behind a
// no-argument helper can be memoized independently of the selected chat.
const $trustedActiveScope = computed(
  [
    $focusedRuntimeId,
    $connection,
    $activeGatewayProfile,
    $sessionTiles,
    $sessions,
    $focusedSessionState,
    $activeSessionId
  ],
  (sessionId, connection, _profile, _tiles, _sessions, _state, primaryRuntime): PlayerAgentScope | null => {
    if (!sessionId || !connection) {return null}

    const owner = knownOwnerForSession(sessionId)

    if (isSessionOwnerRoute(owner) && owner.connectionId.trim()) {
      return {
        connectionId: owner.connectionId,
        profileId: owner.profile.trim() || 'default',
        sessionId
      }
    }

    // Direct URL primaries predate the registry and have no connection id in
    // their socket tags. Match that transport's identity, only for its bound
    // primary runtime. An unresolved tile must never borrow the ambient route.
    if (
      sessionId === primaryRuntime &&
      !connection.connectionId &&
      connection.registryScoped !== true &&
      activeGatewayConnectionId() === null
    ) {
      return { connectionId: LOCAL_CONNECTION_ID, profileId: activeGatewayProfileKey().trim() || 'default', sessionId }
    }

    return null
  }
)

export interface PlayerAgentBridgeController {
  /** Scope derived from the active gateway/session stores; never caller supplied. */
  scope: PlayerAgentScope | null
  linked: boolean
  link(): Promise<PlayerAwarenessLinkState>
  unlink(): Promise<PlayerAwarenessLinkState>
  request(operation: PlayerAgentOperation): Promise<PlayerAgentResponse>
  /** Append a user-approved, unsent prompt only while the linked scope is still active. */
  seedPrompt(text: string): Promise<boolean>
  setupMediaHelpers(): Promise<PlayerMediaJobCapabilities>
  revealMediaJob(jobId: string): Promise<boolean>
  clearLocalData(): Promise<PlayerSnapshot>
}

/** Explicit user link and narrowly typed, scope-bound agent operations for HermesTV. */
export function usePlayerAgentBridge(): PlayerAgentBridgeController {
  const scope = useStore($trustedActiveScope)
  const [currentLinkedScope, setCurrentLinkedScope] = useState<PlayerAgentScope | null>(linkedScope)

  useEffect(() => {
    const listener = (next: PlayerAgentScope | null) => setCurrentLinkedScope(next)
    scopeListeners.add(listener)
    setCurrentLinkedScope(linkedScope)

    return () => {
      scopeListeners.delete(listener)
    }
  }, [])

  const link = useCallback(async () => {
    if (!scope) {throw new Error('No active Hermes session is available for player linking.')}
    const bridge = window.hermesDesktop.player

    if (!bridge) {throw new Error('HermesTV player is unavailable.')}
    const state = await bridge.linkAwareness(scope)
    const accepted = state.linked && sameScope(scope, state.scope)
    publishLinkedScope(accepted ? state.scope : null)
    setCurrentLinkedScope(accepted ? state.scope : null)

    if (!accepted) {throw new Error('HermesTV could not confirm this active session link.')}

    return state
  }, [scope])

  const unlink = useCallback(async () => {
    const bridge = window.hermesDesktop.player

    if (!bridge) {throw new Error('HermesTV player is unavailable.')}
    const state = await bridge.unlinkAwareness()
    publishLinkedScope(null)
    setCurrentLinkedScope(null)

    return state
  }, [])

  const request = useCallback(
    async (operation: PlayerAgentOperation) => {
      const bridge = window.hermesDesktop.player

      if (!bridge || !scope || !sameScope(scope, linkedScope)) {
        throw new Error('Link the active Hermes session to HermesTV before using its player tools.')
      }

      const response = await bridge.agentRequest({ scope, operation })

      if (!sameScope(scope, linkedScope)) {throw new Error('The linked Hermes session changed during the request.')}

      return response
    },
    [scope]
  )

  const setupMediaHelpers = useCallback(async () => {
    const bridge = window.hermesDesktop.player

    if (!bridge || !scope || !sameScope(scope, linkedScope)) {throw new Error('Link the active Hermes session first.')}

    return bridge.setupMediaHelpers()
  }, [scope])

  const seedPrompt = useCallback(
    async (text: string) => {
      const currentScope = $trustedActiveScope.get()

      if (!text.trim() || !scope || !sameScope(scope, linkedScope) || !sameScope(scope, currentScope)) {return false}
      const inserted = await composerHost.insertText(scope.sessionId, text, { mode: 'block' })

      return inserted && sameScope(scope, linkedScope) && sameScope(scope, $trustedActiveScope.get())
    },
    [scope]
  )

  const revealMediaJob = useCallback(
    async (jobId: string) => {
      const bridge = window.hermesDesktop.player

      if (!bridge || !scope || !sameScope(scope, linkedScope)) {return false}

      return bridge.revealMediaJob(scope, jobId)
    },
    [scope]
  )

  const clearLocalData = useCallback(async () => {
    const bridge = window.hermesDesktop.player

    if (!bridge) {throw new Error('HermesTV player is unavailable.')}
    const snapshot = await bridge.clearLocalData()
    publishLinkedScope(null)
    setCurrentLinkedScope(null)

    return snapshot
  }, [])

  return {
    scope,
    linked: sameScope(scope, currentLinkedScope),
    link,
    unlink,
    request,
    seedPrompt,
    setupMediaHelpers,
    revealMediaJob,
    clearLocalData
  }
}

/** Used only by the trusted server-request handler to compare a request's exact route. */
export function isPlayerAgentScopeLinked(scope: PlayerAgentScope): boolean {
  return sameScope(scope, linkedScope)
}

/** The only server→renderer player route: trusted socket tags plus the explicit native link. */
export async function handlePlayerAgentServerRequest(
  scope: PlayerAgentScope,
  operation: unknown
): Promise<PlayerAgentResponse> {
  const bridge = window.hermesDesktop.player
  const decodedOperation = decodePlayerAgentOperation(operation)

  if (!bridge || !decodedOperation || !sameScope(scope, linkedScope)) {
    throw new Error('Player request is unavailable for this linked session.')
  }

  const response = await bridge.agentRequest({ scope, operation: decodedOperation })

  if (!sameScope(scope, linkedScope) || response.operation !== decodedOperation.type) {
    throw new Error('Player request returned for a stale or mismatched session.')
  }

  return response
}

export function subscribePlayerAgentLinkedScope(listener: ScopeListener): () => void {
  scopeListeners.add(listener)

  return () => scopeListeners.delete(listener)
}
