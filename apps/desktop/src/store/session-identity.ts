import { registryBackendScopeKey } from '@hermes/shared'

import { activeGatewayProfileKey } from '@/store/gateway'
import { normalizeProfileKey } from '@/store/profile'
import { $sessions, sessionMatchesStoredId } from '@/store/session'
import {
  $sessionStates,
  $sessionTiles,
  knownOwnerForSession,
  previewScopeForRuntime,
  runtimeSessionOwner,
  storedSessionIdForRuntimeId
} from '@/store/session-states'

export interface SessionRequestSource {
  connectionId?: string
  profile?: string
}

/** Compare source evidence before using a runtime-to-stored alias. A runtime's
 * inbound source must not be laundered through another source's durable tile. */
function sourceMatchesSession(sessionId: string, activeSessionId: string, source?: SessionRequestSource): boolean {
  const owner = knownOwnerForSession(activeSessionId)

  const expected: SessionRequestSource | undefined =
    typeof owner === 'string' ? { profile: owner } : (owner ?? undefined)

  const runtimeOwner = runtimeSessionOwner(sessionId)

  const runtimeSource: SessionRequestSource | undefined =
    typeof runtimeOwner === 'string' ? { profile: runtimeOwner } : (runtimeOwner ?? undefined)

  const expectedProfile = normalizeProfileKey(expected?.profile ?? activeGatewayProfileKey())

  const matches = (candidate?: SessionRequestSource): boolean => {
    if (!candidate) {
      return true
    }

    if (candidate.profile && normalizeProfileKey(candidate.profile) !== expectedProfile) {
      return false
    }

    return (
      !candidate.connectionId ||
      Boolean(
        expected?.connectionId &&
        registryBackendScopeKey(candidate.connectionId, candidate.profile) ===
          registryBackendScopeKey(expected.connectionId, expected.profile)
      )
    )
  }

  if (!matches(source)) {
    return false
  }

  // A matching socket stamp against a known owner outranks the runtime ledger,
  // whose entry may be stale or collide with a runtime on another connection.
  // Partial/untagged sources still need that ledger as corroborating evidence.
  if (expected && source?.profile && (!expected.connectionId || source.connectionId)) {
    return true
  }

  return matches(runtimeSource)
}

interface VisibleSessionIdentity {
  profile: string
  runtimeId: null | string
  sessionId: null | string
}

/** Resolve admission and tab ownership together, from an actual visible binding.
 * The selected stored id alone never proves that a conversation is on screen. */
export function visibleSessionIdentity({
  activeSessionId,
  sessionId,
  source,
  storedIdForRuntimeId = runtimeId => $sessionStates.get()[runtimeId]?.storedSessionId ?? undefined
}: {
  activeSessionId: null | string
  sessionId: string
  source?: SessionRequestSource
  storedIdForRuntimeId?: (runtimeId: string) => string | undefined
}): VisibleSessionIdentity | undefined {
  const candidates = [
    { runtimeId: activeSessionId, storedSessionId: activeSessionId ? storedIdForRuntimeId(activeSessionId) : null },
    ...$sessionTiles.get()
  ]

  for (const candidate of candidates) {
    const sourceId = candidate.runtimeId ?? candidate.storedSessionId

    if (!sourceId || !sourceMatchesSession(sessionId, sourceId, source)) {
      continue
    }

    if (
      !sessionIdsMatch({ activeSessionId: candidate.runtimeId ?? null, sessionId, storedIdForRuntimeId }) &&
      !sessionIdsMatch({ activeSessionId: candidate.storedSessionId ?? null, sessionId, storedIdForRuntimeId })
    ) {
      continue
    }

    return {
      profile: previewScopeForRuntime(candidate.runtimeId ?? candidate.storedSessionId ?? undefined),
      runtimeId: storedIdForRuntimeId(sessionId) ? sessionId : (candidate.runtimeId ?? null),
      sessionId:
        storedIdForRuntimeId(sessionId) ??
        candidate.storedSessionId ??
        storedSessionIdForRuntimeId(sourceId) ??
        storedSessionIdForRuntimeId(sessionId)
    }
  }

  return undefined
}

/** Match a runtime/stored identity to one visible conversation across compression.
 * One session row must answer to both ids: branch siblings sharing only a root
 * are not the same conversation. Unknown runtime ids may already be stored ids. */
export function requestNamesActiveSession({
  activeSessionId,
  sessionId,
  source,
  storedIdForRuntimeId = () => undefined
}: {
  activeSessionId: null | string
  sessionId: string
  source?: SessionRequestSource
  storedIdForRuntimeId?: (runtimeId: string) => string | undefined
}): boolean {
  return Boolean(
    activeSessionId &&
    sourceMatchesSession(sessionId, activeSessionId, source) &&
    sessionIdsMatch({ activeSessionId, sessionId, storedIdForRuntimeId })
  )
}

function sessionIdsMatch({
  activeSessionId,
  sessionId,
  storedIdForRuntimeId
}: {
  activeSessionId: null | string
  sessionId: string
  storedIdForRuntimeId: (runtimeId: string) => string | undefined
}): boolean {
  if (!sessionId || !activeSessionId) {
    return false
  }

  if (sessionId === activeSessionId) {
    return true
  }

  const requestStoredId = storedIdForRuntimeId(sessionId) ?? sessionId
  const activeStoredId = storedIdForRuntimeId(activeSessionId) ?? activeSessionId

  if (requestStoredId === activeStoredId) {
    return true
  }

  return $sessions
    .get()
    .some(
      session => sessionMatchesStoredId(session, requestStoredId) && sessionMatchesStoredId(session, activeStoredId)
    )
}
