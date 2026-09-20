import type { MutableRefObject } from 'react'

import { findStoredIdForRuntimeId } from '@/app/contrib/wiring-routing'
import type { ClientSessionState } from '@/app/types'
import { $sessions, idsShareLineage, setActiveSessionId } from '@/store/session'
import { requestForSessionProfile } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

import type { GatewayRequest } from './utils'

interface SteeringSessionDeps {
  activeSessionIdRef: MutableRefObject<string | null>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  getRoutedStoredSessionId: () => string | null
  requestGateway: GatewayRequest
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

/** Pin a correction to its source, not the selection at retry time. */
export function captureSteeringSession(deps: SteeringSessionDeps) {
  const { activeSessionIdRef, selectedStoredSessionIdRef, runtimeIdByStoredSessionIdRef, getRoutedStoredSessionId } =
    deps

  const sessionId = activeSessionIdRef.current
  const selectedStoredSessionId = selectedStoredSessionIdRef.current
  const routedStoredSessionId = getRoutedStoredSessionId()
  const bindings = runtimeIdByStoredSessionIdRef.current
  const boundStoredSessionId = sessionId ? findStoredIdForRuntimeId(bindings, sessionId) : undefined
  const sessions = $sessions.get()

  const matchesSelection = (id: string) =>
    Boolean(selectedStoredSessionId && idsShareLineage(id, selectedStoredSessionId, sessions))

  // Navigation publishes these identities independently. Unlike ordinary Send,
  // a mid-turn correction must not resolve another target and interrupt it.
  if (
    !sessionId ||
    (routedStoredSessionId && !matchesSelection(routedStoredSessionId)) ||
    (selectedStoredSessionId &&
      selectedStoredSessionId !== sessionId &&
      (!boundStoredSessionId || !matchesSelection(boundStoredSessionId))) ||
    [...bindings].some(([stored, runtime]) => runtime === sessionId && !matchesSelection(stored))
  ) {
    return null
  }

  // A rebuilt runtime can bind the tip while selection/route keep the root.
  // Use its proven stored id for writes; reapplying the root would rotate it back.
  const storedSessionId = boundStoredSessionId ?? selectedStoredSessionId
  const owner = knownOwnerForSession(sessionId) ?? knownOwnerForSession(storedSessionId)
  let adoptedSessionId = sessionId

  const requestGateway: GatewayRequest = (method, params, timeoutMs) =>
    requestForSessionProfile(owner, deps.requestGateway, method, params, timeoutMs)

  return {
    sessionId,
    storedSessionId,
    requestGateway,
    resolveProfile: owner ? async () => (typeof owner === 'string' ? owner : owner.profile) : undefined,
    onRecovered: (recoveredId: string) => {
      // Both visible redirect and hidden steer need the binding BEFORE retry:
      // hidden steer has no optimistic row to establish it incidentally.
      deps.updateSessionState(recoveredId, state => state, storedSessionId)

      // The correction still belongs to its source when navigation backgrounds
      // it. Keep its recovery, but only the unchanged view may adopt its id.
      // A cached recovery can itself expire: the resolver then calls us again.
      if (
        activeSessionIdRef.current === adoptedSessionId &&
        selectedStoredSessionIdRef.current === selectedStoredSessionId &&
        getRoutedStoredSessionId() === routedStoredSessionId
      ) {
        adoptedSessionId = recoveredId
        activeSessionIdRef.current = recoveredId
        setActiveSessionId(recoveredId)
      }
    }
  }
}
