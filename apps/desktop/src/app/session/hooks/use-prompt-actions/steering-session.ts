import type { MutableRefObject } from 'react'

import { findStoredIdForRuntimeId } from '@/app/contrib/wiring-routing'
import type { ClientSessionState } from '@/app/types'
import { $activeSessionStoredIdRotation, $sessions, idsShareLineage, setActiveSessionId } from '@/store/session'
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

  if (!sessionId) {
    return null
  }

  const selectedStoredSessionId = selectedStoredSessionIdRef.current
  const routedStoredSessionId = getRoutedStoredSessionId()
  const bindings = runtimeIdByStoredSessionIdRef.current
  const boundStoredSessionId = findStoredIdForRuntimeId(bindings, sessionId)
  const sessions = $sessions.get()
  const storedIdRotation = $activeSessionStoredIdRotation.get()

  // A compression event can update the runtime binding before the sessions
  // refresh exposes the new tip's lineage. The rotation records the exact
  // runtime and old/new stored ids, so it is enough proof for that brief gap.
  const rotationBridgesUnloadedTip = Boolean(
    storedIdRotation &&
    storedIdRotation.runtimeSessionId === sessionId &&
    storedIdRotation.previousStoredSessionId === selectedStoredSessionId &&
    storedIdRotation.nextStoredSessionId === boundStoredSessionId
  )

  const matchesSelection = (id: string) =>
    Boolean(
      selectedStoredSessionId &&
      (idsShareLineage(id, selectedStoredSessionId, sessions) ||
        (rotationBridgesUnloadedTip && id === boundStoredSessionId))
    )

  // Navigation publishes route, selection and runtime independently. Unlike an
  // ordinary Send, a mid-turn correction must never resolve another target and
  // interrupt it, so any disagreement refuses and the composer queues the text.
  const routeLeftSelection = Boolean(routedStoredSessionId && !matchesSelection(routedStoredSessionId))

  // Selection names a stored chat, yet this runtime proves no binding at all.
  const runtimeUnbound = Boolean(
    selectedStoredSessionId && selectedStoredSessionId !== sessionId && !boundStoredSessionId
  )

  // Any stored id bound to this runtime outside the selected lineage — which is
  // every binding when nothing is selected (a fresh draft next to a live turn).
  const runtimeBoundElsewhere = [...bindings].some(
    ([stored, runtime]) => runtime === sessionId && !matchesSelection(stored)
  )

  if (routeLeftSelection || runtimeUnbound || runtimeBoundElsewhere) {
    return null
  }

  // A rebuilt runtime can bind the tip while selection/route keep the root.
  // Use its proven stored id for writes; reapplying the root would rotate it back.
  const storedSessionId = boundStoredSessionId ?? selectedStoredSessionId

  const owner =
    knownOwnerForSession(sessionId) ??
    knownOwnerForSession(storedSessionId) ??
    (rotationBridgesUnloadedTip ? knownOwnerForSession(storedIdRotation?.previousStoredSessionId) : undefined)

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
