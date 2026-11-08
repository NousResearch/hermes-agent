import type { HermesGateway } from '@/api/client'
import type { ClientSessionState } from '@/app/types'
import { translateNow } from '@/i18n'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionResumeResult } from '@/types/hermes'

import { gatewayRouteState } from './gateway'
import { notifyError, RECOVERY_ACTIONS } from './notifications'
import { $activeSessionId, $selectedStoredSessionId } from './session'
import { isSessionOwnerRoute, type SessionOwnerScope } from './session-request-router'
import { $sessionStates, $sessionTiles, knownOwnerForSession } from './session-states'

interface Viewer {
  gateway: HermesGateway
  connectionId: string | null
  profile: string
  targetProfile: string
  attempted: boolean
  refused: boolean
  retried: boolean
  idleVersion: number
  snapshot?: ClientSessionState
}

const viewers = new Map<string, Viewer>()
const pending = new Map<string, symbol>()
const warned = new WeakSet<HermesGateway>()

export function runtimeReferenced(runtimeId: string, storedSessionId: null | string): boolean {
  return runtimeId === $activeSessionId.get() ||
    (storedSessionId !== null && storedSessionId === $selectedStoredSessionId.get() && !$activeSessionId.get()) ||
    $sessionTiles.get().some(tile => tile.runtimeId === runtimeId ||
      (!tile.runtimeId && storedSessionId !== null && tile.storedSessionId === storedSessionId))
}

export function evictable(runtimeId: string, state: ClientSessionState): boolean {
  return !state.busy && !state.needsInput && !state.awaitingResponse && !runtimeReferenced(runtimeId, state.storedSessionId)
}

/** Capture before a close/drop removes the only exact owner. Never dial a new socket to detach. */
export function captureSessionViewer(runtimeId: string, capturedOwner?: SessionOwnerScope): Viewer | undefined {
  const existing = viewers.get(runtimeId)

  if (existing) { return existing }
  const owner = capturedOwner ?? knownOwnerForSession(runtimeId)

  if (!owner) { return undefined }
  const connectionId = isSessionOwnerRoute(owner) ? owner.connectionId : null
  const profile = typeof owner === 'string' ? owner : owner.profile
  const targetProfile = typeof owner === 'string' ? owner : owner.targetProfile || owner.profile
  const { gateway } = gatewayRouteState(connectionId, profile)

  if (!gateway) { return undefined }
  const viewer = { gateway, connectionId, profile, targetProfile, attempted: false, refused: false, retried: false, idleVersion: 0 }
  viewers.set(runtimeId, viewer)

  return viewer
}

export function observeSessionViewer(runtimeId: string, state: ClientSessionState): void {
  captureSessionViewer(runtimeId)

  if (!evictable(runtimeId, state)) {
    pending.delete(runtimeId)
    const viewer = viewers.get(runtimeId)

    if (viewer && (state.busy || state.awaitingResponse)) {
      viewer.attempted = false
      viewer.refused = false
      viewer.retried = false
    }
  }
}

/** A warm transcript is no longer proof of a live attachment after detach was sent. */
export const sessionViewerNeedsAttach = (runtimeId: string): boolean => viewers.get(runtimeId)?.attempted === true

/** A durable closeout may precede the physical worker's final idle event. */
export function deferCloseoutViewer(runtimeId: string, observedIdleVersion: number): void {
  const viewer = captureSessionViewer(runtimeId)

  if (!viewer) { return }
  viewer.attempted = true
  viewer.refused = true
  const state = $sessionStates.get()[runtimeId]
  viewer.snapshot = state ? { ...state, busy: false, awaitingResponse: false, needsInput: false, messages: [] } : undefined

  if (viewer.idleVersion > observedIdleVersion) { retryRefusedViewer(runtimeId, viewer) }
}

/** The last local reference is gone. Busy/attention work remains attached until a settled publish. */
export function releaseSessionViewer(runtimeId: string, state?: ClientSessionState): void {
  const viewer = captureSessionViewer(runtimeId)
  const snapshot = state ?? $sessionStates.get()[runtimeId]

  if (!viewer || !snapshot || viewer.attempted || snapshot.busy || snapshot.needsInput || snapshot.awaitingResponse) { return }
  const token = Symbol()
  pending.set(runtimeId, token)
  queueMicrotask(() => {
    if (pending.get(runtimeId) !== token) { return }
    pending.delete(runtimeId)
    const latest = $sessionStates.get()[runtimeId] ?? snapshot

    if (!evictable(runtimeId, latest) || gatewayRouteState(viewer.connectionId, viewer.profile).gateway !== viewer.gateway) { return }
    viewer.attempted = true
    viewer.snapshot = { ...latest, messages: [] }
    const idleVersion = viewer.idleVersion
    void viewer.gateway.request<{ detached: boolean }>('session.detach', {
      session_id: runtimeId, profile: viewer.targetProfile
    }).then(result => {
      if (result.detached === false) {
        viewer.refused = true

        if (viewer.idleVersion > idleVersion) { retryRefusedViewer(runtimeId, viewer) }
      }
    }).catch(error => {
      // An older backend keeps its attachment. One actionable notice, with no destructive fallback or retry.
      if (warned.has(viewer.gateway)) { return }
      warned.add(viewer.gateway)
      notifyError(error, translateNow('common.error'), { id: 'session-viewer-detach', action: RECOVERY_ACTIONS.openGateways() })
    })
  })
}

function retryRefusedViewer(runtimeId: string, viewer: Viewer): void {
  if (viewers.get(runtimeId) !== viewer || !viewer.refused || viewer.retried) { return }
  viewer.retried = true
  viewer.refused = false
  viewer.attempted = false
  releaseSessionViewer(runtimeId, viewer.snapshot)
}

/** message.complete precedes backend finalizers. One fresh authoritative idle can retry a refused detach. */
export function noteSessionViewerIdle(runtimeId: string): void {
  const viewer = viewers.get(runtimeId)

  if (!viewer) { return }
  viewer.idleVersion += 1
  queueMicrotask(() => retryRefusedViewer(runtimeId, viewer))
}

/** A pooled resume can finish after its surface closed, without ever publishing locally. */
export function releaseResumedSessionViewer(resumed: SessionResumeResult, storedSessionId: string, owner: SessionOwnerScope): void {
  const viewer = captureSessionViewer(resumed.session_id, owner)

  if (viewer) {
    viewer.attempted = false
    viewer.refused = false
    viewer.retried = false
  }

  releaseSessionViewer(resumed.session_id, {
    ...createClientSessionState(storedSessionId),
    busy: Boolean(resumed.running || resumed.info?.running || resumed.inflight || resumed.queued || resumed.auto_continue),
    needsInput: Boolean(resumed.pending_approval || resumed.open_requests?.length || resumed.pending_connection)
  })
}

export function clearSessionViewers(): void {
  pending.clear()
  viewers.clear()
}
