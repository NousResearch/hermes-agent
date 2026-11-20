import { backendScopeKey, type GatewayEvent, LOCAL_CONNECTION_ID, type VoiceLiveCapturePayload } from '@hermes/shared'

import { hydrateStoredSessionTranscript } from '@/app/contrib/hooks/use-background-sync'
import { translateNow } from '@/i18n'
import { coerceGatewayText } from '@/lib/chat-runtime'
import { parseErrorSurface } from '@/lib/error-surface'
import { executionAdmission } from '@/lib/execution-authority'
import { notify } from '@/store/notifications'
import { normalizeProfileKey } from '@/store/profile'
import { isSessionGone } from '@/store/session-gone-latch'
import { isSessionOwnerRoute } from '@/store/session-request-router'
import { knownOwnerForSession, runtimeSessionOwner } from '@/store/session-states'

import type { GatewayEventDeps } from './types'

/** A capture frame invalidates durable history; it is not live turn output. */
export function handleVoiceLiveCapture(
  event: GatewayEvent,
  deps: GatewayEventDeps,
  fromActiveSource: () => boolean,
  { reads, warnedCalls }: { reads: Map<string, symbol>; warnedCalls: Set<string> }
): void {
  const runtimeSessionId = event.session_id
  const payload = event.payload as VoiceLiveCapturePayload | undefined

  if (!runtimeSessionId || typeof payload?.call_id !== 'string' || !payload.call_id.trim()) {
    return
  }

  const states = deps.sessionStateByRuntimeIdRef.current
  const storedSessionId = states.get(runtimeSessionId)?.storedSessionId

  if (!storedSessionId) {
    return
  }

  const profile = normalizeProfileKey(event.profile ?? deps.activeGatewayProfile)
  const connectionId = event.connectionId || LOCAL_CONNECTION_ID
  const source = backendScopeKey(connectionId, profile)

  const ownsTarget = () => {
    if (deps.sessionStateByRuntimeIdRef.current.get(runtimeSessionId)?.storedSessionId !== storedSessionId || isSessionGone(storedSessionId)) {
      return false
    }

    const owner = knownOwnerForSession(storedSessionId)
    const runtimeOwner = runtimeSessionOwner(runtimeSessionId)
    const exact = isSessionOwnerRoute(owner) ? owner : runtimeOwner

    if (isSessionOwnerRoute(exact)) {
      return source === backendScopeKey(exact.connectionId, normalizeProfileKey(exact.targetProfile ?? exact.profile))
    }

    if (typeof exact === 'string') {
      return connectionId === LOCAL_CONNECTION_ID && profile === normalizeProfileKey(exact)
    }

    return fromActiveSource() && (typeof owner !== 'string' || profile === normalizeProfileKey(owner))
  }

  if (!ownsTarget()) {
    return
  }

  if (payload.state === 'incomplete' && !event.replayed) {
    const key = JSON.stringify([source, storedSessionId, payload.call_id])

    if (!warnedCalls.has(key)) {
      warnedCalls.add(key)
      notify({ kind: 'warning', message: translateNow('notifications.voice.liveCaptureIncomplete') })
    }
  }

  if (payload.state !== 'updated' && payload.state !== 'saved') {
    return
  }

  // A slow read started before a newer commit must not win by returning first.
  const read = Symbol()
  reads.set(runtimeSessionId, read)
  void hydrateStoredSessionTranscript({
    attempts: 1,
    runtimeSessionId,
    storedSessionId,
    storedProfile: { connectionId, profile },
    passive: true,
    isCurrent: () => reads.get(runtimeSessionId) === read && ownsTarget(),
    updateSessionState: deps.updateSessionState
  }).finally(() => {
    if (reads.get(runtimeSessionId) === read) {
      reads.delete(runtimeSessionId)
    }
  })
}

/** A refused recorded GPT-Live claim never ran. The owner settles it as a queued
 *  admission (epoch and admission, no execution generation), which the execution
 *  fence rightly drops. Only the window whose pending turn is that exact submission
 *  renders it, and only while no execution is live there, so it can never settle
 *  another window's or connection's turn. Returns true when consumed. */
export function handleRefusedVoiceClaim(
  event: GatewayEvent,
  sessionId: null | string,
  deps: GatewayEventDeps,
  occurredAt: number
): boolean {
  const payload = event.payload as Record<string, unknown> | undefined
  const surface = parseErrorSurface(payload?.error_surface)

  if (
    event.type !== 'message.complete' ||
    payload?.outcome !== 'rejected' ||
    surface?.code !== 'voice_claim_unverified' ||
    executionAdmission(event) !== null ||
    typeof event.admission_id !== 'string'
  ) {
    return false
  }

  const state = sessionId ? deps.sessionStateByRuntimeIdRef.current.get(sessionId) : undefined
  const submission = payload.submission_id

  if (
    sessionId &&
    state &&
    typeof submission === 'string' &&
    submission &&
    state.voiceClaimSubmissionId === submission &&
    !state.turnLive &&
    !state.streamId
  ) {
    deps.failAssistantMessage(sessionId, coerceGatewayText(payload.error), occurredAt, surface)
    deps.updateSessionState(sessionId, current =>
      current.voiceClaimSubmissionId === submission ? { ...current, voiceClaimSubmissionId: null } : current
    )
  }

  return true
}
