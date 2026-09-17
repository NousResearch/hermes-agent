import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

import type { GatewayRequest } from './utils'

const capturedRequests = new WeakMap<GatewayRequest, SubmissionDestination>()

export interface SubmissionDestination {
  readonly scopeKey: string
  readonly owner: SessionOwnerScope
  readonly requestGateway: GatewayRequest
}

/** Capture before preprocessing, not when the prepared payload finally sends.
 * Unknown legacy owners may use the existing dispatcher only while its ambient
 * connection/profile is unchanged; a write must never fall to a new socket. */
export function captureSubmissionDestination(
  sessionId: string | null | undefined,
  request: GatewayRequest,
  restoredOwner?: { owner: SessionOwnerScope }
): SubmissionDestination {
  const captured = capturedRequests.get(request)

  if (captured && !restoredOwner) {
    return captured
  }

  const knownOwner = restoredOwner ? restoredOwner.owner : knownOwnerForSession(sessionId)
  const owner = knownOwner && typeof knownOwner === 'object' ? Object.freeze({ ...knownOwner }) : knownOwner

  // Window visibility, logs and registry discovery replace this descriptor.
  // Only transport/auth authority may invalidate an in-flight write.
  const authority = () => {
    const connection = $connection.get()

    return (
      connection &&
      JSON.stringify([
        connection.baseUrl,
        connection.wsUrl,
        connection.token,
        connection.connectionId,
        connection.mode,
        connection.profile,
        connection.authMode
      ])
    )
  }

  const connectionAuthority = authority()
  const profile = $activeGatewayProfile.get()

  const guardedRequest: GatewayRequest = (method, params, timeoutMs) => {
    if (connectionAuthority !== authority() || profile !== $activeGatewayProfile.get()) {
      return Promise.reject(new Error('Submission destination changed; retry from the original session'))
    }

    return timeoutMs === undefined ? request(method, params) : request(method, params, timeoutMs)
  }

  const connection = $connection.get()

  const scopeKey = JSON.stringify(
    owner && typeof owner === 'object'
      ? [owner.connectionId, owner.profile]
      : [connection?.connectionId ?? connection?.baseUrl ?? null, profile]
  )

  const destination = Object.freeze({
    scopeKey,
    owner,
    requestGateway: <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) =>
      typeof owner === 'object' && owner !== null
        ? requestForSessionProfile<T>(owner, guardedRequest, method, params, timeoutMs)
        : guardedRequest<T>(method, params, timeoutMs)
  })

  capturedRequests.set(destination.requestGateway, destination)

  return destination
}
