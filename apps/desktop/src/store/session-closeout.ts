import { randomUuid } from '@hermes/shared'

import {
  captureSubmissionDestination,
  type SubmissionDestination
} from '@/app/session/hooks/use-prompt-actions/submission-destination'
import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { isMissingRpcMethod } from '@/lib/gateway-rpc'

import type { SessionCloseoutResult } from '../../../shared/src/gateway-contract.generated'

import { ownerLookupSessionRows, resolveComposerSessionKey } from './session'
import { assertSessionOwnerResolved } from './session-owner-resolution'
import { requestForSessionProfile } from './session-request-router'
import { $sessionStates, $sessionTiles, knownOwnerForSession } from './session-states'
import { captureSessionRestoreFence } from './session-sync'

export type SessionCloseoutReceipt = SessionCloseoutResult
export interface SessionCloseoutOptions {
  intent?: 'archive' | 'new'
  destination?: SubmissionDestination
  requestId?: string
  preserveView?: boolean
  deferUi?: boolean
  receipt?: SessionCloseoutReceipt
  onApplied?: () => void
}

const requests = new Map<string, { id: string; isCurrent: () => boolean }>()
const flights = new Map<string, { promise: Promise<SessionCloseoutReceipt>; isCurrent: () => boolean }>()
const REVIEW_TIMEOUT_MS = 1_800_000

class ReleasedCloseout extends Error {}

export function captureCloseoutTarget(
  storedSessionId: string,
  request: GatewayRequest,
  options: SessionCloseoutOptions = {}
) {
  const owner = options.destination?.owner ?? knownOwnerForSession(storedSessionId)
  assertSessionOwnerResolved(owner, { method: 'session.closeout', sessionId: storedSessionId })
  const destination = options.destination ?? captureSubmissionDestination(storedSessionId, request)

  const runtimeId =
    $sessionTiles.get().find(tile => tile.storedSessionId === storedSessionId)?.runtimeId ??
    Object.entries($sessionStates.get()).find(([, state]) => state.storedSessionId === storedSessionId)?.[0]

  const profile = typeof owner === 'string' ? owner : owner?.targetProfile || owner?.profile

  const fence = captureSessionRestoreFence(
    owner && typeof owner === 'object' ? owner.connectionId : null,
    profile || 'default',
    [storedSessionId, resolveComposerSessionKey(storedSessionId, ownerLookupSessionRows()) ?? storedSessionId]
  )

  return {
    ...fence,
    storedSessionId,
    runtimeId,
    profile,
    destination,
    request: <T>(method: string, params: Record<string, unknown>) =>
      requestForSessionProfile<T>(owner, destination.requestGateway, method, params)
  }
}

type Target = ReturnType<typeof captureCloseoutTarget>

export function validateCloseoutReceipt(receipt: SessionCloseoutReceipt, target: Target): void {
  target.assertCurrent()
  const old = receipt?.old

  const matches =
    old &&
    (old.logical_session_id === target.storedSessionId ||
      old.stored_session_id === target.storedSessionId ||
      Boolean(target.runtimeId && old.runtime_id === target.runtimeId))

  if (!matches || (target.profile && old.profile !== target.profile)) {
    throw new Error('Closeout receipt belongs to another session')
  }

  if (receipt.error?.code === 'closeout_reopened') {
    throw new ReleasedCloseout(receipt.error.message)
  }

  if (receipt.status === 'failed') {
    throw new Error(receipt.error?.message || 'Conversation preservation failed; the transcript is retained')
  }

  if (receipt.status === 'preserved' && (!receipt.archived || !receipt.receipt_id || !receipt.next_create_request_id)) {
    throw new Error('Closeout has no durable archived preservation receipt')
  }
}

async function readUntilPreserved(
  target: Target,
  requestId: string,
  intent: 'archive' | 'new'
): Promise<SessionCloseoutReceipt> {
  const deadline = Date.now() + REVIEW_TIMEOUT_MS

  while (Date.now() < deadline) {
    target.assertCurrent()

    const receipt = await target.request<SessionCloseoutReceipt>('session.closeout', {
      session_id: target.runtimeId || target.storedSessionId,
      request_id: requestId,
      intent,
      ...(target.profile ? { profile: target.profile } : {})
    })

    validateCloseoutReceipt(receipt, target)

    if (receipt.status === 'preserved') {
      return receipt
    }

    await new Promise(resolve => setTimeout(resolve, 1000))
  }

  throw new Error('Closeout review is still pending; retry from this conversation to read the same receipt')
}

export function requestSessionCloseout(
  target: Target,
  options: SessionCloseoutOptions = {}
): Promise<SessionCloseoutReceipt> {
  const key = JSON.stringify([target.destination.scopeKey, target.storedSessionId, options.intent ?? 'archive'])
  const existing = flights.get(key)

  if (existing?.isCurrent()) {
    return existing.promise
  }

  const retained = requests.get(key)
  const requestId = options.requestId ?? (retained?.isCurrent() ? retained.id : randomUuid())
  const identity = { id: requestId, isCurrent: target.isCurrent }
  requests.set(key, identity)

  const flight = readUntilPreserved(target, requestId, options.intent ?? 'archive')
    .then(receipt => {
      target.assertCurrent()

      if (requests.get(key) === identity) {
        requests.delete(key)
      }

      return receipt
    })
    .catch(error => {
      if (error instanceof ReleasedCloseout && requests.get(key) === identity) {
        requests.delete(key)
      }

      if (isMissingRpcMethod(error)) {
        throw new Error(
          'This backend cannot preserve conversations before archival. Update the owning Hermes backend, then retry; the transcript is retained.'
        )
      }

      throw error
    })
    .finally(() => {
      if (flights.get(key)?.promise === flight) {
        flights.delete(key)
      }
    })

  flights.set(key, { promise: flight, isCurrent: target.isCurrent })

  return flight
}
