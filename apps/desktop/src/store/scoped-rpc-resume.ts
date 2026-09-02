/**
 * Resume a stored session before session-scoped background RPCs.
 *
 * After a gateway restart the desktop still holds reaped runtime ids and keeps
 * polling `approval.pending` / `process.list` / `slash.exec`. Those methods
 * resolve only against the in-memory runtime; the gateway's 4001 tells the
 * client to `session.resume` the durable row. This module:
 *
 *  - starts a new epoch on `gateway.ready` so every stored session is resumed
 *    once before the next scoped poll
 *  - reads `session.reclaimed` ids from the payload (envelope `session_id` is
 *    often empty on `ws_orphan_reap`)
 *  - single-flights `session.resume` with the route/tile resume path
 *  - fails closed after one resume miss instead of retrying scoped RPCs for hours
 */

import { singleFlightSessionResume } from '@/app/session/hooks/use-prompt-actions/single-flight-resume'

import { isSessionGone, isSessionGoneForBackgroundPolling, markRuntimeGone, markSessionGone } from './runtime-gone'
import { $sessionStates, $sessionTiles } from './session-states'

export const SCOPED_RPC_METHODS = ['approval.pending', 'process.list', 'slash.exec'] as const

export type ScopedRpcMethod = (typeof SCOPED_RPC_METHODS)[number]

export function isScopedRpcMethod(method: string): method is ScopedRpcMethod {
  return (SCOPED_RPC_METHODS as readonly string[]).includes(method)
}

export interface GatewayRequester {
  request: (method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<unknown>
}

export interface ReclaimedSessionIds {
  runtimeId: string
  storedSessionId: string
}

/** Minted stored keys (`20260814_094231_4e849d`) vs short live runtime ids. */
const STORED_SESSION_KEY_RE = /^\d{8}_\d{6}_[A-Fa-f0-9]{6}$/

let scopedRpcEpoch = 0
const liveIdByStored = new Map<string, { epoch: number; liveId: string }>()
const resumeFailedStoredIds = new Set<string>()

function trimId(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

export function storedIdForRuntime(runtimeId: string): null | string {
  if (!runtimeId) {
    return null
  }

  const cached = $sessionStates.get()[runtimeId]?.storedSessionId

  if (cached) {
    return cached
  }

  return $sessionTiles.get().find(tile => tile.runtimeId === runtimeId)?.storedSessionId ?? null
}

export function durableSessionId(sessionId: string): null | string {
  const trimmed = sessionId.trim()

  if (!trimmed) {
    return null
  }

  if (STORED_SESSION_KEY_RE.test(trimmed)) {
    return trimmed
  }

  return storedIdForRuntime(trimmed)
}

/**
 * `session.reclaimed` puts the real ids in `payload`. The envelope `session_id`
 * is frequently `""` on `ws_orphan_reap` (#100639).
 */
export function idsFromSessionReclaimed(
  payload: { session_id?: unknown; stored_session_id?: unknown } | null | undefined,
  envelopeSessionId?: unknown
): ReclaimedSessionIds {
  const runtimeId = trimId(payload?.session_id) || trimId(envelopeSessionId)
  const storedSessionId = trimId(payload?.stored_session_id) || storedIdForRuntime(runtimeId) || ''

  return { runtimeId, storedSessionId }
}

/** Gateway (re)connected: previous live ids are not valid on the new process. */
export function beginScopedRpcEpoch(): void {
  scopedRpcEpoch += 1
  liveIdByStored.clear()
  resumeFailedStoredIds.clear()
}

export function scopedRpcEpochValue(): number {
  return scopedRpcEpoch
}

function rememberLiveId(storedSessionId: string, liveId: string): void {
  liveIdByStored.set(storedSessionId, { epoch: scopedRpcEpoch, liveId })
  resumeFailedStoredIds.delete(storedSessionId)
}

function cachedLiveId(storedSessionId: string): null | string {
  const hit = liveIdByStored.get(storedSessionId)

  if (!hit || hit.epoch !== scopedRpcEpoch) {
    return null
  }

  return hit.liveId
}

async function resumeStoredSession(gateway: GatewayRequester, storedSessionId: string): Promise<null | string> {
  if (resumeFailedStoredIds.has(storedSessionId)) {
    return null
  }

  const cached = cachedLiveId(storedSessionId)

  if (cached) {
    return cached
  }

  try {
    const resumed = await singleFlightSessionResume(storedSessionId, async () => {
      const stillCached = cachedLiveId(storedSessionId)

      if (stillCached) {
        return { session_id: stillCached }
      }

      return gateway.request('session.resume', {
        omit_messages: true,
        session_id: storedSessionId,
        source: 'desktop'
      }) as Promise<{ session_id?: string } | null>
    })

    const liveId = trimId(resumed?.session_id)

    if (!liveId) {
      resumeFailedStoredIds.add(storedSessionId)

      return null
    }

    rememberLiveId(storedSessionId, liveId)

    return liveId
  } catch (error) {
    if (isSessionGoneForBackgroundPolling(error)) {
      resumeFailedStoredIds.add(storedSessionId)

      return null
    }

    throw error
  }
}

/**
 * Resolve a live runtime id for a session-scoped RPC. Resumes the stored row
 * when this epoch has not yet attached one. Returns null when resume cannot
 * run or already failed — callers must not poll the dead runtime.
 */
export async function ensureLiveSessionIdForScopedRpc(
  gateway: GatewayRequester | null | undefined,
  sessionId: string | null | undefined
): Promise<null | string> {
  if (!gateway || !sessionId) {
    return null
  }

  const trimmed = sessionId.trim()

  if (!trimmed) {
    return null
  }

  const stored = durableSessionId(trimmed)

  if (!stored) {
    // Never-persisted draft: nothing to resume. Let the caller hit the runtime.
    return trimmed
  }

  return resumeStoredSession(gateway, stored)
}

/** Tests only. */
export function resetScopedRpcResume(): void {
  scopedRpcEpoch = 0
  liveIdByStored.clear()
  resumeFailedStoredIds.clear()
}

/** True when a background poller may still try resume / scoped RPC. */
export function canAttemptScopedRpc(sessionId: string | null | undefined): boolean {
  const trimmed = sessionId?.trim() ?? ''

  if (!trimmed) {
    return false
  }

  return !(isSessionGone(trimmed) && !durableSessionId(trimmed))
}

/** Resume failed or there is no stored row: latch the runtime and fail loud. */
export function noteUnresumableScopedSession(sessionId: string): void {
  const stored = durableSessionId(sessionId)

  if (stored) {
    markRuntimeGone(sessionId, stored)

    return
  }

  markSessionGone(sessionId)
}
