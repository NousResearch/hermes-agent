import type {
  GoalContractSnapshot,
  GoalGateSnapshot,
  GoalSnapshot,
  HeartbeatSnapshot,
  LoopSnapshot,
  SessionControlDispatch,
  SessionControlSnapshot,
  WaitBarrierPid,
  WaitBarrierSession,
  WaitBarrierUntil
} from '@hermes/shared'
import { atom } from 'nanostores'

import { $gateway } from './gateway'
import { refreshSessionGoal } from './goals'
import { isSessionGone, isSessionGoneForBackgroundPolling, markSessionGone } from './runtime-gone'
import { ambientRequestFor } from './session-gone-latch'
import { requestForOwnedSession } from './session-states'

export type SessionControlGoalStatus = GoalSnapshot['status']
export type SessionControlLoopMode = LoopSnapshot['mode']
export type SessionControlLoopStatus = LoopSnapshot['status']
export type SessionControlHeartbeatStatus = HeartbeatSnapshot['status']
export type SessionControlGoalContract = GoalContractSnapshot
export type SessionControlGate = GoalGateSnapshot
export type SessionControlWaitBarrier = WaitBarrierPid | WaitBarrierSession | WaitBarrierUntil
export type SessionControlGoal = GoalSnapshot
export type SessionControlLoop = LoopSnapshot
export type SessionControlHeartbeat = HeartbeatSnapshot
export type { SessionControlDispatch, SessionControlSnapshot }

export type SessionControlAction =
  | 'goal.clear'
  | 'goal.pause'
  | 'goal.resume'
  | 'goal.unwait'
  | 'heartbeat.clear'
  | 'heartbeat.pause'
  | 'heartbeat.resume'
  | 'loop.pause'
  | 'loop.resume'
  | 'loop.stop'
  | 'subgoal.add'
  | 'subgoal.clear'
  | 'subgoal.remove'

export type SessionControlActionArgs = { index: number } | { text: string }

export interface SessionControlEntry {
  capability: 'unknown' | 'supported' | 'unsupported'
  error: string | null
  loading: boolean
  pendingAction: SessionControlAction | null
  snapshot: SessionControlSnapshot | null
}

interface RefreshOptions {
  background?: boolean
}

type UnknownRecord = Record<string, unknown>

const ERROR_LIMIT = 240

const versions = new Map<string, number>()
const eventVersions = new Map<string, number>()

export const $sessionControlBySession = atom<Record<string, SessionControlEntry>>({})

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function emptyEntry(): SessionControlEntry {
  return { capability: 'unknown', error: null, loading: false, pendingAction: null, snapshot: null }
}

function currentVersion(sessionId: string): number {
  return versions.get(sessionId) ?? 0
}

function advanceVersion(sessionId: string): number {
  const next = currentVersion(sessionId) + 1
  versions.set(sessionId, next)

  return next
}

function currentEventVersion(sessionId: string): number {
  return eventVersions.get(sessionId) ?? 0
}

function advanceEventVersion(sessionId: string): number {
  const next = currentEventVersion(sessionId) + 1
  eventVersions.set(sessionId, next)

  return next
}

function isCurrent(sessionId: string, token: number): boolean {
  return currentVersion(sessionId) === token
}

function sameEntry(first: SessionControlEntry, second: SessionControlEntry): boolean {
  return (
    first.capability === second.capability &&
    first.error === second.error &&
    first.loading === second.loading &&
    first.pendingAction === second.pendingAction &&
    first.snapshot === second.snapshot
  )
}

function publishEntry(sessionId: string, next: SessionControlEntry): SessionControlEntry {
  const entries = $sessionControlBySession.get()
  const current = entries[sessionId]

  if (current && sameEntry(current, next)) {
    return current
  }

  $sessionControlBySession.set({ ...entries, [sessionId]: next })

  return next
}

function applyParsedSnapshot(sessionId: string, snapshot: SessionControlSnapshot): SessionControlEntry {
  advanceVersion(sessionId)
  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()
  const nextSnapshot = current.snapshot?.revision === snapshot.revision ? current.snapshot : snapshot

  return publishEntry(sessionId, {
    capability: 'supported',
    error: null,
    loading: false,
    pendingAction: null,
    snapshot: nextSnapshot
  })
}

/** Applies a read/action snapshot and marks its session as supported. */
export function applySessionControlSnapshot(
  sessionId: string,
  snapshot: SessionControlSnapshot
): SessionControlEntry | undefined {
  return sessionId ? applyParsedSnapshot(sessionId, snapshot) : undefined
}

/** Applies a `session.control.update` event. */
export function applySessionControlUpdate(
  sessionId: string,
  snapshot: SessionControlSnapshot
): SessionControlEntry | undefined {
  if (!sessionId) {
    return undefined
  }

  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()
  const actionIsPending = current.pendingAction !== null

  advanceEventVersion(sessionId)

  if (!actionIsPending) {
    advanceVersion(sessionId)
  }

  const nextSnapshot = current.snapshot?.revision === snapshot.revision ? current.snapshot : snapshot

  return publishEntry(sessionId, {
    ...current,
    capability: 'supported',
    error: null,
    loading: actionIsPending ? current.loading : false,
    pendingAction: actionIsPending ? current.pendingAction : null,
    snapshot: nextSnapshot
  })
}

/** Drops one runtime session's entry when the session record is closed/deleted. */
export function clearSessionControl(sessionId: string): void {
  if (!sessionId) {
    return
  }

  advanceVersion(sessionId)
  const entries = $sessionControlBySession.get()

  if (!(sessionId in entries)) {
    return
  }

  const { [sessionId]: _removed, ...remaining } = entries
  $sessionControlBySession.set(remaining)
}

/**
 * Wipes every entry — the gateway-switch seam. Entries are keyed by runtime
 * session id and a different backend mints new ids, so nothing here can be
 * reused; in-flight reads/actions from the old backend are invalidated by the
 * version bump so a late response cannot repopulate the map.
 */
export function clearAllSessionControl(): void {
  for (const sessionId of new Set([...versions.keys(), ...Object.keys($sessionControlBySession.get())])) {
    advanceVersion(sessionId)
  }

  $sessionControlBySession.set({})
  versions.clear()
  eventVersions.clear()
}

function beginRead(sessionId: string, background: boolean): number {
  const token = advanceVersion(sessionId)
  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()

  publishEntry(sessionId, {
    ...current,
    error: null,
    loading: background ? current.loading : true
  })

  return token
}

function beginAction(sessionId: string, action: SessionControlAction): number {
  const token = advanceVersion(sessionId)
  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()

  publishEntry(sessionId, {
    ...current,
    error: null,
    loading: true,
    pendingAction: action
  })

  return token
}

function boundedError(error: unknown): string {
  const message =
    error instanceof Error
      ? error.message
      : isRecord(error) && typeof error.message === 'string'
        ? error.message
        : 'Session control request failed'

  return message.trim().slice(0, ERROR_LIMIT) || 'Session control request failed'
}

function publishFailure(sessionId: string, token: number, error: unknown, clearPendingAction: boolean): void {
  if (!isCurrent(sessionId, token)) {
    return
  }

  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()
  publishEntry(sessionId, {
    ...current,
    error: boundedError(error),
    loading: false,
    pendingAction: clearPendingAction ? null : current.pendingAction
  })
}

function finishGoneRequest(sessionId: string, token: number, clearPendingAction: boolean): void {
  if (!isCurrent(sessionId, token)) {
    return
  }

  markSessionGone(sessionId)
  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()
  publishEntry(sessionId, {
    ...current,
    loading: false,
    pendingAction: clearPendingAction ? null : current.pendingAction
  })
}

function markUnsupported(sessionId: string, token: number): boolean {
  if (!isCurrent(sessionId, token)) {
    return false
  }

  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()

  if (current.capability === 'unsupported') {
    return false
  }

  advanceVersion(sessionId)
  publishEntry(sessionId, {
    ...current,
    capability: 'unsupported',
    error: null,
    loading: false,
    pendingAction: null
  })

  return true
}

function isMethodNotFound(error: unknown): boolean {
  if (isRecord(error) && error.code === -32601) {
    return true
  }

  const message =
    error instanceof Error ? error.message : isRecord(error) && typeof error.message === 'string' ? error.message : ''

  return message.toLowerCase().includes('method not found') || message.toLowerCase().includes('method-not-found')
}

/** Hydrates one session's structured controls; background refreshes never flash a loading state. */
export async function refreshSessionControl(
  sessionId: string,
  options: RefreshOptions = {}
): Promise<SessionControlEntry | undefined> {
  const existing = $sessionControlBySession.get()[sessionId]

  if (!sessionId || existing?.capability === 'unsupported' || isSessionGone(sessionId)) {
    return existing
  }

  const gateway = $gateway.get()

  if (!gateway) {
    return existing
  }

  const token = beginRead(sessionId, Boolean(options.background))

  try {
    const response = await requestForOwnedSession(sessionId, ambientRequestFor(gateway), 'session.control.read', {
      session_id: sessionId
    })

    if (!isCurrent(sessionId, token)) {
      return $sessionControlBySession.get()[sessionId]
    }

    return applyParsedSnapshot(sessionId, response.control)
  } catch (error) {
    if (!isCurrent(sessionId, token)) {
      return $sessionControlBySession.get()[sessionId]
    }

    if (isMethodNotFound(error)) {
      const transitioned = markUnsupported(sessionId, token)

      if (transitioned) {
        await refreshSessionGoal(sessionId)
      }

      return $sessionControlBySession.get()[sessionId]
    }

    if (isSessionGoneForBackgroundPolling(error)) {
      finishGoneRequest(sessionId, token, false)

      return $sessionControlBySession.get()[sessionId]
    }

    publishFailure(sessionId, token, error, false)

    return $sessionControlBySession.get()[sessionId]
  }
}

/** Runs an allowlisted backend control action; callers own any composer/UI dispatch. */
export async function runSessionControlAction(
  sessionId: string,
  action: SessionControlAction,
  args?: SessionControlActionArgs
): Promise<SessionControlDispatch> {
  if (!sessionId) {
    throw new Error('A session id is required for session control')
  }

  if (isSessionGone(sessionId)) {
    throw new Error('Session not found')
  }

  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Session control gateway is unavailable')
  }

  const eventVersion = currentEventVersion(sessionId)
  const token = beginAction(sessionId, action)

  try {
    const response = await requestForOwnedSession(sessionId, ambientRequestFor(gateway), 'session.control', {
      action,
      args: { index: null, profile: null, text: null, ...args },
      session_id: sessionId
    })

    if (isCurrent(sessionId, token)) {
      applyParsedSnapshot(sessionId, response.control)
    }

    return response.dispatch
  } catch (error) {
    if (isMethodNotFound(error)) {
      const transitioned = eventVersion === currentEventVersion(sessionId) ? markUnsupported(sessionId, token) : false

      if (transitioned) {
        await refreshSessionGoal(sessionId)
      } else {
        publishFailure(sessionId, token, error, true)
      }
    } else if (isSessionGoneForBackgroundPolling(error)) {
      finishGoneRequest(sessionId, token, true)
    } else {
      publishFailure(sessionId, token, error, true)
    }

    throw error
  }
}

/** Refreshes only sessions already proven to support the structured-control RPC. */
export async function refreshSupportedSessionControlAfterTurn(sessionId: string): Promise<void> {
  if (!sessionId || $sessionControlBySession.get()[sessionId]?.capability !== 'supported') {
    return
  }

  await refreshSessionControl(sessionId, { background: true })
}
