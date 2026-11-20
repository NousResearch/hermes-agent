import { singleFlightSessionResume } from '@/app/session/hooks/use-prompt-actions/single-flight-resume'
import { primaryResumeParams } from '@/app/session/hooks/use-session-actions/resume-owner'
import type { SessionResumeResult } from '@/types/hermes'

import { $gateway, activeGatewayConnectionId } from './gateway'
import { replayPendingApproval } from './prompts'
import { markRuntimeGone } from './runtime-gone'
import { $activeSessionId } from './session'
import { ambientRequestFor, isSessionGoneForBackgroundPolling } from './session-gone-latch'
import type { SessionOwnerRoute } from './session-request-router'
import { $sessionStates, $sessionTiles, knownOwnerForSession, requestForOwnedSession } from './session-states'

/**
 * Resume runtimes on reconnect (t_44f96ba3).
 *
 * A reconnect brings the socket back, not the runtimes behind it. A runtime parked or reaped while
 * the client was away is no longer in the gateway's memory, so the session-scoped follow-ups a
 * reconnect fires bounce `4001 "session_id … not in memory (detached/reaped runtime; client should
 * resume the stored session)"` (282 in one production day, plus `session.control.read` × 7 and
 * `session.activate` × 4), and the approval/clarify card of the open chat stalls until the user
 * reopens it: the replayed events land, the runtime behind them does not.
 *
 * So a reconnect now RE-RESUMES every session with a live UI surface (the primary thread and every
 * open tile) before anything session-scoped follows — the shared channel holds the replayed
 * `open_requests` behind this same gate, and the post-ready `approval.pending` re-pull resolves the
 * session's fresh runtime id through {@link reconnectFollowUpRuntimeId} first. The resume is the
 * shape the desktop already sends when a chat is opened (`primaryResumeParams`: `defer_history` +
 * `omit_messages`), so it attaches a runtime and returns the pending approval without shipping the
 * transcript, and it coalesces with any other resume in flight through the window-wide
 * `singleFlightSessionResume` (#91276).
 *
 * Bounded by construction: one resume per stored session per reconnect epoch, ONE retry after
 * {@link RECONNECT_RESUME_SETTLING_RETRY_MS} for `4009` ("session disconnect interrupt settling"),
 * and a terminal verdict (`4001`/`4007`, canonical `not_found`) drops that runtime's replay cursor
 * and re-arms the surface against its STORED session. Never a loop, and never a rejection that
 * could hold the replay-derived cards hostage.
 */

/** 4009: the gateway is still settling a client-gone interrupt for this session — transient. */
const GATEWAY_SESSION_SETTLING_CODE = 4009
/** 4007: `session.resume`'s own "session not found" (the legacy transport's spelling). */
const GATEWAY_SESSION_RESUME_NOT_FOUND_CODE = 4007

/** How long the ONE settling retry waits: enough for a reap poll to retire the interrupt claim. */
export const RECONNECT_RESUME_SETTLING_RETRY_MS = 250

export interface LiveResumeSurface {
  /** The stored id the resume is keyed on. */
  storedSessionId: string
  /** The runtime id the window currently holds for it, when it has one (the potentially dead one). */
  runtimeId?: string
}

export interface ReconnectResumePorts {
  /** Forget a dead runtime's replay cursor (`JsonRpcGatewayClient.forgetReplayCursor`). */
  dropReplayCursor: (runtimeId: string) => void
  /** Re-arm whichever surface holds the dead runtime against its STORED session. */
  healView: (runtimeId: string) => void
  /** `session.resume` for one surface; resolves null when nothing was resumed. */
  resume: (surface: LiveResumeSurface) => Promise<null | SessionResumeResult>
  sleep: (ms: number) => Promise<void>
  /** The sessions with a live UI surface, deduped by stored id. */
  surfaces: () => LiveResumeSurface[]
}

type ResumeAttempt = { kind: 'failed' } | { kind: 'gone' } | { kind: 'ok'; runtimeId: null | string } | { kind: 'settling' }

interface ReconnectResumeEpoch {
  done: null | Promise<void>
  freshRuntimeByStoredId: Map<string, string>
  goneStoredIds: Set<string>
}

let epoch: ReconnectResumeEpoch | null = null

const errorCode = (error: unknown): null | number => {
  const code = error && typeof error === 'object' ? (error as { code?: unknown }).code : undefined

  return typeof code === 'number' ? code : null
}

/** The gateway is settling a client-gone interrupt for this session: retry once, later. */
export const isReconnectResumeSettling = (error: unknown): boolean => errorCode(error) === GATEWAY_SESSION_SETTLING_CODE

/** The gateway does not have this session at all: nothing will ever resume it. */
export const isReconnectResumeGone = (error: unknown): boolean =>
  errorCode(error) === GATEWAY_SESSION_RESUME_NOT_FOUND_CODE || isSessionGoneForBackgroundPolling(error)

/** Arm a new reconnect epoch. Called once per reconnect (both dial shapes), before the socket opens. */
export function beginReconnectResumeEpoch(): void {
  epoch = { done: null, freshRuntimeByStoredId: new Map(), goneStoredIds: new Set() }
}

/** What every reconnect owes once its socket is open: a fresh epoch, and its re-resumes started.
 *  Synchronous by contract — the caller must not await the resumes on the dial path. */
export function beginReconnectResumeOnSocket(ports?: ReconnectResumePorts): void {
  beginReconnectResumeEpoch()
  void ensureLiveSessionsResumed(ports)
}

/** The sessions with a live UI surface: the primary thread and every open tile, deduped by stored id.
 *  Only surfaces this window's PRIMARY route can serve: a tile owned by another connection is that
 *  connection's reconnect to run, not this one's (resuming it here would dial a backend out of
 *  nowhere, which is what `reconnectSecondaryGateways` owns). */
export function liveResumeSurfaces(): LiveResumeSurface[] {
  const surfaces = new Map<string, LiveResumeSurface>()
  const states = $sessionStates.get()
  const activeRuntimeId = $activeSessionId.get()
  const activeStoredId = activeRuntimeId ? (states[activeRuntimeId]?.storedSessionId ?? '') : ''
  const windowConnectionId = activeGatewayConnectionId()

  const servedHere = (ownerRoute?: SessionOwnerRoute) =>
    !ownerRoute?.connectionId || ownerRoute.connectionId === windowConnectionId

  const add = (storedSessionId: string, runtimeId?: null | string) => {
    if (!storedSessionId) {
      return
    }

    surfaces.set(storedSessionId, { runtimeId: runtimeId ?? undefined, storedSessionId })
  }

  if (activeStoredId) {
    add(activeStoredId, activeRuntimeId)
  }

  for (const tile of $sessionTiles.get()) {
    if (servedHere(tile.ownerRoute)) {
      add(tile.storedSessionId, tile.runtimeId)
    }
  }

  return [...surfaces.values()]
}

async function attemptResume(ports: ReconnectResumePorts, surface: LiveResumeSurface): Promise<ResumeAttempt> {
  try {
    const snapshot = await ports.resume(surface)

    return { kind: 'ok', runtimeId: typeof snapshot?.session_id === 'string' ? snapshot.session_id : null }
  } catch (error) {
    if (isReconnectResumeSettling(error)) {
      return { kind: 'settling' }
    }

    return isReconnectResumeGone(error) ? { kind: 'gone' } : { kind: 'failed' }
  }
}

/** Settle one surface: one attempt, one bounded retry while the gateway is still settling, then the verdict. */
async function resumeSurface(ports: ReconnectResumePorts, surface: LiveResumeSurface): Promise<ResumeAttempt> {
  const first = await attemptResume(ports, surface)

  if (first.kind !== 'settling') {
    return first
  }

  await ports.sleep(RECONNECT_RESUME_SETTLING_RETRY_MS)

  return attemptResume(ports, surface)
}

async function resumeLiveSurfaces(current: ReconnectResumeEpoch, ports: ReconnectResumePorts): Promise<void> {
  const surfaces = ports.surfaces()

  await Promise.all(
    surfaces.map(async surface => {
      const attempt = await resumeSurface(ports, surface)

      if (attempt.kind === 'ok') {
        if (attempt.runtimeId) {
          current.freshRuntimeByStoredId.set(surface.storedSessionId, attempt.runtimeId)
        }

        return
      }

      if (attempt.kind !== 'gone') {
        return
      }

      // Terminal: the runtime is dead for good, so stop replaying under its cursor and hand the
      // surface back to its stored session.
      current.goneStoredIds.add(surface.storedSessionId)

      if (surface.runtimeId) {
        ports.dropReplayCursor(surface.runtimeId)
        ports.healView(surface.runtimeId)
      }
    })
  )
}

/**
 * Re-resume every session with a live UI surface, once per reconnect epoch. Idempotent, never
 * rejects: this is the shared channel's `beforeOpenRequests` gate and the post-reconnect follow-up
 * barrier at once, so an unsettled promise here would hold a user's card off the screen.
 */
export function ensureLiveSessionsResumed(ports: ReconnectResumePorts = liveReconnectResumePorts()): Promise<void> {
  const current = epoch

  if (!current) {
    return Promise.resolve()
  }

  current.done ??= resumeLiveSurfaces(current, ports)

  return current.done
}

/** The runtime id a session-scoped follow-up must use: the fresh one this reconnect's re-resume minted
 *  for the session's stored id, the caller's own id when the session is not part of the epoch, or null
 *  when the gateway does not have the session at all (the stored-session affordance owns it then).
 *
 *  A runtime id rotates on every resume, so a follow-up keyed on the id held BEFORE the re-resume
 *  lands would 4001 against the runtime it just replaced.
 */
export async function reconnectFollowUpRuntimeId(runtimeId: string): Promise<null | string> {
  if (!epoch) {
    return runtimeId
  }

  await ensureLiveSessionsResumed()

  const storedSessionId = storedIdForRuntime(runtimeId)

  if (storedSessionId) {
    const fresh = epoch.freshRuntimeByStoredId.get(storedSessionId)

    if (fresh) {
      return fresh
    }

    if (epoch.goneStoredIds.has(storedSessionId)) {
      return null
    }
  }

  return runtimeId
}

/** The durable identity behind a runtime id (the cached state, or a tile that bound before it landed). */
export function storedIdForRuntime(runtimeId: string): null | string {
  const cached = $sessionStates.get()[runtimeId]?.storedSessionId

  if (cached) {
    return cached
  }

  return $sessionTiles.get().find(tile => tile.runtimeId === runtimeId)?.storedSessionId ?? null
}

/** The gate every reconnect-attached client installs (`beforeOpenRequests`). */
export const reconnectResumeGate = (): Promise<void> => ensureLiveSessionsResumed()

/**
 * `replayPendingApproval`, ordered behind this reconnect's re-resumes: the queue-level re-pull must
 * name the runtime the gateway holds NOW, or it answers the very 4001 this epoch exists to prevent.
 * Null (session gone) is a no-op — the affordance is already up.
 */
export async function replayPendingApprovalAfterReconnectResume(
  gateway: Parameters<typeof replayPendingApproval>[0],
  runtimeId: string
): Promise<void> {
  const live = await reconnectFollowUpRuntimeId(runtimeId)

  if (live === null) {
    return
  }

  await replayPendingApproval(gateway, live)
}

/** The production ports: the owner-routed RPC, the runtime-gone heal, and the two stores. */
export function liveReconnectResumePorts(): ReconnectResumePorts {
  return {
    dropReplayCursor: runtimeId => $gateway.get()?.forgetReplayCursor(runtimeId),
    healView: runtimeId => {
      markRuntimeGone(runtimeId)
    },
    resume: resumeSurfaceOverOwnerRoute,
    sleep: ms => new Promise(resolve => setTimeout(resolve, ms)),
    surfaces: liveResumeSurfaces
  }
}

async function resumeSurfaceOverOwnerRoute(surface: LiveResumeSurface): Promise<null | SessionResumeResult> {
  const gateway = $gateway.get()

  if (!gateway) {
    return null
  }

  const owner = knownOwnerForSession(surface.storedSessionId)
  const profile = typeof owner === 'string' ? owner : owner?.profile

  const snapshot = await singleFlightSessionResume(
    surface.storedSessionId,
    () =>
      requestForOwnedSession<SessionResumeResult>(
        surface.storedSessionId,
        ambientRequestFor(gateway),
        'session.resume',
        primaryResumeParams(surface.storedSessionId, false, undefined, profile)
      ),
    { scope: owner }
  )

  return typeof snapshot?.session_id === 'string' && snapshot.session_id ? snapshot : null
}

/** Tests only: drop the epoch so cases start from a clean slate. */
export function resetReconnectResumeForTests(): void {
  epoch = null
}
