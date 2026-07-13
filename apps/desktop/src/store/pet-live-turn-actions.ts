import { isGatewayTimeoutError, isSessionNotFoundError } from '@/lib/session-errors'

import { clearClarifyRequest } from './clarify'
import type { QueuedPromptEntry } from './composer-queue'
import {
  $petActionCenter,
  type PetActionCenterAllowedAction,
  type PetActionCenterErrorCode,
  type PetActionCenterLiveTurnItem,
  type PetActionCenterLiveTurnStatus,
  setPetActionCenterActionStatus
} from './pet-action-center'
import {
  $petLiveSessions,
  acknowledgePetLiveSession,
  beginPetLiveSession,
  type PetLiveSessionSnapshot,
  replacePetLiveSessionRuntime
} from './pet-live-session'
import type { PetActionCenterControl } from './pet-overlay'
import { normalizeProfileKey } from './profile-key'
import { clearAllPrompts } from './prompts'
import { $cronSessions, $messagingSessions, $sessions } from './session'
import { profileSessionKey } from './session-identity'

export interface PetLiveTurnGateway {
  readonly connectionState: string
  request<T = unknown>(method: string, params?: Record<string, unknown>): Promise<T>
}

export interface PetLiveTurnActionDependencies {
  enqueuePrompt: (
    key: string,
    payload: { text: string; attachments: [] }
  ) => QueuedPromptEntry | null
  gatewayForProfile: (profile: string) => PetLiveTurnGateway | null
}

export type PetLiveTurnControl = Extract<
  PetActionCenterControl,
  {
    type:
      | 'action-center-submit'
      | 'action-center-steer'
      | 'action-center-queue'
      | 'action-center-stop'
      | 'action-center-acknowledge'
  }
>

const liveActionLocks = new Set<string>()

function currentLiveItem(itemId: string): PetActionCenterLiveTurnItem | null {
  const item = $petActionCenter.get().items.find(candidate => candidate.id === itemId)

  return item?.kind === 'live-turn' ? item : null
}

function fail(itemId: string, errorCode: PetActionCenterErrorCode): void {
  setPetActionCenterActionStatus({ status: 'error', itemId, errorCode })
}

function start(itemId: string): void {
  setPetActionCenterActionStatus({ status: 'submitting', itemId })
}

function finish(
  itemId: string,
  status: 'success' | 'steer-rejected' | 'steered' | 'queued' | 'stopped' | 'acknowledged'
): void {
  const action = $petActionCenter.get().action

  if (action?.status === 'submitting' && action.itemId === itemId) {
    setPetActionCenterActionStatus({ status, itemId })
  }
}

function finishError(itemId: string, errorCode: PetActionCenterErrorCode): void {
  const action = $petActionCenter.get().action

  if (action?.status === 'submitting' && action.itemId === itemId) {
    fail(itemId, errorCode)
  }
}

function supports(
  item: PetActionCenterLiveTurnItem,
  action: PetActionCenterAllowedAction,
  statuses: PetActionCenterLiveTurnStatus[]
): boolean {
  return statuses.includes(item.status) && item.allowedActions.includes(action)
}

function resolveCapableItem(
  itemId: string,
  action: PetActionCenterAllowedAction,
  statuses: PetActionCenterLiveTurnStatus[]
): PetActionCenterLiveTurnItem | null {
  const item = currentLiveItem(itemId)

  if (!item) {
    fail(itemId, 'item-not-found')

    return null
  }

  if (!supports(item, action, statuses)) {
    fail(itemId, 'capability-denied')

    return null
  }

  return item
}

function verifiedStoredSession(profile: string, storedSessionId: string | null): boolean {
  if (!storedSessionId) {
    return false
  }

  const normalizedProfile = normalizeProfileKey(profile)

  return [...$sessions.get(), ...$cronSessions.get(), ...$messagingSessions.get()].some(
    session =>
      normalizeProfileKey(session.profile) === normalizedProfile &&
      (session.id === storedSessionId || session._lineage_root_id === storedSessionId)
  )
}

function exactLiveSnapshot(profile: string, runtimeSessionId: string): PetLiveSessionSnapshot | null {
  const key = profileSessionKey(profile, runtimeSessionId)

  return (
    $petLiveSessions
      .get()
      .find(snapshot => profileSessionKey(snapshot.profile, snapshot.runtimeSessionId) === key) ?? null
  )
}

function isIdleSnapshot(snapshot: PetLiveSessionSnapshot, storedSessionId: string): boolean {
  return (
    snapshot.storedSessionId === storedSessionId &&
    !snapshot.busy &&
    !snapshot.needsInput &&
    !snapshot.awaitingResponse &&
    snapshot.outcome === null
  )
}

function actionLockKey(item: PetActionCenterLiveTurnItem): string {
  return profileSessionKey(item.profile, item.storedSessionId ?? item.sessionId ?? '')
}

function textOf(control: Extract<PetLiveTurnControl, { text: string }>): string | null {
  const text = control.text.trim()

  if (!text) {
    fail(control.itemId, 'invalid-text')

    return null
  }

  return text
}

function gatewayFor(
  item: PetActionCenterLiveTurnItem,
  dependencies: PetLiveTurnActionDependencies
): PetLiveTurnGateway | null {
  const gateway = dependencies.gatewayForProfile(item.profile)

  if (!gateway || gateway.connectionState !== 'open') {
    fail(item.id, 'disconnected')

    return null
  }

  return gateway
}

async function send(
  control: Extract<PetLiveTurnControl, { type: 'action-center-submit' }>,
  dependencies: PetLiveTurnActionDependencies
): Promise<void> {
  const item = resolveCapableItem(control.itemId, 'send', ['idle'])

  if (!item) {
    return
  }

  const text = textOf(control)

  if (!text) {
    return
  }

  const gateway = gatewayFor(item, dependencies)

  if (!gateway || !item.sessionId) {
    if (!item.sessionId) {
      fail(item.id, 'invalid-item')
    }

    return
  }

  start(item.id)

  try {
    await gateway.request('prompt.submit', { session_id: item.sessionId, text })
    beginPetLiveSession(item.profile, item.sessionId, item.storedSessionId)
    finish(item.id, 'success')
  } catch (error) {
    if (!isSessionNotFoundError(error) && !isGatewayTimeoutError(error)) {
      finishError(item.id, 'rpc-failed')

      return
    }

    const current = currentLiveItem(item.id)

    if (
      !current ||
      !supports(current, 'send', ['idle']) ||
      current.profile !== item.profile ||
      current.sessionId !== item.sessionId ||
      current.storedSessionId !== item.storedSessionId ||
      !verifiedStoredSession(item.profile, item.storedSessionId)
    ) {
      finishError(item.id, 'stale-runtime')

      return
    }

    try {
      const resumed = await gateway.request<{ session_id?: unknown }>('session.resume', {
        session_id: item.storedSessionId,
        source: 'desktop'
      })

      const recoveredId = typeof resumed?.session_id === 'string' ? resumed.session_id.trim() : ''
      const storedSessionId = item.storedSessionId

      if (
        !recoveredId ||
        !storedSessionId ||
        !verifiedStoredSession(item.profile, storedSessionId)
      ) {
        finishError(item.id, 'stale-runtime')

        return
      }

      const latestOldItem = currentLiveItem(item.id)

      if (
        latestOldItem &&
        (!supports(latestOldItem, 'send', ['idle']) ||
          latestOldItem.profile !== item.profile ||
          latestOldItem.sessionId !== item.sessionId ||
          latestOldItem.storedSessionId !== storedSessionId)
      ) {
        finishError(item.id, 'stale-runtime')

        return
      }

      const oldSnapshot = exactLiveSnapshot(item.profile, item.sessionId)

      if (oldSnapshot && !isIdleSnapshot(oldSnapshot, storedSessionId)) {
        finishError(item.id, 'stale-runtime')

        return
      }

      const recoveredSnapshot = exactLiveSnapshot(item.profile, recoveredId)

      if (recoveredSnapshot && !isIdleSnapshot(recoveredSnapshot, storedSessionId)) {
        finishError(item.id, 'stale-runtime')

        return
      }

      await gateway.request('prompt.submit', { session_id: recoveredId, text })
      replacePetLiveSessionRuntime(item.profile, item.sessionId, recoveredId, storedSessionId)
      finish(item.id, 'success')
    } catch (retryError) {
      finishError(
        item.id,
        isSessionNotFoundError(retryError) || isGatewayTimeoutError(retryError) ? 'stale-runtime' : 'rpc-failed'
      )
    }
  }
}

async function steer(
  control: Extract<PetLiveTurnControl, { type: 'action-center-steer' }>,
  dependencies: PetLiveTurnActionDependencies
): Promise<void> {
  const item = resolveCapableItem(control.itemId, 'steer', ['working', 'reviewing'])

  if (!item) {
    return
  }

  const text = textOf(control)

  if (!text) {
    return
  }

  const gateway = gatewayFor(item, dependencies)

  if (!gateway || !item.sessionId) {
    if (!item.sessionId) {
      fail(item.id, 'invalid-item')
    }

    return
  }

  start(item.id)

  try {
    const result = await gateway.request<{ status?: unknown }>('session.steer', {
      session_id: item.sessionId,
      text
    })

    if (result?.status === 'queued') {
      finish(item.id, 'steered')
    } else if (result?.status === 'rejected') {
      finish(item.id, 'steer-rejected')
    } else {
      finishError(item.id, 'rpc-failed')
    }
  } catch (error) {
    finishError(item.id, isSessionNotFoundError(error) ? 'stale-runtime' : 'rpc-failed')
  }
}

function queue(
  control: Extract<PetLiveTurnControl, { type: 'action-center-queue' }>,
  dependencies: PetLiveTurnActionDependencies
): void {
  const item = resolveCapableItem(control.itemId, 'queue', ['working', 'reviewing'])

  if (!item) {
    return
  }

  const text = textOf(control)

  if (!text || !item.sessionId) {
    if (item && !item.sessionId) {
      fail(item.id, 'invalid-item')
    }

    return
  }

  start(item.id)
  const queueKey = profileSessionKey(item.profile, item.storedSessionId ?? item.sessionId)
  const entry = dependencies.enqueuePrompt(queueKey, { text, attachments: [] })

  if (entry) {
    finish(item.id, 'queued')
  } else {
    finishError(item.id, 'invalid-item')
  }
}

async function stop(
  control: Extract<PetLiveTurnControl, { type: 'action-center-stop' }>,
  dependencies: PetLiveTurnActionDependencies
): Promise<void> {
  const item = resolveCapableItem(control.itemId, 'stop', ['working', 'reviewing'])

  if (!item) {
    return
  }

  const gateway = gatewayFor(item, dependencies)

  if (!gateway || !item.sessionId) {
    if (!item.sessionId) {
      fail(item.id, 'invalid-item')
    }

    return
  }

  start(item.id)

  try {
    await gateway.request('session.interrupt', { session_id: item.sessionId })
    clearAllPrompts({ profile: item.profile, sessionId: item.sessionId })
    clearClarifyRequest({ profile: item.profile, sessionId: item.sessionId })
    finish(item.id, 'stopped')
  } catch (error) {
    finishError(item.id, isSessionNotFoundError(error) ? 'stale-runtime' : 'rpc-failed')
  }
}

function acknowledge(control: Extract<PetLiveTurnControl, { type: 'action-center-acknowledge' }>): void {
  const item = resolveCapableItem(control.itemId, 'acknowledge', ['done', 'failed'])

  if (!item || !item.sessionId) {
    if (item && !item.sessionId) {
      fail(item.id, 'invalid-item')
    }

    return
  }

  start(item.id)

  if (acknowledgePetLiveSession(item.profile, item.sessionId)) {
    finish(item.id, 'acknowledged')
  } else {
    finishError(item.id, 'stale-runtime')
  }
}

export function isPetLiveTurnControl(control: PetActionCenterControl): control is PetLiveTurnControl {
  return (
    control.type === 'action-center-submit' ||
    control.type === 'action-center-steer' ||
    control.type === 'action-center-queue' ||
    control.type === 'action-center-stop' ||
    control.type === 'action-center-acknowledge'
  )
}

export async function handlePetLiveTurnControl(
  control: PetLiveTurnControl,
  dependencies: PetLiveTurnActionDependencies
): Promise<void> {
  const item = currentLiveItem(control.itemId)
  const lockKey = item ? actionLockKey(item) : null

  if (lockKey && liveActionLocks.has(lockKey)) {
    return
  }

  if (lockKey) {
    liveActionLocks.add(lockKey)
  }

  try {
    switch (control.type) {
      case 'action-center-submit':
        await send(control, dependencies)

        return

      case 'action-center-steer':
        await steer(control, dependencies)

        return

      case 'action-center-queue':
        queue(control, dependencies)

        return

      case 'action-center-stop':
        await stop(control, dependencies)

        return

      case 'action-center-acknowledge':
        acknowledge(control)
    }
  } finally {
    if (lockKey) {
      liveActionLocks.delete(lockKey)
    }
  }
}
