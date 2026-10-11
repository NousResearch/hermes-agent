import { clearQueuedPrompts } from '@/store/composer-queue'
import { clearSessionControl } from '@/store/session-control'
import { requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { $sessionStates, closeSessionTile, dropSessionState } from '@/store/session-states'

import type { SessionActionsOptions } from './index'

type RuntimeRefs = Pick<SessionActionsOptions, 'activeSessionIdRef' | 'runtimeIdByStoredSessionIdRef'>
type TileRefs = Pick<SessionActionsOptions, 'runtimeIdByStoredSessionIdRef' | 'sessionStateByRuntimeIdRef'>
type ArchiveRefs = RuntimeRefs & TileRefs & Pick<SessionActionsOptions, 'busyRef'>

/** A stored session's live runtime: the foreground's when it is selected, else
 *  the stored→runtime map's. */
export function liveRuntimeIdFor(storedSessionId: string, wasSelected: boolean, refs: RuntimeRefs): null | string {
  return (
    (wasSelected ? refs.activeSessionIdRef.current : null) ??
    refs.runtimeIdByStoredSessionIdRef.current.get(storedSessionId) ??
    null
  )
}

/** Collapse a stored session's tile and evict its mirrored runtime state.
 *  Returns the tile's runtime id, if it had one. */
export function evictTile(storedSessionId: string, refs: TileRefs): string | undefined {
  const tiledRuntimeId = refs.runtimeIdByStoredSessionIdRef.current.get(storedSessionId)
  closeSessionTile(storedSessionId)

  if (tiledRuntimeId) {
    refs.runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
    refs.sessionStateByRuntimeIdRef.current.delete(tiledRuntimeId)
    dropSessionState(tiledRuntimeId)
  }

  return tiledRuntimeId
}

/** Archive used to only flip the stored flag, so the chat's runtime held its
 *  max_concurrent_sessions slot until the backend exited (#75489).
 *
 *  Call this BEFORE `startFreshSessionDraft`, which clears the foreground
 *  runtime id and `busyRef`. It resolves the runtime now and returns the step to
 *  run once the archive lands: drop the tile, then close the runtime the way
 *  removeSession does. Only an idle runtime is closed, so archiving never ends a
 *  turn that is streaming or waiting on the user. The close isn't awaited,
 *  since finalize runs memory commits and plugin hooks. */
export function prepareArchivedRuntimeRelease(storedSessionId: string, wasSelected: boolean, refs: ArchiveRefs) {
  const runtimeId = liveRuntimeIdFor(storedSessionId, wasSelected, refs)
  const state = runtimeId ? $sessionStates.get()[runtimeId] : undefined
  const live = state && (state.busy || state.awaitingResponse || state.needsInput || state.turnLive)
  const closableRuntimeId = (wasSelected && refs.busyRef.current) || live ? null : runtimeId

  return (owner: SessionOwnerScope, requestGateway: SessionActionsOptions['requestGateway']) => {
    evictTile(storedSessionId, refs)

    if (!closableRuntimeId) {
      return
    }

    void requestForSessionProfile(owner, requestGateway, 'session.close', { session_id: closableRuntimeId })
      .catch(error => console.warn('[archive-close]', closableRuntimeId, error))
      .then(() => {
        clearQueuedPrompts(closableRuntimeId)
        clearSessionControl(closableRuntimeId)
      })
  }
}
