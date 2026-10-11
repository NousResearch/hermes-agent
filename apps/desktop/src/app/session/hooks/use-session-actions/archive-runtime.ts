import type { MutableRefObject } from 'react'

import type { SessionInfo } from '@/hermes'
import { clearQueuedPrompts } from '@/store/composer-queue'
import { clearSessionControl } from '@/store/session-control'
import { requestForSessionProfile, sessionOwnerRouteFromRow } from '@/store/session-request-router'
import { closeSessionTile, dropSessionState } from '@/store/session-states'

import type { ClientSessionState } from '../../../types'

type GatewayRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

export interface ArchiveRuntimeRefs {
  activeSessionIdRef: MutableRefObject<null | string>
  busyRef: MutableRefObject<boolean>
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
}

export interface ArchivedRuntime {
  busy: boolean
  runtimeId: null | string
}

/** Take this BEFORE `startFreshSessionDraft`: the draft reset clears both the
 *  foreground runtime id and `busyRef`. Same runtime lookup as removeSession. */
export function snapshotArchivedRuntime(
  storedSessionId: string,
  wasSelected: boolean,
  refs: ArchiveRuntimeRefs
): ArchivedRuntime {
  const runtimeId =
    (wasSelected ? refs.activeSessionIdRef.current : null) ??
    refs.runtimeIdByStoredSessionIdRef.current.get(storedSessionId) ??
    null

  const state = runtimeId ? refs.sessionStateByRuntimeIdRef.current.get(runtimeId) : undefined
  const foregroundBusy = wasSelected && refs.busyRef.current

  return { busy: foregroundBusy || Boolean(state?.busy || state?.needsInput || state?.awaitingResponse), runtimeId }
}

/** Drop any tile still showing the archived session, then hand back its
 *  max_concurrent_sessions slot. Archive used to only flip the stored flag, so
 *  the runtime held that slot until the backend exited (#75489). The close
 *  mirrors removeSession, but only for an idle runtime: archiving never ends a
 *  turn that is streaming or waiting on the user. The tile teardown runs before
 *  the first await, so a caller that doesn't wait on the close still sees it. */
export async function releaseArchivedRuntime(
  storedSessionId: string,
  { busy, runtimeId }: ArchivedRuntime,
  owner: { row: SessionInfo | undefined; profile: null | string | undefined },
  requestGateway: GatewayRequest,
  refs: ArchiveRuntimeRefs
): Promise<void> {
  // An archived session is hidden from the sidebar; its tile must go too.
  const tiledRuntimeId = refs.runtimeIdByStoredSessionIdRef.current.get(storedSessionId)
  closeSessionTile(storedSessionId)

  if (tiledRuntimeId) {
    refs.runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
    refs.sessionStateByRuntimeIdRef.current.delete(tiledRuntimeId)
    dropSessionState(tiledRuntimeId)
  }

  if (!runtimeId || busy) {
    return
  }

  // A runtime that is already gone has no slot left to free.
  await requestForSessionProfile(
    sessionOwnerRouteFromRow(owner.row) ?? owner.profile,
    requestGateway,
    'session.close',
    { session_id: runtimeId }
  ).catch(() => undefined)
  clearQueuedPrompts(runtimeId)
  clearSessionControl(runtimeId)
}
