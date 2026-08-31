import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $currentCwd,
  $selectedStoredSessionId,
  $workspaceCwdOwner,
  releaseWorkspaceCwdOwner,
  setCurrentCwd
} from '@/store/session'

import { handleSessionInfoEvent } from './session-info'
import type { GatewayEventContext } from './types'

// `_session_info` stamps `stored_session_id: session_key or ""`, so every
// not-yet-persisted session on the gateway emits an UNNAMED session.info that
// still carries a real cwd.
function sessionInfoEvent({
  activeSessionId,
  cwd,
  explicitSid = '',
  storedSessionId = ''
}: {
  activeSessionId: null | string
  cwd: string
  explicitSid?: string
  storedSessionId?: string
}): GatewayEventContext {
  const sessionId = explicitSid || activeSessionId

  return {
    deps: {
      activeGatewayProfile: 'default',
      activeSessionIdRef: { current: activeSessionId },
      hydrateFromStoredSession: vi.fn(),
      lastCwdInfoSessionRef: { current: null },
      queryClient: { invalidateQueries: vi.fn() },
      refreshHermesConfig: vi.fn(),
      scheduleSessionsRefresh: vi.fn(),
      sessionInterrupted: () => false,
      sessionStateByRuntimeIdRef: { current: new Map() },
      updateSessionState: vi.fn(state => state),
      upsertToolCall: vi.fn()
    },
    event: { profile: 'default', session_id: explicitSid, type: 'session.info' },
    explicitSid,
    fromActiveSource: () => true,
    isActiveEvent: !!sessionId && sessionId === activeSessionId,
    occurredAt: Date.now() / 1000,
    payload: { cwd, stored_session_id: storedSessionId },
    scheduleConfigRefresh: vi.fn(),
    sessionId
  } as unknown as GatewayEventContext
}

describe('handleSessionInfoEvent workspace ownership', () => {
  beforeEach(() => {
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    setCurrentCwd('')
  })

  afterEach(() => {
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    setCurrentCwd('')
  })

  // #55831 / the "workspace pane visible with no agent selected" report: with
  // nothing selected an unscoped event is exactly the one that applies, and
  // `broadcast_session_info` re-emits for EVERY live session at once. Adopting
  // those repointed the pane at a stranger's folder and claimed it for the null
  // selection, so the tree/coding rail painted it until the next release
  // un-painted it — a flicker per fan-out, with no agent selected at all.
  it('ignores an unnamed broadcast from a session the pane is not bound to', () => {
    releaseWorkspaceCwdOwner()
    const unowned = $workspaceCwdOwner.get()

    handleSessionInfoEvent(sessionInfoEvent({ activeSessionId: null, cwd: '/repo/someone-elses-worktree' }))

    expect($currentCwd.get()).toBe('')
    expect($workspaceCwdOwner.get()).toBe(unowned)
  })

  it('does not let a fan-out of unnamed broadcasts walk the workspace path', () => {
    const cwds = ['/repo/one', '/repo/two', '/repo/three']

    for (const cwd of cwds) {
      handleSessionInfoEvent(sessionInfoEvent({ activeSessionId: null, cwd }))
    }

    expect($currentCwd.get()).toBe('')
  })

  // The case the absent-id allowance exists for: a lazy session that has not
  // been persisted yet is still the runtime this pane is bound to, so its cwd
  // must be adopted and owned — otherwise the workspace reads as un-owned for
  // the rest of the conversation.
  it('adopts an unnamed session.info from the pane its own runtime', () => {
    $selectedStoredSessionId.set('selected-session')

    handleSessionInfoEvent(
      sessionInfoEvent({ activeSessionId: 'runtime-1', cwd: '/repo/mine', explicitSid: 'runtime-1' })
    )

    expect($currentCwd.get()).toBe('/repo/mine')
    expect($workspaceCwdOwner.get()).toBe('selected-session')
  })
})
