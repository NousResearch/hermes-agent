import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $activeSessionId,
  $currentCwd,
  $currentReasoningEffort,
  $selectedStoredSessionId,
  $workspaceCwdOwner,
  releaseWorkspaceCwdOwner,
  setCurrentCwd,
  setCurrentReasoningEffort
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
  reasoningEffort,
  storedSessionId = ''
}: {
  activeSessionId: null | string
  cwd: string
  explicitSid?: string
  reasoningEffort?: string
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
    payload: { cwd, stored_session_id: storedSessionId, ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}) },
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
    setCurrentReasoningEffort('')
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

  it('keeps runtime state identity when a heartbeat only restates cached fields', () => {
    const original = {
      ...createClientSessionState('stored-1'),
      cwd: '/repo/mine',
      fast: true,
      model: 'model-1',
      provider: 'provider-1'
    }

    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-1',
      cwd: '/repo/mine',
      explicitSid: 'runtime-1',
      storedSessionId: 'stored-1'
    })

    let next: ClientSessionState | undefined

    ctx.payload = {
      ...ctx.payload,
      fast: true,
      model: 'model-1',
      provider: 'provider-1'
    }
    ctx.deps.sessionStateByRuntimeIdRef.current.set('runtime-1', original)
    ctx.deps.updateSessionState = vi.fn(
      (_sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => {
        const updated = updater(original)
        next = updated

        return updated
      }
    )

    handleSessionInfoEvent(ctx)

    expect(next).toBe(original)
  })

  it('does not copy a session.info reasoning_effort into the global composer draft', () => {
    setCurrentReasoningEffort('')

    handleSessionInfoEvent(
      sessionInfoEvent({ activeSessionId: null, cwd: '/repo', reasoningEffort: 'high' })
    )

    // A session's pinned effort is per-session only; it must never seed the
    // persisted composer draft that a fresh chat reads — that would override the
    // profile's `agent.reasoning_effort` default with whatever the last-resumed
    // session happened to pin.
    expect($currentReasoningEffort.get()).toBe('')
  })
})

// #93942 scenario B: a mid-conversation model/provider switch rebuilds the
// runtime, which then speaks under a NEW session_id. Without the re-bind the
// pane keeps listening on the dead id and every later event of the same
// conversation fails the isActiveEvent gate until a full resume.
describe('handleSessionInfoEvent rebuilt-runtime re-bind', () => {
  function rebuiltRuntimeInfo(storedSessionId: unknown, oldState?: Partial<ClientSessionState>): GatewayEventContext {
    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-old',
      cwd: '',
      explicitSid: 'runtime-new',
      storedSessionId: 'stored-1'
    })

    ctx.payload = { ...ctx.payload, stored_session_id: storedSessionId } as typeof ctx.payload

    if (oldState) {
      ctx.deps.sessionStateByRuntimeIdRef.current.set('runtime-old', {
        ...createClientSessionState('stored-1'),
        ...oldState
      })
    }

    return ctx
  }

  beforeEach(() => {
    $selectedStoredSessionId.set('stored-1')
    $activeSessionId.set('runtime-old')
  })

  afterEach(() => {
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  it('adopts the rebuilt runtime id when its lineage matches the open conversation', () => {
    const ctx = rebuiltRuntimeInfo('stored-1', { busy: false })

    handleSessionInfoEvent(ctx)

    expect($activeSessionId.get()).toBe('runtime-new')
    expect(ctx.deps.activeSessionIdRef.current).toBe('runtime-new')
    expect($selectedStoredSessionId.get()).toBe('stored-1')
  })

  it.each([
    ['busy', { busy: true }],
    ['awaiting a response', { awaitingResponse: true }],
    ['streaming', { streamId: 'stream-1' }]
  ] as const)('refuses to hijack the pane while the old runtime is %s', (_label, oldState) => {
    const ctx = rebuiltRuntimeInfo('stored-1', oldState)

    handleSessionInfoEvent(ctx)

    expect($activeSessionId.get()).toBe('runtime-old')
    expect(ctx.deps.activeSessionIdRef.current).toBe('runtime-old')
  })

  it.each([
    ['an empty', ''],
    ['a missing', undefined],
    ['a different conversation', 'stored-other']
  ])('does not re-bind on %s stored_session_id', (_label, storedSessionId) => {
    const ctx = rebuiltRuntimeInfo(storedSessionId)

    handleSessionInfoEvent(ctx)

    expect($activeSessionId.get()).toBe('runtime-old')
    expect(ctx.deps.activeSessionIdRef.current).toBe('runtime-old')
  })
})
