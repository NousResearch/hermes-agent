import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $activeSessionId,
  $currentBranch,
  $currentCwd,
  $selectedStoredSessionId,
  $sessions,
  $workspaceCwdOwner,
  releaseWorkspaceCwdOwner,
  setCurrentBranch,
  setActiveSessionId,
  setCurrentCwd,
  setSessions
} from '@/store/session'
import { clearMainSessionOwner, prepareSessionOwnerRetarget } from '@/store/session-states'

import { handleSessionInfoEvent } from './session-info'
import type { GatewayEventContext } from './types'

// `_session_info` stamps `stored_session_id: session_key or ""`, so every
// not-yet-persisted session on the gateway emits an UNNAMED session.info that
// still carries a real cwd.
function sessionInfoEvent({
  activeSessionId,
  branch,
  connectionId,
  cwd,
  explicitSid = '',
  profile = 'default',
  storedSessionId = ''
}: {
  activeSessionId: null | string
  branch?: string
  connectionId?: string
  cwd: string
  explicitSid?: string
  profile?: string
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
    event: { connectionId, profile, session_id: explicitSid, type: 'session.info' },
    explicitSid,
    fromActiveSource: () => true,
    isActiveEvent: !!sessionId && sessionId === activeSessionId,
    occurredAt: Date.now() / 1000,
    payload: { branch, cwd, stored_session_id: storedSessionId },
    scheduleConfigRefresh: vi.fn(),
    sessionId
  } as unknown as GatewayEventContext
}

describe('handleSessionInfoEvent workspace ownership', () => {
  beforeEach(() => {
    clearMainSessionOwner()
    setActiveSessionId(null)
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    setCurrentCwd('')
    setCurrentBranch('')
    setSessions([])
  })

  afterEach(() => {
    clearMainSessionOwner()
    setActiveSessionId(null)
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    setCurrentCwd('')
    setCurrentBranch('')
    setSessions([])
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

  // #92888: a background Kanban worker's runtime update reaches the pane's
  // active-runtime path while the default Bot Chat stays selected. It names the
  // worker's own stored session and its PR worktree; neither the path nor the
  // branch may move onto the composer, while the selected chat's own update
  // still publishes both.
  it("keeps another session's worktree cwd and branch off the selected chat's composer", () => {
    $selectedStoredSessionId.set('default-bot-chat')
    setCurrentCwd('/repo/main-checkout')
    setCurrentBranch('main')

    handleSessionInfoEvent(
      sessionInfoEvent({
        activeSessionId: 'runtime-1',
        branch: 'kanban/pr-42',
        cwd: '/repo/.worktrees/pr-42',
        explicitSid: 'runtime-1',
        storedSessionId: 'kanban-worker'
      })
    )

    expect($currentCwd.get()).toBe('/repo/main-checkout')
    expect($currentBranch.get()).toBe('main')

    handleSessionInfoEvent(
      sessionInfoEvent({
        activeSessionId: 'runtime-1',
        branch: 'feature/mine',
        cwd: '/repo/main-checkout',
        explicitSid: 'runtime-1',
        storedSessionId: 'default-bot-chat'
      })
    )

    expect($currentBranch.get()).toBe('feature/mine')
    expect($workspaceCwdOwner.get()).toBe('default-bot-chat')
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

  it('carries the gateway source into stored-runtime admission', () => {
    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-a',
      connectionId: 'source-a',
      cwd: '/repo/a',
      explicitSid: 'runtime-a',
      storedSessionId: 'shared-id'
    })

    handleSessionInfoEvent(ctx)

    expect(ctx.deps.updateSessionState).toHaveBeenCalledWith('runtime-a', expect.any(Function), 'shared-id', {
      connectionId: 'source-a',
      profile: 'default'
    })
  })

  it('rejects a stale exact-owner event before it can reclaim the retargeted pane', () => {
    $selectedStoredSessionId.set('shared-id')
    prepareSessionOwnerRetarget('shared-id', { connectionId: 'source-b', profile: 'default' }, true)
    setCurrentCwd('/repo/b')
    const ctx = sessionInfoEvent({
      activeSessionId: null,
      connectionId: 'source-a',
      cwd: '/repo/a',
      explicitSid: 'runtime-a',
      storedSessionId: 'shared-id'
    })

    expect(handleSessionInfoEvent(ctx)).toBe(true)
    expect($activeSessionId.get()).toBeNull()
    expect(ctx.deps.activeSessionIdRef.current).toBeNull()
    expect($currentCwd.get()).toBe('/repo/b')
    expect(ctx.deps.updateSessionState).not.toHaveBeenCalled()
  })

  it('adopts an untagged rebuilt runtime from the active primary source', () => {
    $selectedStoredSessionId.set('shared-id')
    prepareSessionOwnerRetarget('shared-id', { connectionId: 'primary-source', profile: 'default' }, true)
    setActiveSessionId('runtime-old')
    setCurrentCwd('/repo/old')

    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-old',
      cwd: '/repo/rebuilt',
      explicitSid: 'runtime-rebuilt',
      storedSessionId: 'shared-id'
    })

    ctx.deps.sessionStateByRuntimeIdRef.current.set('runtime-old', {
      ...createClientSessionState('shared-id'),
      awaitingResponse: false,
      busy: false,
      streamId: null
    })

    handleSessionInfoEvent(ctx)

    expect($activeSessionId.get()).toBe('runtime-rebuilt')
    expect(ctx.deps.activeSessionIdRef.current).toBe('runtime-rebuilt')
    expect($currentCwd.get()).toBe('/repo/rebuilt')
    expect($workspaceCwdOwner.get()).toBe('shared-id')
  })

  it('adopts a tagged rebuilt runtime from the exact backend target behind a Desktop alias', () => {
    $selectedStoredSessionId.set('shared-id')
    prepareSessionOwnerRetarget(
      'shared-id',
      { connectionId: 'source-a', profile: 'desktop-alias', targetProfile: 'backend-a' },
      true
    )
    setActiveSessionId('runtime-old')
    setCurrentCwd('/repo/old')

    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-old',
      connectionId: 'source-a',
      cwd: '/repo/rebuilt',
      explicitSid: 'runtime-rebuilt',
      profile: 'backend-a',
      storedSessionId: 'shared-id'
    })

    ctx.deps.sessionStateByRuntimeIdRef.current.set('runtime-old', {
      ...createClientSessionState('shared-id'),
      awaitingResponse: false,
      busy: false,
      streamId: null
    })

    handleSessionInfoEvent(ctx)

    expect($activeSessionId.get()).toBe('runtime-rebuilt')
    expect(ctx.deps.activeSessionIdRef.current).toBe('runtime-rebuilt')
    expect($currentCwd.get()).toBe('/repo/rebuilt')
  })

  it("does not let owner A's stale rebuilt-runtime info capture owner B's main pane", () => {
    $selectedStoredSessionId.set('shared-id')
    prepareSessionOwnerRetarget('shared-id', { connectionId: 'source-b', profile: 'default' }, true)
    setActiveSessionId('runtime-b')
    setCurrentCwd('/repo/b')

    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-b',
      connectionId: 'source-a',
      cwd: '/repo/a',
      explicitSid: 'runtime-a-rebuilt',
      storedSessionId: 'shared-id'
    })

    ctx.deps.sessionStateByRuntimeIdRef.current.set('runtime-b', {
      ...createClientSessionState('shared-id'),
      awaitingResponse: false,
      busy: false,
      streamId: null
    })

    handleSessionInfoEvent(ctx)

    expect($activeSessionId.get()).toBe('runtime-b')
    expect(ctx.deps.activeSessionIdRef.current).toBe('runtime-b')
    expect($currentCwd.get()).toBe('/repo/b')
  })

  it("updates only the exact owner's row for a tagged session title", () => {
    setSessions([
      { connection_id: 'source-a', id: 'shared-id', profile: 'profile-a', title: 'Owner A' },
      { connection_id: 'source-b', id: 'shared-id', profile: 'profile-b', title: 'Owner B' }
    ] as never)

    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-b',
      connectionId: 'source-a',
      cwd: '',
      explicitSid: 'runtime-a',
      storedSessionId: 'shared-id'
    })

    ctx.event = { connectionId: 'source-a', profile: 'profile-a', session_id: 'runtime-a', type: 'session.title' }
    ctx.payload = { session_id: 'shared-id', title: 'Updated A' }

    handleSessionInfoEvent(ctx)

    expect($sessions.get().map(session => [session.connection_id, session.title])).toEqual([
      ['source-a', 'Updated A'],
      ['source-b', 'Owner B']
    ])
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
