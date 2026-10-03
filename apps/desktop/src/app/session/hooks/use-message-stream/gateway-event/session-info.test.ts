import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $activeSessionId,
  $currentBranch,
  $currentCwd,
  $selectedStoredSessionId,
  $workspaceCwdOwner,
  releaseWorkspaceCwdOwner,
  setCurrentBranch,
  setCurrentCwd
} from '@/store/session'

import { handleSessionInfoEvent } from './session-info'
import type { GatewayEventContext } from './types'

// The cwd-follow gate's observable: followActiveSessionCwd yanks the sidebar
// into the grouped Projects view (store/projects.ts). Spy on it through a
// partial mock — the handler imports it by name, so a module-object spy set
// AFTER import time can never intercept it.
const { followActiveSessionCwdMock } = vi.hoisted(() => ({ followActiveSessionCwdMock: vi.fn() }))
vi.mock('@/store/projects', async importOriginal => {
  const actual = await importOriginal<typeof import('@/store/projects')>()

  return { ...actual, followActiveSessionCwd: followActiveSessionCwdMock }
})

// `_session_info` stamps `stored_session_id: session_key or ""`, so every
// not-yet-persisted session on the gateway emits an UNNAMED session.info that
// still carries a real cwd.
function sessionInfoEvent({
  activeSessionId,
  branch,
  cwd,
  explicitSid = '',
  storedSessionId = ''
}: {
  activeSessionId: null | string
  branch?: string
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
    payload: { branch, cwd, stored_session_id: storedSessionId },
    scheduleConfigRefresh: vi.fn(),
    sessionId
  } as unknown as GatewayEventContext
}

describe('handleSessionInfoEvent workspace ownership', () => {
  beforeEach(() => {
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    setCurrentCwd('')
    setCurrentBranch('')
  })

  afterEach(() => {
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    setCurrentCwd('')
    setCurrentBranch('')
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

// #72491: the sidebar flipping to the grouped Projects view on restart. The
// cwd-follow gate compares the payload against $currentCwd, which at boot
// holds the remembered/default workspace — not the restored session's own
// cwd. The seed-vs-session reconciliation then reads as a same-session move
// on every heartbeat after the first, and followActiveSessionCwd force-flips
// the sidebar. A real relocation, by contrast, changes the cwd the SAME
// runtime last reported — that (and only that) is followed.
describe('handleSessionInfoEvent cwd-follow gating', () => {
  beforeEach(() => {
    followActiveSessionCwdMock.mockClear()
    $selectedStoredSessionId.set('stored-boot')
    $activeSessionId.set('runtime-boot')
  })

  afterEach(() => {
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
    setCurrentCwd('')
    setCurrentBranch('')
  })

  // The boot-reconcile reproducer (#72491): the FIRST session.info for the
  // restored runtime is skipped by the sameSession guard, but between the
  // first and second info the boot cwd seeding (remembered workspace /
  // ensureDefaultWorkspaceCwd) re-points $currentCwd at the remembered
  // workspace — so the second, same-session heartbeat then looks like a MOVE
  // and flipped the sidebar into Projects view on every relaunch.
  it('does not follow the boot cwd reconciliation against the seeded workspace', () => {
    // The restored session's first info: $currentCwd still holds the
    // remembered workspace from the previous run.
    setCurrentCwd('/remembered/workspace')
    setCurrentBranch('')

    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-boot',
      cwd: '/repo/the-restored-session',
      explicitSid: 'runtime-boot',
      storedSessionId: 'stored-boot'
    })

    // First info: the runtime has no cached cwd yet — learning it is not a move.
    handleSessionInfoEvent(ctx)
    expect(followActiveSessionCwdMock).not.toHaveBeenCalled()
    expect($currentCwd.get()).toBe('/repo/the-restored-session')

    // The boot seeding re-asserts the remembered default workspace between
    // heartbeats (ensureDefaultWorkspaceCwd / seedDefaultCwd, gated on no
    // active session — the restored runtime is not in $activeSessionId yet
    // from the renderer's perspective during restore).
    $activeSessionId.set(null)
    setCurrentCwd('/remembered/workspace')
    $activeSessionId.set('runtime-boot')

    // Second info (the heartbeat/settle edge the old gate tripped on): the
    // payload restates the runtime's own cwd — a reconcile, not a move.
    ctx.deps.sessionStateByRuntimeIdRef.current.set('runtime-boot', {
      ...createClientSessionState('stored-boot'),
      cwd: '/repo/the-restored-session'
    })

    handleSessionInfoEvent(ctx)
    expect(followActiveSessionCwdMock).not.toHaveBeenCalled()

    expect($currentCwd.get()).toBe('/repo/the-restored-session')
    expect($workspaceCwdOwner.get()).toBe('stored-boot')
  })

  it('does not follow a heartbeat restating a cwd the runtime already reported', () => {
    setCurrentCwd('/repo/somewhere-else')

    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-boot',
      cwd: '/repo/the-restored-session',
      explicitSid: 'runtime-boot',
      storedSessionId: 'stored-boot'
    })

    ctx.deps.sessionStateByRuntimeIdRef.current.set('runtime-boot', {
      ...createClientSessionState('stored-boot'),
      cwd: '/repo/the-restored-session'
    })

    handleSessionInfoEvent(ctx)
    handleSessionInfoEvent(ctx)

    expect(followActiveSessionCwdMock).not.toHaveBeenCalled()
  })

  // The feature the gate exists for (#62af32efe7c): the SAME active session's
  // agent relocates (new repo/worktree via the terminal) — its own last
  // reported cwd changes. The sidebar must still follow.
  it('follows a genuine same-session cwd move away from the last reported cwd', () => {
    setCurrentCwd('/repo/old')

    // One ctx shared by both events, exactly like the real hook: the
    // same-session gate reads lastCwdInfoSessionRef, which is per-mount.
    const ctx = sessionInfoEvent({
      activeSessionId: 'runtime-boot',
      cwd: '/repo/old',
      explicitSid: 'runtime-boot',
      storedSessionId: 'stored-boot'
    })

    ctx.deps.sessionStateByRuntimeIdRef.current.set('runtime-boot', {
      ...createClientSessionState('stored-boot'),
      cwd: '/repo/old'
    })

    // Establish the same-session baseline.
    handleSessionInfoEvent(ctx)
    expect(followActiveSessionCwdMock).not.toHaveBeenCalled()

    ctx.payload = { ...ctx.payload, cwd: '/repo/new-worktree' }

    handleSessionInfoEvent(ctx)

    expect(followActiveSessionCwdMock).toHaveBeenCalledTimes(1)
    expect(followActiveSessionCwdMock).toHaveBeenCalledWith('/repo/new-worktree')
    expect($currentCwd.get()).toBe('/repo/new-worktree')
  })
})
