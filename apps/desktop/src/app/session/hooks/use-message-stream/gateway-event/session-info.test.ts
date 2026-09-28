import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createOverviewActions } from '@/app/agents/actions'
import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { AgentRow } from '@/store/agent-overview'
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

import { createPersistedDisplayTranscriptProvenance } from '../../use-session-actions/transcript-provenance'
import { renderMessageStream } from '../test-harness'

import { handleSessionInfoEvent } from './session-info'
import type { GatewayEventContext } from './types'

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
  it.each([
    { connectionId: 'source-b', profile: 'default' },
    { connectionId: 'source-a', profile: 'other' }
  ])('keeps foreground A when an overview reply emits same-id info from $connectionId/$profile', async ownerB => {
    $activeSessionId.set('runtime-a')
    $selectedStoredSessionId.set('collision')
    setCurrentCwd('/source-a')

    const foreground = {
      ...createClientSessionState('collision'),
      ownerRoute: { connectionId: 'source-a', profile: 'default' }
    }

    const stream = renderMessageStream('runtime-a', {
      states: new Map([['runtime-a', foreground]])
    })

    const request = vi.fn(async (connectionId: string, profile: string, method: string) => {
      if (method === 'session.resume') {
        stream.handleEvent({
          connectionId,
          profile,
          type: 'session.info',
          session_id: 'runtime-b',
          payload: { stored_session_id: 'collision', cwd: '/source-b', running: false }
        })

        return { session_id: 'runtime-b' }
      }

      return { status: 'streaming' }
    })

    const actions = createOverviewActions({ request, open: vi.fn() })
    await act(async () => {
      await actions.reply({ ...ownerB, resolvedId: 'collision', canonical: false } as AgentRow, 'continue B')
    })
    expect(request).toHaveBeenLastCalledWith(
      ownerB.connectionId,
      ownerB.profile,
      'prompt.submit',
      { session_id: 'runtime-b', text: 'continue B', profile: ownerB.profile },
      20_000
    )
    expect($activeSessionId.get()).toBe('runtime-a')
    expect($currentCwd.get()).toBe('/source-a')
    expect(stream.states.get('runtime-a')).toBe(foreground)
  })

  it('accepts a proven same-owner rebuild but refuses missing source evidence', () => {
    $activeSessionId.set('runtime-a')
    $selectedStoredSessionId.set('collision')

    const stream = renderMessageStream('runtime-a', {
      states: new Map([
        [
          'runtime-a',
          {
            ...createClientSessionState('collision'),
            ownerRoute: { connectionId: 'source-a', profile: 'default' }
          }
        ]
      ])
    })

    const payload = { stored_session_id: 'collision', cwd: '/rebuilt', running: false }
    act(() => stream.handleEvent({ type: 'session.info', session_id: 'unproven', payload }))
    expect($activeSessionId.get()).toBe('runtime-a')
    act(() =>
      stream.handleEvent({
        connectionId: 'source-a',
        profile: 'default',
        type: 'session.info',
        session_id: 'rebuilt',
        payload
      })
    )
    expect($activeSessionId.get()).toBe('rebuilt')
    expect($currentCwd.get()).toBe('/rebuilt')
  })

  it('preserves untagged primary rebuilds when only legacy transcript provenance is known', () => {
    $activeSessionId.set('runtime-primary')
    $selectedStoredSessionId.set('collision')

    const stream = renderMessageStream('runtime-primary', {
      states: new Map([
        [
          'runtime-primary',
          {
            ...createClientSessionState('collision'),
            transcriptProvenance: createPersistedDisplayTranscriptProvenance({
              storedSessionId: 'collision',
              lineageRootId: null,
              scope: 'default'
            })
          }
        ]
      ])
    })

    act(() =>
      stream.handleEvent({
        type: 'session.info',
        session_id: 'primary-rebuilt',
        payload: { stored_session_id: 'collision', cwd: '/primary-rebuilt' }
      })
    )
    expect($activeSessionId.get()).toBe('primary-rebuilt')
    expect($currentCwd.get()).toBe('/primary-rebuilt')
  })

  beforeEach(() => {
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    setCurrentCwd('')
    setCurrentBranch('')
  })

  afterEach(() => {
    cleanup()
    $activeSessionId.set(null)
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
