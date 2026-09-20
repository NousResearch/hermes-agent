import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $currentCwd,
  $selectedStoredSessionId,
  $workspaceCwdOwner,
  releaseWorkspaceCwdOwner,
  setCurrentCwd
} from '@/store/session'

import { handleSessionInfoEvent } from './session-info'
import {
  consumePendingModelPick,
  registerPendingModelPick,
  resetPendingModelPicksForTests,
  setPrimaryRuntimeProvider
} from '@/lib/model-pick-pending'
import {
  computeLastUsedSelection,
  getCurrentModelSource,
  noteModelSelectionInUse,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider
} from '@/store/session'
import { $currentModel, $currentProvider } from '@/store/session'
import type { GatewayEventContext } from './types'

// `_session_info` stamps `stored_session_id: session_key or ""`, so every
// not-yet-persisted session on the gateway emits an UNNAMED session.info that
// still carries a real cwd.
function sessionInfoEvent({
  activeSessionId,
  cwd,
  explicitSid = '',
  storedSessionId = '',
  model,
  provider
}: {
  activeSessionId: null | string
  cwd: string
  explicitSid?: string
  storedSessionId?: string
  model?: string
  provider?: string
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
    payload: {
        cwd,
        stored_session_id: storedSessionId,
        ...(model !== undefined ? { model } : {}),
        ...(provider !== undefined ? { provider } : {})
      },
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

// v16/v16c port — the pending model-pick reconciler: a transport-lost switch
// may still have been applied by the backend; the authoritative session.info
// is the judge. Same pair + fresh → commit (sticky always; primary composer
// atoms only when the confirming runtime IS the primary's). Divergent/stale →
// drop without touching the sticky. Broadcast (no explicitSid) never consumes.
describe('pendingModelPick reconcile (ported)', () => {
  beforeEach(() => {
    window.localStorage.removeItem('hermes.desktop.composer.last-model')
    window.localStorage.removeItem('hermes.desktop.composer.last-provider')
    window.localStorage.removeItem('hermes.desktop.composer.last-scope')
    setCurrentModel('glm-5.3')
    setCurrentProvider('zai')
    setCurrentModelSource('manual')
    registerPendingModelPick('runtime-a', 'glm-5.3-flash', 'zai')
  })

  afterEach(() => {
    resetPendingModelPicksForTests()
    window.localStorage.removeItem('hermes.desktop.composer.last-model')
    window.localStorage.removeItem('hermes.desktop.composer.last-provider')
    window.localStorage.removeItem('hermes.desktop.composer.last-scope')
  })

  it('confirms the pending: session.info with the expected pair → primary composer + sticky', () => {
    setPrimaryRuntimeProvider(() => 'runtime-a')

    handleSessionInfoEvent(
      sessionInfoEvent({
        activeSessionId: 'runtime-a',
        cwd: '',
        explicitSid: 'runtime-a',
        model: 'glm-5.3-flash',
        provider: 'zai',
        storedSessionId: 'stored-a'
      })
    )

    expect($currentModel.get()).toBe('glm-5.3-flash')
    expect($currentProvider.get()).toBe('zai')
    expect(consumePendingModelPick('runtime-a')).toBeUndefined()
    const sticky = computeLastUsedSelection()
    expect(sticky.model).toBe('glm-5.3-flash')
    expect(sticky.provider).toBe('zai')
  })

  it('tile confirmation: sticky commits, PRIMARY composer untouched (repro)', () => {
    // The primary pane holds a DRAFT (primary runtime null — promotion only
    // happens at submit); the user picked on a FOCUSED TILE (runtime-tile).
    setPrimaryRuntimeProvider(() => null)
    registerPendingModelPick('runtime-tile', 'glm-5.3-flash', 'zai')

    handleSessionInfoEvent(
      sessionInfoEvent({
        activeSessionId: null,
        cwd: '',
        explicitSid: 'runtime-tile',
        model: 'glm-5.3-flash',
        provider: 'zai',
        storedSessionId: 'stored-tile'
      })
    )

    const sticky = computeLastUsedSelection()
    expect(sticky.model).toBe('glm-5.3-flash')
    expect(sticky.provider).toBe('zai')
    expect(consumePendingModelPick('runtime-tile')).toBeUndefined()
    expect($currentModel.get()).toBe('glm-5.3') // untouched primary
    expect($currentProvider.get()).toBe('zai')
    expect(getCurrentModelSource()).toBe('manual')
  })

  it('divergent pair: pending discarded, sticky/composer untouched', () => {
    setPrimaryRuntimeProvider(() => 'runtime-a')
    noteModelSelectionInUse('glm-5.3-flash', 'zai', '')

    handleSessionInfoEvent(
      sessionInfoEvent({
        activeSessionId: 'runtime-a',
        cwd: '',
        explicitSid: 'runtime-a',
        model: 'glm-5.3',
        provider: 'zai',
        storedSessionId: 'stored-a'
      })
    )

    expect($currentModel.get()).toBe('glm-5.3') // no paint happened
    expect(consumePendingModelPick('runtime-a')).toBeUndefined()
    const sticky = computeLastUsedSelection()
    expect(sticky.model).toBe('glm-5.3-flash')
  })

  it('broadcast session.info WITHOUT explicitSid never consumes the pending', () => {
    setPrimaryRuntimeProvider(() => 'runtime-a')

    handleSessionInfoEvent(
      sessionInfoEvent({
        activeSessionId: 'runtime-a',
        cwd: '',
        explicitSid: '', // unnamed broadcast with ANOTHER session's pair
        model: 'grok-4.5',
        provider: 'xai',
        storedSessionId: 'stored-b'
      })
    )

    expect(consumePendingModelPick('runtime-a')).toBeDefined()
    expect($currentModel.get()).toBe('glm-5.3')
    expect(computeLastUsedSelection().model).not.toBe('grok-4.5')
  })
})
